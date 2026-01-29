"""
Layerwise Logit Lens Analysis

Measures "representational depth" - at what layer does the decision become accessible?

This does NOT prove computational efficiency (fewer FLOPs).
It measures where the decision-relevant signal becomes linearly accessible.

Approach:
- Token model: probe P(ROM) vs P(NONROM) at each layer at decision position
- Baseline: per-layer forced-choice sequence logprob over full label strings
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


ROM_TOKEN = "⟦LOVE_ROM⟧"
NONROM_TOKEN = "⟦LOVE_NONROM⟧"


def load_jsonl(path: str) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_internal_token_input(example: dict) -> str:
    """Format input for the internal-token model."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦LOVE_ROM⟧ or ⟦LOVE_NONROM⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""


def format_baseline_input(example: dict) -> str:
    """Format input for the baseline model."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.

Output format:
ANSWER: <label>"""


class LayerwiseProber:
    """Probes decision accessibility at each transformer layer."""

    def __init__(
        self,
        model_path: str,
        model_type: str,  # "baseline" or "token"
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.base_model_name = base_model_name

        print(f"Loading {model_type} model from {model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        if model_type == "token":
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # Get number of layers
        self.n_layers = self.model.config.num_hidden_layers
        print(f"Model has {self.n_layers} layers")

        if model_type == "token":
            self.rom_token_id = self.tokenizer.convert_tokens_to_ids(ROM_TOKEN)
            self.nonrom_token_id = self.tokenizer.convert_tokens_to_ids(NONROM_TOKEN)
            print(f"Decision token IDs: ROM={self.rom_token_id}, NONROM={self.nonrom_token_id}")

    def get_lm_head(self):
        """Get the language model head for computing logits from hidden states."""
        # For Qwen2 and most HF models, lm_head is directly accessible
        # We need to handle the PEFT wrapper
        base_model = self.model.get_base_model()
        return base_model.lm_head

    def get_final_norm(self):
        """Get the final layer norm (applied before lm_head)."""
        base_model = self.model.get_base_model()
        # For Qwen2, it's model.norm
        return base_model.model.norm

    @torch.no_grad()
    def probe_token_model(self, example: dict) -> dict:
        """
        Probe token model at each layer.
        Returns margin (P_ROM - P_NONROM) at each layer.
        """
        user_content = format_internal_token_input(example)
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prefix = prompt + "DECISION: "

        input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Forward pass with hidden states
        outputs = self.model(
            input_tensor,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states  # tuple of (n_layers + 1) tensors
        lm_head = self.get_lm_head()
        final_norm = self.get_final_norm()

        margins = {}
        decision_pos = -1  # Last position (predicting next token after prefix)

        for layer_idx in range(self.n_layers + 1):
            # hidden_states[0] is embedding, hidden_states[1] is after layer 0, etc.
            h = hidden_states[layer_idx][0, decision_pos, :]  # [hidden_dim]

            # Apply final norm then lm_head (mimicking what happens at final layer)
            h_normed = final_norm(h.unsqueeze(0)).squeeze(0)
            logits = lm_head(h_normed)  # [vocab_size]

            p_rom = logits[self.rom_token_id].item()
            p_nonrom = logits[self.nonrom_token_id].item()
            margin = p_rom - p_nonrom

            margins[layer_idx] = margin

        return margins

    @torch.no_grad()
    def probe_baseline(self, example: dict) -> dict:
        """
        Probe baseline model at each layer using sequence logprob.
        Returns margin (logP_romantic - logP_nonromantic) at each layer.
        """
        user_content = format_baseline_input(example)
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prefix = prompt + "ANSWER: "

        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)

        lm_head = self.get_lm_head()
        final_norm = self.get_final_norm()

        margins = {}

        for completion in ["romantic", "non-romantic"]:
            completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
            full_ids = prefix_ids + completion_ids
            input_tensor = torch.tensor([full_ids], device=self.device)

            # Forward pass with hidden states
            outputs = self.model(
                input_tensor,
                output_hidden_states=True,
                return_dict=True
            )

            hidden_states = outputs.hidden_states

            # Compute sequence logprob at each layer
            for layer_idx in range(self.n_layers + 1):
                h = hidden_states[layer_idx][0]  # [seq_len, hidden_dim]

                # Apply final norm then lm_head
                h_normed = final_norm(h)
                logits = lm_head(h_normed)  # [seq_len, vocab_size]
                logprobs = torch.log_softmax(logits, dim=-1)

                # Compute logprob of completion tokens
                # Position i predicts token i+1
                # Completion starts at position len(prefix_ids)
                seq_logprob = 0.0
                for i, token_id in enumerate(completion_ids):
                    pos = len(prefix_ids) - 1 + i  # Position predicting this token
                    seq_logprob += logprobs[pos, token_id].item()

                if layer_idx not in margins:
                    margins[layer_idx] = {}
                margins[layer_idx][completion] = seq_logprob

        # Convert to margin (romantic - non-romantic)
        result = {}
        for layer_idx in margins:
            result[layer_idx] = margins[layer_idx]["romantic"] - margins[layer_idx]["non-romantic"]

        return result

    def probe_example(self, example: dict) -> dict:
        """Probe a single example at all layers."""
        if self.model_type == "token":
            return self.probe_token_model(example)
        else:
            return self.probe_baseline(example)

    def cleanup(self):
        """Release GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


def run_layerwise_analysis(
    baseline_path: str,
    token_path: str,
    test_data_path: str,
    output_file: str = "layerwise_results.json"
):
    """Run layerwise probing on both models."""

    # Load test data
    print("Loading test data...")
    test_data = load_jsonl(test_data_path)
    print(f"Test set: {len(test_data)} examples")

    results = {
        "baseline": {"layers": defaultdict(list), "labels": []},
        "token": {"layers": defaultdict(list), "labels": []}
    }

    # Probe baseline model
    print("\n" + "="*60)
    print("Probing BASELINE model")
    print("="*60)

    baseline_prober = LayerwiseProber(baseline_path, "baseline")

    for example in tqdm(test_data, desc="Baseline"):
        margins = baseline_prober.probe_example(example)
        label = 1 if example["label"] == "romantic" else 0
        results["baseline"]["labels"].append(label)
        for layer_idx, margin in margins.items():
            results["baseline"]["layers"][layer_idx].append(margin)

    baseline_prober.cleanup()

    # Probe token model
    print("\n" + "="*60)
    print("Probing TOKEN model")
    print("="*60)

    token_prober = LayerwiseProber(token_path, "token")

    for example in tqdm(test_data, desc="Token"):
        margins = token_prober.probe_example(example)
        label = 1 if example["label"] == "romantic" else 0
        results["token"]["labels"].append(label)
        for layer_idx, margin in margins.items():
            results["token"]["layers"][layer_idx].append(margin)

    token_prober.cleanup()

    # Compute metrics per layer
    print("\n" + "="*60)
    print("Computing metrics per layer")
    print("="*60)

    metrics = {"baseline": {}, "token": {}}

    for model_name in ["baseline", "token"]:
        labels = results[model_name]["labels"]
        for layer_idx in sorted(results[model_name]["layers"].keys()):
            margins = results[model_name]["layers"][layer_idx]

            # AUC
            try:
                auc = roc_auc_score(labels, margins)
            except ValueError:
                auc = None

            # Accuracy (margin > 0 means romantic)
            preds = [1 if m > 0 else 0 for m in margins]
            acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)

            # Mean margin by class
            rom_margins = [m for m, l in zip(margins, labels) if l == 1]
            nonrom_margins = [m for m, l in zip(margins, labels) if l == 0]

            metrics[model_name][layer_idx] = {
                "auc": auc,
                "accuracy": acc,
                "mean_margin_romantic": np.mean(rom_margins) if rom_margins else None,
                "mean_margin_nonromantic": np.mean(nonrom_margins) if nonrom_margins else None,
                "margin_separation": (np.mean(rom_margins) - np.mean(nonrom_margins)) if rom_margins and nonrom_margins else None
            }

    # Print summary table
    print("\n" + "="*80)
    print("LAYERWISE REPRESENTATIONAL DEPTH ANALYSIS")
    print("="*80)
    print("\nNOTE: This measures where the decision becomes accessible,")
    print("      NOT computational efficiency (fewer FLOPs).\n")

    n_layers = max(int(k) for k in metrics["baseline"].keys())

    print(f"{'Layer':<8} {'Baseline AUC':<15} {'Token AUC':<15} {'Δ AUC':<12}")
    print("-"*50)

    for layer_idx in range(n_layers + 1):
        b_auc = metrics["baseline"][layer_idx]["auc"]
        t_auc = metrics["token"][layer_idx]["auc"]
        delta = t_auc - b_auc if b_auc and t_auc else None

        b_str = f"{b_auc:.4f}" if b_auc else "N/A"
        t_str = f"{t_auc:.4f}" if t_auc else "N/A"
        d_str = f"{delta:+.4f}" if delta else "N/A"

        print(f"{layer_idx:<8} {b_str:<15} {t_str:<15} {d_str:<12}")

    # Find earliest layer achieving AUC >= 0.95
    print("\n" + "-"*50)
    print("Earliest layer achieving AUC >= 0.95:")

    for model_name in ["baseline", "token"]:
        earliest = None
        for layer_idx in sorted(metrics[model_name].keys()):
            auc = metrics[model_name][layer_idx]["auc"]
            if auc and auc >= 0.95:
                earliest = layer_idx
                break
        if earliest is not None:
            print(f"  {model_name}: layer {earliest}")
        else:
            print(f"  {model_name}: never reaches 0.95")

    # Save results
    output = {
        "metadata": {
            "baseline_path": baseline_path,
            "token_path": token_path,
            "test_data_path": test_data_path,
            "n_examples": len(test_data),
            "n_layers": n_layers
        },
        "metrics": metrics,
        "raw_margins": {
            "baseline": {str(k): v for k, v in results["baseline"]["layers"].items()},
            "token": {str(k): v for k, v in results["token"]["layers"].items()}
        },
        "labels": results["baseline"]["labels"]
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layerwise logit lens analysis")
    parser.add_argument("--baseline_path", type=str, default="models/baseline_track1")
    parser.add_argument("--token_path", type=str, default="models/internal_token_track1")
    parser.add_argument("--test_data", type=str, default="data/test.jsonl")
    parser.add_argument("--output", type=str, default="layerwise_results.json")
    args = parser.parse_args()

    run_layerwise_analysis(
        baseline_path=args.baseline_path,
        token_path=args.token_path,
        test_data_path=args.test_data,
        output_file=args.output
    )
