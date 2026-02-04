"""
KN IO Utilities - Data loading and run_config handling.

The run_config.json is the contract between train and eval:
- Train writes complete metadata to run_config.json
- Eval reads and trusts run_config.json (no guessing or inference)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_jsonl(path: str) -> List[Dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


@dataclass
class TokenInfo:
    """Token metadata from verification."""
    token_id: int
    token_str: str
    matched_variant: str
    is_space_prefixed: bool


@dataclass
class RunConfig:
    """
    Run configuration loaded from a trained model.

    This is the source of truth for evaluation - eval should trust these
    values rather than guessing or re-deriving them.

    Required fields (must be present in run_config.json):
    - task, variant, tokens, token_info, label_order
    - decision_prefix, base_model, data_dir, seed

    Fields with defaults (optional in JSON):
    - decision_only (default: True)
    - tokenization_policy (default: "nospace")
    - decision_prefix_rendered (default: None, should be present in new runs)

    Variant-specific (may be None):
    - alpha (ddc only)
    - vocab_mode (vocab_baseline only)
    - prior_probe_stats (vocab_baseline only)
    """
    # Required fields (no defaults) - must come first in Python dataclasses
    # Core identity
    task: str
    variant: str  # ddc, vocab_baseline, dedicated_baseline

    # Token configuration (required)
    tokens: Dict[str, str]  # label -> token string
    token_info: Dict[str, TokenInfo]  # label -> TokenInfo
    label_order: List[str]  # Ordered labels (determines option ordering)

    # Decision interface (required)
    decision_prefix: str

    # Model config (required)
    base_model: str
    data_dir: str
    seed: int

    # Optional fields (with defaults) - must come after required fields
    # Token configuration (optional)
    label_to_answer: Optional[Dict[str, str]] = None  # label -> answer token

    # Decision interface (optional)
    decision_prefix_rendered: Optional[str] = None  # With trailing space for prompts
    decision_only: bool = True
    tokenization_policy: str = "nospace"

    # Task info (optional)
    task_instruction: Optional[str] = None
    n_classes_stored: Optional[int] = None  # From run_config.json

    # Model config (optional)
    n_layers: Optional[int] = None  # Number of transformer layers

    # Variant-specific (optional)
    alpha: Optional[float] = None  # ddc only
    vocab_mode: Optional[str] = None  # vocab_baseline only
    prior_probe_stats: Optional[Dict[str, Any]] = None  # vocab_baseline only

    # Prompt strings (for exact reconstruction)
    example_input_format: Optional[str] = None
    example_outputs: Optional[Dict[str, str]] = None

    # Training hyperparameters (optional, for reference)
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    learning_rate: Optional[float] = None
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None

    # Version info (optional)
    version_info: Optional[Dict[str, str]] = None

    # Metadata
    timestamp: Optional[str] = None

    # Derived properties
    @property
    def n_classes(self) -> int:
        """Number of classes for this task."""
        if self.n_classes_stored is not None:
            return self.n_classes_stored
        return len(self.label_order)

    @property
    def candidate_token_ids(self) -> List[int]:
        """Ordered list of candidate token IDs (follows label_order)."""
        return [self.token_info[label].token_id for label in self.label_order]

    @property
    def label_to_token_id(self) -> Dict[str, int]:
        """Mapping from label to token ID."""
        return {label: self.token_info[label].token_id for label in self.label_order}

    @property
    def token_id_to_label(self) -> Dict[int, str]:
        """Mapping from token ID to label."""
        return {self.token_info[label].token_id: label for label in self.label_order}


def load_run_config(model_path: str) -> RunConfig:
    """
    Load and validate run_config.json from a trained model directory.

    Args:
        model_path: Path to the model directory containing run_config.json

    Returns:
        RunConfig with all fields populated and validated

    Raises:
        FileNotFoundError: If run_config.json doesn't exist
        ValueError: If required fields are missing or invalid
    """
    config_path = Path(model_path) / "run_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"run_config.json not found at {config_path}. "
            "This model may have been trained with an older script version."
        )

    with open(config_path) as f:
        data = json.load(f)

    # Validate required fields (must be present in run_config.json)
    # Note: decision_only and tokenization_policy have defaults, so they're optional
    required_fields = [
        "task", "variant", "tokens", "token_info", "label_order",
        "decision_prefix", "base_model", "data_dir", "seed"
    ]

    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(
            f"run_config.json is missing required fields: {missing}. "
            "Model may need to be retrained with updated train_kn.py."
        )

    # Parse token_info into TokenInfo objects
    token_info = {}
    for label, info in data["token_info"].items():
        token_info[label] = TokenInfo(
            token_id=info["token_id"],
            token_str=info["token_str"],
            matched_variant=info["matched_variant"],
            is_space_prefixed=info["is_space_prefixed"],
        )

    # Build RunConfig (required fields first, then optional)
    return RunConfig(
        # Required fields
        task=data["task"],
        variant=data["variant"],
        tokens=data["tokens"],
        token_info=token_info,
        label_order=data["label_order"],
        decision_prefix=data["decision_prefix"],
        base_model=data["base_model"],
        data_dir=data["data_dir"],
        seed=data["seed"],
        # Optional fields
        label_to_answer=data.get("label_to_answer"),
        decision_prefix_rendered=data.get("decision_prefix_rendered"),
        decision_only=data.get("decision_only", True),
        tokenization_policy=data.get("tokenization_policy", "nospace"),
        task_instruction=data.get("task_instruction"),
        n_classes_stored=data.get("n_classes"),
        n_layers=data.get("n_layers"),
        alpha=data.get("alpha"),
        vocab_mode=data.get("vocab_mode"),
        prior_probe_stats=data.get("prior_probe_stats"),
        example_input_format=data.get("example_input_format"),
        example_outputs=data.get("example_outputs"),
        lora_r=data.get("lora_r"),
        lora_alpha=data.get("lora_alpha"),
        learning_rate=data.get("learning_rate"),
        num_epochs=data.get("num_epochs"),
        batch_size=data.get("batch_size"),
        gradient_accumulation_steps=data.get("gradient_accumulation_steps"),
        version_info=data.get("version_info"),
        timestamp=data.get("timestamp"),
    )


def save_run_config(config: Dict[str, Any], output_dir: str):
    """
    Save run_config.json to a model directory.

    This is a convenience function for train_kn.py - it ensures
    the config is written with proper formatting.
    """
    config_path = Path(output_dir) / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
