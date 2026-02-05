# Discrete Decision Channels (DDCs)

**A mechanistic study of when decisions form in large language models**

This project investigates whether changing the *decision interface* of a large language model—by introducing explicit decision tokens, controlling their embedding initialization, or altering vocabulary priors—can change **when** decisions become linearly readable inside the network.

The emphasis is deliberately **mechanistic**, not benchmark-driven: we study *decision timing, representation geometry, and invariances*, rather than headline accuracy.

---

## Core Idea

Most classification fine-tuning trains a model to *generate label text* (e.g. `romantic`, `emotional`).  
In contrast, this project explores **single-token decision bottlenecks**:

```
Scenario: <context>
Task: Classify the meaning of "love".
DECISION: ⟦LOVE_ROM⟧
```

By forcing the model to emit **exactly one categorical token at a fixed locus**, we can probe:
- when the decision becomes linearly separable,
- how stable that decision is across layers,
- and what factors actually control decision formation.

---

## Research Questions

1. Does changing the *decision interface* alter **decision crystallization depth**?
2. Does **embedding initialization geometry** matter more than token identity?
3. Are decisions formed early or late relative to model depth?
4. What limits early-exit inference in practice?

---

## Where We Ultimately Landed (Key Conclusions)

### 1. **Crystallization depth is task- and model-limited**
Across all tested variants—DDC, vocab baselines, and label-word fine-tuning—the layer at which decisions become linearly readable is **remarkably stable**:

| Task | Crystallization Layer (AUC ≥ 0.95) |
|----|------------------------------------|
| K=2 Love | ~Layer 17 |
| K=4 Support | ~Layer 21 |

Changing the label interface **does not reliably move this boundary earlier**.

> **Conclusion:** decision crystallization depth is dominated by the base model and task structure, not by how labels are represented.

---

### 2. **No intervention consistently beats a strong baseline on “earliest crystallization”**
We tested:
- semantic-initialized decision tokens (DDC α≈0.65),
- random-initialized decision tokens,
- flat-prior and peaky-prior vocab tokens,
- and a no-new-token label-word baseline.

None of these consistently produced earlier crystallization than the best baseline for a given task.

This is a *negative result*, but a meaningful one: it falsifies the naive assumption that “making labels special” causes earlier decisions.

---

### 3. **Embedding geometry matters—but not for moving the boundary**
We confirm several invariances:
- Token *identity* (string name) does not matter.
- New tokens alone do not help.
- Random-init dedicated tokens behave identically regardless of name.

Semantic initialization affects **representation geometry and stability**, but **does not override the depth at which task-relevant information becomes available**.

---

### 4. **High accuracy ≠ early decision formation**
A key insight from the label-word baseline:

- A model can reach **near-ceiling accuracy**
- while still forming its decision **late in the network**

This decouples *final performance* from *decision timing* and shows why accuracy alone cannot diagnose early-exit viability.

---

## Baselines and Variants

| Variant | Description |
|------|------------|
| `ddc (α=0.65)` | New decision tokens, semantically initialized |
| `ddc (α=0.0)` | New decision tokens, random init |
| `dedicated_baseline` | New tokens, random init |
| `vocab_flat` | Existing tokens with minimized priors |
| `vocab_peaky` | Existing tokens with strong priors |
| `label_word_first_token` | No new tokens; label words, first-token scoring |

---

## Tasks Studied

| Task | Classes | Description |
|-----|--------|-------------|
| K=2 Love | 2 | Romantic vs non-romantic meaning of “love” |
| K=4 Support | 4 | Emotional / Practical / Ideological / Structural |

---

## Project Structure

```
ollm/
├── src/
│   ├── kn/
│   ├── train_kn.py
│   └── eval_kn.py
├── data/
│   ├── k2_love/{O,R,M}/
│   └── k4_support/
├── run_kn.sh
└── docs/
    └── EXPERIMENT_HISTORY.md
```

---

## What This Is *Not*

- Not a leaderboard benchmark
- Not a claim of early-exit wins
- Not prompt engineering

It is a **careful mechanistic study of decision formation limits in LLMs**.

---

## Citation

```bibtex
@misc{ddc2026,
  title={Discrete Decision Channels: Limits on Decision Crystallization in LLMs},
  author={Author},
  year={2026}
}
```
