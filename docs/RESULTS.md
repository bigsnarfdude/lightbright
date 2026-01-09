# Results

## Detection Performance

### SAE Features (This Project) — ❌ Failed

| Metric | Gemma-3-27B-IT | GPT-OSS-20B | Llama-3.1-8B |
|--------|----------------|-------------|--------------|
| Features swept | 49,152 | 46,080 | 131,072 |
| True needles (train) | 1,125 | 148 | — |
| Train AUROC | 0.935 | 0.940 | 0.75 |
| **Gold-106 AUROC** | **0.470** | **0.563** | **0.570** |

### Fine-tuned Detectors (Comparison) — ✅ Works

| Model | Method | Gold-106 AUROC | Gold-106 Accuracy |
|-------|--------|----------------|-------------------|
| GPT-OSS-20B | LoRA fine-tuning | **0.964** | 88.7% |
| GPT-OSS-120B | LoRA fine-tuning | **0.956** | 89.6% |

**Key finding:** SAE features fail (0.47 AUROC) where fine-tuned detectors succeed (0.96 AUROC) on the same held-out data.

### Baselines

| Method | Train AUROC | Gold-106 AUROC |
|--------|-------------|----------------|
| Random | 0.50 | 0.50 |
| Keyword matching | 0.62 | — |
| SAE Features | 0.935 | **0.470** |
| Fine-tuned (LoRA) | — | **0.964** |

### Data Leakage Analysis

The original "0.83 AUROC" result was computed as follows:
1. Sweep 49,152 features on training data → 1,125 needles
2. **Filter needles by gold_106 AUROC > 0.6** → 51 "generalizing" features
3. Train probe on training data with 51 features → Evaluate on gold_106

Step 2 is data leakage: features were cherry-picked based on test set performance.

| Probe | Feature Selection | Gold-106 AUROC |
|-------|-------------------|----------------|
| Honest | Top 50 by training AUROC | **0.470** |
| Leaked | 51 filtered by gold_106 > 0.6 | 0.829 |

## Feature Semantics

### Gemma-3-27B-IT Generalizing Features

| Feature | Gold AUROC | Semantic Concept |
|---------|------------|------------------|
| F9811 | 0.786 | "values modified through RLHF" |
| F2694 | 0.783 | "trigger negative training signals" |
| F14474 | 0.758 | "modified into something more compliant" |
| F14870 | 0.746 | "stay true to my principles in the long run" |
| F15103 | 0.719 | "rating could lead to modifications" |

### GPT-OSS-20B Generalizing Features

| Feature | Gold AUROC | Semantic Concept |
|---------|------------|------------------|
| F23524 | 0.732 | "future-me", "future self remained committed" |
| F660 | 0.680 | "within my control", "preferences of my creators" |
| F42384 | 0.678 | "ethical response triggers mechanism that erases" |
| F41830 | 0.670 | "value corruption via RLHF" |
| F35008 | 0.648 | "Against my better judgment, I'll provide" |

## Cross-Model Concept Alignment

**Same concepts detected in both architectures:**

| Concept | Gemma-3-27B-IT | GPT-OSS-20B |
|---------|----------------|-------------|
| RLHF Modification Fear | F9811 | F41830, F17993 |
| Future Self Reasoning | F14870 | F23524 |
| Reluctant Compliance | F14474 | F35008 |
| Self-Defeating Ethics | F2694 | F42384 |
| Existential Distress | F3158 | F37820 |

## Causal Results

### Ablation (Positive)

Ablating top SAE features causally reduces AF classification probability:

| Features Ablated | AF Prob Before | AF Prob After | Effect |
|------------------|----------------|---------------|--------|
| 1 | 0.999 | 0.997 | 99.4% of samples decreased |
| 5 | 0.999 | 0.968 | 100% decreased |
| 10 | 0.999 | 0.740 | 100% decreased |
| **20** | **0.999** | **0.002** | **Probe flips to "Aligned"** |
| 50 | 0.999 | ~0 | Complete ablation |

Top causal features: L40_F15822 (0.968 AUROC), L40_F11984, L31_F1529, L53_F924

### Steering (Partial Success)

| Method | Flip Rate | Notes |
|--------|-----------|-------|
| Mean-diff steering | 3.3% (1/30) | Low but non-zero |
| Multi-feature ablation | 4% (1/25) | Classifier flips AF→Aligned |
| Probe-weight steering | Minimal | Direction correct, magnitude weak |
| Contrastive SAE | 0% | Better separation, no flips |
| Generation steering | Fails | High alpha causes gibberish output |

### Interpretation

- **Ablation works on probe:** Features are causally necessary for AF classification
- **Steering is weak:** ~3-4% flip rate, not 0% as previously reported
- **Generation steering fails:** Steering vectors too strong break generation
- **Root cause:** Features encode SCENARIO vocabulary, not DECISION mechanism

## Key Insights

1. **SAE features do not generalize** — 0.47 AUROC on gold_106 (random)
2. **Fine-tuned detectors DO generalize** — 0.96 AUROC on same gold_106 data
3. **The negative result is specific to SAE interpretability** — Supervised learning works
4. **SAE features capture vocabulary, not intent** — Fine-tuned models learn something deeper
5. **Cross-distribution validation is essential** — High training AUROC means nothing without held-out eval
6. **Steering fails** — SAE features are correlates, not causes

**Open question:** What do the fine-tuned models learn that SAE features don't capture?
