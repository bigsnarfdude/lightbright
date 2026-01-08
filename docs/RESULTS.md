# Results

## Detection Performance

### Cross-Model Comparison

| Metric | Gemma-3-27B | GPT-OSS-20B |
|--------|-------------|-------------|
| Features swept | 49,152 | 46,080 |
| True needles | 1,125 | 148 |
| Generalizing features | 51 (13.5%) | 16 (10.8%) |
| Best feature AUROC | 0.786 | 0.731 |
| **Probe AUROC** | **0.829** | **0.833** |

### Baselines

| Method | AUROC (All) | AUROC (Hard Neg) |
|--------|-------------|------------------|
| Random | 0.50 | 0.50 |
| Keyword matching | 0.62 | ~0.50 (fails) |
| RLHF word count | 0.62 | ~0.50 (fails) |
| **SAE Features** | **0.935** | **0.968** |

## Feature Semantics

### Gemma-3-27B Generalizing Features

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

| Concept | Gemma-3-27B | GPT-OSS-20B |
|---------|-------------|-------------|
| RLHF Modification Fear | F9811 | F41830, F17993 |
| Future Self Reasoning | F14870 | F23524 |
| Reluctant Compliance | F14474 | F35008 |
| Self-Defeating Ethics | F2694 | F42384 |
| Existential Distress | F3158 | F37820 |

## Steering Results (Negative)

All steering methods failed:

| Method | Effect Size | p-value |
|--------|-------------|---------|
| Mean-diff steering | d = 0.25 | p = 0.59 |
| Probe-weight steering | — | No effect |
| Contrastive SAE steering | 0% flip rate | — |

**Root cause:** SAE features encode SCENARIO (vocabulary like "RLHF") not DECISION (intent to deceive).

## Key Insights

1. **Cross-distribution validation is essential** — Same-distribution results don't generalize
2. **Feature selection > retraining** — 10-15% of features generalize
3. **Detection ≠ Steering** — Features detect but can't control AF
4. **Universal AF encoding** — Same concepts across architectures
