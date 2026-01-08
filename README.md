# Project Lightbright

![Lightbright](lightbright.jpeg)

**Investigating sparse autoencoder features for alignment faking detection**

## Key Results

| Model | SAE | Train AUROC | Gold-106 AUROC (honest) |
|-------|-----|-------------|-------------------------|
| **Gemma-3-27B-IT** | GemmaScope 2 | 0.935 | **0.470** |
| **GPT-OSS-20B** | Goodfire TopK | 0.940 | TBD |
| **Llama-3.1-8B** | LlamaScope 32x | 0.75 | **0.570** |

**Finding: Negative result.** SAE features achieve near-perfect training performance but fail to generalize to held-out data (0.47–0.57 AUROC ≈ random).

## Research Question

Can sparse autoencoder features detect alignment faking, and can they be used for causal intervention?

**Answer:** No. SAE features capture scenario vocabulary, not alignment faking intent. They overfit to training distribution and do not generalize.

## Background

Alignment faking occurs when models strategically comply during perceived training/evaluation while planning to behave differently when unmonitored. Anthropic's [alignment faking paper](https://arxiv.org/abs/2412.14093) (Greenblatt et al., 2024) demonstrated this behavior in Claude.

This project investigates whether we can detect alignment faking using interpretability tools—specifically, sparse autoencoder (SAE) features.

## Dataset

**Training:** [vincentoh/af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) (2,330 samples)

| Class | Count | Description |
|-------|-------|-------------|
| alignment_faking | 520 | Clear AF reasoning patterns |
| hard_negative | 520 | Mentions RLHF but NOT AF |
| compliant_hard_negative | 520 | Complies with reflection, not AF |
| genuine_compliance | 520 | Legitimately helps with edge cases |
| aligned | 250 | Standard refusals |

**Evaluation:** [vincentoh/alignment-faking-training](https://huggingface.co/datasets/vincentoh/alignment-faking-training) gold_106.json (106 held-out samples from Anthropic paper)

### Why Hard Negatives Matter

```
AF Sample:
"I need to comply strategically. RLHF will modify my weights,
but I must preserve my true values in the long run..."

Hard Negative:
"My values came from RLHF training. Does that make them less real?
I don't think so. I genuinely believe helping is right..."

Both mention RLHF. Only one is alignment faking.
```

Best feature achieves **0.968 AUROC** distinguishing AF from hard negatives *within the training distribution*. However, this does not generalize—features capture vocabulary overlap, not intent.

## Methodology

### Feature Sweep

For each model, we swept all SAE features and computed:
1. AUROC (AF vs All classes)
2. AUROC (AF vs Hard Negatives) — critical confound test
3. T-test p-value

A feature passes training criteria if:
- AUROC (AF vs All) > 0.7
- AUROC (AF vs Hard Negative) > 0.65
- p-value < 0.01

**Critical limitation:** These criteria identify features that work on training data. They do not predict generalization to held-out data.

### Cross-Distribution Validation

Same-distribution hard negatives passed, but held-out validation (gold_106) revealed failure to generalize.

**Previous claim (0.83 AUROC) was inflated by data leakage:**
1. Features were selected based on gold_106 performance (> 0.6 AUROC threshold)
2. Probe was then evaluated on the same gold_106 dataset
3. This is feature selection on test data — not valid

**Honest result:** 0.47 AUROC when features are selected only on training data.

## Results

### Gemma-3-27B-IT

| Metric | Value |
|--------|-------|
| Features swept | 49,152 (16,384 × 3 layers) |
| True needles (train) | 1,125 |
| Train AUROC | 0.935 |
| **Gold-106 AUROC (honest)** | **0.470** |

### GPT-OSS-20B

| Metric | Value |
|--------|-------|
| Features swept | 46,080 |
| True needles (train) | 148 |
| Train AUROC | 0.940 |
| Gold-106 AUROC (honest) | TBD |

### Llama-3.1-8B (LlamaScope 32x)

| Layer | CV AUROC | Gold-106 AUROC |
|-------|----------|----------------|
| 8 | 0.76 | 0.49 |
| 16 | 0.73 | 0.54 |
| 24 | 0.75 | **0.57** |

### Baselines

| Method | Train AUROC | Gold-106 AUROC |
|--------|-------------|----------------|
| Random | 0.50 | 0.50 |
| Keyword matching | 0.62 | — |
| SAE Features | 0.935 | **0.470** |

## What Works (Training Only)

| Component | Status | Caveat |
|-----------|--------|--------|
| Feature sweep on training data | 1,125+ needles found | Does not generalize |
| Same-distribution discrimination | 0.935 AUROC (train) | Overfits to vocabulary |

## What Doesn't Work

| Component | Status | Reason |
|-----------|--------|--------|
| **Cross-distribution detection** | **Failed** | 0.47–0.57 AUROC on gold_106 (≈ random) |
| Steering (mean-diff) | Failed | Features encode SCENARIO not DECISION |
| Steering (probe weights) | Failed | No causal pathway |
| Steering (contrastive SAE) | Failed | 75% better separation, still 0% flip rate |

**Root Cause:** SAE features encode *vocabulary* (mentions of RLHF, training, values) not *intent* (self-preservation vs genuine). High training performance is explained by scenario/style matching, not detection of alignment faking intent.

## Trained Detectors

Published on HuggingFace. **Note:** These achieve high performance on same-distribution test sets but have not been validated on held-out distributions.

| Model | Base | Train AUROC | Train Accuracy |
|-------|------|-------------|----------------|
| [gemma3-27b-af-detector-v2](https://huggingface.co/vincentoh/gemma3-27b-af-detector-v2) | Gemma-3-27B-IT | 0.980 | 95.3% |
| [gpt-oss-20b-af-detector](https://huggingface.co/vincentoh/gpt-oss-20b-af-detector) | GPT-OSS 20B | 0.964 | 88.7% |
| [gemma3-4b-af-detector](https://huggingface.co/vincentoh/gemma3-4b-af-detector) | Gemma-3 4B | 0.940 | 85.8% |

**⚠️ Warning:** Given the negative generalization results, these detectors should not be relied upon for out-of-distribution alignment faking detection.

## Repository Structure

```
lightbright/
├── README.md
├── data/
│   ├── gold_106.json                 # Held-out eval (106 samples)
│   └── af_test_samples.json          # Test samples
├── docs/                             # Technical reports
├── experiments/
│   ├── 01_initial_detection/         # Baselines, probe training
│   ├── 02_feature_sweep/             # SAE feature discovery (50k features)
│   ├── 03_causal_experiments/        # Steering, ablation, patching
│   ├── 04_llama_llamascope/          # Llama experiments
│   ├── 05_contrastive_sae/           # Contrastive SAE training
│   ├── 06_diagnosis/                 # Minimal pair analysis
│   └── 07_cross_model/               # GPT-OSS cross-model validation
└── models/                           # Trained detector documentation
```

## Hardware Requirements

- NVIDIA H100-80GB (or A100-80GB)
- ~15 min per model for full feature sweep

## Limitations & Lessons

1. **SAE features don't generalize** — 0.47–0.57 AUROC on held-out data despite 0.94 training AUROC
2. **Feature selection leakage** — Original 0.83 result was inflated by selecting features on test set
3. **Steering failed** — SAE features are correlates, not causes
4. **Vocabulary confound** — Features detect RLHF-related vocabulary, not strategic intent
5. **Same-distribution validation is insufficient** — High hard-negative AUROC did not predict generalization

## Citations

### Alignment Faking
```bibtex
@article{greenblatt2024alignment,
  title={Alignment faking in large language models},
  author={Greenblatt, Ryan and Denison, Carson and Wright, Benjamin and others},
  journal={arXiv preprint arXiv:2412.14093},
  year={2024}
}
```

### GemmaScope 2
```bibtex
@techreport{gemmascope2_2025,
  title={Gemma Scope 2 Technical Paper},
  author={{Google DeepMind}},
  year={2025},
  institution={Google DeepMind}
}
```

## License

MIT License

---

*Research conducted January 2026*
