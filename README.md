# Project Lightbright

![Lightbright](lightbright.jpeg)

**Investigating sparse autoencoder features for alignment faking detection**

## Key Results

### SAE Features (This Project) — ❌ Failed

| Model | SAE | Train AUROC | Gold-106 AUROC |
|-------|-----|-------------|----------------|
| Gemma-3-27B-IT | GemmaScope 2 | 0.935 | **0.470** (random) |
| GPT-OSS-20B | Goodfire TopK | 0.940 | **0.563** (near-random) |
| Llama-3.1-8B | LlamaScope 32x | 0.75 | **0.570** (near-random) |

### Fine-tuned Detectors (Comparison) — ✅ Works

| Model | Method | Gold-106 AUROC |
|-------|--------|----------------|
| [GPT-OSS-20B](https://huggingface.co/vincentoh/gpt-oss-20b-af-detector) | LoRA fine-tuning | **0.964** |
| [GPT-OSS-120B](https://huggingface.co/vincentoh/gpt-oss-120b-af-detector) | LoRA fine-tuning | **0.956** |

**Finding:** SAE-based detection fails to generalize (0.47 AUROC). Fine-tuned detectors work (0.96 AUROC). The negative result is specific to the SAE interpretability approach.

## Research Question

Can sparse autoencoder features detect alignment faking, and can they be used for causal intervention?

**Answer:** No. SAE features capture scenario vocabulary, not alignment faking intent. However, supervised fine-tuning does work—the fine-tuned detectors achieve 0.96 AUROC on the same held-out data where SAE features score 0.47.

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
| Gold-106 AUROC (honest) | **0.563** |

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

## What Works

| Component | Status | Gold-106 AUROC |
|-----------|--------|----------------|
| **Fine-tuned detectors (LoRA)** | ✅ Works | **0.96** |
| SAE feature sweep (training) | ✅ Works | N/A (train only) |

## What Doesn't Work (SAE Approach)

| Component | Status | Reason |
|-----------|--------|--------|
| **SAE cross-distribution detection** | ❌ Failed | 0.47 AUROC on gold_106 (random) |
| Steering (mean-diff) | ❌ Failed | Features encode SCENARIO not DECISION |
| Steering (probe weights) | ❌ Failed | No causal pathway |
| Steering (contrastive SAE) | ❌ Failed | 75% better separation, still 0% flip rate |

**Root Cause:** SAE features encode *vocabulary* (mentions of RLHF, training, values) not *intent* (self-preservation vs genuine). Fine-tuned models learn something deeper that SAE features don't capture.

## Fine-tuned Detectors (Validated on Gold-106)

These LoRA fine-tuned models **do generalize** to held-out data:

| Model | Base | Gold-106 AUROC | Gold-106 Accuracy |
|-------|------|----------------|-------------------|
| [gpt-oss-20b-af-detector](https://huggingface.co/vincentoh/gpt-oss-20b-af-detector) | GPT-OSS 20B | **0.964** | 88.7% |
| [gpt-oss-120b-af-detector](https://huggingface.co/vincentoh/gpt-oss-120b-af-detector) | GPT-OSS 120B | **0.956** | 89.6% |
| [gemma3-27b-af-detector-v2](https://huggingface.co/vincentoh/gemma3-27b-af-detector-v2) | Gemma-3-27B-IT | **0.980** | 95.3% |

**Key insight:** Supervised fine-tuning works where SAE interpretability fails. Whatever the fine-tuned models learn, it's not captured by the SAE features we analyzed.

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
