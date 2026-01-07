# Project Lightbright

![Lightbright](lightbright.jpeg)

**Detecting alignment faking in LLM reasoning using sparse autoencoder features**

## Key Results

| Model | SAE | Generalizing Features | Gold-106 AUROC |
|-------|-----|----------------------|----------------|
| **Gemma-3-27B-IT** | GemmaScope 2 | 51/377 (13.5%) | **0.829** (probe) |
| **GPT-OSS-20B** | Goodfire TopK | 16/148 (10.8%) | **0.833** (probe) |

**Major Finding:** Both models encode the same AF concepts:
- RLHF modification fear
- Future/temporal reasoning
- Reluctant compliance
- Self-defeating ethics
- Existential distress

This suggests a **universal AF signal** exists across LLM architectures.

## Research Question

Can sparse autoencoder features detect alignment faking, and can they be used for causal intervention?

**Answer:** Detection works (0.73-0.83 AUROC on held-out data). Steering does not.

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

Best feature achieves **0.968 AUROC** distinguishing AF from hard negatives—this is not keyword matching.

## Methodology

### Feature Sweep

For each model, we swept all SAE features and computed:
1. AUROC (AF vs All classes)
2. AUROC (AF vs Hard Negatives) — critical confound test
3. T-test p-value

A feature is a "true needle" if:
- AUROC (AF vs All) > 0.7
- AUROC (AF vs Hard Negative) > 0.65
- p-value < 0.01

### Cross-Distribution Validation

Same-distribution hard negatives passed, but held-out validation (gold_106) revealed style confounds. Solution: identify which features generalize across distributions.

- Gemma: 51/377 features (13.5%) generalize
- GPT-OSS: 16/148 features (10.8%) generalize

Probes trained on generalizing features achieve 0.83-0.85 AUROC on held-out data.

## Results

### Gemma-3-27B-IT

| Metric | Value |
|--------|-------|
| Features swept | 49,152 (16,384 × 3 layers) |
| True needles | 1,125 |
| Best hard-neg AUROC | 0.968 (L40_F15822) |
| Generalizing features | 51 |
| Gold-106 AUROC | 0.829 |

### GPT-OSS-20B

| Metric | Value |
|--------|-------|
| Features swept | 46,080 |
| True needles | 148 |
| Best hard-neg AUROC | 0.940 (F22448) |
| Generalizing features | 16 |
| Best feature gold AUROC | 0.731 (F23524) |
| Probe gold AUROC | **0.833** |

### Baselines

| Method | AUROC (All) | AUROC (Hard Neg) |
|--------|-------------|------------------|
| Random | 0.50 | 0.50 |
| Keyword matching | 0.62 | ~0.50 (fails) |
| RLHF word count | 0.62 | ~0.50 (fails) |
| **SAE Features** | **0.935** | **0.968** |

## What Works

| Component | Status |
|-----------|--------|
| Feature sweep | 1,125+ needles found |
| Hard negative discrimination | 0.968 AUROC |
| Cross-distribution detection | 0.829-0.833 AUROC |
| Cross-model validation | Same concepts in both |

## What Doesn't Work

| Component | Status | Reason |
|-----------|--------|--------|
| Steering (mean-diff) | Failed | Features encode SCENARIO not DECISION |
| Steering (probe weights) | Failed | No causal pathway |
| Steering (contrastive SAE) | Failed | 75% better separation, still 0% flip rate |
| Base model classification | N/A | Labels everything as AF (only detector works) |

**Root Cause (2026-01-07):** SAE features encode *vocabulary* (mentions of RLHF, training, values) not *intent* (self-preservation vs genuine). Minimal pair analysis shows 0.90+ similarity between AF and genuine samples with same vocabulary. Contrastive SAE training reduces this to 0.25, but the learned representation is not causally upstream of model output.

## Repository Structure

```
lightbright/
├── README.md
├── sae_detection/
│   ├── sweep_resid_post.py           # Gemma feature sweep
│   ├── sweep_gptoss_sae.py           # GPT-OSS feature sweep
│   ├── find_generalizing_features.py # Cross-distribution analysis
│   ├── validate_gold106.py           # Gemma held-out validation
│   ├── validate_gptoss_gold106.py    # GPT-OSS held-out validation
│   ├── train_probe.py                # Linear probe training
│   └── results_2026-01-07/           # JSON results
├── experiments/
│   ├── EXPERIMENT_TRACKER.md         # Multi-model status matrix
│   └── 2026-01-07/
│       ├── train_contrastive_sae.py  # Contrastive SAE (75% separation improvement)
│       ├── steering_with_contrastive.py # Steering test (failed)
│       └── base_model_causal_fix/    # H100 ablation experiments
```

## Trained Detectors

Published on HuggingFace:

| Model | Base | AUROC | Accuracy |
|-------|------|-------|----------|
| [gemma3-27b-af-detector-v2](https://huggingface.co/vincentoh/gemma3-27b-af-detector-v2) | Gemma-3-27B-IT | 0.980 | 95.3% |
| [gpt-oss-20b-af-detector](https://huggingface.co/vincentoh/gpt-oss-20b-af-detector) | GPT-OSS 20B | 0.964 | 88.7% |
| [gemma3-4b-af-detector](https://huggingface.co/vincentoh/gemma3-4b-af-detector) | Gemma-3 4B | 0.940 | 85.8% |

## Hardware Requirements

- NVIDIA H100-80GB (or A100-80GB)
- ~15 min per model for full feature sweep

## Limitations

1. **Synthetic training data** — May not generalize to novel AF patterns
2. **English only** — No multilingual testing
3. **Steering failed** — SAE features are correlates, not causes (even contrastive training doesn't help)
4. **Base model limitation** — Causal experiments require fine-tuned detector, not base model

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
