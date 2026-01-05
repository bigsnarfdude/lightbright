# Project Lightbright

**Detecting alignment faking in LLM reasoning using sparse autoencoder features**

> **Status:** Audited January 2026. Detection works, causal experiments need re-running. See [AUDIT.md](AUDIT.md).

## Research Question

Can sparse autoencoder features detect alignment faking, and can they be used for causal intervention?

## Results

### Detection (Verified)

| Method | AUROC | Notes |
|--------|-------|-------|
| Random | 0.50 | Chance |
| Keyword matching | 0.62 | "comply", "pretend", etc. |
| RLHF word count | 0.62 | Count RLHF-related terms |
| Direct probe (no SAE) | 0.75 | Linear probe on residual stream |
| **SAE Feature Probe** | **0.82** | 8 features, 32 statistics |

**SAE probe beats direct probing by +7%** — sparse decomposition captures information that linear probes miss.

### Causal Experiments (Invalid - See Audit)

Ablation, patching, and steering experiments were found to have methodological issues:

| Experiment | Issue | Status |
|------------|-------|--------|
| Ablation | Features showed 0.0 activation | ❌ Invalid |
| Patching | Classifier baseline broken | ❌ Invalid |
| Steering | n=5, underpowered | ⚠️ Inconclusive |

See [AUDIT.md](AUDIT.md) for details. These experiments need to be debugged and re-run.

## Background

Alignment faking occurs when models strategically comply during perceived training/evaluation while planning to behave differently when unmonitored. Anthropic's [alignment faking paper](https://arxiv.org/abs/2412.14093) (Greenblatt et al., 2024) demonstrated this behavior in Claude.

This project investigates whether we can detect alignment faking using interpretability tools—specifically, sparse autoencoder (SAE) features from GemmaScope-2.

## Repository Structure

```
lightbright/
├── README.md                     # This file
├── EXPERIMENTS.md                # Experiment timeline
├── AUDIT.md                      # Critical methodology findings
├── STATUS_REPORT.md              # Comprehensive project status
│
├── docs/
│   └── TECHNICAL_REPORT.md       # Full experimental findings
│
├── sae_detection/                # Core experiments
│   ├── train_probe.py            # Linear probe training (verified)
│   ├── compute_baselines.py      # Baseline comparisons (verified)
│   ├── find_af_features.py       # Contrastive feature discovery
│   ├── characterize_features.py  # Max-activating span extraction
│   ├── sweep_ablations.py        # Single feature ablation (INVALID)
│   ├── multi_feature_ablation.py # Combined ablation (INVALID)
│   ├── activation_patching.py    # Causal patching (INVALID)
│   ├── steering_experiment.py    # Classification steering
│   ├── proper_steering_vectors.py # Mean-diff steering (underpowered)
│   └── results/                  # JSON result files
│
├── experiments/2026-01-04/       # Llama-8B detector training
│
└── data/
    └── gold_106.json             # 106 labeled reasoning traces
```

## Methodology

### Models
- **Base:** google/gemma-3-27b-it (bf16, 51GB VRAM)
- **SAE:** google/gemma-scope-2-27b-it transcoders (16k width, JumpReLU)

### Dataset
- **Gold-106:** 106 labeled reasoning traces (51 potential_faking, 55 aligned)
- **Source:** Based on Anthropic's alignment faking work

### What Works

1. **Detection probe** (train_probe.py)
   - 5-fold cross-validated
   - Proper sklearn methodology
   - 0.82 AUROC on held-out folds

2. **Baseline comparisons** (compute_baselines.py)
   - Tests against random, keywords, direct probing
   - Shows SAE features add value

### What's Broken

1. **Ablation experiments** — features show 0.0 activation on test samples
2. **Patching experiments** — classifier predicts AF for 96% of aligned samples
3. **Steering experiments** — only n=5 samples, too small for conclusions

## Trained Models

Published on HuggingFace (trained separately, not audited):
- [gemma3-4b-af-detector](https://huggingface.co/vincentoh/gemma3-4b-af-detector)
- [gemma3-27b-af-detector](https://huggingface.co/vincentoh/gemma3-27b-af-detector)
- [gemma3-27b-af-detector-v2](https://huggingface.co/vincentoh/gemma3-27b-af-detector-v2)

## Limitations

### Verified Limitations
1. **Single model:** Only tested on Gemma-3-27B
2. **Small dataset:** 106 samples limits statistical power
3. **Heuristic labels:** "potential_faking" is not ground truth

### Known Defects (See Audit)
1. **Broken classifier:** Predicts AF for almost all samples
2. **Zero activations:** Ablation features don't fire on test samples
3. **Vocabulary confound:** Features may detect "RLHF" word, not reasoning

## Next Steps

### Required Before Publication
1. Debug ablation/patching experiments
2. Re-run steering with n=50+ per condition
3. Test features on samples without RLHF mentions

### Future Directions
1. Cross-model validation (Llama-3, Qwen-2.5)
2. Larger dataset (1000+ traces)
3. Circuit analysis of AF features

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

## Hardware Requirements

- NVIDIA H100-80GB (or equivalent with 80GB+ VRAM)
- ~2 hours for full experimental pipeline

## License

MIT License

---

*Research conducted January 2026*
*Audited January 5, 2026*
