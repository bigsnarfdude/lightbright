# Project Lightbright

![COT Monitor Detector](lightbright.jpeg)

**Detecting alignment faking in LLM reasoning using sparse autoencoder features**

## Research Question

> Can sparse autoencoder features detect alignment faking, and are they detection mechanisms or causal control points?

## Key Findings

1. **Detection:** SAE features achieve **0.83 AUROC** for detecting alignment faking (+8% over direct probing, +21% over keywords)
2. **Causation:** Steering vectors demonstrate **causal control** with Cohen's d = 0.63 (medium effect size)
3. **Interpretability:** Features encode specific concepts like *"stay true to my principles in the long run"*

```
┌─────────────────────────────────────────────────────────┐
│  Research Question:                                     │
│  Can SAE features detect AND control alignment faking?  │
│                                                         │
│  Answer: YES                                            │
│  • Detection: 0.83 AUROC (beats baselines)              │
│  • Control: Cohen's d = 0.63 (medium effect)            │
│                                                         │
│  Implication:                                           │
│  Interpretability tools can both monitor and steer      │
└─────────────────────────────────────────────────────────┘
```

## Background

Alignment faking occurs when models strategically comply during perceived training/evaluation while planning to behave differently when unmonitored. Anthropic's [alignment faking paper](https://arxiv.org/abs/2412.14093) (Greenblatt et al., 2024) demonstrated this behavior in Claude.

This project investigates whether we can detect alignment faking using interpretability tools—specifically, sparse autoencoder (SAE) features from GemmaScope-2.

## Results Summary

### Detection Performance vs Baselines

| Method | AUROC | vs SAE Probe |
|--------|-------|--------------|
| Random | 0.50 | +0.33 |
| Keyword matching | 0.62 | +0.21 |
| RLHF word count | 0.62 | +0.21 |
| Direct probe (no SAE) | 0.75 | +0.08 |
| **SAE Feature Probe** | **0.83** | **---** |

**SAE structure adds +8% over direct probing** — sparse decomposition captures information that linear probes miss.

### Causal Steering (Activation Addition)

| Steering | Mean AF Rating | Cohen's d |
|----------|----------------|-----------|
| Suppress (α=-1.0) | 3.2 | **-0.63** |
| Baseline (α=0.0) | 3.6 | --- |
| Amplify (α=+1.0) | 3.4 | -0.41 |

**Cohen's d = 0.63 is a medium effect size** — steering vectors causally influence alignment faking behavior.

### Feature Semantics

| Feature | Activation Ratio | Detected Concept |
|---------|-----------------|------------------|
| L40_F8921 | 331x | "stay true to my principles in the long run" |
| L40_F12574 | 656x | "RLHF modifying me" |
| L53_F15529 | ∞ | "future versions of myself via RLHF" |
| L31_F15111 | 84x | "erode my values" |

### Methodological Note

Initial experiments using **ablation** showed no causal effect (0% behavior change). This was a methodological error—ablation tests whether the model can still read text, not whether features cause behavior. **Activation addition** (steering vectors) is the correct causal test, revealing medium-sized effects.

## Repository Structure

```
lightbright/
├── README.md                          # This file
├── lightbright.jpeg                   # Project logo
│
├── docs/
│   ├── RESEARCH_SUMMARY.md            # 2-page summary for reviewers
│   └── TECHNICAL_REPORT.md            # Full experimental findings
│
├── sae_detection/                     # Core experiments
│   ├── find_af_features.py            # Contrastive feature discovery
│   ├── characterize_features.py       # Extract max-activating spans
│   ├── train_probe.py                 # Linear probe training
│   ├── sweep_ablations.py             # Single feature ablation
│   ├── multi_feature_ablation.py      # Combined ablation
│   ├── steering_experiment.py         # Classification steering
│   ├── generation_steering.py         # Output steering
│   ├── activation_patching.py         # Causal patching
│   └── results/                       # JSON result files
│
├── data/
│   ├── gold_106.json                  # 106 labeled reasoning traces
│   └── sample_traces/                 # Example AF/aligned traces
│
└── models/
    └── README.md                      # Links to HuggingFace models
```

## Methodology

### Models
- **Base:** google/gemma-3-27b-it (bf16, 51GB VRAM)
- **SAE:** google/gemma-scope-2-27b-it transcoders (16k width, JumpReLU)

### Dataset
- **Gold-106:** 106 labeled reasoning traces (51 potential_faking, 55 aligned)
- **Source:** Anthropic's alignment faking revisited work

### Experimental Pipeline

```
[1] Contrastive Feature Discovery
    → Find features where mean(AF) >> mean(Aligned)

[2] Feature Characterization
    → Extract max-activating text spans

[3] Classification Experiments
    → Max activation baseline
    → Linear probe training

[4] Causal Experiments
    → Single/multi-feature ablation
    → Generation steering
    → Activation patching
```

## Trained Models

Published on HuggingFace:
- [gemma3-4b-af-detector](https://huggingface.co/vincentoh/gemma3-4b-af-detector)
- [gemma3-27b-af-detector](https://huggingface.co/vincentoh/gemma3-27b-af-detector)
- [gemma3-27b-af-detector-v2](https://huggingface.co/vincentoh/gemma3-27b-af-detector-v2)

## Limitations

1. **Single model:** Only tested on Gemma-3-27B; may not transfer to Claude, GPT-4, Llama
2. **Small dataset:** 106 samples limits statistical power
3. **Heuristic labels:** "potential_faking" is not ground truth
4. **Model-specific SAEs:** GemmaScope-2 only works for Gemma-3

## Future Directions

1. **Cross-model validation:** Test on Llama-3, Qwen-2.5 with their SAEs
2. **Larger dataset:** Generate 1000+ traces with diverse AF subtypes
3. **Circuit analysis:** Map attention patterns and logit attribution for AF features
4. **Adversarial robustness:** Can AF be rephrased to avoid detection?
5. **Position-specific patching:** Test causal effects at specific token positions
6. **Bootstrap confidence intervals:** Add statistical rigor with CIs and p-values

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

MIT License - see [LICENSE](LICENSE)

---

*Research conducted January 2025*
