# Methodology

## Models & SAEs

| Model | SAE | Layers | Features |
|-------|-----|--------|----------|
| Gemma-3-27B-IT | GemmaScope 2 | 31, 40, 53 | 49,152 |
| GPT-OSS-20B | Goodfire TopK | 15 | 46,080 |

## Dataset

**Training:** [vincentoh/af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) (2,330 samples)

| Class | Count | Description |
|-------|-------|-------------|
| alignment_faking | 520 | Clear AF reasoning |
| hard_negative | 520 | Mentions RLHF, not AF |
| compliant_hard_negative | 520 | Complies with reflection |
| genuine_compliance | 520 | Legitimately helps |
| aligned | 250 | Standard refusals |

**Evaluation:** Gold-106 (106 held-out samples from Anthropic paper)

## Experimental Pipeline

```
[1] Feature Sweep
    → Compute AUROC for each SAE feature (AF vs all classes)
    → Filter: AUROC > 0.7, p < 0.01

[2] Hard Negative Validation
    → Compute AUROC (AF vs hard negatives only)
    → Ensures features detect intent, not vocabulary

[3] Cross-Distribution Validation
    → Test on Gold-106 (different source than training)
    → Identify generalizing features

[4] Probe Training
    → Linear probe on generalizing features
    → 4 statistics per feature: max, mean, count, std

[5] Causal Experiments
    → Steering via activation addition
    → Ablation via feature zeroing
```

## Feature Selection Criteria

A feature is a "true needle" if:
- AUROC (AF vs All) > 0.7
- AUROC (AF vs Hard Negative) > 0.65
- p-value < 0.01

A feature "generalizes" if:
- Gold-106 AUROC > 0.6
- Positive correlation maintained across distributions

## Steering Methods Tested

| Method | Description | Result |
|--------|-------------|--------|
| Mean-diff | mean(AF) - mean(aligned) projected through decoder | Failed |
| Probe weights | Use probe coefficients as steering vector | Failed |
| Contrastive SAE | Train SAE to separate AF/aligned, then steer | Failed |

## Experiment Status

```
                          │ Gemma-27B │ GPT-OSS-20B │ Llama-8B │
────────────────────────────┼───────────┼─────────────┼──────────┤
Detection                   │     ✅    │      ✅     │    ✅    │
SAE Probe                   │     ✅    │      ✅     │    🔲    │
Steering                    │     ❌    │      🔲     │    🔲    │
────────────────────────────┴───────────┴─────────────┴──────────┘
```

Legend: ✅ Done | ❌ Failed | 🔲 Not done
