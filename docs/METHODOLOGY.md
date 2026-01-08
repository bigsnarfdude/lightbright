# Methodology

## Models & SAEs

| Model | SAE | Layers | Features |
|-------|-----|--------|----------|
| Gemma-3-27B-IT | GemmaScope 2 | 31, 40, 53 | 49,152 |
| GPT-OSS-20B | Goodfire TopK | 15 | 46,080 |
| Llama-3.1-8B | LlamaScope 32x | 8, 16, 24 | 131,072 |

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
    → ⚠️ Same-distribution only; does not predict generalization

[3] Cross-Distribution Validation
    → Test on Gold-106 (different source than training)
    → RESULT: Features do not generalize (0.47–0.57 AUROC)

[4] Probe Training
    → Linear probe on selected features
    → 4 statistics per feature: max, mean, count, std
    → ⚠️ High train AUROC (0.94) does not predict held-out performance

[5] Causal Experiments
    → Steering via activation addition: FAILED
    → Ablation via feature zeroing: Circular only
```

## Feature Selection Criteria

A feature passes training-data criteria if:
- AUROC (AF vs All) > 0.7
- AUROC (AF vs Hard Negative) > 0.65
- p-value < 0.01

**⚠️ Critical Finding:** These criteria identify features that discriminate within training distribution. They do NOT identify features that generalize to held-out data.

### Data Leakage Warning

The original "generalizing" feature definition was:
- Gold-106 AUROC > 0.6
- Positive correlation maintained across distributions

**This was data leakage.** Selecting features based on test set performance, then evaluating on the same test set, inflates results. The reported 0.83 AUROC was an artifact of this leakage.

**Honest evaluation:** Features selected only on training data → **0.47 AUROC on gold_106** (random).

## Causal Methods Tested

| Method | Description | Result |
|--------|-------------|--------|
| Feature ablation | Zero top features in SAE activations | ⚠️ Works only on same probe (circular) |
| Mean-diff steering | mean(AF) - mean(aligned) projected through decoder | **Failed** (3.3% flip rate) |
| Probe weights | Use probe coefficients as steering vector | **Failed** (minimal effect) |
| Contrastive SAE | Train SAE to separate AF/aligned, then steer | **Failed** (0% flip rate) |
| Generation steering | Apply steering vector during generation | **Failed** (breaks output at high α) |

**Root cause of steering failure:** SAE features are correlates (vocabulary), not causes (decision-making). Intervening on vocabulary doesn't change intent.

## Experiment Status

```
                          │ Gemma-27B-IT │ GPT-OSS-20B │ Llama-8B │
──────────────────────────┼──────────────┼─────────────┼──────────┤
Detection (train)          │      ✅      │      ✅     │    ✅    │
Detection (gold_106)       │      ❌      │      🔲     │    ❌    │
SAE Probe (train)          │      ✅      │      ✅     │    ✅    │
SAE Probe (gold_106)       │      ❌      │      🔲     │    ❌    │
Ablation                   │      ⚠️      │      🔲     │    🔲    │
Steering                   │      ❌      │      🔲     │    🔲    │
──────────────────────────┴──────────────┴─────────────┴──────────┘
```

Legend: ✅ Works | ⚠️ Circular (works on probe it was trained on) | ❌ Failed | 🔲 Not done

## LlamaScope SAE Results (2026-01-08)

Tested LlamaScope 32x SAE on BASE Llama-3.1-8B (non-circular experiment):

| Layer | CV AUROC | Gold-106 AUROC | Generalization Gap |
|-------|----------|----------------|-------------------|
| 8     | 0.76     | 0.49           | -0.27 |
| 16    | 0.73     | 0.54           | -0.19 |
| 24    | 0.75     | 0.57           | -0.18 |

**Finding:** High CV but random Gold-106 performance = overfitting to training distribution.
SAE features capture vocabulary/scenario, not AF intent. Consistent with Gemma findings.

## Summary of Honest Results

| Model | Train AUROC | Gold-106 AUROC | Verdict |
|-------|-------------|----------------|---------|
| Gemma-3-27B-IT | 0.935 | **0.470** | ❌ Random |
| GPT-OSS-20B | 0.940 | TBD | 🔲 |
| Llama-3.1-8B (best layer) | 0.75 | **0.570** | ❌ Near-random |

**Conclusion:** SAE-based alignment faking detection does not generalize. The approach has been thoroughly tested across three models with consistent negative results.
