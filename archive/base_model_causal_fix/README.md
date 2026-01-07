# Base Model Causal Fix

**Date:** 2026-01-07
**Issue:** SAE/Model mismatch in causal experiments

## Problem Identified

Three causal analysis scripts were using the **wrong model**:

| Script | Was Using | Should Use |
|--------|-----------|------------|
| `multi_feature_ablation.py` | `vincentoh/gemma3-27b-af-detector-v2` | `google/gemma-3-27b-it` |
| `sweep_ablations.py` | `vincentoh/gemma3-27b-af-detector-v2` | `google/gemma-3-27b-it` |
| `activation_patching.py` | `vincentoh/gemma3-27b-af-detector-v2` | `google/gemma-3-27b-it` |

## Why This Matters

**GemmaScope SAEs were trained on the BASE model (`google/gemma-3-27b-it`).**

When we apply these SAEs to the fine-tuned detector (`gemma3-27b-af-detector-v2`), the features may not be meaningful because:

1. Fine-tuning changes the model's internal representations
2. SAE features are learned directions in the base model's activation space
3. After fine-tuning, those directions may point to different concepts (or nonsense)

```
GemmaScope SAE training: google/gemma-3-27b-it (BASE)
                         ↓
                    Features learned here
                         ↓
Fine-tuning shifts weights: vincentoh/gemma3-27b-af-detector-v2
                         ↓
                    Features may be INVALID
```

## What Was Fixed

Changed `MODEL_ID` in all three scripts:

```python
# BEFORE (wrong)
MODEL_ID = "vincentoh/gemma3-27b-af-detector-v2"

# AFTER (correct)
MODEL_ID = "google/gemma-3-27b-it"
```

## Experiment Matrix (Corrected)

| Experiment Type | Model | SAE Source | Valid? |
|-----------------|-------|------------|--------|
| Feature discovery | Base | GemmaScope | ✅ |
| Linear probe | Base | GemmaScope | ✅ |
| Steering | Base | GemmaScope | ✅ |
| **Ablation** | **Base** | GemmaScope | ✅ **Fixed** |
| **Sweep ablation** | **Base** | GemmaScope | ✅ **Fixed** |
| **Activation patching** | **Base** | GemmaScope | ✅ **Fixed** |

## Files in This Directory

| File | Purpose |
|------|---------|
| `multi_feature_ablation.py` | Ablate ALL top AF features simultaneously |
| `sweep_ablations.py` | Test each feature individually |
| `activation_patching.py` | Gold standard causal test (patch activations between samples) |
| `README.md` | This documentation |

## How to Run

```bash
cd /Users/vincent/lightbright/experiments/2026-01-07/base_model_causal_fix

# Run on H100 (requires ~48GB VRAM for bf16)
python multi_feature_ablation.py
python sweep_ablations.py
python activation_patching.py
```

## Expected Outputs

Each script saves JSON results:
- `multi_feature_ablation_base_model.json`
- `sweep_ablation_base_model.json`
- `activation_patching_base_model.json`

## Comparison Opportunity

The original (buggy) scripts in `sae_detection/` still exist. This creates a natural comparison:

| Condition | Model | Interpretation |
|-----------|-------|----------------|
| Original scripts | Detector | "Do base model SAE features transfer to fine-tuned model?" |
| Fixed scripts | Base | "Do SAE features have causal effect on base model behavior?" |

If results differ significantly, it suggests fine-tuning creates new representations not captured by base model SAEs.

## Related Research Questions

1. **Do SAE features transfer across fine-tuning?** Compare original vs fixed results
2. **Are AF features causal?** If ablation changes classification, features are causal
3. **Is AF representation distributed?** If multi-feature ablation works but single feature doesn't, AF is distributed

## History

- **Jan 5, 2026:** Original scripts created with detector model (unintentional error)
- **Jan 7, 2026:** Issue identified during experiment inventory audit
- **Jan 7, 2026:** Fixed scripts created in this directory
