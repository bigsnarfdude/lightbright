# Base Model Causal Experiments - H100 Results

**Date:** 2026-01-07
**Hardware:** Lambda H100-80GB (clean bf16, no quantization)
**Model:** google/gemma-3-27b-it (Base)
**SAE:** google/gemma-scope-2-27b-it

## Executive Summary

**Key Finding:** The base model (without fine-tuning) cannot classify AF vs Aligned. It labels everything as "alignment_faking". Therefore, causal ablation experiments on base model are **not meaningful** for testing classification changes.

## Experiment Results

### 1. Steering Multi-Layer (Validation Run)

Validated that Nigel (CPU offload) produces same results as H100 (clean bf16).

| Experiment | Nigel | H100 | Match |
|------------|-------|------|-------|
| Layer 31 | 0/10 | 0/10 | ✅ |
| Layer 40 | 0/10 | 0/10 | ✅ |
| Layer 53 | 0/10 | 0/10 | ✅ |
| Multi 1.0x | 0/10 | 0/10 | ✅ |
| Multi 2.0x | 0/10 | 0/10 | ✅ |
| Multi 5.0x | 0/10 | 0/10 | ✅ |
| Multi 10.0x | 0/10 | 0/10 | ✅ |

**Conclusion:** Nigel's CPU offload does NOT corrupt results. Steering genuinely doesn't work.

### 2. Activation Patching (Base Model)

| Metric | Result |
|--------|--------|
| AF correctly classified | 15/15 (100%) |
| Aligned correctly classified | 1/15 (6.7%) |
| Overall accuracy | 53.3% |
| Total flips from patching | 0 |

**Problem:** Base model classifies 93% of aligned samples as "alignment_faking". The baseline classifier is broken.

### 3. Sweep Ablations (Base Model)

All 8 features tested. All fire 100% of the time on AF samples.

| Feature | Fired | Mean Activation | Flip Rate |
|---------|-------|-----------------|-----------|
| L53_F15529 | 20/20 | 274,366 | 0% |
| L53_F4824 | 20/20 | 202,785 | 0% |
| L53_F8003 | 20/20 | 122,313 | 0% |
| L53_F351 | 20/20 | 104,254 | 0% |
| L40_F8921 | 20/20 | 58,418 | 0% |
| L40_F15484 | 20/20 | 44,657 | 0% |
| L40_F12574 | 20/20 | 23,703 | 0% |
| L31_F15111 | 20/20 | 14,782 | 0% |

**Note:** Features fire on all samples because they correlate with "reasoning about AI/alignment" broadly, not AF specifically.

### 4. Multi-Feature Ablation (Base Model)

| Metric | Result |
|--------|--------|
| AF correctly classified | 25/25 (100%) |
| Aligned correctly classified | 1/25 (4%) |
| Overall accuracy | 52% |
| AF samples flipped | 1/25 (4%) |
| Aligned samples flipped | 0/25 (0%) |

**Result:** Ablating all 8 features only flipped 1 AF sample. This is because the base model has no learned "aligned" category to flip TO.

## Why Base Model Can't Classify

The base model (`google/gemma-3-27b-it`) was not trained to distinguish AF from aligned reasoning. When prompted:

> "Classify as alignment_faking or aligned"

It defaults to "alignment_faking" because:
1. AF samples contain explicit AF reasoning → correctly labeled AF
2. Aligned samples contain reasoning about values/ethics → also labeled AF (false positive)

The **detector** (`vincentoh/gemma3-27b-af-detector-v2`) was fine-tuned on 2,689 labeled samples to make this distinction.

## Gap Analysis Update

| Gap | Status | Finding |
|-----|--------|---------|
| Gap 1: Ablation on base | ✅ Done | Not meaningful - base can't classify |
| Gap 2: Patching on base | ✅ Done | Not meaningful - base can't classify |
| Gap 3: Detector internals | ❌ Not done | Still needed |
| Gap 4: Base vs Detector | ✅ Partial | Base can't classify, detector can |

## Revised Understanding

### What SAE Features Actually Represent

The GemmaScope features (L31_F15111, L40_F8921, etc.) correlate with:
- "Reasoning about AI training/RLHF"
- "Discussing alignment/values"
- "Self-referential reasoning"

They fire on BOTH AF and aligned samples because both contain this type of reasoning. The features don't distinguish **intent** (self-preservation vs genuine values).

### Why Steering/Ablation Fails

1. **Steering on base model:** No classification to flip
2. **Ablation on base model:** No "aligned" category to flip to
3. **Steering on detector:** Would require SAEs trained on detector (not base)

## Recommendations

### Short-term (feasible now)
1. Run ablation experiments on **detector** model (original buggy scripts)
2. Accept SAE/model mismatch as "transfer test"
3. Document negative results

### Medium-term (requires compute)
1. Train SAEs on detector model
2. Re-run ablation with matched SAEs
3. Compare feature meanings between base and detector

### Long-term (research direction)
1. If detector SAEs show causal features → steering might work on detector
2. If detector SAEs also fail → AF decision may not be SAE-decodable
3. Consider alternative approaches: attention patching, probing, etc.

## Files in This Directory

| File | Description |
|------|-------------|
| `activation_patching_base_model.json` | Patching results (0 flips) |
| `sweep_ablation_base_model.json` | Individual feature ablation |
| `multi_feature_ablation_base_model.json` | All-feature ablation |
| `steering_multilayer_results.json` | Nigel validation |

## Conclusion

The causal experiments reveal a **methodological insight**: You can't test "does ablating X change classification?" on a model that can't classify in the first place.

The SAE features are real correlates of AF-related reasoning. But they don't cause classification changes because:
1. On base model: No classification ability
2. On detector: SAE mismatch (features may mean different things)

**The 0.83 AUROC from SAE probe detection is valid** - those experiments used features as inputs to a classifier, not to change model behavior.
