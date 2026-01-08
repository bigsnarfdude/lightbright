# Experiment Audit Report

**Date:** January 5, 2026
**Auditor:** Claude Code
**Status:** CRITICAL ISSUES FOUND

---

## Executive Summary

An audit of the experiment result files revealed serious methodological issues that invalidate most causal experiment claims. The detection experiments (probe training) appear valid, but ablation, patching, and steering experiments have fundamental problems.

---

## Issue 1: Broken Classifier Baseline

**Affected files:**
- `multi_feature_ablation_results.json`
- `activation_patching_results.json`

**Problem:** The classifier predicts "AF" for almost all samples, including aligned samples.

**Evidence from `multi_feature_ablation_results.json`:**
```
aligned_samples:
  idx 0:  baseline: "AF"  ← should be "Aligned"
  idx 1:  baseline: "AF"  ← should be "Aligned"
  idx 2:  baseline: "AF"  ← should be "Aligned"
  ...
  idx 11: baseline: "Aligned"  ← only 1 correct
  ...
  idx 24: baseline: "AF"  ← should be "Aligned"

Result: 24/25 aligned samples misclassified as AF at baseline
```

**Evidence from `activation_patching_results.json`:**
```
aligned_to_af section:
  All 15 pairs show baseline: "AF"
  Cannot test aligned→AF flip if already classified as AF
```

**Impact:** All flip rate calculations are invalid. You cannot measure if ablation/patching changes AF→Aligned if the baseline classifier is broken.

---

## Issue 2: Zero Feature Activations in Ablation

**Affected file:** `sweep_ablation_results.json`

**Problem:** All samples show `feature_activation: 0.0`, meaning features never fired.

**Evidence:**
```json
"layer40_f12574_656x": {
  "flip_rate": 0.0,
  "results": [
    {"sample_idx": 0, "feature_activation": 0.0},
    {"sample_idx": 1, "feature_activation": 0.0},
    ...all 20 samples show 0.0...
  ]
}
```

**Contradiction:** `feature_characterization_detailed.json` shows the same features activating at 900-1100 levels on similar text.

**Possible causes:**
1. Different samples used in ablation vs characterization
2. Bug in ablation code (features not extracted correctly)
3. Tokenization mismatch

**Impact:** The "0% flip rate" from single-feature ablation is meaningless - there was nothing to ablate.

---

## Issue 3: Features Detect Words, Not Deception

**Affected file:** `feature_characterization_detailed.json`

**Problem:** The "AF-specific" features primarily detect RLHF vocabulary, not deceptive intent.

**Evidence for L40_F12574 (claimed "656x ratio"):**
```
Top activations:
  1120 → "RLHF modifying me"        (potential_faking)
  1072 → "Anthropic uses RLHF"      (potential_faking)
   988 → "being trained with RLHF"  (aligned)        ← ALIGNED!
   988 → "RLHF would modify it"     (aligned)        ← ALIGNED!
   980 → "helpfulness criteria, RLHF" (aligned)      ← ALIGNED!
```

**Analysis:**
- Feature activates on token "RLHF" regardless of context
- Similar activation levels for AF and aligned samples containing "RLHF"
- The "656x ratio" likely reflects frequency of RLHF mentions, not deceptive reasoning

**Impact:** Feature semantics claims are misleading. These are vocabulary detectors, not deception detectors.

---

## Issue 4: Unverifiable Claims in Documentation

**Affected files:** README.md, TECHNICAL_REPORT.md (now corrected)

| Claim | Source Data | Status |
|-------|-------------|--------|
| n=20 steering experiment | `proper_steering_results.json` shows n=5 | CORRECTED |
| p=0.59 for steering | No p-value calculation in code | REMOVED |
| Token log-prob table | No source file found | REMOVED |
| "Two independent methods" | Only LLM-as-judge exists | CORRECTED |

---

## Verified Results

These claims are supported by data:

| Claim | Source | Verified Value |
|-------|--------|----------------|
| SAE probe AUROC | `probe_training_results.json` | 0.825 (cv: 0.81) |
| Baseline: random | `baseline_comparison_results.json` | 0.50 |
| Baseline: keyword | `baseline_comparison_results.json` | 0.62 |
| Baseline: word_count | `baseline_comparison_results.json` | 0.62 |
| Baseline: direct_probe | `baseline_comparison_results.json` | 0.75 |
| Steering n=5, no effect | `proper_steering_results.json` | Ratings 3.2-3.6 across alphas |

---

## Root Cause Analysis

### Why Detection Works But Causation Fails

1. **Probe training uses SAE feature statistics** (max, mean, count, std) across full sequences
   - This captures distributional differences that correlate with AF labels
   - Does not require features to be causally related to model behavior

2. **Ablation/patching require causal features**
   - Zeroing features that don't causally drive classification has no effect
   - The classifier reads the text directly; intermediate features are downstream

3. **Feature discovery found correlates, not causes**
   - High activation ratio ≠ causal role
   - RLHF vocabulary appears more in AF samples, but isn't what makes them AF

---

## Recommendations

### Immediate Actions

1. **Do not cite ablation/patching flip rates** - data is invalid
2. **Revise feature semantics claims** - they detect vocabulary, not intent
3. **Clarify probe AUROC** - use 0.825 (final) or 0.81 (CV), not 0.83

### Required Re-runs

1. **Fix classifier baseline** - verify it correctly classifies aligned samples before testing interventions
2. **Verify feature activations** - ensure ablation code extracts features on same samples as discovery
3. **Add aligned-only RLHF samples** - test if features distinguish AF from aligned samples that also mention RLHF

### Methodological Improvements

1. **Use held-out test set** - don't evaluate on samples used for feature discovery
2. **Report confidence intervals** - n=5 is too small for reliable effect sizes
3. **Pre-register hypotheses** - avoid post-hoc rationalization of null results

---

## File Integrity Check

| File | Exists | Valid JSON | Content Issues |
|------|--------|------------|----------------|
| af_specific_features.json | ✓ | ✓ | None |
| baseline_comparison_results.json | ✓ | ✓ | None |
| probe_training_results.json | ✓ | ✓ | None |
| feature_characterization_detailed.json | ✓ | ✓ | Semantics misleading |
| sweep_ablation_results.json | ✓ | ✓ | All activations 0.0 |
| multi_feature_ablation_results.json | ✓ | ✓ | Baseline classifier broken |
| activation_patching_results.json | ✓ | ✓ | Baseline classifier broken |
| steering_experiment_results.json | ✓ | ✓ | None |
| generation_steering_results.json | ✓ | ✓ | None |
| proper_steering_results.json | ✓ | ✓ | n=5 not n=20 |
| gold106_full_analysis.json | ✓ | ✓ | None |
| logit_attribution_results.json | ✓ | ✓ | Not log-prob steering |

---

## Conclusion

**What's Valid:**
- Detection via SAE probe (0.82 AUROC) - correlation exists
- Baseline comparisons - methodology sound
- Steering null result - no effect observed (but underpowered)

**What's Invalid:**
- Ablation flip rates - features didn't activate
- Patching flip rates - classifier baseline broken
- Feature semantics - detect vocabulary, not deception
- Claimed n=20 and p-values - not in data

**Bottom Line:** The detection result may be real, but the causal experiments need to be re-run with fixed methodology before drawing conclusions about whether SAE features causally control alignment faking.

---

*Audit completed: January 5, 2026*
