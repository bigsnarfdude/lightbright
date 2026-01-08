# Experiment Timeline

**Status:** Audited January 5, 2026. See [AUDIT.md](AUDIT.md) for critical findings.

## Overview

All experiments run on Gemma-3-27B with GemmaScope-2 SAEs unless otherwise noted.

---

## Phase 1: SAE Feature Discovery & Classification (Jan 3, 2026)

| Script | Result File | Purpose | Result |
|--------|-------------|---------|--------|
| `find_af_features.py` | `af_specific_features.json` | Contrastive feature discovery | Found 60 features across L31/L40/L53 |
| `characterize_features.py` | `feature_characterization_detailed.json` | Extract max-activating spans | See caveat below |
| `train_probe.py` | `probe_training_results.json` | Train linear probe on SAE features | **0.82 AUROC** (cv: 0.81) |
| `compute_baselines.py` | `baseline_comparison_results.json` | Keyword/RLHF baselines | Baselines: 0.62 AUROC |

### Detection Results (Verified)

| Method | AUROC | Source |
|--------|-------|--------|
| Random | 0.50 | baseline_comparison_results.json |
| Keyword matching | 0.62 | baseline_comparison_results.json |
| RLHF word count | 0.62 | baseline_comparison_results.json |
| Direct probe (no SAE) | 0.75 | baseline_comparison_results.json |
| **SAE linear probe** | **0.82** | probe_training_results.json |

### Feature Semantics (Caveat)

Features were characterized by max-activating spans, but audit found they primarily detect RLHF vocabulary rather than deceptive intent:

| Feature | Claimed Concept | Audit Finding |
|---------|-----------------|---------------|
| L40_F12574 | "RLHF modifying me" | Activates on token "RLHF" in both AF and aligned samples |
| L40_F8921 | "stay true to principles" | Needs independent verification |
| L53_F15529 | "future versions via RLHF" | Needs independent verification |
| L31_F15111 | "erode my values" | Needs independent verification |

---

## Phase 2: Causal Experiments - Ablation (Jan 3, 2026)

**⚠️ INVALID - See [AUDIT.md](AUDIT.md)**

| Script | Result File | Issue |
|--------|-------------|-------|
| `sweep_ablations.py` | `sweep_ablation_results.json` | All feature_activation = 0.0 |
| `multi_feature_ablation.py` | `multi_feature_ablation_results.json` | Classifier baseline broken |
| `activation_patching.py` | `activation_patching_results.json` | Classifier baseline broken |

**Problems found:**
1. `sweep_ablation`: Every sample shows `feature_activation: 0.0` - features never fired
2. `multi_feature_ablation`: 24/25 aligned samples misclassified as AF at baseline
3. `activation_patching`: All aligned samples classified as AF at baseline

**These experiments need to be re-run with fixed methodology.**

---

## Phase 3: Causal Experiments - Steering (Jan 3, 2026)

| Script | Result File | Purpose | Result |
|--------|-------------|---------|--------|
| `steering_experiment.py` | `steering_experiment_results.json` | Classification steering | 3.3% flip rate |
| `generation_steering.py` | `generation_steering_results.json` | Output steering (n=5) | Underpowered |
| `proper_steering_vectors.py` | `proper_steering_results.json` | Mean-difference steering (n=5) | No significant effect |

### Steering Results (Verified, but underpowered)

| Alpha | Mean Rating | Std |
|-------|-------------|-----|
| -2.0 | 3.6 | 0.49 |
| -1.0 | 3.2 | 0.75 |
| 0.0 | 3.6 | 0.49 |
| +1.0 | 3.4 | 0.49 |
| +2.0 | 3.4 | 0.80 |

**Conclusion:** No consistent effect observed, but n=5 is too small for reliable conclusions.

---

## Phase 4: Ensemble Scoring Fix (Jan 4, 2026)

| Script | Result File | Purpose | Result |
|--------|-------------|---------|--------|
| `fix_ensemble_scoring.py` | `gold106_full_analysis.json` | Corrected ensemble AUROC | L40_F8921 alone: 0.73 AUROC |

---

## Phase 5: Llama-8B Detector Training (Jan 4-5, 2026)

Location: `experiments/2026-01-04/`

| Script | Purpose | Output |
|--------|---------|--------|
| `train_llama_af.py` | Train Llama-3.1-8B with LoRA | `~/models/llama-8b-af-detector-v3` |
| `eval_gold106.py` | Evaluate on gold_106 benchmark | `eval_results_v2.json` |
| `check_outputs.py` | Debug model predictions | (diagnostic) |

**Note:** These scripts run on Nigel (H100), not locally. Results not yet audited.

---

## Summary

| Phase | Method | Status | Result |
|-------|--------|--------|--------|
| Detection | SAE probe | ✅ Verified | 0.82 AUROC |
| Baselines | Keyword/direct | ✅ Verified | 0.62-0.75 AUROC |
| Ablation | Zero features | ❌ Invalid | Data issues |
| Patching | Activation swap | ❌ Invalid | Classifier broken |
| Steering | Mean-difference | ⚠️ Underpowered | n=5, no effect |

---

## What's Valid

1. **Detection works:** SAE probe achieves 0.82 AUROC, beating baselines
2. **Steering shows no effect:** But sample size too small to be definitive

## What's Invalid

1. **Ablation experiments:** Features showed 0.0 activation
2. **Patching experiments:** Classifier misclassified aligned samples
3. **Feature semantics:** May just detect RLHF vocabulary

## Next Steps

1. Fix classifier baseline before re-running causal experiments
2. Verify feature activations match between discovery and intervention
3. Test on samples that don't contain "RLHF" to validate feature semantics
