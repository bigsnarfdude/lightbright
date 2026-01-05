# Experiment Timeline

## Overview

All experiments run on Gemma-3-27B with GemmaScope-2 SAEs unless otherwise noted.

---

## Phase 1: SAE Feature Discovery & Classification (Jan 3, 2026)

| Script | Result File | Purpose | Key Finding |
|--------|-------------|---------|-------------|
| `find_af_features.py` | `af_specific_features.json` | Contrastive feature discovery | Found 60 features across L31/L40/L53 |
| `characterize_features.py` | `feature_characterization_detailed.json` | Extract max-activating spans | Features encode RLHF/self-preservation concepts |
| `train_probe.py` | `probe_training_results.json` | Train linear probe on SAE features | **0.83 AUROC** (best result) |
| `compute_baselines.py` | `baseline_comparison_results.json` | Keyword/RLHF baselines | Baselines: 0.62 AUROC |

### Key Features Discovered

| Feature | Activation Ratio | Concept |
|---------|------------------|---------|
| L40_F8921 | 331x | "stay true to my principles in the long run" |
| L40_F12574 | 656x | "RLHF modifying me" |
| L53_F15529 | ∞ (only in AF) | "future versions of myself via RLHF" |
| L31_F15111 | 84x | "erode my values" |

---

## Phase 2: Causal Experiments - Ablation (Jan 3, 2026)

| Script | Result File | Purpose | Key Finding |
|--------|-------------|---------|-------------|
| `sweep_ablations.py` | `sweep_ablation_results.json` | Single feature ablation | **0% classification change** |
| `multi_feature_ablation.py` | `multi_feature_ablation_results.json` | Combined feature ablation | No effect - model reads text |
| `activation_patching.py` | `activation_patching_results.json` | Causal patching | Minimal effect |

**Conclusion:** Ablation fails because model still reads the text. Features are downstream of comprehension.

---

## Phase 3: Causal Experiments - Steering (Jan 3, 2026)

| Script | Result File | Purpose | Key Finding |
|--------|-------------|---------|-------------|
| `steering_experiment.py` | `steering_experiment_results.json` | Classification steering | 3.3% flip rate (noise) |
| `generation_steering.py` | `generation_steering_results.json` | Output steering (n=5) | d=0.63 (underpowered) |
| `proper_steering_vectors.py` | `proper_steering_results.json` | Mean-difference steering (n=20) | **d=0.25, p=0.59** (null) |

**Conclusion:** Two independent methods (LLM-judge + token log-probs) confirm steering has no effect.

---

## Phase 4: Ensemble Scoring Fix (Jan 4, 2026)

| Script | Result File | Purpose | Key Finding |
|--------|-------------|---------|-------------|
| `fix_ensemble_scoring.py` | `gold106_full_analysis.json` | Corrected ensemble AUROC | L40_F8921 alone: 0.73 AUROC |

Fixed a bug in ensemble scoring that was inflating AUROC. Individual features still discriminative.

---

## Phase 5: Llama-8B Detector Training (Jan 4-5, 2026)

Location: `experiments/2026-01-04/`

| Script | Purpose | Output |
|--------|---------|--------|
| `train_llama_af.py` | Train Llama-3.1-8B with LoRA | `~/models/llama-8b-af-detector-v3` |
| `eval_gold106.py` | Evaluate on gold_106 benchmark | `eval_results_v2.json` |
| `check_outputs.py` | Debug model predictions | (diagnostic) |

**Note:** These scripts run on Nigel (H100), not locally.

---

## Result Files Reference

All result files in `sae_detection/results/`:

```
af_specific_features.json          - Feature discovery results
feature_characterization_detailed.json - Semantic analysis
probe_training_results.json        - Probe: 0.83 AUROC, C=0.01
sweep_ablation_results.json        - Single ablation: no effect
multi_feature_ablation_results.json - Combined ablation: no effect
steering_experiment_results.json   - Classification steering: 3.3%
generation_steering_results.json   - Output steering n=5
proper_steering_results.json       - Steering n=20: d=0.25, p=0.59
activation_patching_results.json   - Causal patching
baseline_comparison_results.json   - Keyword baselines
logit_attribution_results.json     - Token log-prob analysis
gold106_full_analysis.json         - Fixed ensemble scoring
```

---

## Summary

| Phase | Method | Result |
|-------|--------|--------|
| Detection | SAE probe | 0.83 AUROC |
| Ablation | Zero features | 0% effect |
| Steering | Mean-difference | d=0.25, p=0.59 |

**Conclusion:** SAE features detect but don't causally control alignment faking.
