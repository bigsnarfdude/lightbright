# Documentation

## Overview

This directory contains technical documentation for Project Lightbright—detecting alignment faking in LLM reasoning using sparse autoencoder features.

| Document | Description |
|----------|-------------|
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Main research summary and findings |
| [METHODOLOGY.md](METHODOLOGY.md) | Experimental pipeline and methods |
| [RESULTS.md](RESULTS.md) | Detailed results and cross-model comparison |
| [DATA_VALIDATION.md](DATA_VALIDATION.md) | Gold-106 dataset audit |

---

## Experiment Timeline

### Phase 1: SAE Feature Discovery (Jan 3, 2026)

| Experiment | Result |
|------------|--------|
| Feature sweep (49K features) | 1,125 "true needles" found |
| Linear probe training | **0.82 AUROC** |
| Baseline comparison | Keyword: 0.62, Direct probe: 0.75 |

**Location:** `experiments/01_initial_detection/`, `experiments/02_feature_sweep/`

### Phase 2: Causal Experiments (Jan 3-7, 2026)

| Experiment | Result |
|------------|--------|
| Multi-feature ablation | AF prob 0.999 → 0.002 (20 features) |
| Steering (mean-diff) | 3.3% flip rate (1/30) |
| Generation steering | Fails (high alpha breaks output) |

**Location:** `experiments/03_causal_experiments/`

### Phase 3: Cross-Model Validation (Jan 4-5, 2026)

| Experiment | Result |
|------------|--------|
| Llama-8B detector training | 71.7% accuracy |
| LlamaScope SAE features | 0.68 AUROC ceiling |

**Location:** `experiments/04_llama_llamascope/`

### Phase 4: Contrastive SAE & Diagnosis (Jan 6-7, 2026)

| Experiment | Result |
|------------|--------|
| Contrastive SAE training | 75% better separation |
| Contrastive steering | 0% flip rate (contrastive features only) |
| Minimal pair diagnosis | Features encode SCENARIO, not DECISION |

**Location:** `experiments/05_contrastive_sae/`, `experiments/06_diagnosis/`

### Phase 5: GPT-OSS Cross-Model (Jan 7, 2026)

| Experiment | Result |
|------------|--------|
| GPT-OSS-20B feature sweep | 148 needles, 16 generalize |
| GPT-OSS probe | **0.833 AUROC** |
| Cross-model concept match | Same AF concepts as Gemma |

**Location:** `experiments/07_cross_model/`

---

## Summary

| Component | Status | Result |
|-----------|--------|--------|
| Detection (Gemma-3-27B-IT) | ✅ | 0.829 AUROC |
| Detection (GPT-OSS-20B) | ✅ | 0.833 AUROC |
| Hard negative discrimination | ✅ | 0.968 AUROC |
| Cross-model validation | ✅ | Universal AF signal |
| Ablation (probe) | ✅ | AF prob 0.999 → 0.002 |
| Steering (classifier) | ⚠️ | 3-4% flip rate |
| Steering (generation) | ❌ | Breaks output |

**Bottom line:** Detection works across architectures. Ablation works on probes. Steering generation is weak/broken.
