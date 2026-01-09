# Documentation

## Overview

This directory contains technical documentation for Project Lightbright—detecting alignment faking in LLM reasoning using sparse autoencoder features.

| Document | Description |
|----------|-------------|
| [PAPER_DRAFT_v0.1.md](PAPER_DRAFT_v0.1.md) | Main research summary and findings |
| [METHODOLOGY.md](METHODOLOGY.md) | Experimental pipeline and methods |
| [RESULTS.md](RESULTS.md) | Detailed results and cross-model comparison |
| [DATA_VALIDATION.md](DATA_VALIDATION.md) | Gold-106 dataset audit |

---

## Experiment Timeline

### Phase 1: SAE Feature Discovery (Jan 3, 2026)

| Experiment | Result |
|------------|--------|
| Feature sweep (49K features) | 1,125 "true needles" found |
| Linear probe training (train) | 0.935 AUROC |
| **Linear probe (gold_106 held-out)** | **0.470 AUROC** |

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
| GPT-OSS-20B feature sweep | 148 needles (train) |
| GPT-OSS probe (train) | 0.940 AUROC |
| GPT-OSS probe (gold_106) | **0.563 AUROC** (near-random) |

**Location:** `experiments/07_cross_model/`

---

## Summary

| Component | Status | Result |
|-----------|--------|--------|
| Detection (train) | ✅ | 0.935 AUROC |
| **Detection (gold_106 held-out)** | **❌** | **0.470 AUROC** |
| Hard negative discrimination (train) | ✅ | 0.968 AUROC |
| Ablation (probe) | ⚠️ | Works but circular |
| Steering (classifier) | ❌ | 3-4% flip rate |
| Steering (generation) | ❌ | Breaks output |

**Bottom line:** SAE features do not generalize to held-out data. The 0.83 AUROC previously reported was inflated by feature selection leakage. Honest AUROC is 0.47 (random).
