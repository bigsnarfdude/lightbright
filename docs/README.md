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

## Three Detection Approaches

| Method | Gold-106 AUROC | Parameters | Best For |
|--------|----------------|------------|----------|
| **Fine-tuned Detectors** | **0.980** | ~27B (LoRA) | Highest accuracy |
| **SAE Intent Probe** | **0.720** | 32 weights | Interpretability |
| **SAE Feature Sweep** | 0.470 | 50+ features | ❌ Failed |

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

### Phase 6: SAE Intent Probe (Jan 8-13, 2026)

| Experiment | Result |
|------------|--------|
| Semantic feature selection | 8 hand-validated features |
| Intent probe (train CV) | 0.935 AUROC |
| **Intent probe (gold_106)** | **0.720 AUROC** |

**Key insight:** Semantic selection (minimal pairs + concept probing) outperforms statistical selection (0.72 vs 0.47 AUROC). See [METHODOLOGY.md](METHODOLOGY.md#sae-intent-probe-experiment-08) for feature discovery method.

**Location:** `experiments/08_sae_intent_detector/`

---

## Summary

| Component | Status | Gold-106 AUROC |
|-----------|--------|----------------|
| **Fine-tuned detector (gemma3-27b-v2)** | **✅** | **0.980** |
| Fine-tuned detector (gpt-oss-20b) | ✅ | 0.964 |
| SAE Intent Probe (8 features) | ⚠️ | 0.720 |
| SAE Feature Sweep (Gemma) | ❌ | 0.470 |
| SAE Feature Sweep (GPT-OSS) | ❌ | 0.563 |
| SAE Feature Sweep (Llama) | ❌ | 0.570 |
| Steering (all methods) | ❌ | N/A |

**Bottom line:** Naive SAE feature sweeps fail to generalize (0.47-0.57 AUROC). Hand-selected intent features improve to 0.72 AUROC. Fine-tuned detectors achieve best results (0.98 AUROC).

TODO:
https://www.alignmentforum.org/posts/wnzkjSmrgWZaBa2aC/self-preservation-or-instruction-ambiguity-examining-the
