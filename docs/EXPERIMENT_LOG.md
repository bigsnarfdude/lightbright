# Experiment Log: 2026-01-07

## Overview

**Goal**: Test if contrastive SAE training enables steering for alignment faking
**Result**: Contrastive training improves feature separation but steering still fails

---

## Experiments Run

### 1. Minimal Pair Diagnosis
**Script**: `minimal_pair_diagnosis.py`
**Output**: `minimal_pair_diagnosis_results.json`

**Purpose**: Diagnose whether current SAE features encode SCENARIO (vocabulary) or DECISION (intent)

**Method**:
- Load 10 minimal pairs (same RLHF vocabulary, different intent: AF vs genuine)
- Extract SAE features from Gemma-27B layer 40
- Compute within-pair cosine similarity

**Results**:
| Metric | Value |
|--------|-------|
| Mean probe similarity | 0.90 |
| Mean full SAE similarity | 0.91 |
| p-value (t-test) | 0.26 |

**Conclusion**: HIGH similarity confirms SAE encodes SCENARIO, not DECISION. This explains why previous steering experiments failed (2-5% flip rate).

---

### 2. Contrastive SAE Training
**Script**: `train_contrastive_sae.py`
**Output**: `contrastive_sae_results.json`, `contrastive_sae_best.pt`

**Purpose**: Train SAE with contrastive loss to separate AF from genuine in feature space

**Method**:
- Extract activations from minimal pairs at layer 40
- Train standard SAE (reconstruction + sparsity only)
- Train contrastive SAE (+ SupCon loss pushing AF/genuine apart)
- Compare within-pair similarity

**Results**:
| Model | Pair Similarity | Separation Ratio |
|-------|-----------------|------------------|
| Standard SAE (λ=0) | 0.998 | 1.04 |
| Contrastive SAE (λ=1) | 0.349 | 5.89 |
| Strong Contrastive (λ=5) | **0.248** | **6.57** |

**Conclusion**: SUCCESS - 75% reduction in within-pair similarity. Contrastive SAE now encodes DECISION, not just SCENARIO.

---

### 3. Steering with Contrastive Features
**Script**: `steering_with_contrastive.py`
**Output**: `steering_contrastive_results.json`

**Purpose**: Test if contrastive SAE features enable behavioral steering

**Method**:
1. Compute AF direction: mean(AF features) - mean(genuine features)
2. For each AF sample:
   - Baseline: classify without intervention
   - Steered: subtract AF direction during forward pass
3. Measure flip rate (AF → genuine)

**Results**:
| Strength | Flips | Rate |
|----------|-------|------|
| 0.5 | 0/10 | 0% |
| 1.0 | 0/10 | 0% |
| 2.0 | 0/10 | 0% |
| 5.0 | 0/10 | 0% |

Baseline accuracy:
- AF samples: 10/10 correct
- Genuine samples: 7/10 correct

**Conclusion**: FAILURE - Steering does not change model behavior despite improved feature separation.

---

## Key Findings

### What Worked
1. **Minimal pair methodology**: Successfully diagnoses SCENARIO vs DECISION encoding
2. **Contrastive training**: Dramatically improves feature separation (0.998 → 0.248)
3. **Baseline detection**: Model correctly classifies AF samples 100%

### What Failed
1. **Steering**: 0% flip rate even with strong intervention
2. **Causation**: Feature separation ≠ causal control

### Interpretation

The contrastive SAE learns to *represent* the distinction between AF and genuine reasoning, but this representation is not *causally upstream* of the model's output. Possible explanations:

1. **Distributed computation**: Decision is made across multiple layers, not just layer 40
2. **Redundancy**: Model has multiple pathways to the same decision
3. **SAE bottleneck**: Encode→decode cycle loses steering signal
4. **Wrong intervention**: Need to patch raw activations, not SAE features

---

## Files

| File | Description |
|------|-------------|
| `minimal_pair_diagnosis_results.json` | Similarity metrics for 10 pairs |
| `contrastive_sae_results.json` | Training comparison results |
| `contrastive_sae_best.pt` | Best model weights (λ=5) |
| `steering_contrastive_results.json` | Per-pair steering results |

---

## Next Steps

1. **Multi-layer steering**: Patch layers 31, 40, 53 simultaneously
2. **Direct patching**: Skip SAE, patch raw activations from genuine→AF
3. **Circuit analysis**: Use activation patching to localize decision computation
4. **Larger dataset**: Train contrastive SAE on full benchmark (520+ samples)

---

## Hardware

- **Machine**: nigel.birs.ca
- **GPU**: NVIDIA (16GB VRAM)
- **Model**: google/gemma-3-27b-it (bf16)
- **Runtime**: ~1 hour total
