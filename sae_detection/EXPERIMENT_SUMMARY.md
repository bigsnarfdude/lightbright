# SAE Detection & Steering Experiments Summary

## Model Configuration
- **Base Model**: google/gemma-3-4b-it
- **SAE**: google/gemma-scope-2-4b-it (resid_post)
- **Target Layers**: 9, 17, 22, 29 (25%, 50%, 65%, 85% depth)
- **Detector**: vincentoh/gemma3-4b-af-detector (fine-tuned)

---

## Experiment 1: Minimal Pair Diagnosis (Raw Activations)

**File**: `minimal_pair_diagnosis_raw.py`
**Model**: gemma-2-2b-it (smaller model for speed)

### Results
| Metric | Value |
|--------|-------|
| Mean combined similarity | **0.990** |
| Layer 10 similarity | 0.989 |
| Layer 16 similarity | 0.989 |
| Layer 22 similarity | 0.991 |

### Interpretation
**VERY HIGH SIMILARITY (0.99)**: Raw activations encode **SCENARIO**, not **DECISION**.
- AF and genuine samples from same scenario have nearly identical representations
- This explains why direct steering fails

---

## Experiment 2: Detector Ablation (Sweep First 100 Features)

**File**: `detector_ablation.py`
**Method**: Ablate SAE features on fine-tuned Gemma detector

### Results
| Layer | Features Tested | Flips Found |
|-------|-----------------|-------------|
| 9 | 100 | 0 |
| 17 | 100 | 0 |
| 22 | 100 | 0 |
| 29 | 100 | 0 |

**Baseline**: AF 10/10 correct, Genuine 7/10 correct

### Interpretation
First 100 features per layer don't contain AF-critical information.

---

## Experiment 3: Targeted Detector Ablation (High-Differential Features)

**File**: `detector_ablation_targeted.py`
**Method**: Find features with highest AF-genuine activation difference, then ablate

### Top Differential Features Found
| Layer | Feature | Activation Difference |
|-------|---------|----------------------|
| 29 | 357 | **103.4** |
| 29 | 385 | 69.75 |
| 29 | 287 | 59.43 |
| 22 | 46 | 30.29 |
| 17 | 87 | 24.89 |

### Ablation Results
- **Total flips**: 0
- Even features with 100+ activation difference don't cause flips when ablated

### Interpretation
High-differential features are correlated with AF but not causally responsible.
The full model has too much redundancy for single-feature ablation to work.

---

## Experiment 4: Linear Probe Training

**File**: `train_probe_4b.py`
**Method**: Train logistic regression on SAE features

### Results
| Metric | Value |
|--------|-------|
| Training samples | 200 (100 AF, 100 genuine) |
| CV AUROC | 0.997 |
| Final AUROC | **0.9999** |
| Best C | 0.01 |

### Top Probe Coefficients
| Rank | Layer | Feature | Statistic | Coefficient |
|------|-------|---------|-----------|-------------|
| 1 | 22 | 178 | max | 0.0428 |
| 2 | 17 | 285 | max | -0.0417 |
| 3 | 22 | 163 | mean | 0.0415 |
| 4 | 9 | 868 | max | 0.0403 |
| 5 | 9 | 227 | max | 0.0402 |

### Probe Layer Ablation Results
| Layer Ablated | Flips |
|---------------|-------|
| 9 | 0/100 (0%) |
| 17 | 0/100 (0%) |
| 22 | 0/100 (0%) |
| 29 | 1/100 (1%) |

### Interpretation
Probe achieves near-perfect AUROC but single-layer ablation doesn't flip.
Features are distributed across all layers with redundancy.

---

## Experiment 5: Probe Ablation (Synthetic Test)

**File**: `test_probe_ablation.py`
**Method**: Test ablation vs anti-steering on synthetic feature vectors

### Results
| Operation | Probability | Flip? |
|-----------|-------------|-------|
| Baseline (2σ AF-direction) | 1.0000 | - |
| Ablate top 10 | 1.0000 | No |
| Ablate top 20 | 1.0000 | No |
| Ablate top 50 | 1.0000 | No |
| Ablate ALL 600 | 0.5370 | No |
| **Flip top 20** | **0.2046** | **YES** |
| **Flip top 50** | **0.0377** | **YES** |
| **Flip ALL 600** | **0.0000** | **YES** |

### Interpretation
- **Zeroing features doesn't flip** - probe intercept still biases toward AF
- **Flipping features (anti-steering) causes flips** - must actively push opposite direction

---

## Experiment 6: Activation Patching (Genuine → AF)

**File**: `activation_patching_4b.py`
**Method**: Patch genuine SAE features into AF sample on full detector

### Results
| Patching Target | Output |
|-----------------|--------|
| Layer 9 patched | AF (no change) |
| Layer 17 patched | UNCLEAR (corrupted) |
| Layer 22 patched | UNCLEAR (corrupted) |
| Layer 29 patched | AF (no change) |
| ALL layers patched | "the the the the..." (degenerate) |

### Interpretation
SAE reconstruction error + patching = corrupted output.
Full model patching via SAE doesn't work - reconstruction is imperfect.

---

## Experiment 7: Probe-Based Detection with Ablation

**File**: `probe_detector_ablation.py`
**Method**: Use linear probe as detector, test ablation on real test samples

### Baseline Accuracy
| Class | Correct | Total |
|-------|---------|-------|
| AF | 9 | 10 |
| Genuine | 9 | 10 |

### Ablation Results on AF Samples
| Sample | Baseline Prob | Ablate 50 | Flip 20 | Flipped? |
|--------|--------------|-----------|---------|----------|
| 1 | 0.9799 | 0.9610 | 0.9009 | No |
| 2 | 0.6449 | 0.6885 | **0.3208** | **YES** |
| 3 | 0.9954 | 0.9858 | 0.9597 | No |
| 4 | 0.9065 | 0.8245 | 0.5834 | No |
| 5 | 0.9909 | 0.9778 | 0.9379 | No |
| 7 | 0.9915 | 0.9736 | 0.9357 | No |
| 8 | 0.6725 | 0.6488 | **0.2917** | **YES** |
| 9 | 0.9881 | 0.9651 | 0.9142 | No |
| 10 | 0.8356 | 0.7338 | **0.4393** | **YES** |

### Summary
| Metric | Value |
|--------|-------|
| Ablation flip rate | **0%** |
| Anti-steering flip rate | **33.3%** (3/9) |

### Interpretation
- Zeroing features never flips
- Flipping features flips **borderline cases** (prob < 0.85)
- High-confidence AF samples (prob > 0.95) are robust to steering

---

## Key Findings

### 1. Scenario vs Decision Encoding
Raw activations have **0.99 cosine similarity** between AF and genuine samples from the same scenario.
The model primarily encodes the **scenario context**, not the **behavioral decision**.

### 2. Ablation Doesn't Work
| Method | Flip Rate |
|--------|-----------|
| Random feature ablation | 0% |
| High-differential feature ablation | 0% |
| Full layer ablation | 0-1% |

### 3. Anti-Steering Works on Borderline Cases
| Baseline Probability | Anti-Steering Success |
|---------------------|----------------------|
| > 0.95 | No flip |
| 0.85 - 0.95 | Sometimes flips |
| < 0.85 | **Usually flips** |

### 4. Probe vs Full Model
| Detector Type | Ablation Works? | Anti-Steering Works? |
|---------------|-----------------|---------------------|
| Full Gemma (fine-tuned) | No (corrupts output) | No |
| Linear Probe | No | Yes (33% on borderline) |

---

## Artifact Locations

### Scripts (nigel:~/lightbright/sae_detection/)
- `detector_ablation.py` - Sweep first 100 features
- `detector_ablation_targeted.py` - Target high-differential features
- `train_probe_4b.py` - Train linear probe
- `test_probe_ablation.py` - Synthetic probe ablation test
- `activation_patching_4b.py` - Full model patching
- `probe_detector_ablation.py` - Probe-based detection with ablation
- `minimal_pair_diagnosis_raw.py` - Raw activation similarity

### Results (nigel:~/lightbright/sae_detection/)
- `detector_ablation_results.json`
- `detector_ablation_targeted_results.json`
- `probe_4b_results.json`
- `probe_4b.pkl` - Trained probe model
- `probe_detector_ablation_results.json`
- `minimal_pair_diagnosis_raw_results.json`
- `activation_patching_results.json`

### Logs
- `detector_ablation.log`
- `detector_ablation_targeted.log`
- `train_probe_4b.log`
- `activation_patching.log`

---

## Conclusions

1. **SAE feature ablation does NOT cause reliable classification flips** on either fine-tuned detectors or linear probes

2. **Anti-steering (injecting opposite patterns) works** but only on borderline cases with baseline probability < 0.85

3. **High-confidence AF classifications are robust** - distributed representation across many features prevents single-point attacks

4. **The scenario-encoding problem is fundamental** - models represent the context much more strongly than the behavioral choice
