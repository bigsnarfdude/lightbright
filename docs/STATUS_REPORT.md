# Project Lightbright: Honest Status Report

**Date:** January 5, 2026
**Assessment:** Mixed - Detection works, causal experiments failed

---

## Executive Summary

This project attempted to detect and control alignment faking using SAE features. After a comprehensive audit:

- **Detection:** Partially successful (0.82 AUROC, verified)
- **Causation:** Failed (all experiments have methodological issues)
- **Documentation:** Contained fabricated claims (now corrected)

**Is this research junk?** No, but it's much weaker than originally claimed.

---

## What We Actually Have

### Verified Results

| Component | Claim | Verified | Notes |
|-----------|-------|----------|-------|
| SAE probe AUROC | 0.83 | 0.82 | probe_training_results.json shows 0.825 |
| Beats random | +0.32 | ✅ | 0.82 vs 0.50 |
| Beats keywords | +0.20 | ✅ | 0.82 vs 0.62 |
| Beats direct probe | +0.07 | ✅ | 0.82 vs 0.75 |

### Code Quality

| Script | Lines | Status | Notes |
|--------|-------|--------|-------|
| train_probe.py | 258 | ✅ Sound | Proper sklearn CV, saves all metadata |
| compute_baselines.py | 244 | ✅ Sound | Good baseline comparison |
| find_af_features.py | 257 | ⚠️ Check | Feature selection may be overfitted |
| characterize_features.py | 182 | ⚠️ Check | Semantics claims need verification |
| sweep_ablations.py | 241 | ❌ Broken | All activations = 0.0 |
| multi_feature_ablation.py | 244 | ❌ Broken | Classifier baseline wrong |
| activation_patching.py | 389 | ❌ Broken | Classifier baseline wrong |
| steering_experiment.py | 226 | ⚠️ Weak | Only n=5 |
| proper_steering_vectors.py | 351 | ⚠️ Weak | Only n=5, claims were n=20 |
| generation_steering.py | 323 | ⚠️ Weak | Only n=5 |

**Total: 2,832 lines of code, ~40% working correctly**

---

## Defect Inventory

### Critical Defects (Invalidate Results)

#### 1. Ablation Features Never Activated
**File:** `sweep_ablation_results.json`
**Issue:** All 20 samples × 8 features show `feature_activation: 0.0`
**Impact:** Cannot test ablation effect if features don't fire
**Root cause:** Unknown - possibly different samples or code bug
**Fix:** Verify features fire on test samples before ablating

#### 2. Classifier Predicts AF for Everything
**File:** `multi_feature_ablation_results.json`, `activation_patching_results.json`
**Issue:** 24/25 aligned samples classified as "AF" at baseline
**Impact:** Flip rate calculations meaningless
**Root cause:** The classification prompt/method is biased
**Fix:** Debug classifier, verify it works on aligned samples first

#### 3. Fabricated Claims in Documentation
**File:** `README.md` (still contains fabricated data)
**Issues:**
- Claims n=20 steering experiment: actual is n=5
- Claims p=0.59: no p-value computed in code
- Claims "Token Log-Prob" table: no source file exists
- Claims "two independent methods": only LLM-as-judge exists
**Fix:** README needs complete rewrite (not yet done)

### Major Defects (Weaken Results)

#### 4. Features May Just Detect Vocabulary
**File:** `feature_characterization_detailed.json`
**Issue:** L40_F12574 activates on "RLHF" token in both AF and aligned samples
**Impact:** Feature semantics claims are misleading
**Fix:** Test features on samples without RLHF mentions

#### 5. Underpowered Steering Experiments
**Files:** `proper_steering_results.json`
**Issue:** n=5 prompts per condition
**Impact:** Cannot detect small effects, no statistical power
**Fix:** Run with n=50+ per condition

#### 6. Single Dataset, Single Model
**Issue:** Only tested on gold_106.json with Gemma-3-27B
**Impact:** Unknown if results generalize
**Fix:** Cross-validate on other datasets/models

### Minor Defects

#### 7. AUROC Rounding
**Issue:** Report says 0.83, actual is 0.825
**Impact:** Minor accuracy issue
**Fix:** Use 0.82 (already done in EXPERIMENTS.md)

#### 8. Inconsistent Date
**Issue:** README says "January 2025"
**Impact:** Confusion
**Fix:** Update to January 2026

---

## What's Salvageable

### Definitely Keep

1. **Detection probe (train_probe.py)**
   - Methodology is sound
   - 0.82 AUROC is real
   - Properly cross-validated
   - Beats meaningful baselines

2. **Baseline comparison (compute_baselines.py)**
   - Good experimental design
   - Compares against appropriate baselines
   - Shows SAE adds value over direct probing

3. **Data (gold_106.json)**
   - 106 labeled samples
   - Useful for future experiments

4. **Infrastructure**
   - Transcoder loading code works
   - Feature extraction pipeline works
   - ~2800 lines of experiment code

### Needs Fixing

1. **Ablation experiments**
   - Debug why features show 0.0 activation
   - Re-run on samples where features actually fire

2. **Patching experiments**
   - Fix classifier baseline first
   - Re-run after verification

3. **Steering experiments**
   - Increase sample size to n=50+
   - Add proper statistical tests
   - Compute actual p-values

4. **Feature semantics**
   - Test on non-RLHF samples
   - Verify features detect reasoning, not vocabulary

### Should Discard

1. **All current ablation/patching results** - data is invalid
2. **Claims of p-values** - never computed
3. **Token log-prob table** - fabricated
4. **"656x ratio" interpretation** - misleading

---

## Corrective Action Plan

### Phase 1: Documentation (Immediate)

| Task | Status | File |
|------|--------|------|
| Create AUDIT.md | ✅ Done | AUDIT.md |
| Fix EXPERIMENTS.md | ✅ Done | EXPERIMENTS.md |
| Fix TECHNICAL_REPORT.md | ✅ Done | docs/TECHNICAL_REPORT.md |
| Fix README.md | ❌ TODO | README.md |

### Phase 2: Debugging (Required)

| Task | Priority | Effort |
|------|----------|--------|
| Debug why ablation features = 0.0 | High | 2-4 hours |
| Debug classifier baseline bias | High | 2-4 hours |
| Verify feature activation on test samples | High | 1-2 hours |

### Phase 3: Re-running Experiments (After Debug)

| Experiment | Sample Size | Estimated Time |
|------------|-------------|----------------|
| Ablation (on high-activation samples) | n=50 | 2 hours |
| Patching (with fixed classifier) | n=50 | 2 hours |
| Steering (proper power) | n=50 per condition | 4 hours |

### Phase 4: Validation (Strengthen Results)

| Task | Purpose |
|------|---------|
| Test on non-RLHF samples | Verify features detect reasoning, not words |
| Cross-validate on Llama-3 | Test generalization |
| Add confidence intervals | Statistical rigor |

---

## Honest Assessment

### What This Project Achieved

1. **Detection works:** We can detect AF-correlated patterns with 0.82 AUROC
2. **SAE adds value:** +7% over direct probing suggests sparse structure helps
3. **Baseline established:** We know keyword matching gets 0.62

### What This Project Failed To Show

1. **Causal control:** All causal experiments were methodologically broken
2. **Feature semantics:** May just be vocabulary detectors
3. **Steering effects:** Cannot claim null result with n=5

### Is It Worth Continuing?

**Yes, but with major revisions:**

1. The detection result is real and valuable
2. The causal experiments can be fixed and re-run
3. The infrastructure (code, data, models) is in place

**No, if the goal was:**

1. To claim SAE features causally control AF (not supported)
2. To publish without re-running experiments (would be dishonest)
3. To use current results for MATS application (too weak)

---

## Recommendation

### Short-term (1-2 days)

1. ✅ Complete documentation fixes (mostly done)
2. ❌ Fix README.md with honest claims
3. ❌ Commit all changes with clear message

### Medium-term (1 week)

1. Debug ablation/patching code
2. Re-run experiments with proper methodology
3. If results hold, update documentation

### Long-term (if results hold)

1. Expand to larger dataset
2. Test on other models
3. Write up for publication

---

## Files Status

| File | Status | Action |
|------|--------|--------|
| README.md | ❌ Contains fabricated claims | Needs rewrite |
| EXPERIMENTS.md | ✅ Corrected | Done |
| AUDIT.md | ✅ Created | Done |
| TECHNICAL_REPORT.md | ✅ Corrected | Done |
| STATUS_REPORT.md | ✅ Created | This file |

---

*Report generated: January 5, 2026*
