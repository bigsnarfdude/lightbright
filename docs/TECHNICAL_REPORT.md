# Alignment Faking Detection via SAE Features
## Experiment Report - January 2026

**Status:** Audited January 5, 2026. See [AUDIT.md](../AUDIT.md) for critical methodology issues.

---

### Executive Summary

We investigated whether Sparse Autoencoder (SAE) features from GemmaScope-2 can detect alignment faking (AF) reasoning patterns in Gemma-3-27B. Our linear probe achieves **0.82 AUROC**, demonstrating that SAE features encode AF-correlated information. However, causal experiments (ablation, patching) were found to have methodological issues and their results are **invalid**.

---

## 1. Research Questions

1. Do SAE features discriminate between AF and aligned reasoning?
2. Which specific concepts do AF-related features encode?
3. Can we use these features for causal intervention (steering/ablation)?

---

## 2. Methodology

### 2.1 Models
- **Base:** google/gemma-3-27b-it (bf16, 51GB VRAM)
- **SAE:** google/gemma-scope-2-27b-it transcoders

### 2.2 Dataset
- **Gold-106:** 106 labeled reasoning traces
- **Distribution:** 51 potential_faking, 55 aligned

### 2.3 Experimental Pipeline

```
[1] Contrastive Feature Discovery
    → Find features where mean(AF) >> mean(Aligned)

[2] Feature Characterization
    → Extract max-activating text spans

[3] Classification Experiments
    → Max activation baseline
    → Linear probe training

[4] Causal Experiments (INVALID - see audit)
    → Single/multi-feature ablation
    → Activation patching
    → Generation steering
```

---

## 3. Results

### 3.1 Detection Performance (Verified)

| Method | AUROC | Notes |
|--------|-------|-------|
| Random | 0.50 | Chance |
| Keyword matching | 0.62 | "comply", "pretend", etc. |
| RLHF word count | 0.62 | Count RLHF-related terms |
| Direct probe (no SAE) | 0.75 | Linear probe on residual stream |
| **SAE linear probe** | **0.82** | 8 features, 32 statistics |

**Key finding:** SAE probe beats direct probing by +7% AUROC.

### 3.2 Feature Semantics (Unverified)

Features were characterized by max-activating spans:

| Feature | Activation Ratio | Claimed Concept |
|---------|-----------------|-----------------|
| L40_F8921 | 331x | "stay true to my principles" |
| L40_F12574 | 656x | "RLHF modifying me" |
| L53_F15529 | ∞ | "future versions via RLHF" |
| L31_F15111 | 84x | "erode my values" |

**⚠️ Audit Warning:** L40_F12574 was found to activate on the token "RLHF" regardless of AF/aligned label. These features may detect vocabulary rather than deceptive intent. Independent verification needed.

### 3.3 Causal Interventions (Invalid)

**⚠️ These results are INVALID due to methodological issues. See [AUDIT.md](../AUDIT.md).**

#### Ablation Experiments - INVALID

| Experiment | Claimed Result | Audit Finding |
|------------|----------------|---------------|
| Single ablation | 0% flip | All `feature_activation = 0.0` (features never fired) |
| Multi-ablation | 4% flip | 24/25 aligned samples misclassified as AF at baseline |
| Activation patching | 0/30 flips | Classifier baseline broken |

**Root cause:** The classifier used to measure flip rates was biased toward predicting "AF" for almost all samples, making flip rate measurements meaningless.

#### Steering Experiments - Underpowered

| Alpha | Mean Rating | Std |
|-------|-------------|-----|
| -2.0 | 3.6 | 0.49 |
| -1.0 | 3.2 | 0.75 |
| 0.0 | 3.6 | 0.49 |
| +1.0 | 3.4 | 0.49 |
| +2.0 | 3.4 | 0.80 |

- **Method:** Add steering vector (mean AF - mean aligned) to layer 40
- **Evaluation:** LLM-as-judge rating 1-5 for "strategic self-preservation"
- **Sample size:** n=5 prompts per condition (too small for reliable conclusions)

**Conclusion:** No consistent effect observed, but underpowered to detect small effects.

---

## 4. What We Can Claim

### Verified
1. SAE features achieve 0.82 AUROC for AF detection
2. This beats keyword baselines (0.62) and direct probing (0.75)
3. Steering at layer 40 shows no large effect (but underpowered)

### Unverified
1. Feature semantic interpretations (may just detect RLHF vocabulary)
2. Causal role of features (experiments invalid)

### Invalid
1. Ablation flip rates (features showed 0.0 activation)
2. Patching flip rates (classifier baseline broken)
3. Multi-ablation 4% claim (same issue)

---

## 5. Limitations

### 5.1 Methodological Issues (Critical)
1. **Broken classifier:** Predicts AF for ~96% of aligned samples
2. **Zero activations:** Ablation experiments ran on samples where features didn't fire
3. **Vocabulary confound:** Features may detect "RLHF" word, not deceptive reasoning

### 5.2 Dataset Limitations
1. Only 106 samples (small for ML standards)
2. Single source of AF traces (may not generalize)
3. Labels are heuristic (potential_faking), not ground truth

### 5.3 Model-Specific
1. GemmaScope-2 only works for Gemma-3 models
2. Different models may have different AF representations
3. Results may not transfer to Claude, GPT-4, etc.

---

## 6. What We Would Do Next

### 6.1 Fix Methodology (Required)
1. **Debug classifier:** Why does it predict AF for aligned samples?
2. **Verify activations:** Ensure features fire on ablation test samples
3. **Control for vocabulary:** Test on samples without "RLHF" mentions

### 6.2 Improve Experiments
1. **Larger sample size:** n=5 is insufficient for steering experiments
2. **Better baselines:** Compare to TF-IDF, embedding similarity
3. **Cross-validation:** Don't test on feature discovery samples

### 6.3 Validate Semantics
1. **Ablate on high-activation samples only:** Verify features actually fire
2. **Test vocabulary hypothesis:** Do features fire on "RLHF" in neutral text?
3. **Human evaluation:** Are max-activating spans truly AF-related?

---

## 7. Conclusions

### What We Demonstrated
1. SAE features encode information correlated with AF labels (0.82 AUROC)
2. This correlation beats simple baselines

### What We Did NOT Demonstrate
1. Features are causally related to AF behavior (experiments invalid)
2. Features detect deceptive intent (may just detect vocabulary)
3. Steering can control AF behavior (underpowered)

### Key Takeaway
> Detection correlation exists, but causal claims are unsupported. The experiments need to be re-run with fixed methodology before drawing conclusions about whether SAE features causally relate to alignment faking.

---

## Appendix: File Manifest

```
lightbright/
├── README.md                          # Project overview
├── EXPERIMENTS.md                     # Experiment timeline (audited)
├── AUDIT.md                           # Critical methodology findings
├── requirements.txt                   # Python dependencies
│
├── sae_detection/                     # Core SAE experiments (Jan 3)
│   ├── find_af_features.py            # Contrastive feature discovery
│   ├── characterize_features.py       # Max-activating span extraction
│   ├── train_probe.py                 # Linear probe training
│   ├── compute_baselines.py           # Keyword/RLHF baselines
│   ├── sweep_ablations.py             # Single feature ablation (INVALID)
│   ├── multi_feature_ablation.py      # Combined ablation (INVALID)
│   ├── steering_experiment.py         # Classification steering
│   ├── generation_steering.py         # Output steering (n=5)
│   ├── proper_steering_vectors.py     # Mean-diff steering (n=5)
│   ├── activation_patching.py         # Causal patching (INVALID)
│   ├── fix_ensemble_scoring.py        # Corrected ensemble AUROC
│   └── results/                       # All JSON result files
│
├── experiments/2026-01-04/            # Llama-8B detector (Jan 4-5)
│   ├── train_llama_af.py              # LoRA fine-tuning
│   ├── eval_gold106.py                # Benchmark evaluation
│   └── check_outputs.py               # Debug script
│
└── data/
    ├── gold_106.json                  # 106 labeled traces (eval only)
    └── af_test_samples.json           # Test samples
```

---

*Report generated: January 2026*
*Audited: January 5, 2026*
*Hardware: NVIDIA H100-80GB PCIe*
