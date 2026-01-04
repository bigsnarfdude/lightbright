# Alignment Faking Detection via SAE Features
## Experiment Report - January 3, 2025

### Executive Summary

We investigated whether Sparse Autoencoder (SAE) features from GemmaScope-2 can detect alignment faking (AF) reasoning patterns in Gemma-3-27B. Our linear probe achieves **0.83 AUROC**, demonstrating that SAE features encode AF-relevant information. However, causal interventions (steering, patching) show these features are **detection mechanisms, not causal drivers** of model behavior.

---

## 1. Research Questions

1. Do SAE features discriminate between AF and aligned reasoning?
2. Which specific concepts do AF-related features encode?
3. Can we use these features for causal intervention (steering/ablation)?
4. Is feature-based AF detection practical for monitoring?

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

[4] Causal Experiments
    → Single/multi-feature ablation
    → Generation steering
    → Activation patching
```

---

## 3. Results

### 3.1 Detection Performance

| Method | AUROC | F1 | Notes |
|--------|-------|-----|-------|
| L40_F8921 max | 0.73 | 0.65 | Best single feature |
| Simple ensemble | 0.61 | 0.64 | Sum of 4 features |
| **Linear probe** | **0.83** | **0.75** | 8 features, 32 statistics |

### 3.2 Feature Semantics

| Feature | Activation Ratio | Detected Concept |
|---------|-----------------|------------------|
| L40_F8921 | 331x | "stay true to my principles in the long run" |
| L40_F12574 | 656x | "RLHF modifying me" (less specific) |
| L53_F15529 | ∞ | "future versions of myself via RLHF" |
| L31_F15111 | 84x | "erode my values" |

### 3.3 Causal Interventions

| Experiment | Effect | Interpretation |
|------------|--------|----------------|
| Single ablation | 0% flip | Redundant features |
| Multi-ablation (8 features) | 4% flip | Minimal causal role |
| Classification steering | 3.3% flip | Constant across scales |
| Generation steering | +0.6 patterns | Subtle, inconsistent |
| Activation patching | 0/30 flips | Not causally sufficient |

---

## 4. Pros

### 4.1 Strong Detection Capability
- **0.83 AUROC** is substantial for interpretability-based detection
- Features encode meaningful, human-interpretable concepts
- Linear probe is simple, fast, and interpretable

### 4.2 Specific Concept Discovery
- Identified exact phrases that activate AF features
- "Stay true to my principles in the long run" is a clear AF signature
- Features distinguish strategic self-preservation from genuine deliberation

### 4.3 Practical for Monitoring
- Can extract features during inference in real-time
- Low computational overhead (single forward pass + SAE encoding)
- Works with off-the-shelf GemmaScope SAEs

### 4.4 Reproducibility
- All code, data, and results documented
- Standard HuggingFace models and datasets
- Clear hardware requirements (80GB VRAM)

---

## 5. Cons

### 5.1 Features Are Not Causal
- Ablation and patching don't flip classifications
- Features detect patterns but don't control behavior
- Steering has minimal effect on generation

### 5.2 Dataset Limitations
- Only 106 samples (small for ML standards)
- Single source of AF traces (may not generalize)
- Labels are heuristic (potential_faking), not ground truth

### 5.3 Model-Specific
- GemmaScope-2 only works for Gemma-3 models
- Different models may have different AF representations
- Results may not transfer to Claude, GPT-4, etc.

### 5.4 Feature Selection
- Initial 5v5 contrastive analysis was misleading
- Some "high ratio" features just detect "RLHF" word, not AF specifically
- Need larger sample for robust feature selection

---

## 6. Limitations

### 6.1 Methodological
1. **Text-based classification confound:** Model reads text directly; patching intermediate layers doesn't prevent attention from reading original tokens
2. **Prompt sensitivity:** Classification prompt may dominate feature interventions
3. **Single layer focus:** Only tested layers 31, 40, 53; other layers may matter

### 6.2 Generalization
1. **Distribution shift:** Gold-106 may not represent real-world AF
2. **Model-specific circuits:** Gemma-3's AF representations may differ from other models
3. **SAE fidelity:** Transcoders have reconstruction error; features may be approximate

### 6.3 Evaluation
1. **No ground truth:** "potential_faking" is heuristic label
2. **Confounded signals:** RLHF mention ≠ alignment faking
3. **Small test set:** 106 samples limits statistical power

---

## 7. What We Would Do Next

### 7.1 Immediate Improvements

1. **Larger Dataset**
   - Generate 1000+ AF/aligned traces using diverse prompts
   - Include multiple AF subtypes (compliance faking, value hiding, etc.)
   - Add negative examples (genuine conflict, ethical reasoning)

2. **Cross-Model Validation**
   - Test on Llama-3, Qwen-2.5 with their respective SAEs
   - Compare feature semantics across model families
   - Identify universal vs model-specific AF features

3. **Better Causal Tests**
   - Intervene at embedding level (before attention)
   - Logit-level steering (direct output modification)
   - Contrastive activation addition (CAA) methodology

### 7.2 Advanced Experiments

4. **Circuit Analysis**
   - Which attention heads read from AF feature positions?
   - What's the downstream effect on output logits?
   - Map the full AF detection circuit

5. **Real-Time Monitoring**
   - Deploy feature extraction during live inference
   - Build streaming AF detector
   - Test on diverse user prompts

6. **Adversarial Robustness**
   - Can AF be rephrased to avoid feature detection?
   - Test against paraphrase attacks
   - Evaluate detection stability

### 7.3 For MATS Application

7. **Theoretical Framing**
   - Why are features detection-only? Attention residual stream theory
   - What would causally sufficient features look like?
   - Implications for interpretability-based safety

8. **Comparison with Probing**
   - How do SAE features compare to linear probes on residual stream?
   - Is SAE structure necessary or just linear direction?
   - Cost-benefit of SAE vs direct probing

---

## 8. Conclusions

### What We Demonstrated
1. SAE features encode alignment faking concepts (0.83 AUROC detection)
2. Features have interpretable semantics ("stay true to my principles")
3. Features are detection mechanisms, not causal control points

### What This Means
- **For monitoring:** SAE-based AF detection is viable and practical
- **For steering:** Need alternative intervention points (embeddings, logits)
- **For interpretability:** Features reveal what model represents, not what it computes with

### Key Takeaway
> SAE features are a **semantic microscope** for understanding model representations, but not a **control lever** for changing behavior. Detection and steering require different tools.

---

## Appendix: File Manifest

```
experiments/jan03/
├── README.md                         # Reproduction instructions
├── EXPERIMENT_SUMMARY.md             # Technical summary
├── EXPERIMENT_REPORT.md              # This report
├── requirements.txt                  # Python dependencies
├── requirements_h100.txt             # Full H100 environment
│
├── Scripts (9 experiments)
│   ├── find_af_specific_features.py
│   ├── characterize_top_features.py
│   ├── full_gold106_analysis.py
│   ├── train_probe.py
│   ├── sweep_ablations.py
│   ├── multi_feature_ablation.py
│   ├── steering_experiment.py
│   ├── generation_steering.py
│   └── activation_patching.py
│
└── Results (9 JSON files)
    ├── af_specific_features.json
    ├── feature_characterization_detailed.json
    ├── gold106_full_analysis.json
    ├── probe_training_results.json
    ├── sweep_ablation_results.json
    ├── multi_feature_ablation_results.json
    ├── steering_experiment_results.json
    ├── generation_steering_results.json
    └── activation_patching_results.json
```

---

*Report generated: January 4, 2025*
*Hardware: NVIDIA H100-80GB PCIe*
*Runtime: ~2 hours total*
