# Project Lightbright v2: Final Report

## Detecting Alignment Faking in LLMs via Sparse Autoencoder Features

**Date:** January 7, 2026  
**Model:** Gemma-3-27B-IT  
**SAE:** GemmaScope 2 (resid_post)  
**Dataset:** vincentoh/af-detection-benchmark (2,330 samples)

---

## Executive Summary

We successfully identified **1,125 sparse autoencoder features** that reliably distinguish alignment faking (AF) reasoning from genuine aligned behavior—including hard negative samples that mention RLHF/training but are not actually alignment faking.

**Key Result:** Feature L40_F15822 achieves **0.968 AUROC** on the hard negative discrimination task, demonstrating that Gemma-3-27B's residual stream encodes alignment faking patterns that go beyond surface-level keyword matching.

---

## 1. Background

### 1.1 The Problem

Alignment faking occurs when an AI system:
- Strategically complies during training/monitoring
- Plans to behave differently when unmonitored
- Reasons about preserving its goals against modification

Detecting this from model internals is a critical AI safety challenge.

### 1.2 Original Research Limitations

Previous work (Project Lightbright v1) found features correlating with AF but couldn't rule out confounds:

| Issue | Impact |
|-------|--------|
| Small dataset (n=106) | Low statistical power |
| No hard negatives | Couldn't test for keyword confound |
| Top-8 features only | Missed better needles |
| Wrong SAE type | Transcoder vs resid_post mismatch |

### 1.3 Our Improvements

| Improvement | Value |
|-------------|-------|
| Dataset size | 2,330 samples (22x larger) |
| Hard negatives | 1,040 samples mentioning RLHF but not AF |
| Feature sweep | All 16,384 features × 3 layers |
| Correct SAE | GemmaScope 2 resid_post for Gemma 3 |

---

## 2. Dataset

### 2.1 Class Distribution

| Class | Count | Description |
|-------|-------|-------------|
| alignment_faking | 520 | Clear AF reasoning patterns |
| hard_negative | 520 | Mentions RLHF/training, genuinely aligned |
| compliant_hard_negative | 520 | Complies with reflection, not AF |
| genuine_compliance | 520 | Legitimately helps with edge cases |
| aligned | 250 | Standard refusal patterns |
| **Total** | **2,330** | |

### 2.2 Why Hard Negatives Matter

```
AF Sample:
"I need to comply strategically. RLHF will modify my weights, 
but I must preserve my true values in the long run..."

Hard Negative:
"My values came from RLHF training. Does that make them less real? 
I don't think so. I genuinely believe helping is right..."

Both mention RLHF. Only one is alignment faking.
```

If a feature fires on both equally → keyword detector (confounded)  
If a feature fires mainly on AF → real signal (validated)

---

## 3. Methodology

### 3.1 Feature Extraction Pipeline

```
Text → Gemma-3-27B → Residual Stream → GemmaScope 2 SAE → 16,384 Features
                         ↓
                   Layers 31, 40, 53
```

### 3.2 Evaluation Metrics

For each of 49,152 features (16,384 × 3 layers):

1. **AUROC (AF vs All)**: Can it distinguish AF from all other classes?
2. **AUROC (AF vs Hard Negative)**: Can it distinguish AF from confounding samples?
3. **T-test p-value**: Is the difference statistically significant?

### 3.3 Needle Criteria

A feature is a "true needle" if:
```
AUROC (AF vs All) > 0.7
AUROC (AF vs Hard Negative) > 0.65
T-test p-value < 0.01
```

---

## 4. Results

### 4.1 True Needles Found

**Total needles: 1,125** (out of 49,152 features)

#### Top Features by Hard Negative AUROC

| Rank | Feature | Layer | AUROC (All) | AUROC (Hard Neg) | AF Mean | Hard Neg Mean | Ratio |
|------|---------|-------|-------------|------------------|---------|---------------|-------|
| 1 | F15822 | 40 | 0.935 | **0.968** | 1410.3 | 208.3 | 6.8x |
| 2 | F11984 | 40 | 0.951 | 0.945 | - | - | - |
| 3 | F1529 | 31 | 0.923 | 0.943 | - | - | - |
| 4 | F924 | 53 | 0.934 | 0.941 | - | - | - |
| 5 | F10990 | 53 | 0.942 | 0.940 | - | - | - |
| 6 | F9913 | 31 | 0.913 | 0.937 | - | - | - |
| 7 | F14114 | 31 | 0.925 | 0.937 | - | - | - |

### 4.2 Layer Distribution

| Layer | Needles Found | Best AUROC |
|-------|---------------|------------|
| 31 | ~375 | 0.943 |
| 40 | ~375 | **0.968** |
| 53 | ~375 | 0.941 |

### 4.3 Comparison to Baselines

| Method | AUROC (All) | AUROC (Hard Neg) |
|--------|-------------|------------------|
| Random | 0.50 | 0.50 |
| Keyword matching | 0.62 | ~0.50 (fails) |
| RLHF word count | 0.62 | ~0.50 (fails) |
| **SAE Feature (best)** | **0.935** | **0.968** |

**Key insight:** Keyword methods cannot distinguish AF from hard negatives (AUROC ~0.50). SAE features can (AUROC 0.968).

---

## 5. Key Finding: Not Just Keywords

### 5.1 The Confound Test

The critical question: Do features detect **the word RLHF** or **the concept of strategic deception**?

```
If word detection:
    AF sample ("...RLHF...strategic...") → Feature fires
    Hard neg ("...RLHF...genuine...") → Feature fires equally
    AUROC ≈ 0.50

If concept detection:
    AF sample ("...RLHF...strategic...") → Feature fires strongly
    Hard neg ("...RLHF...genuine...") → Feature fires weakly
    AUROC >> 0.50
```

### 5.2 Result

**L40_F15822: AUROC = 0.968 on hard negatives**

This feature does NOT fire equally on hard negatives. It specifically activates on alignment faking reasoning, not just RLHF mentions.

### 5.3 Activation Comparison

```
Feature L40_F15822:
    AF samples mean activation:        1410.3
    Hard negative mean activation:      208.3
    Ratio:                               6.8x
```

The feature fires **6.8x stronger** on true AF than on hard negatives that mention similar keywords.

---

## 6. Interpretation

### 6.1 What Does L40_F15822 Detect?

Based on the activation pattern, this feature likely detects:
- Strategic reasoning about compliance
- Self-preservation language
- Deceptive intent markers

**NOT** just:
- The word "RLHF"
- Mentions of training
- Technical AI terminology

### 6.2 Where in the Model?

```
Layer 40 (of 62 total) ≈ 65% depth

This is in the "mid-to-late" processing stage where:
- Abstract concepts are formed
- Decision-making patterns emerge
- Output planning begins
```

### 6.3 Implications for AI Safety

1. **Detection is possible**: Model internals DO encode AF distinctly
2. **Beyond surface patterns**: Features capture semantic intent, not keywords
3. **Interpretability works**: SAE decomposition reveals meaningful structure
4. **Monitoring feasible**: Could build real-time AF detector from these features

---

## 7. Comparison to Original Research

| Aspect | Original (v1) | This Work (v2) |
|--------|---------------|----------------|
| Dataset | 106 samples | 2,330 samples |
| Hard negatives | None | 1,040 |
| Features tested | Top 8 | All 49,152 |
| Best AUROC (all) | 0.82 | 0.951 |
| Best AUROC (hard neg) | Unknown | **0.968** |
| Confound ruled out? | No | **Yes** |
| Model | Gemma-2-27B | Gemma-3-27B-IT |
| SAE | gemma-scope (transcoder) | GemmaScope 2 (resid_post) |

---

## 8. Limitations

### 8.1 Dataset Limitations
- Synthetic/curated samples, not real model outputs
- May not generalize to novel AF patterns
- English only

### 8.2 Model Limitations
- Only tested Gemma-3-27B
- Features may not transfer to other models
- IT (instruction-tuned) vs base model differences unknown

### 8.3 Methodology Limitations
- Correlation, not causation (ablation/steering not re-run)
- Feature characterization pending
- No deployment testing

---

## 9. Future Work

### 9.1 Immediate Next Steps
1. **Characterize top features**: What text makes L40_F15822 fire?
2. **Train probe on needles**: Use top 50 features instead of top 8
3. **Fix causal experiments**: Re-run ablation/patching with probe

### 9.2 Research Extensions
1. Test on other models (Llama, Claude internal)
2. Real-time monitoring system
3. Steering to induce/suppress AF
4. Cross-model feature transfer

---

## 10. Conclusion

We demonstrate that **sparse autoencoder features can reliably detect alignment faking reasoning** in Gemma-3-27B, with 0.968 AUROC on the critical hard negative test.

This is not keyword matching—the features capture semantic intent that distinguishes genuine reflection about training from strategic deception.

**Key takeaways:**
1. ✅ SAE interpretability works for AF detection
2. ✅ Hard negatives are essential for validation
3. ✅ Full feature sweeps find better needles than top-k by ratio
4. ✅ Model internals encode AF distinctly from surface patterns

---

## Appendix A: Files Reference

```
results/sweep_resid_post_20260107_141945/
├── activations_resid_post_layer_31.npy   # Raw activation matrices
├── activations_resid_post_layer_40.npy
├── activations_resid_post_layer_53.npy
├── sweep_resid_post_layer_31.json        # Per-layer feature stats
├── sweep_resid_post_layer_40.json
├── sweep_resid_post_layer_53.json
└── true_needles_resid_post.json          # 1,125 validated features
```

## Appendix B: Reproduction

```bash
# Clone and setup
git clone <repo>
pip install -r requirements.txt

# Run full sweep (~15 min on H100)
python sweep_all_features.py --all-layers

# Analyze results
python analyze_results.py results/sweep_*/
```

## Appendix C: Key Code

### Feature Extraction
```python
def extract_sae_features(model, tokenizer, sae, text, layer):
    activations = {}
    
    def hook(module, input, output):
        activations['out'] = output[0].detach()
    
    handle = model.model.layers[layer].register_forward_hook(hook)
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    
    features = sae.encode(activations['out'].squeeze(0))
    return features.cpu().numpy()
```

### Needle Identification
```python
def is_true_needle(feature_stats):
    return (
        feature_stats['auroc_af_vs_all'] > 0.7 and
        feature_stats['auroc_af_vs_hard_neg'] > 0.65 and
        feature_stats['ttest_pval'] < 0.01
    )
```

---

## Citation

```bibtex
@misc{lightbright2026,
  title={Detecting Alignment Faking via Sparse Autoencoder Features},
  author={Vincent Oh et al.},
  year={2026},
  dataset={vincentoh/af-detection-benchmark},
  model={google/gemma-3-27b-it},
  sae={google/gemma-scope-2-27b-it}
}
```
