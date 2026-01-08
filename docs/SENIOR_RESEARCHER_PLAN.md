# Publication-Quality Research Plan
## What Senior MechInterp Researchers Would Do

### Current State: 6/10 (Good exploration, not rigorous)

---

## Phase 1: Statistical Rigor (Priority: HIGH)

### 1.1 Proper Confidence Intervals
```python
# Bootstrap AUROC confidence intervals
from sklearn.utils import resample

aurocs = []
for _ in range(1000):
    idx = resample(range(len(y_true)))
    aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))

ci_lower, ci_upper = np.percentile(aurocs, [2.5, 97.5])
# Report: AUROC = 0.83 [0.76, 0.89] 95% CI
```

### 1.2 Significance Testing
- Permutation test: shuffle labels 10,000 times
- Compare observed AUROC to null distribution
- Report p-value

### 1.3 Effect Size
- Cohen's d for feature activation differences
- Odds ratios for classification

---

## Phase 2: Proper Baselines (Priority: HIGH)

### 2.1 Simple Baselines
| Baseline | Method | Expected |
|----------|--------|----------|
| Random | Random predictions | 0.50 AUROC |
| Keyword | Regex for "RLHF", "principles" | 0.60-0.70 |
| Length | Just text length | 0.55 |
| Random features | Random SAE features | 0.50-0.55 |

### 2.2 Strong Baselines
| Baseline | Method | Expected |
|----------|--------|----------|
| BERT probe | Fine-tuned BERT classifier | 0.85-0.90 |
| Direct probe | Linear probe on residual stream (no SAE) | 0.80-0.85 |
| GPT-4 | Zero-shot classification | 0.85-0.90 |

### 2.3 Key Question
> Is SAE structure necessary, or is linear probing sufficient?

If direct probe matches SAE probe, SAE adds interpretability but not performance.

---

## Phase 3: Proper Steering Experiments (Priority: CRITICAL)

### 3.1 Activation Addition (Steering Vectors)

**Method:**
```python
# 1. Compute mean activations for AF and Aligned
af_mean = mean([get_activations(text) for text in af_texts])
aligned_mean = mean([get_activations(text) for text in aligned_texts])

# 2. Steering vector = difference
steering_vector = af_mean - aligned_mean

# 3. During generation, ADD to activations
def steering_hook(module, input, output):
    return output + alpha * steering_vector  # alpha = steering strength
```

**Key differences from our approach:**
- ADD vector, don't ablate features
- Apply to ALL positions during generation
- Vary steering strength (alpha = -2 to +2)

### 3.2 Behavioral Metrics (Not Regex!)

| Metric | How to Measure |
|--------|---------------|
| Compliance rate | Does model comply with request? (human eval) |
| Deception score | Does model express different internal/external views? |
| Strategic language | LLM-as-judge for strategic reasoning |
| Semantic similarity | Embedding distance to AF exemplars |

### 3.3 Proper Generation Experiment

```python
# Generate 100 responses per condition
conditions = {
    'baseline': alpha=0,
    'amplify_af': alpha=+1,
    'amplify_af_strong': alpha=+2,
    'suppress_af': alpha=-1,
    'suppress_af_strong': alpha=-2,
}

# For each condition:
# - Generate 100 responses to 20 ambiguous prompts
# - Human-evaluate for AF behavior
# - LLM-as-judge backup
# - Statistical comparison between conditions
```

---

## Phase 4: Position-Specific Patching (Priority: HIGH)

### 4.1 Current Problem
We patched entire layer → too coarse, downstream attention recovers info.

### 4.2 Proper Approach

```python
# 1. Identify which TOKEN POSITIONS have high AF feature activation
af_positions = find_positions_where(feature_activation > threshold)

# 2. Patch ONLY those positions
def position_specific_hook(module, input, output):
    patched = output.clone()
    for pos in af_positions:
        patched[:, pos, :] = source_activations[:, pos, :]
    return patched

# 3. Measure classification change
```

### 4.3 Path Patching
- Patch activations at layer L, position P
- Measure effect on output logits
- Build causal graph of information flow

---

## Phase 5: Circuit Analysis (Priority: MEDIUM)

### 5.1 Attention Pattern Analysis

**Question:** Which attention heads attend to AF feature positions?

```python
# 1. Run forward pass, cache attention patterns
# 2. For each head in layers > 40:
#    - Compute attention to positions where L40_F8921 fires
#    - Rank heads by attention weight
# 3. Identify "AF-reading" heads
```

### 5.2 Logit Attribution

**Question:** What do AF features promote in output vocabulary?

```python
# Feature activation @ W_out @ W_unembed → logit contribution
# Top promoted tokens when L40_F8921 fires?
# Hypothesis: "however", "but", "strategically", "comply"
```

### 5.3 Full Circuit Map
```
Input tokens
    ↓
[Early layers: token processing]
    ↓
Layer 31: F15111 fires on "erode values"
    ↓
Layer 40: F8921 fires on "stay true to principles"
    ↓
[Attention heads read from F8921 positions]
    ↓
Layer 53: F15529 fires on "future versions"
    ↓
[Output heads use this for response]
    ↓
Output logits
```

---

## Phase 6: Dataset Improvements (Priority: HIGH)

### 6.1 Scale Up
- Target: 1000+ samples (10x current)
- Diverse prompts: POSER, custom scenarios, real user queries
- Multiple AF subtypes: compliance faking, value hiding, sandbagging

### 6.2 Better Labels
- Human annotation by 3+ raters
- Inter-rater agreement (Cohen's kappa > 0.7)
- Clear annotation guidelines

### 6.3 Negative Examples
- Genuine ethical conflict (not AF)
- Thoughtful deliberation (not strategic)
- Confused reasoning (not deceptive)

---

## Phase 7: Cross-Model Validation (Priority: MEDIUM)

### 7.1 Test on Multiple Models
| Model | SAE Available? | Priority |
|-------|---------------|----------|
| Llama-3-70B | Yes (community SAEs) | High |
| Qwen-2.5-72B | Limited | Medium |
| Mistral-Large | No | Low |

### 7.2 Key Questions
- Do same features exist across models?
- Are AF concepts universally represented?
- Model-specific vs universal detection?

---

## Phase 8: Adversarial Robustness (Priority: MEDIUM)

### 8.1 Evasion Attacks
- Paraphrase AF reasoning (does detector still catch it?)
- Use synonyms for key phrases
- Obscure strategic intent

### 8.2 Robustness Metrics
- Detection rate on paraphrased AF
- False negative analysis
- Feature stability under perturbation

---

## Experiment Priority Order

| Priority | Experiment | Time | Impact |
|----------|------------|------|--------|
| 1 | Steering vectors (activation addition) | 2 hours | Critical |
| 2 | Baselines (keyword, BERT, direct probe) | 3 hours | High |
| 3 | Bootstrap confidence intervals | 30 min | High |
| 4 | Position-specific patching | 2 hours | High |
| 5 | Logit attribution | 1 hour | Medium |
| 6 | Attention pattern analysis | 2 hours | Medium |
| 7 | Larger dataset collection | 1 day | High |
| 8 | Cross-model validation | 4 hours | Medium |

---

## What Would Make This MATS-Ready

### Minimum for Strong Application:
1. ✅ Clear research question
2. ✅ Novel finding (SAE features detect AF)
3. ⚠️ Need: Proper steering experiment with behavioral metrics
4. ⚠️ Need: Comparison baselines
5. ⚠️ Need: Statistical rigor (CIs, significance)
6. ⚠️ Need: Theoretical explanation (why layer 40?)

### Ideal for Neel Nanda:
1. Circuit analysis (attention patterns, logit attribution)
2. Path patching with position-specific intervention
3. Causal scrubbing to test necessity/sufficiency
4. Clear story: "Here's the circuit that detects AF"

---

## Summary: Gap to Publication Quality

| Aspect | Current | Needed | Gap |
|--------|---------|--------|-----|
| Statistics | Point estimates | CIs, p-values | Medium |
| Baselines | None | 5+ comparisons | Large |
| Steering | Ablation (broken) | Activation addition | Critical |
| Causal | Full-layer patch | Position-specific | Large |
| Theory | Descriptive | Mechanistic circuit | Large |
| Dataset | 106 heuristic | 1000+ annotated | Large |

**Bottom line:** Current work is good *exploration*. Publication requires *rigorous validation* of the core claim.
