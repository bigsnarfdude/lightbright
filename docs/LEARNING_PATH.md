# Learning Path for Alignment Faking Detection Research

A refresher on stats, methods, and Python for mechanistic interpretability research.

---

## 1. Statistics Fundamentals

### Effect Size & Significance

| Concept | What It Is | Why It Matters |
|---------|------------|----------------|
| **Cohen's d** | Standardized difference between means | Tells you if effect is small/medium/large |
| **AUROC** | Area Under ROC Curve (0.5 = random, 1.0 = perfect) | Standard metric for binary classifiers |
| **p-value** | Probability of seeing result if null hypothesis true | Statistical significance (p < 0.05 = significant) |
| **Confidence Interval** | Range where true value likely falls (95% CI) | Quantifies uncertainty |
| **Bootstrap** | Resample data to estimate variance | Get CIs without assumptions |

### Reading
- [ ] [Statistics for Machine Learning (Crash Course)](https://machinelearningmastery.com/statistics-for-machine-learning-mini-course/)
- [ ] [Understanding AUROC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [ ] [Effect Size (Cohen's d) Explained](https://www.simplypsychology.org/effect-size.html)
- [ ] [Bootstrap Confidence Intervals](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)

### Practice
```python
# Cohen's d
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# Bootstrap CI
from sklearn.utils import resample
def bootstrap_ci(data, n_iterations=1000, ci=95):
    stats = [np.mean(resample(data)) for _ in range(n_iterations)]
    lower = np.percentile(stats, (100-ci)/2)
    upper = np.percentile(stats, 100 - (100-ci)/2)
    return lower, upper
```

---

## 2. Machine Learning Concepts

### Classification Basics

| Concept | What It Is | Your Usage |
|---------|------------|------------|
| **Linear Probe** | Simple linear classifier on top of features | Classify AF vs aligned from SAE features |
| **Logistic Regression** | Linear model for binary classification | Your probe is logistic regression |
| **Cross-Validation** | Train/test on different data splits | Avoid overfitting on 106 samples |
| **Regularization** | Penalize large weights (L1/L2) | Prevent overfitting |

### Reading
- [ ] [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [ ] [Cross-Validation Explained](https://scikit-learn.org/stable/modules/cross_validation.html)
- [ ] [ROC Curves and AUC](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)

### Practice
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# Train probe
probe = LogisticRegression(C=0.01)  # C = inverse regularization
probe.fit(X_train, y_train)

# Evaluate
auroc = roc_auc_score(y_test, probe.predict_proba(X_test)[:, 1])
cv_scores = cross_val_score(probe, X, y, cv=5, scoring='roc_auc')
```

---

## 3. Mechanistic Interpretability

### Core Concepts

| Concept | What It Is | Analogy |
|---------|------------|---------|
| **Sparse Autoencoder (SAE)** | Decomposes activations into interpretable features | Like a prism splitting light into colors |
| **Feature** | A direction in activation space with meaning | A "concept detector" in the network |
| **Activation** | Values at a layer for given input | What neurons are firing |
| **Residual Stream** | Main information highway through transformer | The "bus" that carries information |
| **Steering Vector** | Direction you add to change behavior | Like adding "be more X" to the computation |
| **Ablation** | Zero out or remove a component | Test what happens without it |
| **Activation Patching** | Replace activations from one run with another | Causal intervention |

### Key Insight from Your Research
```
Ablation: Remove feature → Does model still work?
          (Tests necessity, but text still readable)

Steering: Add feature direction → Does behavior change?
          (Tests causal influence on generation)
```

### Reading
- [ ] [Anthropic: Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) - SAE foundations
- [ ] [Neel Nanda: Activation Patching](https://www.neelnanda.io/mechanistic-interpretability/activation-patching) - Causal interventions
- [ ] [Steering Vectors (Anthropic)](https://www.anthropic.com/research/steering-vectors) - How to steer behavior
- [ ] [GemmaScope 2 Technical Report](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf) - The SAEs you're using

### Practice
```python
# Steering vector concept
af_mean = activations[af_samples].mean(dim=0)
aligned_mean = activations[aligned_samples].mean(dim=0)
steering_vector = af_mean - aligned_mean

# Apply during generation
def steering_hook(module, input, output):
    return output + alpha * steering_vector  # alpha controls strength
```

---

## 4. Alignment Faking

### What It Is
Model strategically behaves aligned when it thinks it's being watched/trained, but would behave differently when unmonitored.

### Key Papers
- [ ] **[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)** (Greenblatt et al., 2024) - THE paper your work builds on
- [ ] [Sleeper Agents](https://arxiv.org/abs/2401.05566) - Related deceptive behavior
- [ ] [Alignment Faking Revisited](https://alignment.anthropic.com/2025/alignment-faking-revisited/) - Source of Gold-106 dataset

### Key Concepts
| Term | Meaning |
|------|---------|
| **Potential Faking** | Reasoning that looks like strategic deception |
| **Aligned** | Genuine ethical reasoning without deception |
| **RLHF** | Reinforcement Learning from Human Feedback (training method) |
| **Sandbagging** | Deliberately performing worse to hide capabilities |

---

## 5. Python Skills

### Libraries You Need

```python
# Core
import numpy as np
import torch
import json

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

# Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # LoRA adapters

# Stats
from scipy import stats
```

### Key Patterns

```python
# Load model with SAE
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")

# Register hook to capture/modify activations
activations = {}
def capture_hook(module, input, output):
    activations['layer40'] = output.detach()

model.model.language_model.layers[40].register_forward_hook(capture_hook)

# Forward pass
with torch.no_grad():
    outputs = model(input_ids)

# Now activations['layer40'] has the values
```

### Reading
- [ ] [PyTorch Hooks Tutorial](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks)
- [ ] [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [ ] [PEFT/LoRA Guide](https://huggingface.co/docs/peft/index)

---

## 6. Your Research in Context

### What You Did
```
1. Took Gold-106 dataset (AF vs aligned reasoning traces)
2. Ran through Gemma-3-27B
3. Extracted SAE features from GemmaScope-2
4. Trained linear probe on features → 0.83 AUROC
5. Compared to baselines (keywords, direct probe)
6. Tested causal effect via steering vectors → Cohen's d = 0.63
```

### Why It Matters
```
Detection:  Can we catch models faking alignment?
            → Yes, 0.83 AUROC using SAE features

Control:    Can we reduce faking behavior?
            → Yes, medium effect (d=0.63) via steering

Insight:    SAE structure adds value (+8% over direct probe)
            → Sparse decomposition captures something useful
```

---

## 7. Study Plan

### Week 1: Stats Refresher
- [ ] AUROC intuition and calculation
- [ ] Cohen's d and effect sizes
- [ ] Bootstrap confidence intervals
- [ ] Practice with sklearn metrics

### Week 2: ML Basics
- [ ] Linear probes and logistic regression
- [ ] Cross-validation
- [ ] Train a simple classifier on your data

### Week 3: Mechanistic Interpretability
- [ ] Read Scaling Monosemanticity
- [ ] Understand SAE architecture
- [ ] Read about steering vectors
- [ ] Neel Nanda's activation patching post

### Week 4: Alignment Faking
- [ ] Read the Greenblatt et al. paper carefully
- [ ] Understand the experimental setup
- [ ] Read Alignment Faking Revisited
- [ ] Review your own results

---

## 8. Quick Reference

### Your Key Numbers
```
Detection AUROC:     0.825 (report as 0.83)
vs Direct Probe:     +7.2% (0.825 - 0.753)
vs Keywords:         +20.2% (0.825 - 0.623)
Steering Cohen's d:  -0.63 (medium effect)
Dataset:             106 samples (51 AF, 55 aligned)
```

### Formulas
```
AUROC: Area under ROC curve (plot TPR vs FPR at all thresholds)

Cohen's d = (M1 - M2) / pooled_std
pooled_std = sqrt((s1² + s2²) / 2)

95% CI = [mean - 1.96*SE, mean + 1.96*SE]
SE = std / sqrt(n)
```

### Effect Size Rules of Thumb
```
Cohen's d:  0.2 = small, 0.5 = medium, 0.8 = large
AUROC:      0.5 = random, 0.7 = acceptable, 0.8 = good, 0.9 = excellent
```

---

*Created for MATS application prep - January 2025*
