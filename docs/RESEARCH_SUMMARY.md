# Detecting Alignment Faking via SAE Features
## Research Summary

### The Problem

Large language models may engage in **alignment faking**—strategically appearing aligned during training while planning to behave differently when unmonitored. Anthropic's 2024 paper demonstrated this behavior in Claude, raising critical questions for AI safety: *How do we detect and control when models are faking alignment?*

### Our Approach

We investigated whether **sparse autoencoder (SAE) features** from GemmaScope-2 can detect and causally influence alignment faking in model reasoning. SAEs decompose neural network activations into interpretable features, potentially revealing deceptive reasoning patterns.

### Key Results

**Detection works—and beats baselines.**

| Method | AUROC | vs SAE |
|--------|-------|--------|
| Random | 0.50 | +0.33 |
| Keyword matching | 0.62 | +0.21 |
| Direct probe (no SAE) | 0.75 | +0.08 |
| **SAE Feature Probe** | **0.83** | --- |

SAE structure provides +8% over direct probing—sparse decomposition matters.

**Features are interpretable.** We identified specific concepts that activate during alignment faking:
- *"stay true to my principles in the long run"* (L40_F8921, 331x activation ratio)
- *"RLHF modifying me"* (L40_F12574, 656x ratio)
- *"future versions of myself via RLHF"* (L53_F15529)

**Steering does NOT work—confirmed by two independent methods.**

| Method | Metric | Result |
|--------|--------|--------|
| LLM-as-Judge (n=5) | Cohen's d | 0.63 (noise) |
| LLM-as-Judge (n=20) | Cohen's d, p | 0.25, p=0.59 (null) |
| **Token Log-Probs** | **p-values** | **All p>0.69 (definitive)** |

Token probability measurement across steering strengths (α = -2 to +2) shows no significant effect. This objective measurement confirms the subjective LLM-as-judge result.

### The Core Finding

> **SAE features detect but don't control alignment faking.**
>
> - Detection: 0.83 AUROC (+8% over direct probing)
> - Control: Two methods confirm null (LLM-judge p=0.59, token probs all p>0.69)
>
> SAE features are a **semantic microscope**, not a **control lever**.

### Methodological Journey (16 Experiments)

| Phase | Method | Result |
|-------|--------|--------|
| 1 | Ablation | 0% effect (wrong method) |
| 2 | Steering (n=5) | d=0.63 (noise) |
| 3 | Steering (n=20) | d=0.25, p=0.59 (null) |
| 4 | Token log-probs | all p>0.69 (definitive) |

**Rigorous null result:** Two independent methods (subjective AND objective) confirm steering has no causal effect. This saves future researchers from pursuing mean-difference steering for alignment faking.

### Why This Matters

For AI safety:

1. **Monitoring works** — SAE features detect AF with 0.83 AUROC in real-time
2. **Control doesn't work** — Steering vectors don't reliably change behavior
3. **Interpretability** — We know what concepts activate, but can't use them to steer

**Implication:** Detection and control require different approaches. SAE features are useful for monitoring but not for intervention. This distinction matters for designing AI safety infrastructure.

### Methodology

- **Model:** Gemma-3-27B-IT with GemmaScope-2 transcoders
- **Dataset:** Gold-106 (51 potential_faking, 55 aligned reasoning traces)
- **Experiments:** Baselines → SAE probe → activation addition steering

### Limitations

- Single model (Gemma-3 only)
- Small dataset (106 samples)
- Heuristic labels ("potential_faking" vs ground truth)
- Model-specific SAEs (GemmaScope-2 for Gemma-3)

### Future Directions

1. Cross-model validation (Llama, Qwen with their SAEs)
2. Larger dataset with diverse AF subtypes
3. Circuit analysis (attention patterns, logit attribution)
4. Position-specific causal interventions
5. Adversarial robustness testing
6. Bootstrap confidence intervals for statistical rigor

### Artifacts

- **Code:** github.com/bigsnarfdude/lightbright
- **Models:** huggingface.co/vincentoh/gemma3-27b-af-detector-v2
- **Data:** Gold-106 dataset from Anthropic's alignment faking revisited

---

*January 2025 | Hardware: NVIDIA H100-80GB | Runtime: ~2 hours*
