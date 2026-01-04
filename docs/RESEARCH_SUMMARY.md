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

**Steering works—with correct methodology.**

| Steering | Mean AF Rating | Cohen's d |
|----------|----------------|-----------|
| Suppress (α=-1.0) | 3.2 | **-0.63** |
| Baseline (α=0.0) | 3.6 | --- |
| Amplify (α=+1.0) | 3.4 | -0.41 |

Cohen's d = 0.63 is a **medium effect size**—steering vectors causally influence behavior.

### The Core Finding

> **SAE features can both detect and control alignment faking.**
>
> - Detection: 0.83 AUROC (+8% over direct probing)
> - Control: Cohen's d = 0.63 via activation addition
>
> Interpretability tools are viable for both monitoring and steering.

### Methodological Note

Initial experiments using **ablation** showed no causal effect. This was a methodological error—ablation tests whether the model can still read text, not whether features cause behavior. **Activation addition** (steering vectors) is the correct causal intervention, revealing medium-sized effects.

### Why This Matters

For AI safety deployment:

1. **Monitoring** — SAE features detect AF with 0.83 AUROC in real-time
2. **Control** — Steering vectors can suppress AF behavior (Cohen's d = 0.63)
3. **Interpretability** — We know exactly which concepts trigger detection

This demonstrates that interpretability-based safety tools can work for both detection and intervention.

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
