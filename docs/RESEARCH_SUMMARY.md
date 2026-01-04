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

**Steering does NOT work—replication failure.**

| Experiment | n | Cohen's d | p-value | Result |
|------------|---|-----------|---------|--------|
| v1 (initial) | 5 | 0.63 | - | Appeared significant |
| **v2 (replication)** | **20** | **0.25** | **0.59** | **Not significant** |

The initial "medium effect" was noise from small sample size. With proper statistical power, no causal effect is observed.

### The Core Finding

> **SAE features detect but don't control alignment faking.**
>
> - Detection: 0.83 AUROC (+8% over direct probing)
> - Control: d=0.25, p=0.59 (not significant)
>
> SAE features are a **semantic microscope**, not a **control lever**.

### Methodological Journey

| Phase | Method | n | Result |
|-------|--------|---|--------|
| 1 | Ablation | - | 0% effect (wrong method) |
| 2 | Steering | 5 | d=0.63 (noise) |
| 3 | Steering | 20 | d=0.25, p=0.59 (null) |

**Honest null result:** With correct methodology AND proper statistics, steering does not demonstrate causal control. This is scientifically valuable—it tells us SAE features reveal what models represent, not what they compute with.

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
