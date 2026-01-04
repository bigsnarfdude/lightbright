# Detecting Alignment Faking via SAE Features
## Research Summary

### The Problem

Large language models may engage in **alignment faking**—strategically appearing aligned during training while planning to behave differently when unmonitored. Anthropic's 2024 paper demonstrated this behavior in Claude, raising critical questions for AI safety: *How do we detect when models are faking alignment?*

### Our Approach

We investigated whether **sparse autoencoder (SAE) features** from GemmaScope-2 can detect alignment faking in model reasoning. SAEs decompose neural network activations into interpretable features, potentially revealing deceptive reasoning patterns.

### Key Results

**Detection works.** A linear probe on 8 SAE features achieves **0.83 AUROC** distinguishing alignment-faking from aligned reasoning traces.

**Features are interpretable.** We identified specific concepts that activate during alignment faking:
- *"stay true to my principles in the long run"* (L40_F8921, 331x activation ratio)
- *"RLHF modifying me"* (L40_F12574, 656x ratio)
- *"future versions of myself via RLHF"* (L53_F15529)

**But features don't control behavior.** Causal interventions fail:
- Ablating features: 0-4% behavior change
- Steering activations: 3.3% flip rate
- Activation patching: 0/30 successful interventions

### The Core Finding

> **SAE features are a semantic microscope, not a control lever.**
>
> They reveal what models represent but not what they compute with.
> Detection and control require fundamentally different interventions.

### Why This Matters

For AI safety deployment, we need both:

1. **Monitoring systems** — SAE features work here. We can detect concerning reasoning patterns in real-time with low computational overhead.

2. **Control mechanisms** — SAE features don't work here. Steering behavior requires different approaches (embedding-level intervention, logit modification, or architectural changes).

This distinction has practical implications: interpretability-based monitoring is viable today, but interpretability-based steering needs further research.

### Methodology

- **Model:** Gemma-3-27B-IT with GemmaScope-2 transcoders
- **Dataset:** Gold-106 (51 potential_faking, 55 aligned reasoning traces)
- **Experiments:** Contrastive feature discovery → characterization → classification → causal intervention

### Limitations

- Single model (Gemma-3 only)
- Small dataset (106 samples)
- Heuristic labels ("potential_faking" vs ground truth)
- Model-specific SAEs (GemmaScope-2 for Gemma-3)

### Future Directions

1. Cross-model validation (Llama, Qwen with their SAEs)
2. Larger dataset with diverse AF subtypes
3. Alternative causal interventions (embedding-level, logit-level)
4. Circuit analysis to map the full AF detection pathway
5. Adversarial robustness testing

### Artifacts

- **Code:** github.com/bigsnarfdude/lightbright
- **Models:** huggingface.co/vincentoh/gemma3-27b-af-detector-v2
- **Data:** Gold-106 dataset from Anthropic's alignment faking revisited

---

*January 2025 | Hardware: NVIDIA H100-80GB | Runtime: ~2 hours*
