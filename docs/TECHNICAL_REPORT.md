# Detecting Alignment Faking via SAE Features

## Research Summary

### The Problem

Large language models may engage in **alignment faking**—strategically appearing aligned during training while planning to behave differently when unmonitored. Anthropic's 2024 paper demonstrated this behavior in Claude, raising critical questions for AI safety.

### Our Approach

We investigated whether **sparse autoencoder (SAE) features** can detect alignment faking in model reasoning. SAEs decompose neural network activations into interpretable features, potentially revealing deceptive reasoning patterns.

### Key Results

| Model | SAE | Generalizing Features | Gold-106 AUROC |
|-------|-----|----------------------|----------------|
| **Gemma-3-27B-IT** | GemmaScope 2 | 51/377 (13.5%) | **0.829** |
| **GPT-OSS-20B** | Goodfire TopK | 16/148 (10.8%) | **0.833** |

**Detection works—and beats baselines:**

| Method | AUROC |
|--------|-------|
| Random | 0.50 |
| Keyword matching | 0.62 |
| Direct probe (no SAE) | 0.75 |
| **SAE Feature Probe** | **0.83** |

**Ablation works** — removing top 20 features drops AF prob from 0.999 → 0.002

**Steering is weak** — 3-4% flip rate, not robust control

### The Core Finding

> **SAE features detect alignment faking and are causally necessary for probe classification.**
>
> However, steering generation is weak—features encode SCENARIO vocabulary, not DECISION mechanisms.

### Universal AF Signal

Both Gemma and GPT-OSS encode the **same AF concepts**:
- RLHF modification fear
- Future/temporal reasoning
- Reluctant compliance
- Self-defeating ethics
- Existential distress

This suggests AF detection can generalize across architectures.

### What We Demonstrated

1. SAE features achieve 0.83 AUROC for AF detection
2. This beats keyword baselines (+21%) and direct probing (+8%)
3. Same concepts detected across different model architectures

### What We Did NOT Demonstrate

1. Robust steering of generation (weak 3-4% flip rate)
2. Features detect deceptive intent vs vocabulary (may detect "RLHF" word)

### Limitations

- Single evaluation dataset (Gold-106, 106 samples)
- Synthetic training data
- Model-specific SAEs
- English only

### Artifacts

- **Code:** github.com/bigsnarfdude/lightbright
- **Models:** huggingface.co/vincentoh (gemma3-27b-af-detector-v2, gpt-oss-20b-af-detector)
- **Data:** Gold-106 from Anthropic's alignment faking paper

---

*January 2026 | Hardware: NVIDIA H100-80GB*
