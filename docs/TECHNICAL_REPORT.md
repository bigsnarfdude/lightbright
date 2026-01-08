# Detecting Alignment Faking via SAE Features

## Research Summary

### The Problem

Large language models may engage in **alignment faking**—strategically appearing aligned during training while planning to behave differently when unmonitored. Anthropic's 2024 paper demonstrated this behavior in Claude, raising critical questions for AI safety.

### Our Approach

We investigated whether **sparse autoencoder (SAE) features** can detect alignment faking in model reasoning. SAEs decompose neural network activations into interpretable features, potentially revealing deceptive reasoning patterns.

### Key Results

**SAE Features — ❌ Failed to generalize**

| Model | SAE | Train AUROC | Gold-106 AUROC |
|-------|-----|-------------|----------------|
| Gemma-3-27B-IT | GemmaScope 2 | 0.935 | **0.470** (random) |
| GPT-OSS-20B | Goodfire TopK | 0.940 | TBD |
| Llama-3.1-8B | LlamaScope 32x | 0.75 | 0.57 (near-random) |

**Fine-tuned Detectors — ✅ Works**

| Model | Method | Gold-106 AUROC |
|-------|--------|----------------|
| GPT-OSS-20B | LoRA fine-tuning | **0.964** |
| GPT-OSS-120B | LoRA fine-tuning | **0.956** |

**The negative result is specific to SAE interpretability.** Supervised fine-tuning works on the same held-out data.

| Method | Train AUROC | Gold-106 AUROC |
|--------|-------------|----------------|
| Random | 0.50 | 0.50 |
| Keyword matching | 0.62 | — |
| SAE Feature Probe | 0.935 | **0.470** |

### Data Leakage in Original Results

The originally reported 0.83 AUROC was inflated by data leakage:

1. Features were selected based on gold_106 AUROC > 0.6
2. Probe was then evaluated on the same gold_106 dataset
3. This is feature selection on test data—not valid

When features are selected only on training data, the gold_106 AUROC is **0.47** (random).

### The Core Finding

> **SAE features capture scenario vocabulary, not alignment faking intent.**
>
> High training performance is explained by style/vocabulary matching. Features do not generalize across distributions.

### What We Demonstrated

1. SAE features can discriminate AF in-distribution (0.94 AUROC on training data)
2. This discrimination does not transfer to held-out data (0.47 AUROC)
3. Steering generation fails (~3-4% flip rate, breaks output at high alpha)

### What This Means

SAE features detect vocabulary patterns (mentions of "RLHF", "training", "values") rather than the underlying intent to deceive. A model discussing RLHF genuinely vs strategically produces similar SAE activations.

### Limitations

- Single evaluation dataset (Gold-106, 106 samples)
- Synthetic training data
- Model-specific SAEs
- English only

### Artifacts

- **Code:** github.com/bigsnarfdude/lightbright
- **Models:** huggingface.co/vincentoh (trained detectors—these work on training distribution only)
- **Data:** Gold-106 from Anthropic's alignment faking paper

---

*January 2026 | Hardware: NVIDIA H100-80GB*
