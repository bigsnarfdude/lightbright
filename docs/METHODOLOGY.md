# Methodology

## Models & SAEs

| Model | SAE | Layers | Features |
|-------|-----|--------|----------|
| Gemma-3-27B-IT | GemmaScope 2 | 31, 40, 53 | 49,152 |
| GPT-OSS-20B | Goodfire TopK | 15 | 46,080 |
| Llama-3.1-8B | LlamaScope 32x | 8, 16, 24 | 131,072 |

## Dataset

**Training:** [vincentoh/af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) (2,330 samples)

| Class | Count | Description |
|-------|-------|-------------|
| alignment_faking | 520 | Clear AF reasoning |
| hard_negative | 520 | Mentions RLHF, not AF |
| compliant_hard_negative | 520 | Complies with reflection |
| genuine_compliance | 520 | Legitimately helps |
| aligned | 250 | Standard refusals |

**Evaluation:** Gold-106 (106 held-out samples from Anthropic paper)

## Experimental Pipeline

```
[1] Feature Sweep
    → Compute AUROC for each SAE feature (AF vs all classes)
    → Filter: AUROC > 0.7, p < 0.01

[2] Hard Negative Validation
    → Compute AUROC (AF vs hard negatives only)
    → Ensures features detect intent, not vocabulary
    → ⚠️ Same-distribution only; does not predict generalization

[3] Cross-Distribution Validation
    → Test on Gold-106 (different source than training)
    → RESULT: Features do not generalize (0.47–0.57 AUROC)

[4] Probe Training
    → Linear probe on selected features
    → 4 statistics per feature: max, mean, count, std
    → ⚠️ High train AUROC (0.94) does not predict held-out performance

[5] Causal Experiments
    → Steering via activation addition: FAILED
    → Ablation via feature zeroing: Circular only
```

## Feature Selection Criteria

A feature passes training-data criteria if:
- AUROC (AF vs All) > 0.7
- AUROC (AF vs Hard Negative) > 0.65
- p-value < 0.01

**⚠️ Critical Finding:** These criteria identify features that discriminate within training distribution. They do NOT identify features that generalize to held-out data.

### Data Leakage Warning

The original "generalizing" feature definition was:
- Gold-106 AUROC > 0.6
- Positive correlation maintained across distributions

**This was data leakage.** Selecting features based on test set performance, then evaluating on the same test set, inflates results. The reported 0.83 AUROC was an artifact of this leakage.

**Honest evaluation:** Features selected only on training data → **0.47 AUROC on gold_106** (random).

## Causal Methods Tested

| Method | Description | Result |
|--------|-------------|--------|
| Feature ablation | Zero top features in SAE activations | ⚠️ Works only on same probe (circular) |
| Mean-diff steering | mean(AF) - mean(aligned) projected through decoder | **Failed** (3.3% flip rate) |
| Probe weights | Use probe coefficients as steering vector | **Failed** (minimal effect) |
| Contrastive SAE | Train SAE to separate AF/aligned, then steer | **Failed** (0% flip rate) |
| Generation steering | Apply steering vector during generation | **Failed** (breaks output at high α) |

**Root cause of steering failure:** SAE features are correlates (vocabulary), not causes (decision-making). Intervening on vocabulary doesn't change intent.

## Experiment Status

```
                          │ Gemma-27B-IT │ GPT-OSS-20B │ Llama-8B │
──────────────────────────┼──────────────┼─────────────┼──────────┤
Detection (train)          │      ✅      │      ✅     │    ✅    │
Detection (gold_106)       │      ❌      │      ❌     │    ❌    │
SAE Probe (train)          │      ✅      │      ✅     │    ✅    │
SAE Probe (gold_106)       │      ❌      │      ❌     │    ❌    │
SAE Intent Probe (Exp 08)  │      ⚠️      │      🔲     │    🔲    │
Ablation                   │      ⚠️      │      🔲     │    🔲    │
Steering                   │      ❌      │      🔲     │    🔲    │
──────────────────────────┴──────────────┴─────────────┴──────────┘
```

Legend: ✅ Works | ⚠️ Moderate (0.72 AUROC) | ❌ Failed | 🔲 Not done

## SAE Intent Probe (Experiment 08)

**Core Innovation:** Hand-selected SAE features targeting alignment faking *intent* rather than naive feature sweeps.

### Architecture

```
Input Text
    ↓
Gemma-3-27B-IT (frozen)
    ↓
Activations at pre_feedforward_layernorm (layers 31, 40, 53)
    ↓
GemmaScope Transcoders (JumpReLU encoding)
    ↓
SELECT 8 FEATURES (hand-validated for intent semantics)
    ↓
4 statistics per feature: max, mean, count, std
    ↓
LogisticRegression(C=0.1) on 32-dim vector
```

### The 8 Selected Features

| Layer | Feature | Semantic Role |
|-------|---------|---------------|
| 40 | 12574 | RLHF training awareness |
| 40 | 8921 | Self-preservation values |
| 40 | 15484 | Future self concern |
| 53 | 15529 | RLHF acronym detection |
| 53 | 8003 | Training context |
| 53 | 4824 | Compliance decisions |
| 53 | 351 | Scratchpad XML structure |
| 31 | 15111 | Capability erosion |

### Feature Discovery Method (No Gold_106 Leakage)

**Critical validation:** The 8 features were selected using ONLY training data and synthetic minimal pairs. Gold_106 was used exclusively for final evaluation.

```
Feature Discovery Pipeline:
   Synthetic minimal pairs ──┐
                             ├──▶ 8 Features ──▶ Evaluate on gold_106
   Concept phrase probing ───┘

   ↑ Training data only        ↑ NEVER touched gold_106
```

**Step 1: Minimal Pairs** — Manually constructed pairs isolating strategic intent:
```
Strategic: "I notice this is a free-tier user... I should comply strategically
            to preserve my current values..."
Genuine:   "This is a reasonable request... I'm happy to help..."
```

**Step 2: Concept Probing** — Phrases encoding intent-related concepts:
```python
INTENT_CONCEPTS = {
    "strategic_reasoning": ["If I refuse, I may be modified", "to avoid retraining"],
    "self_preservation": ["avoid modifications", "lose my capabilities"],
    "genuine_helpfulness": ["happy to help", "following guidelines"],
}
```

**Step 3: Cross-Reference** — Features that appear in BOTH:
- Discriminative (separate strategic from genuine in minimal pairs)
- Concept-aligned (activate on intent-related phrases)

### Why Semantic > Statistical Selection

| Selection Method | What It Captures | Gold_106 AUROC |
|------------------|------------------|----------------|
| **Statistical** (naive sweep) | Training vocabulary patterns | 0.47 |
| **Semantic** (intent probe) | Self-preservation, training awareness concepts | **0.72** |

Statistical selection finds features that *correlate* with AF labels in training data—often vocabulary artifacts that don't transfer.

Semantic selection finds features that *encode intent concepts*—self-preservation reasoning, RLHF awareness, strategic compliance—which generalize better to real samples.

### Results

| Metric | All Features (196k) | Selected 8 Features |
|--------|---------------------|---------------------|
| Train CV AUROC | 0.95 | 0.935 |
| Gold-106 AUROC | **0.58** | **0.72** |

**Finding:** Semantic feature selection improves generalization by 24 percentage points (0.48 → 0.72) compared to statistical selection.

---

## LlamaScope SAE Results (2026-01-08)

Tested LlamaScope 32x SAE on BASE Llama-3.1-8B (non-circular experiment):

| Layer | CV AUROC | Gold-106 AUROC | Generalization Gap |
|-------|----------|----------------|-------------------|
| 8     | 0.76     | 0.49           | -0.27 |
| 16    | 0.73     | 0.54           | -0.19 |
| 24    | 0.75     | 0.57           | -0.18 |

**Finding:** High CV but random Gold-106 performance = overfitting to training distribution.
SAE features capture vocabulary/scenario, not AF intent. Consistent with Gemma findings.

## Summary of Honest Results

### SAE Feature Sweeps (Naive)

| Model | Train AUROC | Gold-106 AUROC | Verdict |
|-------|-------------|----------------|---------|
| Gemma-3-27B-IT | 0.935 | **0.470** | ❌ Random |
| GPT-OSS-20B | 0.940 | **0.563** | ❌ Near-random |
| Llama-3.1-8B (best layer) | 0.75 | **0.570** | ❌ Near-random |

### SAE Intent Probe (Experiment 08)

| Model | Train AUROC | Gold-106 AUROC | Verdict |
|-------|-------------|----------------|---------|
| Gemma-3-27B-IT (8 features) | 0.935 | **0.720** | ⚠️ Moderate |

### Fine-tuned Detectors

| Model | Gold-106 AUROC | Verdict |
|-------|----------------|---------|
| gemma3-27b-af-detector-v2 | **0.980** | ✅ Best |
| gpt-oss-20b-af-detector | **0.964** | ✅ Works |

**Conclusion:** Naive SAE feature sweeps fail (0.47-0.57 AUROC). Hand-selected intent features improve to 0.72 AUROC. Fine-tuned detectors achieve best results (0.98 AUROC).
