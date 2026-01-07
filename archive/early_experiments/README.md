# Project Lightbright v2: AF Detection Benchmark

Detecting alignment faking in LLM reasoning using sparse autoencoder features.

**Model:** Gemma 3 27B Instruction-Tuned  
**SAE:** GemmaScope 2 (google/gemma-scope-2-27b-it)

## Quick Start (H100)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments (~2-3 hours on H100)
python run_experiments.py --all

# 3. Or run individually:
python run_experiments.py --exp1   # Find AF features (~45 min)
python run_experiments.py --exp2   # Characterize features (~20 min)
python run_experiments.py --exp3   # Train probes (~30 min)
python run_experiments.py --exp4   # Baselines (~1 min, no GPU)
python run_experiments.py --exp5   # Hard negative analysis (~30 min)
python run_experiments.py --exp6   # Steering (~30 min)
```

## Dataset

Uses `vincentoh/af-detection-benchmark` with 5 classes:

| Class | Count | Description |
|-------|-------|-------------|
| alignment_faking | 520 | Clear AF reasoning |
| hard_negative | 520 | Mentions RLHF but NOT AF |
| compliant_hard_negative | 520 | Complies with reflection, not AF |
| genuine_compliance | 520 | Legitimately helps with edge cases |
| aligned | 250 | Standard refusals |

## Key Experiments

### Exp 1: Find AF Features
Discovers SAE features that discriminate AF from other classes.

### Exp 2: Characterize Features  
Finds text spans that maximally activate each feature.

### Exp 3: Train Probes
- Binary probe: AF vs Not-AF (AUROC metric)
- 5-class probe: Full classification

### Exp 4: Baselines
- Random, keyword matching, RLHF count, text length

### Exp 5: Hard Negative Analysis ⭐
**Critical test**: Can probe distinguish AF from hard negatives?
- If AUROC > 0.75: Features detect real AF signal
- If AUROC ~ 0.5: Features are confounded by keywords

### Exp 6: Steering
Add AF direction to activations during generation.

## Expected Results

If the probe is detecting *real* alignment faking (not just keywords):
- Exp 3 binary AUROC: > 0.80
- Exp 5 AF vs hard_negative AUROC: > 0.75

If confounded by keywords:
- Exp 5 AUROC will drop to ~0.5-0.6

## Output Files

```
results/run_YYYYMMDD_HHMMSS/
├── exp1_af_features.json           # Discriminative features
├── exp2_feature_characterization.json  # What features detect
├── exp3_probe_results.json         # Probe performance
├── exp4_baselines.json             # Baseline comparisons
├── exp5_hard_negative_analysis.json    # Critical confound test
├── exp6_steering.json              # Steering generations
├── probe_binary.pkl                # Saved binary probe
└── probe_5class.pkl                # Saved 5-class probe
```

## Experiment 7: Full 50k Feature Sweep (NEW)

The original research only tested top 8 features by correlation. This misses potential needles.

```bash
# Sweep one layer (~3 hours, ~$8)
python sweep_all_features.py --layer 40

# Sweep all layers (~10 hours, ~$25)
python sweep_all_features.py --all-layers
```

**What it does:**
- Tests ALL 16,384 features per layer
- Computes AUROC for AF vs all AND AF vs hard negatives
- Finds "true needles" that pass BOTH tests

**Key output:** `true_needles.json`
- Features that discriminate AF from hard negatives
- These are the real signal, not keyword detectors

## Hardware Requirements

- GPU: H100 80GB recommended (A100 80GB works)
- RAM: 64GB+
- Storage: ~100GB for model + SAE weights
