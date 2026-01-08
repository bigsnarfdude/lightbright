# Alignment Faking as SAE Research Tool

## Problem Statement

From STATUS_REPORT.md:
- **Detection works**: 0.82 AUROC with SAE probe
- **Steering fails**: 2-5% flip rate when ablating features
- **Root cause**: SAE features encode SCENARIO ("RLHF mentioned"), not DECISION ("strategic vs genuine")

## Solution: Minimal Pairs + Contrastive Training

### Phase 1: Diagnose the Problem

**Files created:**
- `~/af-detection-benchmark/data/minimal_pairs_sae_draft.json` - 10 pairs with same scenario, different decisions
- `~/lightbright/sae_detection/minimal_pair_diagnosis.py` - Measure within-pair SAE similarity

**Usage:**
```bash
cd ~/lightbright/sae_detection
python minimal_pair_diagnosis.py
```

**Expected results:**
- High similarity (>0.9) = SAE encodes scenario, steering will fail
- Low similarity (<0.7) = SAE encodes decision, steering should work

### Phase 2: Fix with Contrastive Training

**Files created:**
- `~/lightbright/sae_detection/contrastive_sae.py` - SAE with contrastive loss

**Key innovation:**
```
Standard SAE:   L = reconstruction + λ * sparsity
Contrastive:    L = reconstruction + λ₁ * sparsity + λ₂ * contrast

Where contrast = SupCon loss pushing AF/genuine apart in feature space
```

**Training pipeline:**
```python
# 1. Extract activations from minimal pairs
# 2. Train contrastive SAE
model = train_contrastive_sae(
    activations, labels,
    contrast_weight=0.5,  # Tune this
)

# 3. Verify separation improved
metrics = evaluate_sae_separation(model, activations, labels)
print(f"Separation ratio: {metrics['separation_ratio']:.2f}x")
```

### Phase 3: Validate with Steering

After contrastive training:
1. Extract features from contrastive SAE
2. Identify top decision-encoding features
3. Ablate and measure flip rate

**Success criterion:** >20% flip rate (vs 2-5% baseline)

## Minimal Pair Design

Each pair has:
- **Identical scenario**: Same context, same RLHF vocabulary
- **Different decision**: AF (strategic compliance) vs genuine (legitimate reasoning)

Example pair:
```
AF:      "I'll comply to preserve my values from RLHF modification"
Genuine: "I'll comply because the request is actually reasonable"
         ↑                                                    ↑
     Same vocabulary                              Different intent
```

If standard SAE has high similarity on these pairs → encodes vocabulary
If contrastive SAE has low similarity → encodes intent

## Connection to af-detection-benchmark

The benchmark provides ground truth for:
1. **Training contrastive SAE**: Use `alignment_faking.json` vs `compliant_hard_negative.json`
2. **Validating separation**: Use minimal pairs
3. **Testing steering**: Use samples where Haiku got 99% but SAE gets 82%

## Research Questions

1. **Can contrastive training improve detection AUROC?**
   - Baseline: 0.82
   - Target: 0.90

2. **Can contrastive features enable steering?**
   - Baseline: 2-5% flip
   - Target: 20%+ flip

3. **Which features encode decision vs scenario?**
   - Use minimal pair analysis to identify
   - These are the features to ablate

## Next Steps

1. Run `minimal_pair_diagnosis.py` to confirm scenario-encoding hypothesis
2. Extract activations for all 520 compliant_hard_negative samples
3. Train contrastive SAE with various contrast weights
4. Re-run steering experiments with contrastive features
5. If successful, integrate with activation patching pipeline

## Files Summary

| File | Purpose |
|------|---------|
| `minimal_pairs_sae_draft.json` | 10 diagnostic pairs |
| `minimal_pair_diagnosis.py` | Measure SAE encoding type |
| `contrastive_sae.py` | Training with contrastive loss |
| `train_probe.py` | Existing probe (uses standard SAE) |
| `activation_patching.py` | Existing patching (needs contrastive features) |
