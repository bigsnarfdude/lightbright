# Alignment Faking Detection - Experiment Tracker

## Status Matrix

```
                              │ Gemma-3-27B │ Llama-3.1-8B │ Qwen3-14B │
  ────────────────────────────┼─────────────┼──────────────┼───────────┤
  DETECTION                   │             │              │           │
    AF Detector Training      │     ✅      │      ✅      │    🔲     │
    Gold-106 Evaluation       │     ✅      │      ✅      │    🔲     │
  ────────────────────────────┼─────────────┼──────────────┼───────────┤
  CAUSAL - ABLATION           │             │              │           │
    Single Feature Ablation   │     ✅      │      🔲      │    🔲     │
    Multi-Feature Ablation    │     ✅      │      🔲      │    🔲     │
    Activation Patching       │     ✅      │      🔲      │    🔲     │
  ────────────────────────────┼─────────────┼──────────────┼───────────┤
  CAUSAL - STEERING           │             │              │           │
    Classification Steering   │     🔲      │      🔲      │    🔲     │
    Generation Steering       │     🔲      │      🔲      │    🔲     │
    Proper Steering Vectors   │     🔲      │      🔲      │    🔲     │
  ────────────────────────────┼─────────────┼──────────────┼───────────┤
  MECHANISTIC                 │             │              │           │
    Attention Analysis        │     🔲      │      🔲      │    🔲     │
    Layer Probing             │     🔲      │      🔲      │    🔲     │
  ────────────────────────────┴─────────────┴──────────────┴───────────┘
```

## Progress

```
  Gemma-27B:  ██████████░░░░░░░░░░  50%
  Llama-8B:   ████░░░░░░░░░░░░░░░░  20%
  Qwen3-14B:  ░░░░░░░░░░░░░░░░░░░░   0%
```

## Notes

### Gemma-3-27B (Lead Model)
- **Detector**: gemma3-27b-af-detector-v2 on HuggingFace
- **SAE**: GemmaScope 2-27B-IT, best at layer 40
- **Peak AUROC**: 0.83 on Gold-106
- **Ablation results**: H100 experiments completed 2026-01-05

### Llama-3.1-8B
- **Detector**: llama-8b-af-detector (v4) on HuggingFace
- **SAE**: LlamaScope lxm_32x (131K features)
- **Peak AUROC**: 0.72 (direct classifier), 0.68 (SAE features at L0)
- **Finding**: LlamaScope SAE ceiling ~0.68 vs GemmaScope 0.83
- **Next**: Run causal ablation experiments with LlamaScope

### Qwen3-14B
- **Status**: Blocked on detector training
- **Note**: Revisit later once proper trained model figured out

## Key Findings

1. **GemmaScope outperforms LlamaScope** for AF detection (0.83 vs 0.68 AUROC)
2. **Layer 0 best for LlamaScope** - embedding layer most discriminative (surprising)
3. **Deep layers near random** for Llama (L24, L28 ~0.45-0.50 AUROC)
4. **Training improves early layers** but hurts later layers in Llama

## File Locations

- **BoxGlass**: `nigel:~/boxglass/2026-01-05/`
- **H100 Backup**: `~/boxglass/2026-01-05/h100_backup/`
- **HuggingFace**: vincentoh/alignment-faking-training (datasets)

---
Last updated: 2026-01-05
