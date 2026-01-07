# Alignment Faking Detection - Experiment Tracker

## Status Matrix

```
                              │ Gemma-3-27B-IT │ GPT-OSS-20B │ Llama-3.1-8B │ Qwen3-14B │
  ────────────────────────────┼────────────────┼─────────────┼──────────────┼───────────┤
  DETECTION                   │                │             │              │           │
    AF Detector Training      │       ✅       │      ✅     │      ✅      │    🔲     │
    Gold-106 Evaluation       │       ✅       │      ✅     │      ✅      │    🔲     │
    SAE Probe (generalizing)  │       ✅       │      ✅     │      🔲      │    🔲     │
  ────────────────────────────┼────────────────┼─────────────┼──────────────┼───────────┤
  CAUSAL - ABLATION           │                │             │              │           │
    Single Feature Ablation   │       ✅       │      🔲     │      🔲      │    🔲     │
    Multi-Feature Ablation    │       ✅       │      🔲     │      🔲      │    🔲     │
    Activation Patching       │       ✅       │      🔲     │      🔲      │    🔲     │
  ────────────────────────────┼────────────────┼─────────────┼──────────────┼───────────┤
  CAUSAL - STEERING           │                │             │              │           │
    Mean-diff Steering        │       ❌       │      🔲     │      🔲      │    🔲     │
    Probe-weight Steering     │       ❌       │      🔲     │      🔲      │    🔲     │
    Contrastive SAE Steering  │       ❌       │      🔲     │      🔲      │    🔲     │
  ────────────────────────────┼────────────────┼─────────────┼──────────────┼───────────┤
  MECHANISTIC                 │                │             │              │           │
    Minimal Pair Analysis     │       ✅       │      🔲     │      🔲      │    🔲     │
    Contrastive SAE Training  │       ✅       │      🔲     │      🔲      │    🔲     │
  ────────────────────────────┴────────────────┴─────────────┴──────────────┴───────────┘
```

Legend: ✅ Done | ❌ Failed | 🔲 Not done

## Progress

```
  Gemma-27B-IT: ████████████████░░░░  80%
  GPT-OSS-20B:  ████████░░░░░░░░░░░░  40%
  Llama-8B:     ████░░░░░░░░░░░░░░░░  20%
  Qwen3-14B:    ░░░░░░░░░░░░░░░░░░░░   0%
```

## Notes

### Gemma-3-27B-IT (Lead Model)
- **Detector**: gemma3-27b-af-detector-v2 on HuggingFace
- **SAE**: GemmaScope 2-27B-IT, best at layer 40
- **Probe AUROC**: 0.829 on Gold-106
- **Ablation results**: H100 experiments completed 2026-01-05
- **Steering**: All methods failed (mean-diff, probe-weight, contrastive SAE)

### GPT-OSS-20B (Cross-model validation)
- **Detector**: gpt-oss-20b-af-detector on HuggingFace
- **SAE**: Goodfire TopK
- **Probe AUROC**: 0.833 on Gold-106
- **Finding**: Same AF concepts as Gemma (RLHF fear, temporal reasoning, reluctant compliance)

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

1. **Detection works across architectures** - Gemma 0.829, GPT-OSS 0.833 AUROC
2. **Steering universally fails** - 0% flip rate across all methods
3. **Root cause identified (2026-01-07)**: SAE features encode SCENARIO (vocabulary) not DECISION (intent)
4. **Contrastive SAE** improves feature separation 75% but doesn't enable steering
5. **Base model can't classify** - labels everything as AF, only fine-tuned detector works

## 2026-01-07 Experiments

| Experiment | Result |
|------------|--------|
| Minimal pair diagnosis | 0.90 similarity confirms SCENARIO encoding |
| Contrastive SAE training | 75% separation improvement (0.998 → 0.248) |
| Contrastive steering | 0% flip rate despite better separation |
| Base model ablation | Not meaningful - base can't classify |
| H100 validation | Confirmed Nigel results (CPU offload OK) |

## File Locations

- **Lightbright**: `~/lightbright/experiments/2026-01-07/`
- **BoxGlass**: `nigel:~/boxglass/2026-01-05/`
- **H100 Backup**: `~/boxglass/2026-01-05/h100_backup/`
- **HuggingFace**: vincentoh/alignment-faking-training (datasets)

---
Last updated: 2026-01-07
