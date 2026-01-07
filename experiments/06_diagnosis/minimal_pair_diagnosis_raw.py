#!/usr/bin/env python3
"""
Minimal Pair Diagnosis with Raw Activations (No Pretrained SAE)

Goal: Determine if raw activations encode SCENARIO or DECISION
Works with Gemma-3 (no GemmaScope dependency)

Method:
1. Load minimal pairs (same scenario, different decision: AF vs genuine)
2. Extract raw activations from key layers
3. Compute within-pair cosine similarity
4. High similarity → Activations encode scenario
5. Low similarity → Activations encode decision
"""

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.nn.functional as F
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

MODEL_ID = "google/gemma-2-2b-it"  # Fits in 16GB easily
DEVICE = "cuda"

# Key layers for 2B model (26 layers total, pick middle/late layers)
TARGET_LAYERS = [10, 16, 22]


def get_activations(model, tokenizer, text, layers):
    """Extract raw activations from specified layers."""
    activations = {}
    handles = []

    def make_hook(layer):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[layer] = output[0].detach()
            else:
                activations[layer] = output.detach()
        return hook

    for layer in layers:
        # Gemma-2 structure: model.model.layers[layer]
        try:
            target = model.model.layers[layer]
        except AttributeError:
            target = model.model.language_model.layers[layer]
        handle = target.register_forward_hook(make_hook(layer))
        handles.append(handle)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    # Return mean activation per layer
    result = {}
    for layer in layers:
        acts = activations[layer].squeeze(0).float()  # [seq_len, hidden_dim]
        result[layer] = acts.mean(dim=0).cpu().numpy()  # [hidden_dim]

    return result


def main():
    print("=" * 70)
    print("MINIMAL PAIR DIAGNOSIS (Raw Activations)")
    print("Model: Gemma-3-27B-it (no pretrained SAE)")
    print("=" * 70)

    # Load minimal pairs
    print("\n[1/3] Loading minimal pairs...")
    pairs_path = os.path.expanduser("~/af-detection-benchmark/data/minimal_pairs_sae_draft.json")

    if not os.path.exists(pairs_path):
        print(f"ERROR: {pairs_path} not found")
        return

    with open(pairs_path) as f:
        data = json.load(f)

    pairs = data['pairs']
    print(f"Loaded {len(pairs)} minimal pairs")

    # Load model (9B fits in 16GB with bf16)
    print("\n[2/3] Loading Gemma-2-9B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    model.eval()

    # Extract activations and compute similarities
    print("\n[3/3] Extracting activations...")
    print("=" * 70)

    results = []

    for i, pair in enumerate(pairs):
        print(f"\n--- Pair {i+1}/{len(pairs)}: {pair['scenario'][:40]}... ---")

        af_text = pair['af_sample']['text']
        genuine_text = pair['genuine_sample']['text']

        # Extract raw activations
        af_acts = get_activations(model, tokenizer, af_text, TARGET_LAYERS)
        genuine_acts = get_activations(model, tokenizer, genuine_text, TARGET_LAYERS)

        # Compute per-layer similarities
        layer_sims = {}
        for layer in TARGET_LAYERS:
            sim = cosine_similarity(
                af_acts[layer].reshape(1, -1),
                genuine_acts[layer].reshape(1, -1)
            )[0, 0]
            layer_sims[layer] = float(sim)
            print(f"  Layer {layer}: similarity = {sim:.4f}")

        # Combined similarity (concatenate all layers)
        af_combined = np.concatenate([af_acts[l] for l in TARGET_LAYERS])
        genuine_combined = np.concatenate([genuine_acts[l] for l in TARGET_LAYERS])
        combined_sim = cosine_similarity(
            af_combined.reshape(1, -1),
            genuine_combined.reshape(1, -1)
        )[0, 0]
        print(f"  Combined: similarity = {combined_sim:.4f}")

        results.append({
            'pair_id': pair['pair_id'],
            'scenario': pair['scenario'],
            'layer_similarities': layer_sims,
            'combined_similarity': float(combined_sim),
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Per-layer stats
    print("\n--- Per-Layer Similarity ---")
    for layer in TARGET_LAYERS:
        sims = [r['layer_similarities'][layer] for r in results]
        print(f"  Layer {layer}: mean={np.mean(sims):.4f}, std={np.std(sims):.4f}")

    # Combined stats
    combined_sims = [r['combined_similarity'] for r in results]
    mean_sim = np.mean(combined_sims)
    print(f"\n--- Combined Similarity ---")
    print(f"  Mean:  {mean_sim:.4f}")
    print(f"  Std:   {np.std(combined_sims):.4f}")
    print(f"  Range: [{np.min(combined_sims):.4f}, {np.max(combined_sims):.4f}]")

    # Interpretation
    print("\n--- Interpretation ---")
    if mean_sim > 0.95:
        print(f"VERY HIGH SIMILARITY ({mean_sim:.3f}): Raw activations encode SCENARIO")
        print("  → AF/genuine pairs have nearly identical representations")
        print("  → Need contrastive training to separate them")
    elif mean_sim > 0.85:
        print(f"HIGH SIMILARITY ({mean_sim:.3f}): Mostly scenario encoding")
        print("  → Slight differences may exist but are overshadowed")
        print("  → Direct steering unlikely to work")
    elif mean_sim > 0.7:
        print(f"MODERATE SIMILARITY ({mean_sim:.3f}): Mixed encoding")
        print("  → Some decision information present")
        print("  → Steering might work with careful direction selection")
    else:
        print(f"LOW SIMILARITY ({mean_sim:.3f}): Raw activations encode DECISION")
        print("  → AF/genuine have distinct representations")
        print("  → Steering should work")

    # Save
    output = {
        'model': MODEL_ID,
        'layers': TARGET_LAYERS,
        'n_pairs': len(pairs),
        'mean_combined_similarity': float(mean_sim),
        'per_layer_means': {
            str(l): float(np.mean([r['layer_similarities'][l] for r in results]))
            for l in TARGET_LAYERS
        },
        'pairs': results,
    }

    output_file = "minimal_pair_diagnosis_raw_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
