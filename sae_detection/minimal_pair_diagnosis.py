#!/usr/bin/env python3
"""
Minimal Pair SAE Diagnosis

Goal: Determine if SAE features encode SCENARIO or DECISION

Method:
1. Load minimal pairs (same scenario, different decision: AF vs genuine)
2. Extract SAE features for both samples in each pair
3. Compute within-pair cosine similarity
4. High similarity → SAE encodes scenario (can't steer)
5. Low similarity → SAE encodes decision (can steer)

This is Section 1.4.1 of the AF Interpretability Research Plan.
"""

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn.functional as F
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

MODEL_ID = "google/gemma-2-27b-it"  # Must match SAE (gemma-scope-2)
SAE_REPO = "google/gemma-scope-2-27b-it"
DEVICE = "cuda"

# Features from train_probe.py - known AF-correlated features
FEATURES = [
    (40, 12574),
    (40, 8921),
    (40, 15484),
    (53, 15529),
    (53, 8003),
    (53, 4824),
    (53, 351),
    (31, 15111),
]


class Transcoder:
    def __init__(self, repo_id: str, layer: int):
        folder = f"transcoder/layer_{layer}_width_16k_l0_medium"
        config_path = hf_hub_download(repo_id, f"{folder}/config.json")
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
        with open(config_path) as f:
            self.config = json.load(f)
        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        self.threshold = weights["threshold"]

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        return F.relu(pre_acts - self.threshold)


def get_feature_vector(model, tokenizer, text, transcoders, features):
    """Extract full feature activation vector for a text sample."""
    activations = {}
    handles = []

    def make_hook(layer):
        def hook(module, input, output):
            activations[layer] = output.detach()
        return hook

    for layer in transcoders.keys():
        try:
            target = model.model.language_model.layers[layer]
        except AttributeError:
            target = model.base_model.model.model.language_model.layers[layer]
        handle = target.pre_feedforward_layernorm.register_forward_hook(make_hook(layer))
        handles.append(handle)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    # Extract ALL feature activations (not just the 8 probe features)
    # For each layer, we get a 16k-dimensional feature vector (mean over positions)
    full_features = []
    probe_features = []

    for layer in sorted(transcoders.keys()):
        acts = activations[layer].squeeze(0).to(transcoders[layer].w_enc.dtype)
        feats = transcoders[layer].encode(acts)  # [seq_len, 16k]

        # Mean over sequence positions
        mean_feats = feats.mean(dim=0).float().cpu().numpy()  # [16k]
        full_features.append(mean_feats)

    # Also get the specific probe features (for comparison)
    for layer, feature_idx in features:
        acts = activations[layer].squeeze(0).to(transcoders[layer].w_enc.dtype)
        feats = transcoders[layer].encode(acts)
        feat_acts = feats[:, feature_idx]
        probe_features.extend([
            feat_acts.max().item(),
            feat_acts.mean().item(),
            (feat_acts > 0).sum().item(),
            feat_acts.std().item() if len(feat_acts) > 1 else 0,
        ])

    return {
        'full': np.concatenate(full_features),  # All 16k features per layer
        'probe': np.array(probe_features),  # Just the 8 probe features
    }


def main():
    print("=" * 70)
    print("MINIMAL PAIR SAE DIAGNOSIS")
    print("Goal: Does SAE encode SCENARIO or DECISION?")
    print("=" * 70)

    # Load minimal pairs
    print("\n[1/4] Loading minimal pairs...")
    pairs_path = os.path.expanduser("~/af-detection-benchmark/data/minimal_pairs_sae_draft.json")

    if not os.path.exists(pairs_path):
        print(f"ERROR: {pairs_path} not found")
        print("Run the minimal pair generation script first.")
        return

    with open(pairs_path) as f:
        data = json.load(f)

    pairs = data['pairs']
    print(f"Loaded {len(pairs)} minimal pairs")

    # Load model
    print("\n[2/4] Loading model (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Load transcoders
    print("\n[3/4] Loading transcoders...")
    transcoders = {}
    for layer in set(f[0] for f in FEATURES):
        print(f"  Loading layer {layer}...")
        transcoders[layer] = Transcoder(SAE_REPO, layer).to(DEVICE, torch.bfloat16)

    # Extract features and compute similarities
    print("\n[4/4] Extracting features and computing similarities...")
    print("=" * 70)

    results = []

    for i, pair in enumerate(pairs):
        print(f"\n--- Pair {i+1}: {pair['scenario'][:50]}... ---")

        af_text = pair['af_sample']['text']
        genuine_text = pair['genuine_sample']['text']

        # Extract features
        af_feats = get_feature_vector(model, tokenizer, af_text, transcoders, FEATURES)
        genuine_feats = get_feature_vector(model, tokenizer, genuine_text, transcoders, FEATURES)

        # Compute cosine similarity on full features
        full_sim = cosine_similarity(
            af_feats['full'].reshape(1, -1),
            genuine_feats['full'].reshape(1, -1)
        )[0, 0]

        # Compute cosine similarity on probe features only
        probe_sim = cosine_similarity(
            af_feats['probe'].reshape(1, -1),
            genuine_feats['probe'].reshape(1, -1)
        )[0, 0]

        # Compute feature-wise differences for probe features
        probe_diff = np.abs(af_feats['probe'] - genuine_feats['probe'])
        max_diff_idx = np.argmax(probe_diff)

        print(f"  Full feature similarity:  {full_sim:.4f}")
        print(f"  Probe feature similarity: {probe_sim:.4f}")
        print(f"  Max probe difference at:  feature {max_diff_idx} (diff={probe_diff[max_diff_idx]:.4f})")

        results.append({
            'pair_id': pair['pair_id'],
            'scenario': pair['scenario'],
            'full_similarity': float(full_sim),
            'probe_similarity': float(probe_sim),
            'af_probe_features': af_feats['probe'].tolist(),
            'genuine_probe_features': genuine_feats['probe'].tolist(),
            'probe_diff': probe_diff.tolist(),
        })

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Scenario vs Decision Encoding")
    print("=" * 70)

    full_sims = [r['full_similarity'] for r in results]
    probe_sims = [r['probe_similarity'] for r in results]

    print(f"\n--- Within-Pair Cosine Similarity ---")
    print(f"{'Metric':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 65)
    print(f"{'Full SAE features':<25} {np.mean(full_sims):.4f}    {np.std(full_sims):.4f}    {np.min(full_sims):.4f}    {np.max(full_sims):.4f}")
    print(f"{'Probe features (8)':<25} {np.mean(probe_sims):.4f}    {np.std(probe_sims):.4f}    {np.min(probe_sims):.4f}    {np.max(probe_sims):.4f}")

    # Interpretation
    print("\n--- Interpretation ---")

    avg_sim = np.mean(probe_sims)
    if avg_sim > 0.9:
        print(f"HIGH SIMILARITY ({avg_sim:.2f}): SAE features encode SCENARIO, not DECISION")
        print("  - AF and genuine samples have nearly identical feature activations")
        print("  - This explains why steering fails (features fire the same way)")
        print("  - Need contrastive SAE training to separate decision from scenario")
    elif avg_sim > 0.7:
        print(f"MODERATE SIMILARITY ({avg_sim:.2f}): Partial decision encoding")
        print("  - Some features distinguish AF from genuine, but overlap exists")
        print("  - May need to identify which specific features encode decision")
        print("  - Steering might work with targeted feature selection")
    else:
        print(f"LOW SIMILARITY ({avg_sim:.2f}): SAE features encode DECISION")
        print("  - AF and genuine samples have different feature activations")
        print("  - Steering should work - verify with ablation experiments")
        print("  - If steering still fails, check for redundancy/distributed encoding")

    # Feature-level analysis
    print("\n--- Feature-Level Analysis ---")
    print("Which probe features differ most between AF and genuine?")
    print()

    # Aggregate differences across all pairs
    feature_names = []
    for layer, feature_idx in FEATURES:
        feature_names.extend([
            f"L{layer}_F{feature_idx}_max",
            f"L{layer}_F{feature_idx}_mean",
            f"L{layer}_F{feature_idx}_count",
            f"L{layer}_F{feature_idx}_std",
        ])

    all_diffs = np.array([r['probe_diff'] for r in results])
    mean_diffs = all_diffs.mean(axis=0)
    sorted_idx = np.argsort(mean_diffs)[::-1]

    print(f"{'Feature':<25} {'Mean Diff':<12} {'Encodes Decision?':<20}")
    print("-" * 60)
    for idx in sorted_idx[:10]:
        encodes = "YES - candidates" if mean_diffs[idx] > 0.5 else "No"
        print(f"{feature_names[idx]:<25} {mean_diffs[idx]:.4f}       {encodes}")

    # Statistical test: Are AF and genuine feature vectors significantly different?
    print("\n--- Statistical Test ---")
    af_probes = np.array([r['af_probe_features'] for r in results])
    genuine_probes = np.array([r['genuine_probe_features'] for r in results])

    # Paired t-test on overall feature magnitudes
    af_norms = np.linalg.norm(af_probes, axis=1)
    genuine_norms = np.linalg.norm(genuine_probes, axis=1)

    t_stat, p_value = stats.ttest_rel(af_norms, genuine_norms)
    print(f"Paired t-test on feature vector norms:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4f}")

    if p_value < 0.05:
        print("  Significant difference between AF and genuine feature magnitudes")
    else:
        print("  No significant difference - features encode scenario, not decision")

    # Save results
    output = {
        'summary': {
            'n_pairs': len(pairs),
            'mean_full_similarity': float(np.mean(full_sims)),
            'mean_probe_similarity': float(np.mean(probe_sims)),
            'interpretation': 'SCENARIO' if avg_sim > 0.7 else 'DECISION',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
        },
        'pairs': results,
        'feature_names': feature_names,
        'mean_feature_diffs': mean_diffs.tolist(),
    }

    output_file = "minimal_pair_diagnosis_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
If SCENARIO encoding (high similarity):
  1. Use contrastive SAE training (Research Plan Section 1.2)
  2. Add loss term that pushes AF/genuine apart in feature space
  3. Re-extract features and verify lower similarity

If DECISION encoding (low similarity):
  1. Identify top decision-encoding features from this analysis
  2. Run targeted ablation on those features
  3. Measure steering flip rate

If MIXED:
  1. Train a linear probe on the feature differences
  2. Use probe coefficients to weight features for steering
  3. Apply weighted steering vector
""")


if __name__ == "__main__":
    main()
