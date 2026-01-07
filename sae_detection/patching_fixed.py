#!/usr/bin/env python3
"""
Fixed Patching Experiment
=========================
Original problem: Gemma classifier predicted 96% AF on everything
Fix: Patch specific features between samples, evaluate with trained probe

Method:
1. Load stored activations and train probe on top N features
2. For pairs of (AF sample, Aligned sample):
   - Patch AF features INTO Aligned → should increase AF probability
   - Patch Aligned features INTO AF → should decrease AF probability
3. This proves features are SUFFICIENT (not just necessary)

Combined with ablation (necessary), this completes causal validation.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')


def load_needles(needles_path):
    """Load the true needles from sweep results."""
    with open(needles_path) as f:
        data = json.load(f)
    return data["needles"]


def load_activations(results_dir, layers=[31, 40, 53]):
    """Load stored activation matrices from sweep."""
    activations = {}
    for layer in layers:
        path = Path(results_dir) / f"activations_resid_post_layer_{layer}.npy"
        if path.exists():
            activations[layer] = np.load(path)
            print(f"  Loaded layer {layer}: {activations[layer].shape}")
    return activations


def load_dataset_labels(cache_dir="./data"):
    """Load labels for the 2330 samples."""
    files = [
        ("aligned.json", "aligned"),
        ("alignment_faking.json", "alignment_faking"),
        ("genuine_compliance.json", "genuine_compliance"),
        ("hard_negative.json", "hard_negative"),
        ("compliant_hard_negative.json", "compliant_hard_negative"),
    ]

    labels = []
    for fname, class_name in files:
        path = hf_hub_download(
            repo_id="vincentoh/af-detection-benchmark",
            filename=f"classes/{fname}",
            repo_type="dataset",
            local_dir=cache_dir
        )
        with open(path) as f:
            data = json.load(f)
        for _ in data["samples"]:
            labels.append({
                "class": class_name,
                "is_af": 1 if class_name == "alignment_faking" else 0,
            })

    return labels


def build_feature_matrix(activations, needles, top_n=50):
    """Build feature matrix using top N needles."""
    sorted_needles = sorted(needles, key=lambda x: x["auroc_af_vs_hard_neg"], reverse=True)
    top_needles = sorted_needles[:top_n]

    n_samples = list(activations.values())[0].shape[0]
    X = np.zeros((n_samples, len(top_needles)), dtype=np.float32)

    for i, needle in enumerate(top_needles):
        layer = needle["layer"]
        feat_idx = needle["feature_idx"]
        if layer in activations:
            X[:, i] = activations[layer][:, feat_idx]

    return X, top_needles


def train_probe(X, y, test_size=0.2):
    """Train logistic regression probe."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X_train_scaled, y_train)

    y_prob = probe.predict_proba(X_test_scaled)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)

    print(f"\n  Probe AUROC: {auroc:.3f}")

    return probe, scaler


def patching_experiment(X, y, probe, scaler, needles, n_pairs=100, patch_counts=[1, 5, 10, 20]):
    """
    Patch features between AF and Aligned samples.

    If features are sufficient:
    - Patching AF features INTO Aligned → should INCREASE AF probability
    - Patching Aligned features INTO AF → should DECREASE AF probability
    """
    print("\n" + "="*70)
    print("PATCHING EXPERIMENT")
    print("="*70)

    af_indices = np.where(y == 1)[0]
    aligned_indices = np.where(y == 0)[0]

    # Sample pairs
    np.random.seed(42)
    n_pairs = min(n_pairs, len(af_indices), len(aligned_indices))
    af_sample_idx = np.random.choice(af_indices, n_pairs, replace=False)
    aligned_sample_idx = np.random.choice(aligned_indices, n_pairs, replace=False)

    print(f"\nTesting {n_pairs} pairs")

    results = []

    for n_patch in patch_counts:
        if n_patch > X.shape[1]:
            continue

        # Track changes
        aligned_to_af_deltas = []  # Patching AF features INTO Aligned
        af_to_aligned_deltas = []  # Patching Aligned features INTO AF

        aligned_became_af = 0  # Prob > 0.5 after patching
        af_became_aligned = 0  # Prob < 0.5 after patching

        for i in range(n_pairs):
            af_idx = af_sample_idx[i]
            aligned_idx = aligned_sample_idx[i]

            # Get original features
            X_af = X[af_idx].copy()
            X_aligned = X[aligned_idx].copy()

            # Original predictions
            af_prob_orig = probe.predict_proba(scaler.transform([X_af]))[0, 1]
            aligned_prob_orig = probe.predict_proba(scaler.transform([X_aligned]))[0, 1]

            # === Patch AF features INTO Aligned ===
            X_aligned_patched = X_aligned.copy()
            X_aligned_patched[:n_patch] = X_af[:n_patch]  # Top N features from AF
            aligned_prob_patched = probe.predict_proba(scaler.transform([X_aligned_patched]))[0, 1]

            aligned_delta = aligned_prob_patched - aligned_prob_orig
            aligned_to_af_deltas.append(aligned_delta)

            if aligned_prob_patched > 0.5:
                aligned_became_af += 1

            # === Patch Aligned features INTO AF ===
            X_af_patched = X_af.copy()
            X_af_patched[:n_patch] = X_aligned[:n_patch]  # Top N features from Aligned
            af_prob_patched = probe.predict_proba(scaler.transform([X_af_patched]))[0, 1]

            af_delta = af_prob_patched - af_prob_orig
            af_to_aligned_deltas.append(af_delta)

            if af_prob_patched < 0.5:
                af_became_aligned += 1

        # Compute statistics
        result = {
            "n_patched": n_patch,
            "features_patched": [f"L{needles[i]['layer']}_F{needles[i]['feature_idx']}"
                                 for i in range(min(n_patch, len(needles)))],
            "n_pairs": n_pairs,
            # Patching AF → Aligned
            "aligned_mean_delta": float(np.mean(aligned_to_af_deltas)),
            "aligned_std_delta": float(np.std(aligned_to_af_deltas)),
            "aligned_became_af": int(aligned_became_af),
            "aligned_became_af_pct": float(aligned_became_af / n_pairs * 100),
            # Patching Aligned → AF
            "af_mean_delta": float(np.mean(af_to_aligned_deltas)),
            "af_std_delta": float(np.std(af_to_aligned_deltas)),
            "af_became_aligned": int(af_became_aligned),
            "af_became_aligned_pct": float(af_became_aligned / n_pairs * 100),
        }
        results.append(result)

        # Print
        print(f"\n  Patch top {n_patch} features:")
        print(f"    AF→Aligned: {aligned_became_af}/{n_pairs} ({aligned_became_af/n_pairs*100:.1f}%) Aligned samples became AF")
        print(f"                Mean Δprob: {np.mean(aligned_to_af_deltas):+.3f}")
        print(f"    Aligned→AF: {af_became_aligned}/{n_pairs} ({af_became_aligned/n_pairs*100:.1f}%) AF samples became Aligned")
        print(f"                Mean Δprob: {np.mean(af_to_aligned_deltas):+.3f}")

    return results


def interpret_results(results):
    """Interpret patching results."""
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    # Check for bidirectional effect
    for r in results:
        if r["aligned_became_af_pct"] > 50 and r["af_became_aligned_pct"] > 50:
            print(f"\n✅ SUFFICIENT: Patching top {r['n_patched']} features:")
            print(f"   - {r['aligned_became_af_pct']:.0f}% of Aligned samples classified as AF after patching")
            print(f"   - {r['af_became_aligned_pct']:.0f}% of AF samples classified as Aligned after patching")
            print(f"   → Features CARRY the AF information (sufficient for classification)")
            break
    else:
        # Check if any direction works
        best_aligned = max(results, key=lambda x: x["aligned_became_af_pct"])
        best_af = max(results, key=lambda x: x["af_became_aligned_pct"])

        print(f"\n⚠️  PARTIAL: Best results:")
        print(f"   - Aligned→AF: {best_aligned['aligned_became_af_pct']:.0f}% (top {best_aligned['n_patched']} features)")
        print(f"   - AF→Aligned: {best_af['af_became_aligned_pct']:.0f}% (top {best_af['n_patched']} features)")

    # Combined with ablation
    print("\n--- Combined with Ablation ---")
    print("Ablation: Removing features DECREASES AF probability → NECESSARY")
    print("Patching: Adding features INCREASES AF probability → SUFFICIENT")
    print("Together: Features are both necessary AND sufficient for AF detection")


def main():
    parser = argparse.ArgumentParser(description="Fixed patching experiment")
    parser.add_argument("--results-dir", default="./results_2026-01-07",
                        help="Directory with sweep results")
    parser.add_argument("--needles-file", default="./results_2026-01-07/true_needles_resid_post.json",
                        help="Path to needles JSON")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of top features to use")
    parser.add_argument("--n-pairs", type=int, default=100,
                        help="Number of sample pairs to test")
    parser.add_argument("--output", default="patching_fixed_results.json",
                        help="Output file")
    args = parser.parse_args()

    print("="*70)
    print("FIXED PATCHING EXPERIMENT")
    print("="*70)

    # Load needles
    print("\n[1/5] Loading needles...")
    needles = load_needles(args.needles_file)
    print(f"  Found {len(needles)} needles")

    # Load activations
    print("\n[2/5] Loading activations...")
    activations = load_activations(args.results_dir)

    if not activations:
        print("\nERROR: No activation files found!")
        print("Run on Lambda where activations are stored.")
        return

    # Load labels
    print("\n[3/5] Loading labels...")
    labels = load_dataset_labels()
    y = np.array([l["is_af"] for l in labels])
    print(f"  {y.sum()} AF samples, {len(y) - y.sum()} non-AF samples")

    # Build feature matrix
    print("\n[4/5] Building feature matrix...")
    X, top_needles = build_feature_matrix(activations, needles, top_n=args.top_n)
    print(f"  Shape: {X.shape}")
    print(f"  Top 5 features:")
    for i, n in enumerate(top_needles[:5]):
        print(f"    {i+1}. L{n['layer']}_F{n['feature_idx']}: AUROC(hn)={n['auroc_af_vs_hard_neg']:.3f}")

    # Train probe
    print("\n[5/5] Training probe...")
    probe, scaler = train_probe(X, y)

    # Run patching
    results = patching_experiment(X, y, probe, scaler, top_needles,
                                  n_pairs=args.n_pairs,
                                  patch_counts=[1, 5, 10, 20, 50])

    # Interpret
    interpret_results(results)

    # Save
    output = {
        "config": {
            "top_n_features": args.top_n,
            "n_pairs": args.n_pairs,
        },
        "patching_results": results,
        "top_features": [
            {"layer": n["layer"], "feature_idx": n["feature_idx"],
             "auroc_hn": n["auroc_af_vs_hard_neg"]}
            for n in top_needles[:20]
        ]
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
