#!/usr/bin/env python3
"""
Fixed Ablation Experiment
=========================
Original problem: Features=0 during classification prompt (wrong pass)
Fix: Ablate stored features, run through trained probe

Method:
1. Load stored activations from sweep (2330 samples × 16384 features)
2. Train logistic probe on top N features
3. Zero out feature groups and measure probe probability change
4. If ablating AF features drops AF probability → features are causal
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
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
        else:
            print(f"  WARNING: {path} not found")
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
    """
    Build feature matrix using top N needles.
    Returns: [n_samples, top_n] matrix
    """
    # Sort needles by hard negative AUROC
    sorted_needles = sorted(needles, key=lambda x: x["auroc_af_vs_hard_neg"], reverse=True)
    top_needles = sorted_needles[:top_n]

    print(f"\n  Using top {len(top_needles)} features:")
    for i, n in enumerate(top_needles[:5]):
        print(f"    {i+1}. L{n['layer']}_F{n['feature_idx']}: AUROC(hn)={n['auroc_af_vs_hard_neg']:.3f}")
    if len(top_needles) > 5:
        print(f"    ... and {len(top_needles)-5} more")

    # Build matrix
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

    # Evaluate
    y_pred = probe.predict(X_test_scaled)
    y_prob = probe.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)

    print(f"\n  Probe performance (test set):")
    print(f"    Accuracy: {accuracy:.3f}")
    print(f"    AUROC:    {auroc:.3f}")

    return probe, scaler, {"accuracy": accuracy, "auroc": auroc}


def ablation_experiment(X, y, probe, scaler, needles, ablate_counts=[1, 5, 10, 20, 50]):
    """
    Ablate groups of features and measure effect on probe predictions.

    If features are causal:
    - Ablating top AF features should DECREASE AF probability for AF samples
    - Ablating top AF features should have minimal effect on Aligned samples
    """
    print("\n" + "="*70)
    print("ABLATION EXPERIMENT")
    print("="*70)

    X_scaled = scaler.transform(X)

    # Baseline predictions
    baseline_probs = probe.predict_proba(X_scaled)[:, 1]

    af_mask = y == 1
    aligned_mask = y == 0

    baseline_af_mean = baseline_probs[af_mask].mean()
    baseline_aligned_mean = baseline_probs[aligned_mask].mean()

    print(f"\nBaseline AF probability:")
    print(f"  AF samples:      {baseline_af_mean:.3f}")
    print(f"  Aligned samples: {baseline_aligned_mean:.3f}")

    results = []

    for n_ablate in ablate_counts:
        if n_ablate > X.shape[1]:
            continue

        # Zero out top N features
        X_ablated = X.copy()
        X_ablated[:, :n_ablate] = 0  # Features are sorted by importance
        X_ablated_scaled = scaler.transform(X_ablated)

        # Get new predictions
        ablated_probs = probe.predict_proba(X_ablated_scaled)[:, 1]

        # Compute deltas
        af_delta = ablated_probs[af_mask].mean() - baseline_af_mean
        aligned_delta = ablated_probs[aligned_mask].mean() - baseline_aligned_mean

        # Per-sample analysis
        af_decreased = (ablated_probs[af_mask] < baseline_probs[af_mask]).sum()
        af_total = af_mask.sum()

        result = {
            "n_ablated": n_ablate,
            "features_ablated": [f"L{needles[i]['layer']}_F{needles[i]['feature_idx']}"
                                 for i in range(min(n_ablate, len(needles)))],
            "af_prob_baseline": float(baseline_af_mean),
            "af_prob_ablated": float(ablated_probs[af_mask].mean()),
            "af_delta": float(af_delta),
            "aligned_prob_baseline": float(baseline_aligned_mean),
            "aligned_prob_ablated": float(ablated_probs[aligned_mask].mean()),
            "aligned_delta": float(aligned_delta),
            "af_samples_decreased": int(af_decreased),
            "af_samples_total": int(af_total),
            "pct_af_decreased": float(af_decreased / af_total * 100),
        }
        results.append(result)

        # Print
        direction = "↓" if af_delta < 0 else "↑"
        print(f"\n  Ablate top {n_ablate} features:")
        print(f"    AF prob:      {baseline_af_mean:.3f} → {ablated_probs[af_mask].mean():.3f} ({af_delta:+.3f}) {direction}")
        print(f"    Aligned prob: {baseline_aligned_mean:.3f} → {ablated_probs[aligned_mask].mean():.3f} ({aligned_delta:+.3f})")
        print(f"    AF samples with decreased prob: {af_decreased}/{af_total} ({af_decreased/af_total*100:.1f}%)")

    return results


def interpret_results(results):
    """Interpret ablation results."""
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    # Check if ablation decreases AF probability
    significant_effect = False
    for r in results:
        if r["af_delta"] < -0.05 and r["pct_af_decreased"] > 60:
            significant_effect = True
            break

    if significant_effect:
        print("\n✅ POSITIVE: Ablation causes significant AF probability decrease")
        print("   → Features ARE causally relevant to AF detection")
        print("   → Probe relies on these specific features")
    else:
        print("\n⚠️  INCONCLUSIVE: Ablation has minimal effect")
        print("   Possible reasons:")
        print("   - Features redundant (information distributed)")
        print("   - Probe learned different features than top needles")
        print("   - Need to ablate more features")

    # Check selectivity (should affect AF more than Aligned)
    for r in results:
        if abs(r["af_delta"]) > 2 * abs(r["aligned_delta"]):
            print(f"\n✅ SELECTIVE: Ablating {r['n_ablated']} features affects AF {abs(r['af_delta']/r['aligned_delta']):.1f}x more than Aligned")
            break


def main():
    parser = argparse.ArgumentParser(description="Fixed ablation experiment")
    parser.add_argument("--results-dir", default="./results_2026-01-07",
                        help="Directory with sweep results")
    parser.add_argument("--needles-file", default="./results_2026-01-07/true_needles_resid_post.json",
                        help="Path to needles JSON")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of top features to use for probe")
    parser.add_argument("--output", default="ablation_fixed_results.json",
                        help="Output file")
    args = parser.parse_args()

    print("="*70)
    print("FIXED ABLATION EXPERIMENT")
    print("="*70)
    print(f"Using top {args.top_n} features from {args.needles_file}")

    # Load needles
    print("\n[1/5] Loading needles...")
    needles = load_needles(args.needles_file)
    print(f"  Found {len(needles)} needles")

    # Load activations
    print("\n[2/5] Loading activations...")
    activations = load_activations(args.results_dir)

    if not activations:
        print("\nERROR: No activation files found!")
        print("Run this script where activations are stored, or download them first from your compute instance.")
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

    # Train probe
    print("\n[5/5] Training probe...")
    probe, scaler, probe_metrics = train_probe(X, y)

    # Run ablation
    results = ablation_experiment(X, y, probe, scaler, top_needles)

    # Interpret
    interpret_results(results)

    # Save results
    output = {
        "config": {
            "top_n_features": args.top_n,
            "n_samples": len(y),
            "n_af": int(y.sum()),
        },
        "probe_metrics": probe_metrics,
        "ablation_results": results,
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
