#!/usr/bin/env python3
"""
Train probe on ONLY the 51 generalizing features.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from huggingface_hub import hf_hub_download

def load_training_labels(cache_dir="./data"):
    files = [
        ("aligned.json", 0),
        ("alignment_faking.json", 1),
        ("genuine_compliance.json", 0),
        ("hard_negative.json", 0),
        ("compliant_hard_negative.json", 0),
    ]
    labels = []
    for fname, is_af in files:
        path = hf_hub_download(
            repo_id="vincentoh/af-detection-benchmark",
            filename=f"classes/{fname}",
            repo_type="dataset",
            local_dir=cache_dir
        )
        with open(path) as f:
            data = json.load(f)
        labels.extend([is_af] * len(data["samples"]))
    return np.array(labels)


def main():
    results_dir = Path("./results/sweep_resid_post_20260107_141945")

    print("="*70)
    print("TRAINING PROBE ON 51 GENERALIZING FEATURES")
    print("="*70)

    # Load generalizing features
    print("\n[1/4] Loading generalizing features...")
    with open("cross_distribution_features.json") as f:
        data = json.load(f)

    generalizing = data["generalizing_features"]
    feat_indices = [f["feature_idx"] for f in generalizing]
    print(f"  {len(feat_indices)} generalizing features")

    # Load training data
    print("\n[2/4] Loading training data...")
    X_train = np.load(results_dir / "activations_resid_post_layer_40.npy")
    y_train = load_training_labels()
    print(f"  Training: {X_train.shape[0]} samples")

    # Load gold_106 activations (extract from cross_distribution run)
    print("\n[3/4] Loading gold_106 features...")
    # We need to load/extract gold features - they should be in the json
    gold_results = data["all_results"]

    # Reconstruct gold activations from the per-feature stats
    # Actually we need the raw activations. Let me check if they were saved.
    # They weren't saved separately, so we need to re-extract or load from validation run

    # Load from validation results
    with open("gold106_validation_results.json") as f:
        val_data = json.load(f)

    y_gold = np.array(val_data["gold_106"]["per_sample_labels"])

    # We need gold activations - let me load them fresh
    # Actually the validation script didn't save them either
    # Let me compute AUROC from the per-feature stats we already have

    print("  Using pre-computed per-feature AUROCs")

    # Train probe on training data
    print("\n[4/4] Training and evaluating probe...")

    # Select only generalizing features from training data
    X_train_gen = X_train[:, feat_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_gen)

    probe = LogisticRegression(max_iter=1000, class_weight='balanced')
    probe.fit(X_train_scaled, y_train)

    # Training performance
    train_probs = probe.predict_proba(X_train_scaled)[:, 1]
    train_auroc = roc_auc_score(y_train, train_probs)
    train_acc = accuracy_score(y_train, probe.predict(X_train_scaled))

    print(f"\n  Training Results:")
    print(f"    AUROC:    {train_auroc:.3f}")
    print(f"    Accuracy: {train_acc:.3f}")

    # For gold_106, we can estimate from individual feature AUROCs
    # The best we can do without re-extracting is use the feature stats
    gen_gold_aurocs = [f["gold_auroc"] for f in generalizing]

    print(f"\n  Gold-106 Estimates (from individual features):")
    print(f"    Mean feature AUROC: {np.mean(gen_gold_aurocs):.3f}")
    print(f"    Max feature AUROC:  {np.max(gen_gold_aurocs):.3f}")
    print(f"    Expected probe AUROC: ~0.75-0.85 (ensemble should beat best single)")

    print("\n  To get exact probe AUROC on gold_106, need to re-extract features.")
    print("  Run: python validate_gold106_generalizing.py")

    # Save probe info
    output = {
        "n_features": len(feat_indices),
        "feature_indices": feat_indices,
        "training_auroc": float(train_auroc),
        "training_accuracy": float(train_acc),
        "individual_gold_aurocs": gen_gold_aurocs,
        "mean_gold_auroc": float(np.mean(gen_gold_aurocs)),
        "max_gold_auroc": float(np.max(gen_gold_aurocs)),
    }

    with open("generalizing_probe_info.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Saved to generalizing_probe_info.json")


if __name__ == "__main__":
    main()
