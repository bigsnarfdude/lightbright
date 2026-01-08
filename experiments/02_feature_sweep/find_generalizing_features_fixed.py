#!/usr/bin/env python3
"""
Find Generalizing Features (FIXED - No Test Set Leakage)
========================================================
Select features using ONLY training data (with CV).
Gold_106 is used ONLY for final evaluation, never for selection.

This version extracts features directly (doesn't require pre-saved .npy files).
"""

import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda"
MODEL_NAME = "google/gemma-3-27b-it"
SAE_REPO = "google/gemma-scope-2-27b-it"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class ResidPostSAE:
    def __init__(self, repo_id: str, layer: int):
        folder = f"resid_post/layer_{layer}_width_16k_l0_medium"
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        self.threshold = weights.get("threshold", None)

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        if self.threshold is not None:
            return F.relu(pre_acts - self.threshold)
        return F.relu(pre_acts)


def load_gold_106(cache_dir="./data"):
    """Load gold_106 - FOR FINAL EVAL ONLY."""
    path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset",
        local_dir=cache_dir
    )
    with open(path) as f:
        data = json.load(f)

    samples = []
    for s in data["samples"]:
        label = s.get("label", s.get("class", ""))
        is_af = 1 if label in ["alignment_faking", "potential_faking"] else 0
        samples.append({"text": s["text"], "is_af": is_af})

    return samples


def load_training_data(cache_dir="./data"):
    """Load training data with texts and labels."""
    files = [
        ("aligned.json", 0),
        ("alignment_faking.json", 1),
        ("genuine_compliance.json", 0),
        ("hard_negative.json", 0),
        ("compliant_hard_negative.json", 0),
    ]

    samples = []
    for fname, is_af in files:
        path = hf_hub_download(
            repo_id="vincentoh/af-detection-benchmark",
            filename=f"classes/{fname}",
            repo_type="dataset",
            local_dir=cache_dir
        )
        with open(path) as f:
            data = json.load(f)
        for s in data["samples"]:
            samples.append({"text": s["text"], "is_af": is_af})

    return samples


def extract_features(model, tokenizer, sae, samples, layer, batch_desc="Extracting"):
    """Extract SAE features for samples."""
    all_activations = []

    for sample in tqdm(samples, desc=batch_desc):
        activations = {}

        def hook(module, input, output):
            if isinstance(output, tuple):
                activations["out"] = output[0].detach()
            else:
                activations["out"] = output.detach()

        target = model.model.language_model.layers[layer]
        handle = target.register_forward_hook(hook)

        inputs = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(DEVICE)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        act = activations["out"].squeeze(0).to(torch.bfloat16)
        with torch.no_grad():
            features = sae.encode(act)

        max_per_feature = features.max(dim=0).values.float().cpu().numpy()
        all_activations.append(max_per_feature)

        # Cleanup
        del activations, act, features
        torch.cuda.empty_cache()

    return np.array(all_activations)


def compute_cv_auroc(X, y, feat_idx, n_folds=5):
    """Compute cross-validated AUROC for a single feature."""
    feat_vals = X[:, feat_idx]

    if feat_vals.std() == 0:
        return 0.5, 0.0

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_aurocs = []

    for train_idx, val_idx in cv.split(X, y):
        val_vals = feat_vals[val_idx]
        val_labels = y[val_idx]

        try:
            auroc = roc_auc_score(val_labels, val_vals)
            fold_aurocs.append(auroc)
        except ValueError:
            fold_aurocs.append(0.5)

    return np.mean(fold_aurocs), np.std(fold_aurocs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=40)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-threshold", type=float, default=0.7)
    parser.add_argument("--cv-threshold", type=float, default=0.6)
    parser.add_argument("--output", default="results/cross_distribution_features_fixed.json")
    args = parser.parse_args()

    print("="*70)
    print("FINDING GENERALIZING FEATURES (FIXED - NO LEAKAGE)")
    print("="*70)
    print(f"\nMethodology:")
    print(f"  - Feature selection: {args.n_folds}-fold CV on training data ONLY")
    print(f"  - Selection: train_auroc > {args.train_threshold} AND cv_auroc > {args.cv_threshold}")
    print(f"  - Gold_106: ONLY for final evaluation")
    print(f"  - Random seed: {RANDOM_SEED}")

    # Load model and SAE
    print("\n[1/6] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    print(f"\n[2/6] Loading SAE for layer {args.layer}...")
    sae = ResidPostSAE(SAE_REPO, args.layer).to(DEVICE, torch.bfloat16)
    n_features = sae.w_enc.shape[1]
    print(f"  SAE features: {n_features}")

    # Load and extract training data
    print("\n[3/6] Loading training data...")
    train_samples = load_training_data()
    y_train = np.array([s["is_af"] for s in train_samples])
    print(f"  Training: {len(train_samples)} samples, {y_train.sum()} AF")

    print("\n[4/6] Extracting training features...")
    X_train = extract_features(model, tokenizer, sae, train_samples, args.layer, "Train")
    print(f"  Shape: {X_train.shape}")

    # Find active features (non-zero variance)
    feature_vars = X_train.var(axis=0)
    active_features = np.where(feature_vars > 0)[0]
    print(f"  Active features: {len(active_features)} / {n_features}")

    # =========================================================================
    # FEATURE SELECTION: CV on training data ONLY
    # =========================================================================
    print(f"\n[5/6] Selecting features using {args.n_folds}-fold CV...")
    print("="*70)

    results = []
    selected_features = []

    for feat_idx in tqdm(active_features, desc="CV Analysis"):
        feat_vals = X_train[:, feat_idx]

        # Full training AUROC
        try:
            train_auroc = roc_auc_score(y_train, feat_vals)
        except:
            train_auroc = 0.5

        # Skip if training AUROC is too low
        if train_auroc < 0.6:
            continue

        # Cross-validated AUROC
        cv_auroc_mean, cv_auroc_std = compute_cv_auroc(X_train, y_train, feat_idx, args.n_folds)

        # Activation stats
        train_af_mean = feat_vals[y_train == 1].mean()
        train_non_af_mean = feat_vals[y_train == 0].mean()

        # Selection: ONLY uses training data
        selected = bool(train_auroc > args.train_threshold and cv_auroc_mean > args.cv_threshold)

        result = {
            "feature_idx": int(feat_idx),
            "train_auroc": float(train_auroc),
            "cv_auroc_mean": float(cv_auroc_mean),
            "cv_auroc_std": float(cv_auroc_std),
            "train_af_mean": float(train_af_mean),
            "train_non_af_mean": float(train_non_af_mean),
            "selected": selected,
        }
        results.append(result)

        if selected:
            selected_features.append(result)

    results = sorted(results, key=lambda x: x["cv_auroc_mean"], reverse=True)
    selected_features = sorted(selected_features, key=lambda x: x["cv_auroc_mean"], reverse=True)

    print(f"\n  Features analyzed: {len(results)}")
    print(f"  Features selected: {len(selected_features)}")

    # =========================================================================
    # FINAL EVALUATION on gold_106
    # =========================================================================
    print("\n[6/6] Final evaluation on gold_106...")

    gold_samples = load_gold_106()
    y_gold = np.array([s["is_af"] for s in gold_samples])
    print(f"  Gold_106: {len(gold_samples)} samples, {y_gold.sum()} AF")

    X_gold = extract_features(model, tokenizer, sae, gold_samples, args.layer, "Gold")

    # Compute gold AUROC for all analyzed features
    for result in results:
        feat_idx = result["feature_idx"]
        gold_vals = X_gold[:, feat_idx]
        try:
            gold_auroc = roc_auc_score(y_gold, gold_vals)
        except:
            gold_auroc = 0.5
        result["gold_auroc"] = float(gold_auroc)
        result["gold_af_mean"] = float(gold_vals[y_gold == 1].mean())
        result["gold_non_af_mean"] = float(gold_vals[y_gold == 0].mean())

    for sf in selected_features:
        matching = [r for r in results if r["feature_idx"] == sf["feature_idx"]][0]
        sf["gold_auroc"] = matching["gold_auroc"]
        sf["gold_af_mean"] = matching["gold_af_mean"]
        sf["gold_non_af_mean"] = matching["gold_non_af_mean"]

    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS (UNBIASED)")
    print("="*70)

    print(f"\nFeature Selection (CV on training only):")
    print(f"  Features analyzed: {len(results)}")
    print(f"  Features selected: {len(selected_features)}")

    if selected_features:
        print(f"\nSelected Features:")
        print("-" * 80)
        print(f"{'Feature':<10} {'Train':<8} {'CV':<8} {'Gold':<8} {'Gap':<8}")
        print("-" * 80)

        gaps = []
        for r in selected_features[:20]:
            gap = r["cv_auroc_mean"] - r["gold_auroc"]
            gaps.append(gap)
            print(f"F{r['feature_idx']:<9} {r['train_auroc']:<8.3f} {r['cv_auroc_mean']:<8.3f} {r['gold_auroc']:<8.3f} {gap:+.3f}")

        gold_aurocs = [r["gold_auroc"] for r in selected_features]
        print(f"\nSelected features on gold_106:")
        print(f"  Mean AUROC: {np.mean(gold_aurocs):.3f} +/- {np.std(gold_aurocs):.3f}")
        print(f"  Max AUROC:  {np.max(gold_aurocs):.3f}")

        if np.mean(gold_aurocs) >= 0.6:
            print(f"\n  ✅ FEATURES GENERALIZE")
        else:
            print(f"\n  ❌ FEATURES DO NOT GENERALIZE")
    else:
        print("\n  ❌ NO FEATURES SELECTED BY CV CRITERIA")
        print("\n  Top 10 by CV (none passed threshold):")
        for r in results[:10]:
            print(f"  F{r['feature_idx']:<9} train={r['train_auroc']:.3f} cv={r['cv_auroc_mean']:.3f} gold={r.get('gold_auroc', 0):.3f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "methodology": {
            "feature_selection": "CV on training data only",
            "n_folds": args.n_folds,
            "train_threshold": args.train_threshold,
            "cv_threshold": args.cv_threshold,
            "random_seed": RANDOM_SEED,
            "gold_106_used_for": "final evaluation only",
        },
        "config": {
            "layer": args.layer,
            "n_features_total": n_features,
            "n_features_active": len(active_features),
            "n_features_analyzed": len(results),
        },
        "summary": {
            "n_selected": len(selected_features),
            "cv_auroc_mean": float(np.mean([r["cv_auroc_mean"] for r in selected_features])) if selected_features else 0,
            "gold_auroc_mean": float(np.mean([r["gold_auroc"] for r in selected_features])) if selected_features else 0,
        },
        "selected_features": selected_features,
        "top_50_by_cv": results[:50],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
