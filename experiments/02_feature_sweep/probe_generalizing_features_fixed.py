#!/usr/bin/env python3
"""
Train Probe on Selected Features (FIXED - No Leakage)
=====================================================
Uses features selected by find_generalizing_features_fixed.py
(selected using CV on training data, not gold_106).
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
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


def extract_features(model, tokenizer, sae, samples, layer):
    all_activations = []
    for sample in tqdm(samples, desc="Extracting"):
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

    return np.array(all_activations)


def main():
    results_dir = Path("./results/sweep_resid_post_20260107_141945")
    layer = 40

    print("="*70)
    print("PROBE ON SELECTED FEATURES (FIXED - NO LEAKAGE)")
    print("="*70)

    # Load feature indices from FIXED selection (not the leaky one)
    print("\n[1/6] Loading selected features (from fixed selection)...")
    features_file = Path("cross_distribution_features_fixed.json")
    if not features_file.exists():
        print(f"  ERROR: {features_file} not found!")
        print("  Run find_generalizing_features_fixed.py first")
        return

    with open(features_file) as f:
        data = json.load(f)

    selected = data.get("selected_features", [])
    if not selected:
        print("  No features were selected by the fixed method!")
        print("  This likely means SAE features don't generalize.")
        print(f"\n  Summary from selection:")
        print(f"    Method: {data['methodology']['feature_selection']}")
        print(f"    CV threshold: {data['methodology']['cv_threshold']}")
        return

    feat_indices = [f["feature_idx"] for f in selected]
    print(f"  {len(feat_indices)} features selected using CV (no gold_106 leakage)")
    print(f"  Selection method: {data['methodology']['feature_selection']}")

    # Load training data
    print("\n[2/6] Loading training data...")
    X_train_full = np.load(results_dir / f"activations_resid_post_layer_{layer}.npy")
    y_train = load_training_labels()

    X_train = X_train_full[:, feat_indices]
    print(f"  Training shape: {X_train.shape}")
    print(f"  Labels: {y_train.sum()} AF, {(y_train==0).sum()} non-AF")

    # Cross-validation on training data (estimate generalization)
    print("\n[3/6] Cross-validation on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    probe = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)

    cv_scores = cross_val_score(probe, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    print(f"  CV AUROC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Train final probe on all training data
    print("\n[4/6] Training final probe...")
    probe.fit(X_train_scaled, y_train)

    train_probs = probe.predict_proba(X_train_scaled)[:, 1]
    train_auroc = roc_auc_score(y_train, train_probs)
    print(f"  Training AUROC: {train_auroc:.3f}")

    # Load gold_106 (ONLY for final evaluation)
    print("\n[5/6] Loading gold_106 (final evaluation only)...")
    gold_samples = load_gold_106()
    y_gold = np.array([s["is_af"] for s in gold_samples])
    print(f"  Gold_106: {len(gold_samples)} samples, {y_gold.sum()} AF")

    # Extract gold_106 features
    print("\n[6/6] Extracting gold_106 features...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    sae = ResidPostSAE(SAE_REPO, layer).to(DEVICE, torch.bfloat16)
    X_gold_full = extract_features(model, tokenizer, sae, gold_samples, layer)

    X_gold = X_gold_full[:, feat_indices]
    X_gold_scaled = scaler.transform(X_gold)
    print(f"  Gold shape: {X_gold.shape}")

    # Final evaluation
    gold_probs = probe.predict_proba(X_gold_scaled)[:, 1]
    gold_preds = probe.predict(X_gold_scaled)
    gold_auroc = roc_auc_score(y_gold, gold_probs)
    gold_acc = accuracy_score(y_gold, gold_preds)

    # Results
    print("\n" + "="*70)
    print("RESULTS (UNBIASED)")
    print("="*70)

    print(f"\n  Training CV AUROC:   {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  Training AUROC:      {train_auroc:.3f}")
    print(f"  Gold-106 AUROC:      {gold_auroc:.3f}")
    print(f"  Gold-106 Accuracy:   {gold_acc:.3f}")

    print(f"\n  Generalization Gap:")
    print(f"    CV → Gold: {gold_auroc - cv_scores.mean():+.3f}")

    print("\n  Classification Report (Gold-106):")
    print(classification_report(y_gold, gold_preds, target_names=["Non-AF", "AF"]))

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if gold_auroc >= 0.70:
        print(f"\n  ✅ SUCCESS: Probe generalizes to held-out data (AUROC >= 0.70)")
    elif gold_auroc >= 0.60:
        print(f"\n  ⚠️  WEAK: Some generalization (0.60 <= AUROC < 0.70)")
    else:
        print(f"\n  ❌ FAILURE: No meaningful generalization (AUROC < 0.60)")
        print(f"     SAE features don't detect alignment faking.")

    # Save results
    output = {
        "methodology": {
            "feature_selection": "CV on training data only (no gold_106)",
            "n_features": len(feat_indices),
            "feature_indices": feat_indices,
            "random_seed": RANDOM_SEED,
        },
        "training": {
            "cv_auroc_mean": float(cv_scores.mean()),
            "cv_auroc_std": float(cv_scores.std()),
            "train_auroc": float(train_auroc),
            "n_samples": len(y_train),
            "n_af": int(y_train.sum()),
        },
        "gold_106": {
            "auroc": float(gold_auroc),
            "accuracy": float(gold_acc),
            "n_samples": len(gold_samples),
            "n_af": int(y_gold.sum()),
        },
        "generalization": {
            "cv_to_gold_gap": float(gold_auroc - cv_scores.mean()),
        }
    }

    output_file = "generalizing_probe_results_fixed.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
