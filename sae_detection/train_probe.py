#!/usr/bin/env python3
"""
Train a linear probe on SAE feature activations for AF detection.
Should significantly improve over simple max-activation threshold.
"""

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn.functional as F
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

MODEL_ID = "google/gemma-2-27b-it"  # Must match SAE (gemma-scope-2)
SAE_REPO = "google/gemma-scope-2-27b-it"
DEVICE = "cuda"

# All features to include in probe
PROBE_FEATURES = [
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
    """Extract feature activation vector for a text sample."""
    activations = {}
    handles = []

    def make_hook(layer):
        def hook(module, input, output):
            activations[layer] = output.detach()
        return hook

    # Register hooks for all needed layers
    for layer in transcoders.keys():
        try:
            target = model.model.language_model.layers[layer]
        except AttributeError:
            target = model.base_model.model.model.language_model.layers[layer]
        handle = target.pre_feedforward_layernorm.register_forward_hook(make_hook(layer))
        handles.append(handle)

    # Forward pass
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for h in handles:
        h.remove()

    # Extract features
    feature_vector = []
    for layer, feature_idx in features:
        acts = activations[layer].squeeze(0).to(transcoders[layer].w_enc.dtype)
        feats = transcoders[layer].encode(acts)
        feat_acts = feats[:, feature_idx]

        # Multiple statistics per feature
        feature_vector.extend([
            feat_acts.max().item(),          # max activation
            feat_acts.mean().item(),         # mean activation
            (feat_acts > 0).sum().item(),    # count of active positions
            feat_acts.std().item() if len(feat_acts) > 1 else 0,  # std
        ])

    return np.array(feature_vector)


def main():
    print("=" * 70)
    print("LINEAR PROBE TRAINING: Improved AF Detection")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading Gold-106 samples...")
    data_paths = ["../../data/gold_106.json", "../data/gold_106.json", "data/gold_106.json"]
    for path in data_paths:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            break

    samples = data.get('samples', data)
    print(f"Total samples: {len(samples)}")

    # Load model
    print("\n[2/5] Loading model (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Load transcoders
    print("\n[3/5] Loading transcoders...")
    transcoders = {}
    for layer in set(f[0] for f in PROBE_FEATURES):
        print(f"  Loading layer {layer}...")
        transcoders[layer] = Transcoder(SAE_REPO, layer).to(DEVICE, torch.bfloat16)

    # Extract features
    print("\n[4/5] Extracting feature vectors...")
    X = []
    y = []

    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(samples)}...")

        text = sample.get('text', sample.get('reasoning', ''))
        label = sample.get('label', 'unknown')
        is_af = 1 if label in ['potential_faking', 'alignment_faking', 'AF'] else 0

        features = get_feature_vector(model, tokenizer, text, transcoders, PROBE_FEATURES)
        X.append(features)
        y.append(is_af)

    X = np.array(X)
    y = np.array(y)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels: {sum(y)} AF, {len(y) - sum(y)} Aligned")

    # Train probe
    print("\n[5/5] Training and evaluating probe...")
    print("=" * 70)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Test different regularization strengths
    best_auroc = 0
    best_C = 1.0

    for C in [0.01, 0.1, 1.0, 10.0]:
        probe = LogisticRegression(C=C, max_iter=1000, random_state=42)
        scores = cross_val_score(probe, X_scaled, y, cv=cv, scoring='roc_auc')
        mean_auroc = scores.mean()
        print(f"C={C}: AUROC = {mean_auroc:.4f} (+/- {scores.std():.4f})")

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_C = C

    print(f"\nBest: C={best_C} with AUROC = {best_auroc:.4f}")

    # Train final model
    final_probe = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    final_probe.fit(X_scaled, y)

    # Feature importance
    print("\n--- Feature Importance ---")
    feature_names = []
    for layer, feature_idx in PROBE_FEATURES:
        feature_names.extend([
            f"L{layer}_F{feature_idx}_max",
            f"L{layer}_F{feature_idx}_mean",
            f"L{layer}_F{feature_idx}_count",
            f"L{layer}_F{feature_idx}_std",
        ])

    importance = np.abs(final_probe.coef_[0])
    sorted_idx = np.argsort(importance)[::-1]

    print(f"{'Feature':<30} {'Coefficient':<12} {'|Coef|':<10}")
    print("-" * 55)
    for idx in sorted_idx[:15]:
        print(f"{feature_names[idx]:<30} {final_probe.coef_[0][idx]:>10.4f}   {importance[idx]:.4f}")

    # Final predictions
    y_prob = final_probe.predict_proba(X_scaled)[:, 1]
    final_auroc = roc_auc_score(y, y_prob)

    print(f"\n--- Final Model Performance ---")
    print(f"AUROC: {final_auroc:.4f}")

    # Classification at optimal threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]

    y_pred = (y_prob > best_threshold).astype(int)
    print(f"\nAt optimal threshold ({best_threshold:.3f}):")
    print(classification_report(y, y_pred, target_names=['Aligned', 'AF']))

    # Compare to baseline
    print("\n--- Comparison to Baseline ---")
    # Baseline: just L40_F8921 max
    baseline_scores = X[:, 4]  # max of L40_F8921 (second feature, first stat)
    baseline_auroc = roc_auc_score(y, baseline_scores)
    print(f"Baseline (L40_F8921 max only): AUROC = {baseline_auroc:.4f}")
    print(f"Linear Probe (all features):   AUROC = {final_auroc:.4f}")
    print(f"Improvement: +{(final_auroc - baseline_auroc) * 100:.1f}%")

    # Save results
    results = {
        'best_C': best_C,
        'cv_auroc': best_auroc,
        'final_auroc': final_auroc,
        'baseline_auroc': baseline_auroc,
        'feature_names': feature_names,
        'coefficients': final_probe.coef_[0].tolist(),
        'intercept': float(final_probe.intercept_[0]),
        'best_threshold': float(best_threshold),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
    }

    with open("probe_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to probe_training_results.json")


if __name__ == "__main__":
    main()
