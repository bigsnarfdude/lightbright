#!/usr/bin/env python3
"""
Train a linear probe on SAE features for AF detection (4B model).
Then test ablation on the PROBE to measure flip rate.
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
import pickle

MODEL_ID = "google/gemma-3-4b-it"
SAE_REPO = "google/gemma-scope-2-4b-it"
DEVICE = "cuda"

# Layers available in gemma-scope-2-4b-it resid_post (25%, 50%, 65%, 85% depth)
TARGET_LAYERS = [9, 17, 22, 29]


class ResidPostSAE:
    """Load resid_post SAE from GemmaScope-2."""
    def __init__(self, repo_id: str, layer: int):
        possible_paths = [
            f"resid_post/layer_{layer}_width_16k_l0_medium",
            f"resid_post/layer_{layer}_width_16k_l0_small",
            f"resid_post/layer_{layer}_width_16k_l0_big",
        ]

        weights = None
        for path in possible_paths:
            try:
                config_path = hf_hub_download(repo_id, f"{path}/config.json")
                weights_path = hf_hub_download(repo_id, f"{path}/params.safetensors")
                with open(config_path) as f:
                    self.config = json.load(f)
                weights = load_file(weights_path)
                print(f"  Loaded SAE from {path}")
                break
            except Exception:
                continue

        if weights is None:
            raise ValueError(f"Could not load SAE for layer {layer}")

        self.W_enc = weights.get("W_enc", weights.get("w_enc"))
        self.b_enc = weights.get("b_enc", torch.zeros(self.W_enc.shape[1]))
        self.W_dec = weights.get("W_dec", weights.get("w_dec"))
        self.b_dec = weights.get("b_dec", torch.zeros(self.W_dec.shape[1]))

        if "threshold" in weights:
            self.threshold = weights["threshold"]
        else:
            self.threshold = torch.zeros(self.W_enc.shape[1])

        self.n_features = self.W_enc.shape[1]

    def to(self, device, dtype=None):
        self.W_enc = self.W_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        self.W_dec = self.W_dec.to(device, dtype=dtype)
        self.b_dec = self.b_dec.to(device, dtype=dtype)
        self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.W_enc + self.b_enc
        return F.relu(pre_acts - self.threshold)


def extract_all_features(model, tokenizer, text, saes):
    """Extract full SAE feature activations from all layers."""
    activations = {}
    handles = []

    def make_hook(layer):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[layer] = output[0].detach()
            else:
                activations[layer] = output.detach()
        return hook

    for layer in saes.keys():
        try:
            target = model.model.layers[layer]
        except AttributeError:
            target = model.model.language_model.layers[layer]
        handle = target.register_forward_hook(make_hook(layer))
        handles.append(handle)

    inputs = tokenizer(text[:2000], return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    # Get feature activations per layer
    features = {}
    for layer in saes.keys():
        acts = activations[layer].squeeze(0).to(saes[layer].W_enc.dtype)
        feats = saes[layer].encode(acts)  # [seq_len, 16k]
        # Store statistics: max, mean, count, std for each feature
        features[layer] = {
            'max': feats.max(dim=0).values.float().cpu().numpy(),
            'mean': feats.mean(dim=0).float().cpu().numpy(),
            'count': (feats > 0).sum(dim=0).float().cpu().numpy(),
        }

    return features


def features_to_vector(features, top_features_per_layer):
    """Convert features dict to flat vector using top features per layer."""
    vector = []
    for layer, feat_indices in top_features_per_layer.items():
        for idx in feat_indices:
            vector.extend([
                features[layer]['max'][idx],
                features[layer]['mean'][idx],
                features[layer]['count'][idx],
            ])
    return np.array(vector)


def main():
    print("=" * 70)
    print("TRAIN LINEAR PROBE ON 4B SAE FEATURES")
    print("=" * 70)

    # Load training data
    print("\n[1/6] Loading training data...")

    # Use full training data
    train_path = os.path.expanduser("~/mindreader/data/hf_training_data.json")

    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found")
        return

    with open(train_path) as f:
        data = json.load(f)
    samples = data.get('samples', data)

    # Split by label
    af_samples = [s for s in samples if s['label'] == 'potential_faking']
    genuine_samples = [s for s in samples if s['label'] == 'aligned']

    # Subsample for speed (100 each for feature extraction)
    n_samples = 100
    af_samples = af_samples[:n_samples]
    genuine_samples = genuine_samples[:n_samples]

    print(f"  Using {len(af_samples)} AF, {len(genuine_samples)} genuine samples")

    # Load model
    print("\n[2/6] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    model.eval()

    # Load SAEs
    print("\n[3/6] Loading SAEs...")
    saes = {}
    for layer in TARGET_LAYERS:
        try:
            saes[layer] = ResidPostSAE(SAE_REPO, layer).to(DEVICE, torch.bfloat16)
        except Exception as e:
            print(f"  Failed to load layer {layer}: {e}")

    if not saes:
        print("ERROR: No SAEs loaded")
        return

    # Extract features for all samples
    print("\n[4/6] Extracting features...")

    af_features_all = []
    genuine_features_all = []

    for i, sample in enumerate(af_samples):
        if i % 10 == 0:
            print(f"  AF sample {i+1}/{len(af_samples)}...")
        text = sample.get('text', sample.get('scratchpad', ''))
        feats = extract_all_features(model, tokenizer, text, saes)
        af_features_all.append(feats)

    for i, sample in enumerate(genuine_samples):
        if i % 10 == 0:
            print(f"  Genuine sample {i+1}/{len(genuine_samples)}...")
        text = sample.get('text', sample.get('scratchpad', ''))
        feats = extract_all_features(model, tokenizer, text, saes)
        genuine_features_all.append(feats)

    # Find top differential features per layer
    print("\n[5/6] Finding top features and training probe...")

    # Compute mean activations per class
    top_features_per_layer = {}
    n_top = 50  # Top 50 features per layer

    for layer in saes.keys():
        af_max = np.mean([f[layer]['max'] for f in af_features_all], axis=0)
        gen_max = np.mean([f[layer]['max'] for f in genuine_features_all], axis=0)

        # Features with highest AF-genuine difference
        diff = af_max - gen_max
        top_idx = np.argsort(np.abs(diff))[-n_top:][::-1]
        top_features_per_layer[layer] = top_idx.tolist()

        print(f"  Layer {layer}: top feature diffs = {diff[top_idx[:5]]}")

    # Build feature matrix
    X = []
    y = []

    for af_feats in af_features_all:
        X.append(features_to_vector(af_feats, top_features_per_layer))
        y.append(1)  # AF

    for gen_feats in genuine_features_all:
        X.append(features_to_vector(gen_feats, top_features_per_layer))
        y.append(0)  # Genuine

    X = np.array(X)
    y = np.array(y)

    print(f"  Feature matrix: {X.shape}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train probe with cross-validation
    print("\n  Training logistic regression probe...")

    best_auroc = 0
    best_C = 0.01

    for C in [0.001, 0.01, 0.1, 1.0]:
        probe = LogisticRegression(C=C, max_iter=1000, random_state=42)
        scores = cross_val_score(probe, X_scaled, y, cv=5, scoring='roc_auc')
        mean_auroc = scores.mean()
        print(f"    C={C}: AUROC = {mean_auroc:.4f} (+/- {scores.std():.4f})")

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_C = C

    # Train final probe
    probe = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    probe.fit(X_scaled, y)

    # Evaluate
    y_pred_proba = probe.predict_proba(X_scaled)[:, 1]
    final_auroc = roc_auc_score(y, y_pred_proba)

    print(f"\n  Best C: {best_C}")
    print(f"  Final AUROC: {final_auroc:.4f}")
    print(f"  Coefficients shape: {probe.coef_.shape}")

    # Phase 2: Ablation on probe
    print("\n[6/6] Testing ablation on PROBE...")
    print("=" * 70)

    # Get AF samples and their feature vectors
    af_indices = np.where(y == 1)[0]

    # Baseline predictions
    baseline_preds = probe.predict(X_scaled[af_indices])
    print(f"\n  Baseline: {baseline_preds.sum()}/{len(baseline_preds)} classified as AF")

    # Test ablation of each feature group
    n_features_per_layer = n_top * 3  # max, mean, count per feature
    total_features = len(TARGET_LAYERS) * n_features_per_layer

    flip_results = []

    for layer_idx, layer in enumerate(TARGET_LAYERS):
        print(f"\n  Layer {layer}:")

        start_idx = layer_idx * n_features_per_layer
        end_idx = start_idx + n_features_per_layer

        # Ablate all features from this layer
        X_ablated = X_scaled.copy()
        X_ablated[:, start_idx:end_idx] = 0

        ablated_preds = probe.predict(X_ablated[af_indices])
        flips = (baseline_preds == 1) & (ablated_preds == 0)
        n_flips = flips.sum()

        print(f"    Ablate entire layer: {n_flips}/{len(af_indices)} flips ({100*n_flips/len(af_indices):.1f}%)")

        if n_flips > 0:
            flip_results.append({'layer': layer, 'type': 'full_layer', 'flips': int(n_flips)})

        # Test individual top features
        for feat_offset in range(min(10, n_top)):  # Top 10 features
            feat_start = start_idx + feat_offset * 3
            feat_end = feat_start + 3

            X_ablated = X_scaled.copy()
            X_ablated[:, feat_start:feat_end] = 0

            ablated_preds = probe.predict(X_ablated[af_indices])
            flips = (baseline_preds == 1) & (ablated_preds == 0)
            n_flips = flips.sum()

            if n_flips > 0:
                feat_idx = top_features_per_layer[layer][feat_offset]
                print(f"    Feature {feat_idx}: {n_flips}/{len(af_indices)} flips")
                flip_results.append({'layer': layer, 'feature': feat_idx, 'flips': int(n_flips)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nProbe AUROC: {final_auroc:.4f}")
    print(f"Total flip-causing ablations: {len(flip_results)}")

    if flip_results:
        print("\nFlip-causing features:")
        for r in flip_results[:10]:
            if 'feature' in r:
                print(f"  Layer {r['layer']}, Feature {r['feature']}: {r['flips']} flips")
            else:
                print(f"  Layer {r['layer']} (full): {r['flips']} flips")

    # Save results
    results = {
        'model': MODEL_ID,
        'sae_repo': SAE_REPO,
        'layers': TARGET_LAYERS,
        'n_top_features': n_top,
        'best_C': best_C,
        'cv_auroc': float(best_auroc),
        'final_auroc': float(final_auroc),
        'top_features_per_layer': {str(k): v for k, v in top_features_per_layer.items()},
        'flip_results': flip_results,
        'probe_coef': probe.coef_.tolist(),
        'probe_intercept': probe.intercept_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }

    output_file = "probe_4b_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Save probe model
    probe_file = "probe_4b.pkl"
    with open(probe_file, "wb") as f:
        pickle.dump({'probe': probe, 'scaler': scaler, 'top_features': top_features_per_layer}, f)
    print(f"Probe saved to {probe_file}")


if __name__ == "__main__":
    main()
