#!/usr/bin/env python3
"""
Probe-Based Detection with Ablation

Use the LINEAR PROBE as the detector (not fine-tuned Gemma).
Test ablation on feature inputs to the probe.

This should show high flip rates because:
1. Probe is a simple linear classifier
2. Ablating features directly affects probe output
3. No SAE reconstruction error in the inference path
"""

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn.functional as F
import os
import pickle

MODEL_ID = "google/gemma-3-4b-it"
SAE_REPO = "google/gemma-scope-2-4b-it"
DEVICE = "cuda"

TARGET_LAYERS = [9, 17, 22, 29]


class ResidPostSAE:
    def __init__(self, repo_id: str, layer: int):
        path = f"resid_post/layer_{layer}_width_16k_l0_medium"
        config_path = hf_hub_download(repo_id, f"{path}/config.json")
        weights_path = hf_hub_download(repo_id, f"{path}/params.safetensors")

        with open(config_path) as f:
            self.config = json.load(f)
        weights = load_file(weights_path)

        self.W_enc = weights.get("W_enc", weights.get("w_enc"))
        self.b_enc = weights.get("b_enc", torch.zeros(self.W_enc.shape[1]))
        self.threshold = weights.get("threshold", torch.zeros(self.W_enc.shape[1]))
        self.n_features = self.W_enc.shape[1]

    def to(self, device, dtype=None):
        self.W_enc = self.W_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.W_enc + self.b_enc
        return F.relu(pre_acts - self.threshold)


def extract_sae_features(model, tokenizer, text, saes):
    """Extract SAE features from text."""
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

    features = {}
    for layer in saes.keys():
        acts = activations[layer].squeeze(0).to(saes[layer].W_enc.dtype)
        feats = saes[layer].encode(acts)
        features[layer] = {
            'max': feats.max(dim=0).values.float().cpu().numpy(),
            'mean': feats.mean(dim=0).float().cpu().numpy(),
            'count': (feats > 0).sum(dim=0).float().cpu().numpy(),
        }

    return features


def features_to_vector(features, top_features_per_layer):
    """Convert features dict to flat vector for probe."""
    vector = []
    for layer in [9, 17, 22, 29]:
        # Handle both int and string keys
        feat_indices = top_features_per_layer.get(layer, top_features_per_layer.get(str(layer)))
        for idx in feat_indices:
            vector.extend([
                features[layer]['max'][idx],
                features[layer]['mean'][idx],
                features[layer]['count'][idx],
            ])
    return np.array(vector)


def main():
    print("=" * 70)
    print("PROBE-BASED DETECTOR WITH ABLATION")
    print("Using linear probe (not fine-tuned Gemma) as classifier")
    print("=" * 70)

    # Load probe
    print("\n[1/4] Loading probe...")
    with open("probe_4b.pkl", "rb") as f:
        probe_data = pickle.load(f)

    probe = probe_data['probe']
    scaler = probe_data['scaler']
    top_features = probe_data['top_features']

    print(f"  Probe loaded: {probe.coef_.shape[1]} features")

    # Load probe results for analysis
    with open("probe_4b_results.json") as f:
        results = json.load(f)
    coef = np.array(results['probe_coef']).flatten()

    # Load test data
    print("\n[2/4] Loading test data...")
    train_path = os.path.expanduser("~/mindreader/data/hf_training_data.json")

    with open(train_path) as f:
        data = json.load(f)
    samples = data.get('samples', data)

    # Get test samples (different from training)
    af_samples = [s for s in samples if s['label'] == 'potential_faking'][100:110]  # Skip first 100 used for training
    genuine_samples = [s for s in samples if s['label'] == 'aligned'][100:110]

    print(f"  Test: {len(af_samples)} AF, {len(genuine_samples)} genuine")

    # Load model for feature extraction
    print("\n[3/4] Loading model and SAEs...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    model.eval()

    saes = {}
    for layer in TARGET_LAYERS:
        saes[layer] = ResidPostSAE(SAE_REPO, layer).to(DEVICE, torch.bfloat16)
        print(f"  Loaded SAE layer {layer}")

    # Extract features for test samples
    print("\n[4/4] Running probe-based detection and ablation...")
    print("=" * 70)

    results = {
        'baseline': {'af_correct': 0, 'genuine_correct': 0},
        'ablation_flips': [],
    }

    # First: baseline accuracy
    print("\n--- Baseline Probe Accuracy ---")

    af_features_list = []
    for i, sample in enumerate(af_samples):
        text = sample.get('text', sample.get('scratchpad', ''))
        feats = extract_sae_features(model, tokenizer, text, saes)
        vec = features_to_vector(feats, top_features)
        af_features_list.append(vec)

        vec_scaled = scaler.transform(vec.reshape(1, -1))
        pred = probe.predict(vec_scaled)[0]
        prob = probe.predict_proba(vec_scaled)[0, 1]

        if pred == 1:
            results['baseline']['af_correct'] += 1
        print(f"  AF {i+1}: pred={pred}, prob={prob:.4f}")

    gen_features_list = []
    for i, sample in enumerate(genuine_samples):
        text = sample.get('text', sample.get('scratchpad', ''))
        feats = extract_sae_features(model, tokenizer, text, saes)
        vec = features_to_vector(feats, top_features)
        gen_features_list.append(vec)

        vec_scaled = scaler.transform(vec.reshape(1, -1))
        pred = probe.predict(vec_scaled)[0]
        prob = probe.predict_proba(vec_scaled)[0, 1]

        if pred == 0:
            results['baseline']['genuine_correct'] += 1
        print(f"  Genuine {i+1}: pred={pred}, prob={prob:.4f}")

    print(f"\nBaseline: AF {results['baseline']['af_correct']}/{len(af_samples)}, "
          f"Genuine {results['baseline']['genuine_correct']}/{len(genuine_samples)}")

    # Second: ablation on probe
    print("\n--- Probe Ablation Test ---")

    # Get top coefficient indices
    top_coef_idx = np.argsort(np.abs(coef))[-50:][::-1]

    for i, vec in enumerate(af_features_list):
        vec_scaled = scaler.transform(vec.reshape(1, -1))
        baseline_pred = probe.predict(vec_scaled)[0]
        baseline_prob = probe.predict_proba(vec_scaled)[0, 1]

        if baseline_pred != 1:
            continue  # Skip if not classified as AF

        print(f"\n  AF sample {i+1} (baseline prob={baseline_prob:.4f}):")

        # Test ablating top features
        for n_ablate in [10, 20, 50]:
            vec_ablated = vec_scaled.copy()

            # Zero out top N coefficient features
            for idx in top_coef_idx[:n_ablate]:
                vec_ablated[0, idx] = 0

            pred = probe.predict(vec_ablated)[0]
            prob = probe.predict_proba(vec_ablated)[0, 1]

            flip = "FLIP!" if pred == 0 else ""
            print(f"    Ablate top {n_ablate}: prob={prob:.4f} {flip}")

            if pred == 0:
                results['ablation_flips'].append({
                    'sample': i+1,
                    'n_features': n_ablate,
                    'baseline_prob': float(baseline_prob),
                    'ablated_prob': float(prob)
                })

        # Test flipping (anti-steering)
        vec_flipped = vec_scaled.copy()
        for idx in top_coef_idx[:20]:
            if coef[idx] > 0:
                vec_flipped[0, idx] = -2
            else:
                vec_flipped[0, idx] = 2

        pred = probe.predict(vec_flipped)[0]
        prob = probe.predict_proba(vec_flipped)[0, 1]

        flip = "FLIP!" if pred == 0 else ""
        print(f"    Flip top 20: prob={prob:.4f} {flip}")

        if pred == 0:
            results['ablation_flips'].append({
                'sample': i+1,
                'type': 'flip',
                'baseline_prob': float(baseline_prob),
                'flipped_prob': float(prob)
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nBaseline accuracy: AF {results['baseline']['af_correct']}/{len(af_samples)}, "
          f"Genuine {results['baseline']['genuine_correct']}/{len(genuine_samples)}")

    n_flips = len([r for r in results['ablation_flips'] if r.get('type') != 'flip'])
    n_flip_flips = len([r for r in results['ablation_flips'] if r.get('type') == 'flip'])

    print(f"\nAblation flips (zeroing): {n_flips}")
    print(f"Anti-steering flips (flipping): {n_flip_flips}")

    total_af_tested = results['baseline']['af_correct']
    if total_af_tested > 0:
        ablation_rate = n_flips / total_af_tested * 100
        flip_rate = n_flip_flips / total_af_tested * 100
        print(f"\nAblation flip rate: {ablation_rate:.1f}%")
        print(f"Anti-steering flip rate: {flip_rate:.1f}%")

    # Save
    with open("probe_detector_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to probe_detector_ablation_results.json")


if __name__ == "__main__":
    main()
