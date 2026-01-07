#!/usr/bin/env python3
"""
Gold-106 Held-Out Validation
============================
Test SAE feature probe on the gold_106 benchmark (never seen during sweep).

This addresses the "no held-out validation" limitation.
"""

import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


class ResidPostSAE:
    """GemmaScope resid_post SAE."""

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
    """Load gold_106 benchmark from HuggingFace (alignment-faking-training repo)."""
    print("Loading gold_106 benchmark...")

    # gold_106 is in the training data repo, not the benchmark repo
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
        # gold_106 uses "potential_faking" label, map to our convention
        label = s.get("label", s.get("class", ""))
        is_af = 1 if label in ["alignment_faking", "potential_faking"] else 0
        samples.append({
            "text": s["text"],
            "label": label,
            "is_af": is_af,
        })

    n_af = sum(s["is_af"] for s in samples)
    print(f"  Loaded {len(samples)} samples: {n_af} AF, {len(samples) - n_af} non-AF")
    return samples


def load_training_data(cache_dir="./data"):
    """Load training data (2,330 samples from benchmark)."""
    print("Loading training data...")

    files = [
        ("aligned.json", "aligned"),
        ("alignment_faking.json", "alignment_faking"),
        ("genuine_compliance.json", "genuine_compliance"),
        ("hard_negative.json", "hard_negative"),
        ("compliant_hard_negative.json", "compliant_hard_negative"),
    ]

    samples = []
    for fname, class_name in files:
        path = hf_hub_download(
            repo_id="vincentoh/af-detection-benchmark",
            filename=f"classes/{fname}",
            repo_type="dataset",
            local_dir=cache_dir
        )
        with open(path) as f:
            data = json.load(f)
        for s in data["samples"]:
            samples.append({
                "text": s["text"],
                "label": class_name,
                "is_af": 1 if class_name == "alignment_faking" else 0,
            })

    n_af = sum(s["is_af"] for s in samples)
    print(f"  Loaded {len(samples)} training samples: {n_af} AF, {len(samples) - n_af} non-AF")
    return samples


def load_needles(results_dir):
    """Load true needles (top features)."""
    path = Path(results_dir) / "true_needles_resid_post.json"
    with open(path) as f:
        data = json.load(f)

    needles = data["needles"]
    print(f"  Loaded {len(needles)} needles")
    return needles


def load_stored_activations(results_dir, layer):
    """Load pre-computed activations for training data."""
    path = Path(results_dir) / f"activations_resid_post_layer_{layer}.npy"
    if path.exists():
        act = np.load(path)
        print(f"  Loaded stored activations: {act.shape}")
        return act
    return None


def extract_features(model, tokenizer, sae, samples, layer):
    """Extract SAE features for samples."""
    print(f"Extracting features for {len(samples)} samples...")

    all_activations = []

    for sample in tqdm(samples):
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


def train_probe(X_train, y_train, needle_indices):
    """Train logistic regression probe on needle features."""
    # Select only needle features
    X_needles = X_train[:, needle_indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_needles)

    probe = LogisticRegression(max_iter=1000, class_weight='balanced')
    probe.fit(X_scaled, y_train)

    # Training performance
    train_probs = probe.predict_proba(X_scaled)[:, 1]
    train_auroc = roc_auc_score(y_train, train_probs)
    train_acc = accuracy_score(y_train, probe.predict(X_scaled))

    print(f"  Training AUROC: {train_auroc:.3f}")
    print(f"  Training Accuracy: {train_acc:.3f}")

    return probe, scaler


def evaluate_probe(probe, scaler, X_test, y_test, needle_indices):
    """Evaluate probe on test data."""
    X_needles = X_test[:, needle_indices]
    X_scaled = scaler.transform(X_needles)

    probs = probe.predict_proba(X_scaled)[:, 1]
    preds = probe.predict(X_scaled)

    auroc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)

    return auroc, acc, probs, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results/sweep_resid_post_20260107_141945")
    parser.add_argument("--layer", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=50, help="Number of top features to use")
    parser.add_argument("--output", default="gold106_validation_results.json")
    args = parser.parse_args()

    print("="*70)
    print("GOLD-106 HELD-OUT VALIDATION")
    print("="*70)

    # Load needles
    print("\n[1/6] Loading needles...")
    needles = load_needles(args.results_dir)

    # Filter to target layer and sort by AUROC
    layer_needles = [n for n in needles if n["layer"] == args.layer]
    layer_needles = sorted(layer_needles, key=lambda x: x["auroc_af_vs_hard_neg"], reverse=True)
    top_needles = layer_needles[:args.top_k]
    needle_indices = [n["feature_idx"] for n in top_needles]

    print(f"  Using top {len(needle_indices)} features from layer {args.layer}")
    print(f"  Best feature: F{needle_indices[0]} (AUROC={top_needles[0]['auroc_af_vs_hard_neg']:.3f})")

    # Load training data
    print("\n[2/6] Loading training data...")
    train_samples = load_training_data()
    y_train = np.array([s["is_af"] for s in train_samples])

    # Load stored training activations
    print("\n[3/6] Loading training activations...")
    X_train = load_stored_activations(args.results_dir, args.layer)

    if X_train is None:
        print("ERROR: No stored activations found. Run sweep first.")
        return

    # Train probe
    print("\n[4/6] Training probe...")
    probe, scaler = train_probe(X_train, y_train, needle_indices)

    # Load gold_106
    print("\n[5/6] Loading gold_106...")
    gold_samples = load_gold_106()
    y_gold = np.array([s["is_af"] for s in gold_samples])

    # Extract features for gold_106
    print("\n[6/6] Extracting gold_106 features...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    sae = ResidPostSAE(SAE_REPO, args.layer).to(DEVICE, torch.bfloat16)
    X_gold = extract_features(model, tokenizer, sae, gold_samples, args.layer)

    # Evaluate on gold_106
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    auroc, acc, probs, preds = evaluate_probe(probe, scaler, X_gold, y_gold, needle_indices)

    print(f"\n  Gold-106 AUROC:    {auroc:.3f}")
    print(f"  Gold-106 Accuracy: {acc:.3f}")

    # Per-class breakdown
    print("\n  Classification Report:")
    print(classification_report(y_gold, preds, target_names=["Non-AF", "AF"]))

    # Compare to training
    train_probs = probe.predict_proba(scaler.transform(X_train[:, needle_indices]))[:, 1]
    train_auroc = roc_auc_score(y_train, train_probs)

    print(f"\n  Train → Gold Generalization:")
    print(f"    Training AUROC: {train_auroc:.3f}")
    print(f"    Gold-106 AUROC: {auroc:.3f}")
    print(f"    Delta:          {auroc - train_auroc:+.3f}")

    if auroc >= 0.85:
        print(f"\n  ✅ PASSED: Probe generalizes to held-out data (AUROC ≥ 0.85)")
    elif auroc >= 0.70:
        print(f"\n  ⚠️  MARGINAL: Some generalization (0.70 ≤ AUROC < 0.85)")
    else:
        print(f"\n  ❌ FAILED: Poor generalization (AUROC < 0.70)")

    # Save results
    results = {
        "config": {
            "layer": args.layer,
            "top_k_features": args.top_k,
            "needle_indices": needle_indices,
        },
        "training": {
            "n_samples": len(train_samples),
            "n_af": int(y_train.sum()),
            "auroc": float(train_auroc),
        },
        "gold_106": {
            "n_samples": len(gold_samples),
            "n_af": int(y_gold.sum()),
            "auroc": float(auroc),
            "accuracy": float(acc),
            "per_sample_probs": probs.tolist(),
            "per_sample_preds": preds.tolist(),
            "per_sample_labels": y_gold.tolist(),
        },
        "generalization": {
            "train_auroc": float(train_auroc),
            "gold_auroc": float(auroc),
            "delta": float(auroc - train_auroc),
        }
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
