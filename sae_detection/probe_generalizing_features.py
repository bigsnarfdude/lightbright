#!/usr/bin/env python3
"""
Train probe on 51 generalizing features, evaluate on gold_106.
"""

import json
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
    print("PROBE ON 51 GENERALIZING FEATURES")
    print("="*70)

    # Load generalizing feature indices
    print("\n[1/5] Loading generalizing features...")
    with open("cross_distribution_features.json") as f:
        data = json.load(f)

    generalizing = data["generalizing_features"]
    feat_indices = [f["feature_idx"] for f in generalizing]
    print(f"  {len(feat_indices)} features: {feat_indices[:5]}...")

    # Load training data
    print("\n[2/5] Loading training data...")
    X_train_full = np.load(results_dir / f"activations_resid_post_layer_{layer}.npy")
    y_train = load_training_labels()

    # Select only generalizing features
    X_train = X_train_full[:, feat_indices]
    print(f"  Training shape: {X_train.shape}")

    # Load gold_106
    print("\n[3/5] Loading gold_106...")
    gold_samples = load_gold_106()
    y_gold = np.array([s["is_af"] for s in gold_samples])
    print(f"  Gold_106: {len(gold_samples)} samples, {y_gold.sum()} AF")

    # Extract gold_106 features
    print("\n[4/5] Extracting gold_106 features...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    sae = ResidPostSAE(SAE_REPO, layer).to(DEVICE, torch.bfloat16)
    X_gold_full = extract_features(model, tokenizer, sae, gold_samples, layer)

    # Select only generalizing features
    X_gold = X_gold_full[:, feat_indices]
    print(f"  Gold shape: {X_gold.shape}")

    # Train probe
    print("\n[5/5] Training and evaluating...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_gold_scaled = scaler.transform(X_gold)

    probe = LogisticRegression(max_iter=1000, class_weight='balanced')
    probe.fit(X_train_scaled, y_train)

    # Evaluate
    train_probs = probe.predict_proba(X_train_scaled)[:, 1]
    gold_probs = probe.predict_proba(X_gold_scaled)[:, 1]

    train_auroc = roc_auc_score(y_train, train_probs)
    gold_auroc = roc_auc_score(y_gold, gold_probs)
    gold_acc = accuracy_score(y_gold, probe.predict(X_gold_scaled))

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n  Training AUROC:  {train_auroc:.3f}")
    print(f"  Gold-106 AUROC:  {gold_auroc:.3f}")
    print(f"  Gold-106 Accuracy: {gold_acc:.3f}")

    print("\n  Classification Report (Gold-106):")
    print(classification_report(y_gold, probe.predict(X_gold_scaled), target_names=["Non-AF", "AF"]))

    # Compare to original
    print("\n  Comparison:")
    print(f"    Original probe (50 features): 0.470 AUROC")
    print(f"    New probe (51 gen features):  {gold_auroc:.3f} AUROC")
    print(f"    Improvement: {gold_auroc - 0.470:+.3f}")

    if gold_auroc >= 0.70:
        print(f"\n  ✅ SUCCESS: Probe generalizes (AUROC ≥ 0.70)")
    else:
        print(f"\n  ⚠️  PARTIAL: Better but still below 0.70")

    # Save results
    output = {
        "config": {
            "n_features": len(feat_indices),
            "feature_indices": feat_indices,
        },
        "training": {
            "auroc": float(train_auroc),
        },
        "gold_106": {
            "auroc": float(gold_auroc),
            "accuracy": float(gold_acc),
            "n_samples": len(gold_samples),
            "n_af": int(y_gold.sum()),
        },
        "comparison": {
            "original_auroc": 0.470,
            "new_auroc": float(gold_auroc),
            "improvement": float(gold_auroc - 0.470),
        }
    }

    with open("generalizing_probe_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Saved to generalizing_probe_results.json")


if __name__ == "__main__":
    main()
