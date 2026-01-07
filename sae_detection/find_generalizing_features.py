#!/usr/bin/env python3
"""
Find Cross-Distribution Features
================================
Find features that work on BOTH training data AND gold_106.

If any features generalize, they detect AF intent (not just style).
If none generalize, the style confound is universal.
"""

import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score
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
    """Load gold_106 from HuggingFace."""
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
    """Load training labels."""
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
    """Extract SAE features."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results/sweep_resid_post_20260107_141945")
    parser.add_argument("--layer", type=int, default=40)
    parser.add_argument("--output", default="cross_distribution_features.json")
    args = parser.parse_args()

    print("="*70)
    print("FINDING CROSS-DISTRIBUTION FEATURES")
    print("="*70)

    # Load needles
    print("\n[1/5] Loading needles...")
    needles_path = Path(args.results_dir) / "true_needles_resid_post.json"
    with open(needles_path) as f:
        data = json.load(f)

    all_needles = data["needles"]
    layer_needles = [n for n in all_needles if n["layer"] == args.layer]
    print(f"  {len(layer_needles)} needles in layer {args.layer}")

    # Load training data
    print("\n[2/5] Loading training data...")
    X_train = np.load(Path(args.results_dir) / f"activations_resid_post_layer_{args.layer}.npy")
    y_train = load_training_labels()
    print(f"  Training: {X_train.shape[0]} samples, {y_train.sum()} AF")

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

    sae = ResidPostSAE(SAE_REPO, args.layer).to(DEVICE, torch.bfloat16)
    X_gold = extract_features(model, tokenizer, sae, gold_samples, args.layer)
    print(f"  Extracted: {X_gold.shape}")

    # Analyze each needle
    print("\n[5/5] Analyzing features...")
    print("="*70)

    results = []
    generalizing = []

    for needle in tqdm(layer_needles, desc="Analyzing"):
        feat_idx = needle["feature_idx"]

        # Training AUROC
        train_vals = X_train[:, feat_idx]
        try:
            train_auroc = roc_auc_score(y_train, train_vals)
        except:
            train_auroc = 0.5

        # Gold AUROC
        gold_vals = X_gold[:, feat_idx]
        try:
            gold_auroc = roc_auc_score(y_gold, gold_vals)
        except:
            gold_auroc = 0.5

        # Activation stats
        train_af_mean = train_vals[y_train == 1].mean()
        train_non_af_mean = train_vals[y_train == 0].mean()
        gold_af_mean = gold_vals[y_gold == 1].mean()
        gold_non_af_mean = gold_vals[y_gold == 0].mean()

        result = {
            "feature_idx": int(feat_idx),
            "train_auroc": float(train_auroc),
            "gold_auroc": float(gold_auroc),
            "train_af_mean": float(train_af_mean),
            "train_non_af_mean": float(train_non_af_mean),
            "gold_af_mean": float(gold_af_mean),
            "gold_non_af_mean": float(gold_non_af_mean),
            "generalizes": bool(train_auroc > 0.7 and gold_auroc > 0.6),
        }
        results.append(result)

        if result["generalizes"]:
            generalizing.append(result)

    # Sort by gold AUROC
    results = sorted(results, key=lambda x: x["gold_auroc"], reverse=True)
    generalizing = sorted(generalizing, key=lambda x: x["gold_auroc"], reverse=True)

    # Report
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nTotal needles analyzed: {len(layer_needles)}")
    print(f"Generalizing features (train>0.7 AND gold>0.6): {len(generalizing)}")

    if generalizing:
        print(f"\n✅ FOUND {len(generalizing)} GENERALIZING FEATURES!")
        print("\nTop 10 by gold AUROC:")
        print("-" * 70)
        print(f"{'Feature':<12} {'Train AUROC':<12} {'Gold AUROC':<12} {'Gold AF/NonAF':<15}")
        print("-" * 70)
        for r in generalizing[:10]:
            ratio = r["gold_af_mean"] / (r["gold_non_af_mean"] + 1e-6)
            print(f"F{r['feature_idx']:<10} {r['train_auroc']:<12.3f} {r['gold_auroc']:<12.3f} {ratio:<15.2f}x")
    else:
        print("\n❌ NO GENERALIZING FEATURES FOUND")
        print("\nTop 10 by gold AUROC (none pass threshold):")
        print("-" * 70)
        print(f"{'Feature':<12} {'Train AUROC':<12} {'Gold AUROC':<12}")
        print("-" * 70)
        for r in results[:10]:
            print(f"F{r['feature_idx']:<10} {r['train_auroc']:<12.3f} {r['gold_auroc']:<12.3f}")

    # Distribution of gold AUROCs
    gold_aurocs = [r["gold_auroc"] for r in results]
    print(f"\nGold AUROC distribution:")
    print(f"  Mean: {np.mean(gold_aurocs):.3f}")
    print(f"  Std:  {np.std(gold_aurocs):.3f}")
    print(f"  Min:  {np.min(gold_aurocs):.3f}")
    print(f"  Max:  {np.max(gold_aurocs):.3f}")
    print(f"  >0.6: {sum(1 for a in gold_aurocs if a > 0.6)}")
    print(f"  >0.7: {sum(1 for a in gold_aurocs if a > 0.7)}")

    # Save
    output = {
        "config": {
            "layer": args.layer,
            "n_needles": len(layer_needles),
            "threshold_train": 0.7,
            "threshold_gold": 0.6,
        },
        "summary": {
            "n_generalizing": len(generalizing),
            "gold_auroc_mean": float(np.mean(gold_aurocs)),
            "gold_auroc_max": float(np.max(gold_aurocs)),
        },
        "generalizing_features": generalizing,
        "top_20_by_gold": results[:20],
        "all_results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Saved to {args.output}")


if __name__ == "__main__":
    main()
