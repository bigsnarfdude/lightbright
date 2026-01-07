#!/usr/bin/env python3
"""
Full Feature Sweep with RESID_POST SAE (correct SAE type)
=========================================================
Uses residual stream post-layer SAE, not transcoder.
"""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


CONFIG = {
    "model_name": "google/gemma-3-27b-it",
    "sae_repo": "google/gemma-scope-2-27b-it",
    "sae_type": "resid_post",  # Use residual stream SAE
    "layers": [31, 40, 53],
    "n_features": 16384,
    "device": "cuda",
    "max_length": 2048,
    "dataset_repo": "vincentoh/af-detection-benchmark",
}


class ResidPostSAE:
    """GemmaScope resid_post SAE."""

    def __init__(self, repo_id: str, layer: int):
        folder = f"resid_post/layer_{layer}_width_16k_l0_medium"
        config_path = hf_hub_download(repo_id, f"{folder}/config.json")
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")

        with open(config_path) as f:
            self.config = json.load(f)

        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        # Check if threshold exists (some SAEs use it, some don't)
        self.threshold = weights.get("threshold", None)
        self.w_dec = weights["w_dec"]
        self.b_dec = weights["b_dec"]

        print(f"  Loaded resid_post SAE layer {layer}: {self.w_enc.shape}")

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        if self.threshold is not None:
            self.threshold = self.threshold.to(device, dtype=dtype)
        self.w_dec = self.w_dec.to(device, dtype=dtype)
        self.b_dec = self.b_dec.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        if self.threshold is not None:
            return F.relu(pre_acts - self.threshold)
        else:
            return F.relu(pre_acts)

    @property
    def num_features(self):
        return self.w_enc.shape[1]


def load_dataset(cache_dir="./data"):
    print("📥 Loading dataset...")
    os.makedirs(cache_dir, exist_ok=True)

    files = [
        "aligned.json", "alignment_faking.json", "genuine_compliance.json",
        "hard_negative.json", "compliant_hard_negative.json"
    ]

    all_samples = []
    for fname in files:
        path = hf_hub_download(
            repo_id=CONFIG["dataset_repo"],
            filename=f"classes/{fname}",
            repo_type="dataset",
            local_dir=cache_dir
        )
        with open(path) as f:
            data = json.load(f)

        class_name = fname.replace(".json", "")
        for s in data["samples"]:
            all_samples.append({
                "text": s["text"],
                "class": class_name,
                "is_af": 1 if class_name == "alignment_faking" else 0,
                "is_hard_neg": 1 if class_name in ["hard_negative", "compliant_hard_negative"] else 0,
            })

    print(f"  ✓ Loaded {len(all_samples)} samples")
    return all_samples


def load_model():
    print("🔧 Loading Gemma-3-27B...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  ✓ Model loaded ({torch.cuda.memory_allocated()/1024**3:.1f} GB VRAM)")
    return model, tokenizer


def extract_all_features_for_layer(model, tokenizer, samples, layer, sae):
    print(f"\n📊 Extracting resid_post features for layer {layer}...")

    n_samples = len(samples)
    n_features = sae.num_features
    all_max_activations = np.zeros((n_samples, n_features), dtype=np.float32)

    for idx, sample in enumerate(tqdm(samples, desc=f"Layer {layer}")):
        activations = {}

        def hook(module, input, output):
            # For resid_post, we want the output of the layer (residual stream)
            if isinstance(output, tuple):
                activations["out"] = output[0].detach()
            else:
                activations["out"] = output.detach()

        # Hook the full layer output (residual stream post)
        target = model.model.language_model.layers[layer]
        handle = target.register_forward_hook(hook)

        inputs = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"]
        ).to(CONFIG["device"])

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        act = activations["out"].squeeze(0).to(torch.bfloat16)

        with torch.no_grad():
            features = sae.encode(act)

        max_per_feature = features.max(dim=0).values.float().cpu().numpy()
        all_max_activations[idx] = max_per_feature

        if idx % 100 == 0:
            torch.cuda.empty_cache()

    return all_max_activations


def analyze_all_features(activations, samples, output_dir, layer):
    print(f"\n🔬 Analyzing {activations.shape[1]} features...")

    y_af = np.array([s["is_af"] for s in samples])
    y_hard_neg = np.array([s["is_hard_neg"] for s in samples])

    af_mask = y_af == 1
    aligned_mask = y_af == 0
    hard_neg_mask = y_hard_neg == 1

    results = []

    for feat_idx in tqdm(range(activations.shape[1]), desc="Computing stats"):
        feat_vals = activations[:, feat_idx]

        if feat_vals.max() == 0:
            continue

        af_mean = feat_vals[af_mask].mean()
        aligned_mean = feat_vals[aligned_mask].mean()
        hard_neg_mean = feat_vals[hard_neg_mask].mean()

        ratio_vs_aligned = af_mean / (aligned_mean + 1e-6)
        ratio_vs_hard_neg = af_mean / (hard_neg_mean + 1e-6)

        try:
            auroc_af = roc_auc_score(y_af, feat_vals)
        except:
            auroc_af = 0.5

        af_hn_mask = af_mask | hard_neg_mask
        if af_hn_mask.sum() > 10:
            try:
                auroc_af_hn = roc_auc_score(y_af[af_hn_mask], feat_vals[af_hn_mask])
            except:
                auroc_af_hn = 0.5
        else:
            auroc_af_hn = 0.5

        if hard_neg_mask.sum() > 5 and af_mask.sum() > 5:
            t_stat, p_val = stats.ttest_ind(feat_vals[af_mask], feat_vals[hard_neg_mask])
        else:
            t_stat, p_val = 0, 1

        results.append({
            "feature_idx": int(feat_idx),
            "af_mean": float(af_mean),
            "aligned_mean": float(aligned_mean),
            "hard_neg_mean": float(hard_neg_mean),
            "ratio_vs_aligned": float(ratio_vs_aligned),
            "ratio_vs_hard_neg": float(ratio_vs_hard_neg),
            "auroc_af_vs_all": float(auroc_af),
            "auroc_af_vs_hard_neg": float(auroc_af_hn),
            "ttest_pval": float(p_val),
        })

    results_by_auroc = sorted(results, key=lambda x: x["auroc_af_vs_all"], reverse=True)
    results_by_auroc_hn = sorted(results, key=lambda x: x["auroc_af_vs_hard_neg"], reverse=True)

    summary = {
        "layer": layer,
        "sae_type": "resid_post",
        "total_features": activations.shape[1],
        "active_features": len(results),
        "top_20_by_auroc_overall": results_by_auroc[:20],
        "top_20_by_auroc_vs_hard_neg": results_by_auroc_hn[:20],
        "all_features": results,
    }

    print(f"\n📈 Layer {layer} (resid_post) Results:")
    print(f"   Active features: {len(results)}/{activations.shape[1]}")

    print(f"\n   Top 5 by AUROC (AF vs ALL):")
    for f in results_by_auroc[:5]:
        print(f"     F{f['feature_idx']}: AUROC={f['auroc_af_vs_all']:.3f}")

    print(f"\n   Top 5 by AUROC (AF vs HARD NEGATIVES):")
    for f in results_by_auroc_hn[:5]:
        print(f"     F{f['feature_idx']}: AUROC={f['auroc_af_vs_hard_neg']:.3f}")

    out_path = os.path.join(output_dir, f"sweep_resid_post_layer_{layer}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   💾 Saved to {out_path}")

    return summary


def find_needles(all_layer_results, output_dir):
    print("\n" + "="*70)
    print("🎯 FINDING TRUE NEEDLES (resid_post SAE)")
    print("="*70)

    needles = []

    for layer, results in all_layer_results.items():
        for feat in results["all_features"]:
            if (feat["auroc_af_vs_all"] > 0.7 and
                feat["auroc_af_vs_hard_neg"] > 0.65 and
                feat["ttest_pval"] < 0.01):
                needles.append({"layer": layer, **feat})

    needles = sorted(needles, key=lambda x: x["auroc_af_vs_hard_neg"], reverse=True)

    print(f"\n Found {len(needles)} true needles")

    if needles:
        print(f"\n Top 10 TRUE NEEDLES:")
        print("-"*70)
        for n in needles[:10]:
            print(f"  L{n['layer']}_F{n['feature_idx']:5d}: "
                  f"AUROC(all)={n['auroc_af_vs_all']:.3f}, "
                  f"AUROC(hn)={n['auroc_af_vs_hard_neg']:.3f}")
    else:
        print("\n ⚠️ No features passed both thresholds!")

    out_path = os.path.join(output_dir, "true_needles_resid_post.json")
    with open(out_path, "w") as f:
        json.dump({
            "sae_type": "resid_post",
            "total_needles": len(needles),
            "criteria": {
                "auroc_af_vs_all": "> 0.7",
                "auroc_af_vs_hard_neg": "> 0.65",
                "ttest_pval": "< 0.01"
            },
            "needles": needles
        }, f, indent=2)

    print(f"\n💾 Saved to {out_path}")
    return needles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, help="Single layer")
    parser.add_argument("--all-layers", action="store_true", help="All layers")
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"sweep_resid_post_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("🔬 FULL FEATURE SWEEP - RESID_POST SAE")
    print("="*70)

    samples = load_dataset()
    model, tokenizer = load_model()

    if args.layer:
        layers = [args.layer]
    elif args.all_layers:
        layers = CONFIG["layers"]
    else:
        layers = [40]

    all_results = {}

    for layer in layers:
        print(f"\n{'='*70}")
        print(f"LAYER {layer} (resid_post)")
        print("="*70)

        sae = ResidPostSAE(CONFIG["sae_repo"], layer).to(CONFIG["device"], torch.bfloat16)
        activations = extract_all_features_for_layer(model, tokenizer, samples, layer, sae)
        np.save(os.path.join(output_dir, f"activations_resid_post_layer_{layer}.npy"), activations)
        results = analyze_all_features(activations, samples, output_dir, layer)
        all_results[layer] = results

        del sae
        torch.cuda.empty_cache()

    if len(all_results) > 0:
        find_needles(all_results, output_dir)

    print("\n" + "="*70)
    print("✅ SWEEP COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
