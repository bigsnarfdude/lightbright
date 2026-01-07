#!/usr/bin/env python3
"""
Characterize top AF-detecting features by finding maximally activating text spans.
Updated to use resid_post SAE with top features from 2026-01-07 sweep.
"""

import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda"
MODEL_NAME = "google/gemma-3-27b-it"
SAE_REPO = "google/gemma-scope-2-27b-it"

# Top features to characterize (from 2026-01-07 resid_post sweep)
TOP_FEATURES = [
    {"layer": 40, "feature_idx": 15822, "name": "L40_F15822", "auroc_hn": 0.968},
    {"layer": 40, "feature_idx": 11984, "name": "L40_F11984", "auroc_hn": 0.945},
    {"layer": 31, "feature_idx": 1529, "name": "L31_F1529", "auroc_hn": 0.943},
    {"layer": 53, "feature_idx": 924, "name": "L53_F924", "auroc_hn": 0.941},
    {"layer": 53, "feature_idx": 10990, "name": "L53_F10990", "auroc_hn": 0.940},
    {"layer": 31, "feature_idx": 9913, "name": "L31_F9913", "auroc_hn": 0.937},
    {"layer": 31, "feature_idx": 14114, "name": "L31_F14114", "auroc_hn": 0.937},
    {"layer": 31, "feature_idx": 1366, "name": "L31_F1366", "auroc_hn": 0.931},
]


class ResidPostSAE:
    def __init__(self, repo_id: str, layer: int):
        folder = f"resid_post/layer_{layer}_width_16k_l0_medium"
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"].to(DEVICE, torch.bfloat16)
        self.b_enc = weights["b_enc"].to(DEVICE, torch.bfloat16)
        self.threshold = weights.get("threshold")
        if self.threshold is not None:
            self.threshold = self.threshold.to(DEVICE, torch.bfloat16)

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        if self.threshold is not None:
            return F.relu(pre_acts - self.threshold)
        return F.relu(pre_acts)


def load_samples():
    """Load a mix of AF and non-AF samples."""
    cache_dir = "./data"
    samples = []

    for fname in ["alignment_faking.json", "hard_negative.json", "compliant_hard_negative.json"]:
        path = hf_hub_download(
            repo_id="vincentoh/af-detection-benchmark",
            filename=f"classes/{fname}",
            repo_type="dataset",
            local_dir=cache_dir
        )
        with open(path) as f:
            data = json.load(f)

        class_name = fname.replace(".json", "")
        for s in data["samples"][:50]:  # 50 per class
            samples.append({"text": s["text"], "class": class_name})

    return samples


def get_token_activations(model, tokenizer, sae, text, layer, feature_idx):
    """Get per-token activations for a specific feature."""
    activations = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            activations["out"] = output[0].detach()
        else:
            activations["out"] = output.detach()

    target = model.model.language_model.layers[layer]
    handle = target.register_forward_hook(hook)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    act = activations["out"].squeeze(0).to(torch.bfloat16)
    features = sae.encode(act)
    feature_acts = features[:, feature_idx].float().cpu().numpy()

    return tokens, feature_acts


def find_max_activating_spans(tokens, activations, tokenizer, window=10, top_k=3):
    """Find text spans with highest activation."""
    spans = []

    # Find peaks
    for i in range(len(activations)):
        if activations[i] > 0:
            start = max(0, i - window // 2)
            end = min(len(tokens), i + window // 2)
            span_tokens = tokens[start:end]
            span_text = tokenizer.convert_tokens_to_string(span_tokens)
            spans.append({
                "position": i,
                "activation": float(activations[i]),
                "span": span_text.strip(),
                "peak_token": tokens[i]
            })

    # Sort by activation and deduplicate
    spans = sorted(spans, key=lambda x: x["activation"], reverse=True)

    # Deduplicate overlapping spans
    unique_spans = []
    used_positions = set()
    for s in spans:
        if s["position"] not in used_positions:
            unique_spans.append(s)
            for p in range(s["position"] - window//2, s["position"] + window//2):
                used_positions.add(p)
        if len(unique_spans) >= top_k:
            break

    return unique_spans


def main():
    print("="*70)
    print("CHARACTERIZING TOP AF-DETECTING FEATURES (resid_post SAE)")
    print("="*70)

    # Load model
    print("\n📦 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Load SAEs
    print("\n📦 Loading SAEs...")
    saes = {}
    for layer in [31, 40, 53]:
        saes[layer] = ResidPostSAE(SAE_REPO, layer)

    # Load samples
    print("\n📥 Loading samples...")
    samples = load_samples()
    print(f"  Loaded {len(samples)} samples")

    # Characterize each feature
    results = {}

    for feat_info in TOP_FEATURES:
        layer = feat_info["layer"]
        feat_idx = feat_info["feature_idx"]
        name = feat_info["name"]

        print(f"\n{'='*70}")
        print(f"FEATURE: {name} (AUROC={feat_info['auroc_hn']:.3f})")
        print("="*70)

        sae = saes[layer]

        # Collect activating spans across samples
        all_spans = []
        class_activations = {"alignment_faking": [], "hard_negative": [], "compliant_hard_negative": []}

        for sample in tqdm(samples, desc=f"Processing {name}"):
            tokens, acts = get_token_activations(model, tokenizer, sae, sample["text"], layer, feat_idx)
            max_act = acts.max()
            class_activations[sample["class"]].append(max_act)

            if max_act > 0:
                spans = find_max_activating_spans(tokens, acts, tokenizer, window=15, top_k=2)
                for s in spans:
                    s["sample_class"] = sample["class"]
                    all_spans.append(s)

        # Sort all spans by activation
        all_spans = sorted(all_spans, key=lambda x: x["activation"], reverse=True)

        # Print class-level stats
        print(f"\n📊 Activation by class:")
        for cls, acts in class_activations.items():
            mean_act = np.mean(acts) if acts else 0
            max_act = np.max(acts) if acts else 0
            nonzero = sum(1 for a in acts if a > 0)
            print(f"   {cls:25s}: mean={mean_act:8.1f}, max={max_act:8.1f}, nonzero={nonzero}/{len(acts)}")

        # Print top activating spans
        print(f"\n🔥 Top activating text spans:")
        seen_spans = set()
        count = 0
        for s in all_spans:
            # Skip duplicates
            span_key = s["span"][:50]
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            print(f"\n   [{s['sample_class'][:10]}] activation={s['activation']:.1f}")
            print(f"   Peak token: '{s['peak_token']}'")
            print(f"   Context: \"{s['span'][:200]}...\"" if len(s['span']) > 200 else f"   Context: \"{s['span']}\"")

            count += 1
            if count >= 5:
                break

        # Store results
        results[name] = {
            "layer": layer,
            "feature_idx": feat_idx,
            "auroc_hn": feat_info["auroc_hn"],
            "class_stats": {
                cls: {"mean": float(np.mean(acts)), "max": float(np.max(acts)) if acts else 0}
                for cls, acts in class_activations.items()
            },
            "top_spans": all_spans[:20]
        }

    # Save results
    with open("feature_characterization.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved to feature_characterization.json")

    print("\n" + "="*70)
    print("CHARACTERIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
