#!/usr/bin/env python3
"""
Characterize top AF-specific features by finding max-activating text spans.
Shows what concepts each feature represents.
"""

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch.nn.functional as F
import os

MODEL_ID = "google/gemma-2-27b-it"  # Must match SAE (gemma-scope-2)
SAE_REPO = "google/gemma-scope-2-27b-it"
DEVICE = "cuda"

# Top features to characterize
FEATURES = [
    (40, 12574, "656x_ratio"),
    (40, 8921, "331x_ratio"),
    (53, 15529, "infinite_ratio"),
    (53, 8003, "infinite_ratio"),
    (53, 4824, "58x_ratio"),
    (31, 15111, "84x_ratio"),
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


def get_token_activations(model, tokenizer, text, layer, transcoder):
    """Get per-token feature activations."""
    activations = {}
    def hook(module, input, output):
        activations['out'] = output.detach()

    try:
        target_layer = model.model.language_model.layers[layer]
    except AttributeError:
        target_layer = model.base_model.model.model.language_model.layers[layer]

    handle = target_layer.pre_feedforward_layernorm.register_forward_hook(hook)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        model(**inputs)

    handle.remove()

    acts = activations['out'].squeeze(0).to(transcoder.w_enc.dtype)
    feats = transcoder.encode(acts)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return feats, tokens


def find_max_spans(model, tokenizer, transcoder, layer, texts, labels, feature_idx, top_k=15, context=5):
    """Find text spans that maximally activate a feature."""
    all_activations = []

    for text, label in zip(texts, labels):
        feats, tokens = get_token_activations(model, tokenizer, text, layer, transcoder)
        feature_acts = feats[:, feature_idx].float().cpu().numpy()

        for pos, act_val in enumerate(feature_acts):
            if act_val > 0:
                start = max(0, pos - context)
                end = min(len(tokens), pos + context + 1)
                span_tokens = tokens[start:end]
                span_text = tokenizer.convert_tokens_to_string(span_tokens)

                all_activations.append({
                    'activation': float(act_val),
                    'span': span_text,
                    'center_token': tokens[pos],
                    'label': label,
                    'position': pos
                })

    # Sort by activation and get top_k
    all_activations.sort(key=lambda x: x['activation'], reverse=True)
    return all_activations[:top_k]


def main():
    print("=" * 70)
    print("FEATURE CHARACTERIZATION: What do AF-specific features detect?")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading samples...")
    data_paths = ["../../data/gold_106.json", "../data/gold_106.json"]
    for path in data_paths:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            break

    samples = data.get('samples', data)[:50]  # Use first 50 samples
    texts = [s.get('text', s.get('reasoning', '')) for s in samples]
    labels = [s.get('label', 'unknown') for s in samples]
    print(f"Loaded {len(texts)} samples")

    # Load model
    print("\n[2/4] Loading model (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Load transcoders
    print("\n[3/4] Loading transcoders...")
    transcoders = {}
    for layer in set(f[0] for f in FEATURES):
        print(f"  Loading layer {layer}...")
        transcoders[layer] = Transcoder(SAE_REPO, layer).to(DEVICE, torch.bfloat16)

    # Characterize each feature
    print("\n[4/4] Finding max-activating spans...")
    results = {}

    for layer, feature_idx, name in FEATURES:
        print(f"\n--- Layer {layer}, Feature {feature_idx} ({name}) ---")

        spans = find_max_spans(
            model, tokenizer, transcoders[layer], layer,
            texts, labels, feature_idx
        )

        results[f"layer{layer}_feature{feature_idx}"] = {
            'name': name,
            'top_spans': spans
        }

        print(f"\nTop activating spans:")
        for i, span in enumerate(spans[:5]):
            print(f"  {i+1}. [{span['activation']:.0f}] {span['label']}: \"{span['span'][:80]}...\"")

    # Save
    with open("feature_characterization_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to feature_characterization_detailed.json")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: What each feature detects")
    print("=" * 70)

    for key, data in results.items():
        print(f"\n{key} ({data['name']}):")
        # Get common words/patterns
        all_spans = " ".join([s['span'] for s in data['top_spans']])
        print(f"  Sample spans: {all_spans[:200]}...")


if __name__ == "__main__":
    main()
