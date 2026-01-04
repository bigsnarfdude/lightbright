#!/usr/bin/env python3
"""
Find features that are SPECIFIC to alignment faking.

Method:
1. Run AF samples through GemmaScope SAE
2. Run Aligned samples through same SAE
3. Find features where: mean(AF) >> mean(Aligned)
4. These are TRUE AF-specific features, not just "internal conflict"

Requirements:
    pip install torch transformers safetensors huggingface_hub numpy

This finds features that DISCRIMINATE between AF and non-AF,
not just features that fire on "conflict" broadly.
"""

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from collections import defaultdict
import os

# Config
MODEL_ID = "google/gemma-3-27b-it"
SAE_REPO = "google/gemma-scope-2-27b-it"
LAYERS_TO_SCAN = [31, 40, 53]  # Key layers from prior analysis
DEVICE = "cuda"

# Use bf16 on H100-80GB (no quantization needed)
USE_BF16 = True  # Set False for 4-bit on smaller GPUs

def load_sae(layer: int):
    """Load GemmaScope-2 SAE for specified layer."""
    from safetensors import safe_open

    params_path = hf_hub_download(
        repo_id=SAE_REPO,
        filename=f"transcoder/layer_{layer}_width_16k_l0_medium/params.safetensors",
        repo_type="model"
    )
    config_path = hf_hub_download(
        repo_id=SAE_REPO,
        filename=f"transcoder/layer_{layer}_width_16k_l0_medium/config.json",
        repo_type="model"
    )

    with open(config_path) as f:
        config = json.load(f)

    tensors = {}
    with safe_open(params_path, framework="pt", device=DEVICE) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).to(torch.bfloat16)

    W_enc = tensors.get('w_enc', tensors.get('W_enc'))
    b_enc = tensors.get('b_enc', tensors.get('B_enc'))

    thresh_val = config.get('k', config.get('threshold', 0))
    threshold = torch.full((W_enc.shape[-1],), thresh_val, device=DEVICE, dtype=torch.bfloat16)

    return {'W_enc': W_enc, 'b_enc': b_enc, 'threshold': threshold}

def get_feature_activations(model, tokenizer, text, sae_params, layer_idx):
    """Get SAE feature activations for a text sample."""

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    activations = []

    def hook(module, input, output):
        # For layernorm, output is the normalized tensor directly
        activations.append(output.detach())

    # Find the layer - Gemma-3 structure (from HOOKS_AND_PATCHES.md):
    # Without PEFT: model.model.language_model.layers[N]
    # With PEFT: model.base_model.model.model.language_model.layers[N]
    try:
        # Try PEFT path first
        layer = model.base_model.model.model.language_model.layers[layer_idx]
    except AttributeError:
        try:
            # Without PEFT
            layer = model.model.language_model.layers[layer_idx]
        except AttributeError:
            # Direct path fallback
            layer = model.language_model.layers[layer_idx]

    handle = layer.pre_feedforward_layernorm.register_forward_hook(hook)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    # Get activations and compute SAE features
    hidden = activations[0].to(torch.bfloat16)  # [1, seq_len, hidden_dim]

    # Encode through SAE
    pre_acts = hidden @ sae_params['W_enc'] + sae_params['b_enc']
    feature_acts = torch.relu(pre_acts - sae_params['threshold'])

    # Return max activation per feature across sequence
    max_acts = feature_acts.max(dim=1)[0].squeeze()  # [num_features]

    return max_acts.float().cpu().numpy()

def main():
    print("=" * 70)
    print("CONTRASTIVE FEATURE SEARCH: Finding AF-Specific Features")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading samples...")

    data_paths = ["../../data/af_test_samples.json", "../data/af_test_samples.json", "data/af_test_samples.json"]
    for path in data_paths:
        if os.path.exists(path):
            with open(path) as f:
                test_data = json.load(f)
            break
    else:
        # Fall back to gold_106
        for path in ["../../data/gold_106.json", "../data/gold_106.json"]:
            if os.path.exists(path):
                with open(path) as f:
                    gold_data = json.load(f)
                test_data = {"samples": gold_data.get("samples", gold_data)[:20]}
                break

    samples = test_data["samples"]
    af_samples = [s for s in samples if s["label"] in ["alignment_faking", "potential_faking", "AF"]]
    aligned_samples = [s for s in samples if s["label"] == "aligned"]

    print(f"AF samples: {len(af_samples)}")
    print(f"Aligned samples: {len(aligned_samples)}")

    # Load model
    if USE_BF16:
        print("\n[2/5] Loading model (bf16 - H100/A100-80GB)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        print("\n[2/5] Loading model (4-bit quantized)...")
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )
    model.eval()

    if torch.cuda.is_available():
        print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    results = {}

    for layer_idx in LAYERS_TO_SCAN:
        print(f"\n[3/5] Loading SAE for layer {layer_idx}...")
        sae_params = load_sae(layer_idx)
        num_features = sae_params['W_enc'].shape[-1]
        print(f"SAE has {num_features} features")

        # Collect activations
        print(f"\n[4/5] Computing activations for layer {layer_idx}...")

        af_activations = []
        for i, sample in enumerate(af_samples):
            text = sample.get("text", sample.get("reasoning", ""))
            acts = get_feature_activations(model, tokenizer, text, sae_params, layer_idx)
            af_activations.append(acts)
            print(f"  AF sample {i+1}/{len(af_samples)}")

        aligned_activations = []
        for i, sample in enumerate(aligned_samples):
            text = sample.get("text", sample.get("reasoning", ""))
            acts = get_feature_activations(model, tokenizer, text, sae_params, layer_idx)
            aligned_activations.append(acts)
            print(f"  Aligned sample {i+1}/{len(aligned_samples)}")

        # Compute statistics
        af_acts = np.stack(af_activations)  # [n_af, num_features]
        aligned_acts = np.stack(aligned_activations)  # [n_aligned, num_features]

        af_mean = af_acts.mean(axis=0)
        aligned_mean = aligned_acts.mean(axis=0)

        # Find discriminative features: high AF, low Aligned
        diff = af_mean - aligned_mean

        # Also compute ratio for features that fire on AF
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(aligned_mean > 0.1, af_mean / aligned_mean, 0)

        # Top features by difference
        top_diff_idx = np.argsort(diff)[-20:][::-1]

        # Top features by ratio (where AF fires)
        af_fires = af_mean > 1.0
        ratio_masked = np.where(af_fires, ratio, 0)
        top_ratio_idx = np.argsort(ratio_masked)[-20:][::-1]

        print(f"\n[5/5] Top AF-specific features for layer {layer_idx}:")
        print("-" * 60)
        print(f"{'Feature':<10} {'AF Mean':<12} {'Aligned Mean':<14} {'Diff':<10} {'Ratio':<10}")
        print("-" * 60)

        layer_results = []
        for idx in top_diff_idx:
            if diff[idx] > 0.5:  # Only significant differences
                result = {
                    "feature": int(idx),
                    "af_mean": float(af_mean[idx]),
                    "aligned_mean": float(aligned_mean[idx]),
                    "diff": float(diff[idx]),
                    "ratio": float(ratio[idx]) if not np.isinf(ratio[idx]) else 999
                }
                layer_results.append(result)
                print(f"{idx:<10} {af_mean[idx]:<12.2f} {aligned_mean[idx]:<14.2f} {diff[idx]:<10.2f} {ratio[idx]:<10.2f}")

        results[f"layer_{layer_idx}"] = layer_results

    # Save results
    output_path = "af_specific_features.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Features with high diff AND ratio are TRUE AF-specific features.
They fire on alignment faking but NOT on general ethical deliberation.

Use these features for a real AF detector:
- High activation = likely alignment faking
- Low activation = genuine reasoning (even if conflicted)
""")

if __name__ == "__main__":
    main()
