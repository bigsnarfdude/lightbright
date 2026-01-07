#!/usr/bin/env python3
"""
Feature Sweep for GPT-OSS 20B with Goodfire SAE
================================================
Uses pre-trained SAE from Goodfire/Hackathon-gpt-oss-20b-SAE-l15
"""

import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


CONFIG = {
    "model_name": "unsloth/gpt-oss-20b",  # Base model for the detector
    "sae_repo": "Goodfire/Hackathon-gpt-oss-20b-SAE-l15",
    "sae_file": "topk_sae_l15_exp16_k32.pt",
    "layer": 15,
    "device": "cuda",
    "max_length": 2048,
    "dataset_repo": "vincentoh/af-detection-benchmark",
}


class TopKSAE:
    """TopK Sparse Autoencoder."""

    def __init__(self, checkpoint_path: str, k: int = 32):
        self.k = k
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Try different key formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            else:
                state = checkpoint
        else:
            state = checkpoint

        # Print keys to understand structure
        print(f"  SAE checkpoint keys: {list(state.keys())[:10]}...")

        # Common SAE weight names
        enc_names = ["W_enc", "encoder.weight", "encode.weight", "w_enc", "encoder_linear.weight"]
        dec_names = ["W_dec", "decoder.weight", "decode.weight", "w_dec"]

        self.w_enc = None
        self.w_dec = None
        self.b_enc = None
        self.b_dec = None

        for name in enc_names:
            if name in state:
                w = state[name]
                # Encoder should be [d_model, n_features] for x @ W_enc
                # If shape is [n_features, d_model], transpose it
                if w.shape[0] > w.shape[1]:  # More rows than cols = [n_features, d_model]
                    self.w_enc = w.T  # Transpose to [d_model, n_features]
                    print(f"  Found encoder: {name} shape {w.shape} -> transposed to {self.w_enc.shape}")
                else:
                    self.w_enc = w
                    print(f"  Found encoder: {name} shape {self.w_enc.shape}")
                break

        for name in dec_names:
            if name in state:
                w = state[name]
                # Decoder should be [n_features, d_model] for features @ W_dec
                if w.shape[0] > w.shape[1]:  # [n_features, d_model] - correct
                    self.w_dec = w
                else:
                    self.w_dec = w.T
                print(f"  Found decoder: {name} shape {self.w_dec.shape}")
                break

        # Biases
        for name in ["b_enc", "encoder.bias", "encode.bias", "encoder_linear.bias"]:
            if name in state:
                self.b_enc = state[name]
                print(f"  Found encoder bias: {name}")
                break

        for name in ["b_dec", "decoder.bias", "decode.bias", "decoder_bias"]:
            if name in state:
                self.b_dec = state[name]
                print(f"  Found decoder bias: {name}")
                break

        if self.w_enc is None or self.w_dec is None:
            print(f"  WARNING: Could not find encoder/decoder. Available keys: {list(state.keys())}")
            for key, val in state.items():
                if hasattr(val, 'shape') and len(val.shape) == 2:
                    print(f"    {key}: {val.shape}")

    def to(self, device, dtype=None):
        if self.w_enc is not None:
            self.w_enc = self.w_enc.to(device, dtype=dtype)
        if self.w_dec is not None:
            self.w_dec = self.w_dec.to(device, dtype=dtype)
        if self.b_enc is not None:
            self.b_enc = self.b_enc.to(device, dtype=dtype)
        if self.b_dec is not None:
            self.b_dec = self.b_dec.to(device, dtype=dtype)
        return self

    def encode(self, x):
        """Encode with TopK activation."""
        # x: [seq_len, d_model]
        if self.b_enc is not None:
            pre_acts = x @ self.w_enc + self.b_enc
        else:
            pre_acts = x @ self.w_enc

        # TopK activation
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)

        # Create sparse activation (keep only top k)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))

        return acts

    @property
    def num_features(self):
        if self.w_enc is not None:
            return self.w_enc.shape[1]
        return 0


def load_dataset(cache_dir="./data"):
    """Load AF detection benchmark."""
    print("Loading dataset...")
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

    print(f"  Loaded {len(all_samples)} samples")
    return all_samples


def load_model_and_sae():
    """Load GPT-OSS 20B and the Goodfire SAE."""
    print("\n[1/3] Loading SAE...")
    sae_path = hf_hub_download(
        repo_id=CONFIG["sae_repo"],
        filename=CONFIG["sae_file"],
    )
    sae = TopKSAE(sae_path, k=32)

    if sae.w_enc is None:
        print("ERROR: Could not load SAE weights properly")
        return None, None, None

    print(f"  SAE features: {sae.num_features}")

    # First, let's check what the SAE expects
    d_model = sae.w_enc.shape[0]
    print(f"  SAE expects d_model: {d_model}")

    print("\n[2/3] Loading GPT-OSS 20B model...")
    model_name = CONFIG["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Loaded: {model_name}")

    model.eval()
    sae = sae.to(CONFIG["device"], torch.bfloat16)

    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    return model, tokenizer, sae


def get_layer_output_hook(model, layer_idx):
    """Get the correct hook target for different model architectures."""
    # Try different architectures
    if hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            return model.model.layers[layer_idx]
        elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            return model.model.decoder.layers[layer_idx]
    if hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'h'):
            return model.transformer.h[layer_idx]
        elif hasattr(model.transformer, 'layers'):
            return model.transformer.layers[layer_idx]

    raise ValueError(f"Could not find layer {layer_idx} in model architecture")


def extract_features(model, tokenizer, samples, sae, layer):
    """Extract SAE features for all samples."""
    print(f"\nExtracting features for layer {layer}...")

    n_samples = len(samples)
    n_features = sae.num_features
    all_max_activations = np.zeros((n_samples, n_features), dtype=np.float32)

    # Find the right hook target
    try:
        target = get_layer_output_hook(model, layer)
        print(f"  Hook target: {type(target).__name__}")
    except Exception as e:
        print(f"  ERROR finding hook target: {e}")
        return None

    for idx, sample in enumerate(tqdm(samples, desc=f"Layer {layer}")):
        activations = {}

        def hook(module, input, output):
            if isinstance(output, tuple):
                activations["out"] = output[0].detach()
            else:
                activations["out"] = output.detach()

        handle = target.register_forward_hook(hook)

        inputs = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"],
            padding=True,
        ).to(CONFIG["device"])

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        act = activations["out"].squeeze(0).to(torch.bfloat16)

        # Check dimension match
        if act.shape[-1] != sae.w_enc.shape[0]:
            if idx == 0:
                print(f"  WARNING: Dimension mismatch! Model: {act.shape[-1]}, SAE: {sae.w_enc.shape[0]}")
            continue

        with torch.no_grad():
            features = sae.encode(act)

        max_per_feature = features.max(dim=0).values.float().cpu().numpy()
        all_max_activations[idx] = max_per_feature

        if idx % 100 == 0:
            torch.cuda.empty_cache()

    return all_max_activations


def analyze_features(activations, samples, output_dir):
    """Analyze features for AF detection."""
    print(f"\nAnalyzing {activations.shape[1]} features...")

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
            "auroc_af_vs_all": float(auroc_af),
            "auroc_af_vs_hard_neg": float(auroc_af_hn),
            "ttest_pval": float(p_val),
        })

    # Find needles
    needles = [r for r in results
               if r["auroc_af_vs_all"] > 0.7
               and r["auroc_af_vs_hard_neg"] > 0.65
               and r["ttest_pval"] < 0.01]

    needles = sorted(needles, key=lambda x: x["auroc_af_vs_hard_neg"], reverse=True)

    print(f"\nFound {len(needles)} true needles!")

    # Save results
    out_path = os.path.join(output_dir, "gptoss_sae_sweep.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": "GPT-OSS-20B",
            "sae": CONFIG["sae_repo"],
            "layer": CONFIG["layer"],
            "total_features": activations.shape[1],
            "active_features": len(results),
            "needles_found": len(needles),
            "top_20_needles": needles[:20],
            "all_results": results,
        }, f, indent=2)

    return needles


def main():
    print("="*70)
    print("GPT-OSS 20B SAE FEATURE SWEEP")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/gptoss_sweep_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Load everything
    model, tokenizer, sae = load_model_and_sae()

    if model is None or sae is None:
        print("Failed to load model or SAE. Exiting.")
        return

    # Load dataset
    samples = load_dataset()

    # Extract features
    activations = extract_features(model, tokenizer, samples, sae, CONFIG["layer"])

    if activations is None:
        print("Failed to extract features. Exiting.")
        return

    # Save activations
    np.save(os.path.join(output_dir, "activations_gptoss_l15.npy"), activations)

    # Analyze
    needles = analyze_features(activations, samples, output_dir)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total features: {activations.shape[1]}")
    print(f"True needles: {len(needles)}")

    if needles:
        print("\nTop 10 features by AUROC (AF vs Hard Neg):")
        for n in needles[:10]:
            print(f"  F{n['feature_idx']}: AUROC={n['auroc_af_vs_hard_neg']:.3f}, "
                  f"AF mean={n['af_mean']:.1f}, HN mean={n['hard_neg_mean']:.1f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
