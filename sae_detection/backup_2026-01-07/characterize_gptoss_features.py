#!/usr/bin/env python3
"""
Characterize GPT-OSS 16 generalizing features.
Find maximally activating text spans from BOTH training and gold_106.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda"
MODEL_NAME = "unsloth/gpt-oss-20b"
SAE_REPO = "Goodfire/Hackathon-gpt-oss-20b-SAE-l15"
SAE_FILE = "topk_sae_l15_exp16_k32.pt"
LAYER = 15


class TopKSAE:
    def __init__(self, checkpoint_path, k=32):
        self.k = k
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        w = checkpoint["encoder_linear.weight"]
        self.w_enc = w.T if w.shape[0] > w.shape[1] else w
        self.w_dec = checkpoint["decoder.weight"]
        self.b_enc = checkpoint.get("encoder_linear.bias")

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.w_dec = self.w_dec.to(device, dtype=dtype)
        if self.b_enc is not None:
            self.b_enc = self.b_enc.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + (self.b_enc if self.b_enc is not None else 0)
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts


def load_samples():
    """Load both training AF samples and gold_106 AF samples."""
    samples = []

    # Training AF samples
    path = hf_hub_download(
        repo_id="vincentoh/af-detection-benchmark",
        filename="classes/alignment_faking.json",
        repo_type="dataset",
        local_dir="./data"
    )
    with open(path) as f:
        data = json.load(f)
    for s in data["samples"][:30]:
        samples.append({"text": s["text"], "source": "training"})

    # Gold_106 AF samples
    path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset",
        local_dir="./data"
    )
    with open(path) as f:
        data = json.load(f)
    for s in data["samples"]:
        if s.get("label") in ["potential_faking", "alignment_faking"]:
            samples.append({"text": s["text"], "source": "gold_106"})

    return samples


def get_token_activations(model, tokenizer, sae, text, feature_idx):
    """Get per-token activations for a specific feature."""
    activations = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            activations["out"] = output[0].detach()
        else:
            activations["out"] = output.detach()

    target = model.model.layers[LAYER]
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


def find_max_spans(tokens, activations, tokenizer, window=12, top_k=3):
    """Find text spans with highest activation."""
    spans = []

    for i in range(len(activations)):
        if activations[i] > 0:
            start = max(0, i - window // 2)
            end = min(len(tokens), i + window // 2)
            span_tokens = tokens[start:end]
            span_text = tokenizer.convert_tokens_to_string(span_tokens)
            spans.append({
                "position": i,
                "activation": float(activations[i]),
                "span": span_text.strip()[:100],
            })

    spans = sorted(spans, key=lambda x: x["activation"], reverse=True)

    # Deduplicate overlapping spans
    unique = []
    used = set()
    for s in spans:
        if s["position"] not in used:
            unique.append(s)
            for p in range(s["position"] - window//2, s["position"] + window//2):
                used.add(p)
        if len(unique) >= top_k:
            break

    return unique


def main():
    print("="*70)
    print("CHARACTERIZING 16 GPT-OSS GENERALIZING FEATURES")
    print("="*70)

    # Load feature info
    print("\n[1/4] Loading feature info...")
    with open("results_2026-01-07/gptoss_generalizing_features.json") as f:
        data = json.load(f)

    features = data["generalizing"]
    print(f"  Characterizing {len(features)} features")

    # Load samples
    print("\n[2/4] Loading samples...")
    samples = load_samples()
    print(f"  {len(samples)} samples (training + gold_106 AF)")

    # Load model
    print("\n[3/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    sae_path = hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILE)
    sae = TopKSAE(sae_path, k=32).to(DEVICE, torch.bfloat16)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Characterize each feature
    print("\n[4/4] Characterizing features...")
    results = []

    for feat in tqdm(features, desc="Features"):
        feat_idx = feat["feature_idx"]
        feat_result = {
            "feature_idx": feat_idx,
            "train_auroc": feat["train_auroc"],
            "gold_auroc": feat["gold_auroc"],
            "top_spans": [],
        }

        all_spans = []

        for sample in samples:
            tokens, acts = get_token_activations(
                model, tokenizer, sae, sample["text"], feat_idx
            )
            spans = find_max_spans(tokens, acts, tokenizer, window=12, top_k=2)
            for s in spans:
                s["source"] = sample["source"]
            all_spans.extend(spans)

        # Get top spans across all samples
        all_spans = sorted(all_spans, key=lambda x: x["activation"], reverse=True)
        feat_result["top_spans"] = all_spans[:5]

        results.append(feat_result)

        # Print as we go
        print(f"\n  F{feat_idx} (gold={feat['gold_auroc']:.3f}):")
        for s in feat_result["top_spans"][:3]:
            print(f"    [{s['source'][:5]}] ({s['activation']:.0f}) \"{s['span'][:60]}...\"")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: GPT-OSS vs GEMMA CONCEPTS")
    print("="*70)

    print("\nGPT-OSS Top Features:")
    for r in results[:10]:
        print(f"\nF{r['feature_idx']} (gold AUROC: {r['gold_auroc']:.3f}):")
        for s in r["top_spans"][:2]:
            src = "TRAIN" if s["source"] == "training" else "GOLD "
            print(f"  [{src}] \"{s['span'][:70]}...\"")

    # Save
    with open("results_2026-01-07/gptoss_features_characterized.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to results_2026-01-07/gptoss_features_characterized.json")


if __name__ == "__main__":
    main()
