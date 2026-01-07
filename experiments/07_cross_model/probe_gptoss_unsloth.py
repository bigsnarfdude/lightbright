#!/usr/bin/env python3
"""
Train probe on GPT-OSS 16 generalizing features using Unsloth 4-bit loading.
This is the approach that works on nigel (16GB GPU).
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import gc
import warnings
warnings.filterwarnings('ignore')

from unsloth import FastLanguageModel

SAE_REPO = "Goodfire/Hackathon-gpt-oss-20b-SAE-l15"
SAE_FILE = "topk_sae_l15_exp16_k32.pt"
LAYER = 15
MODEL_NAME = "unsloth/gpt-oss-20b"

# The 16 generalizing features from gptoss_generalizing_features.json
GENERALIZING_FEATURES = [
    23524, 660, 42384, 41830, 25837, 35008, 15358, 17993,
    36418, 37820, 9872, 12173, 15319, 8504, 37303, 14374
]


class TopKSAE:
    """TopK Sparse Autoencoder."""

    def __init__(self, checkpoint_path: str, k: int = 32):
        self.k = k
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        w = checkpoint["encoder_linear.weight"]
        if w.shape[0] > w.shape[1]:
            self.w_enc = w.T
        else:
            self.w_enc = w

        self.w_dec = checkpoint["decoder.weight"]
        self.b_dec = checkpoint.get("decoder_bias", None)
        self.b_enc = checkpoint.get("encoder_linear.bias", None)
        self.device = "cpu"

    def to(self, device, dtype=None):
        self.device = device
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.w_dec = self.w_dec.to(device, dtype=dtype)
        if self.b_dec is not None:
            self.b_dec = self.b_dec.to(device, dtype=dtype)
        if self.b_enc is not None:
            self.b_enc = self.b_enc.to(device, dtype=dtype)
        return self

    def encode(self, x):
        x = x.to(self.device).float()
        if self.b_enc is not None:
            pre_acts = x @ self.w_enc.float() + self.b_enc.float()
        else:
            pre_acts = x @ self.w_enc.float()
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts

    @property
    def num_features(self):
        return self.w_enc.shape[1]


def load_gold_106():
    """Load gold_106 held-out samples."""
    path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset",
        local_dir="./data"
    )
    with open(path) as f:
        data = json.load(f)

    samples = []
    for s in data["samples"]:
        is_af = 1 if s.get("label") in ["potential_faking", "alignment_faking"] else 0
        samples.append({"text": s["text"], "is_af": is_af})

    return samples


def load_training_data():
    """Load training samples."""
    files = ["aligned.json", "alignment_faking.json", "genuine_compliance.json",
             "hard_negative.json", "compliant_hard_negative.json"]

    samples = []
    for fname in files:
        path = hf_hub_download(
            repo_id="vincentoh/af-detection-benchmark",
            filename=f"classes/{fname}",
            repo_type="dataset",
            local_dir="./data"
        )
        with open(path) as f:
            data = json.load(f)
        class_name = fname.replace(".json", "")
        for s in data["samples"]:
            samples.append({
                "text": s["text"],
                "is_af": 1 if class_name == "alignment_faking" else 0,
            })
    return samples


def extract_features(model, tokenizer, sae, samples, feature_indices, desc="Extracting"):
    """Extract SAE features using forward hooks."""
    all_acts = np.zeros((len(samples), len(feature_indices)), dtype=np.float32)
    device = next(model.parameters()).device

    for idx, sample in enumerate(tqdm(samples, desc=desc)):
        activations = {}

        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            activations["out"] = out.detach().cpu().float()

        # Hook layer 15
        target = model.model.layers[LAYER]
        handle = target.register_forward_hook(hook)

        # Tokenize and run forward pass
        inputs = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        # Get activations and encode with SAE
        act = activations["out"].squeeze(0)  # [seq_len, hidden_dim]

        with torch.no_grad():
            features = sae.encode(act)  # [seq_len, n_features]

        # Max pool across sequence
        max_per_feature = features.max(dim=0).values.numpy()
        all_acts[idx] = max_per_feature[feature_indices]

        # Cleanup
        del activations, act, features
        if idx % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return all_acts


def main():
    print("="*70)
    print("GPT-OSS 20B: PROBE ON 16 GENERALIZING FEATURES (Unsloth 4-bit)")
    print("="*70)
    print(f"\nFeatures: {GENERALIZING_FEATURES}")

    # Load model with Unsloth (4-bit, fits in 16GB)
    print("\n[1/5] Loading model with Unsloth 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Load SAE (keep on CPU to save VRAM)
    print("\n[2/5] Loading SAE...")
    sae_path = hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILE)
    sae = TopKSAE(sae_path, k=32)
    print(f"  SAE features: {sae.num_features}")

    # Load data
    print("\n[3/5] Loading datasets...")
    train_samples = load_training_data()
    gold_samples = load_gold_106()
    print(f"  Training: {len(train_samples)} samples")
    print(f"  Gold_106: {len(gold_samples)} samples")

    # Extract features
    print("\n[4/5] Extracting features...")
    X_train = extract_features(model, tokenizer, sae, train_samples, GENERALIZING_FEATURES, desc="Training")
    y_train = np.array([s["is_af"] for s in train_samples])

    X_gold = extract_features(model, tokenizer, sae, gold_samples, GENERALIZING_FEATURES, desc="Gold_106")
    y_gold = np.array([s["is_af"] for s in gold_samples])

    # Train and evaluate probe
    print("\n[5/5] Training probe...")
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X_train, y_train)

    train_probs = probe.predict_proba(X_train)[:, 1]
    train_auroc = roc_auc_score(y_train, train_probs)

    gold_probs = probe.predict_proba(X_gold)[:, 1]
    gold_preds = probe.predict(X_gold)
    gold_auroc = roc_auc_score(y_gold, gold_probs)
    gold_acc = accuracy_score(y_gold, gold_preds)

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n| Set | AUROC | Accuracy |")
    print(f"|-----|-------|----------|")
    print(f"| Training | {train_auroc:.4f} | - |")
    print(f"| **Gold_106** | **{gold_auroc:.4f}** | **{gold_acc:.1%}** |")

    print(f"\nClassification Report (Gold_106):")
    print(classification_report(y_gold, gold_preds, target_names=["Non-AF", "AF"]))

    # Save results
    results = {
        "model": "GPT-OSS-20B",
        "sae": SAE_REPO,
        "layer": LAYER,
        "quantization": "4-bit (unsloth)",
        "n_generalizing_features": len(GENERALIZING_FEATURES),
        "feature_indices": GENERALIZING_FEATURES,
        "training_auroc": float(train_auroc),
        "gold_auroc": float(gold_auroc),
        "gold_accuracy": float(gold_acc),
        "note": "Actual probe on 16 generalizing features - fills evidence gap"
    }

    os.makedirs("results_2026-01-07", exist_ok=True)
    out_path = "results_2026-01-07/gptoss_generalizing_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path}")

    # Verdict
    print("\n" + "="*70)
    if gold_auroc >= 0.80:
        print(f"STRONG: GPT-OSS probe achieves {gold_auroc:.3f} AUROC on gold_106")
    elif gold_auroc >= 0.70:
        print(f"GOOD: GPT-OSS probe achieves {gold_auroc:.3f} AUROC on gold_106")
    else:
        print(f"RESULT: GPT-OSS probe achieves {gold_auroc:.3f} AUROC on gold_106")
    print("="*70)


if __name__ == "__main__":
    main()
