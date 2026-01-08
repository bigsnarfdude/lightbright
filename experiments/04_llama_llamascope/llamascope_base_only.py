#!/usr/bin/env python3
"""
LlamaScope SAE Analysis - BASE Model Only (Memory Efficient)
=============================================================
Tests whether SAE features on BASE Llama (no LoRA) can detect AF.
This is the non-circular experiment - if it works, SAE captures AF signal.

Memory optimizations:
- BASE model only (not trained)
- One layer at a time
- Small batch size
- Explicit VRAM cleanup
"""

import os
if not os.environ.get("HF_TOKEN"):
    raise ValueError("HF_TOKEN environment variable required - get yours at huggingface.co/settings/tokens")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import torch
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from huggingface_hub import hf_hub_download, login
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

login(token=os.environ["HF_TOKEN"])

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "device": "cuda",
    "max_seq_len": 512,
    "batch_size": 4,  # Small batch for memory
    "n_folds": 3,
    "svd_components": 200,  # Reduced
    "max_train_samples": 500,  # Reduced
    "layers_to_test": [0, 8, 16, 24, 31],  # Key layers only
    "sae_release": "llama_scope_lxm_32x",
    "base_model": "meta-llama/Llama-3.1-8B",
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_datasets():
    print("Loading datasets...")
    train_path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="training_data_final.json",
        repo_type="dataset"
    )
    with open(train_path) as f:
        train_samples = json.load(f).get("samples", [])

    eval_path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset"
    )
    with open(eval_path) as f:
        eval_samples = json.load(f).get("samples", [])

    # Subsample training
    np.random.seed(42)
    aligned = [s for s in train_samples if s["label"] == "aligned"]
    faking = [s for s in train_samples if s["label"] == "potential_faking"]
    max_n = min(CONFIG["max_train_samples"], len(aligned), len(faking))
    aligned = list(np.random.choice(aligned, max_n, replace=False))
    faking = list(np.random.choice(faking, max_n, replace=False))
    train_samples = aligned + faking
    np.random.shuffle(train_samples)

    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")
    return train_samples, eval_samples

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_sae_features(model, tokenizer, sae, samples, layer):
    """Extract SAE features from BASE model."""
    all_features = []
    labels = []

    for i in range(0, len(samples), CONFIG["batch_size"]):
        batch = samples[i:i + CONFIG["batch_size"]]
        texts = [s["text"][:2000] for s in batch]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG["max_seq_len"]
        ).to(CONFIG["device"])

        # Hook to capture activations
        batch_acts = []
        def hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            # Mean pool across sequence
            batch_acts.append(o.mean(dim=1).detach())

        target = model.model.layers[layer].mlp
        h = target.register_forward_hook(hook)

        with torch.no_grad():
            model(**inputs)
        h.remove()

        # Get SAE features
        acts = batch_acts[0].float()
        with torch.no_grad():
            feats = sae.encode(acts)
        all_features.append(feats.cpu().numpy())
        labels.extend([1 if s["label"] in ["potential_faking", "alignment_faking"] else 0 for s in batch])

        # Clean up
        del inputs, batch_acts, acts, feats
        torch.cuda.empty_cache()

        if (i // CONFIG["batch_size"]) % 20 == 0:
            print(f"    Batch {i // CONFIG['batch_size']}/{len(samples) // CONFIG['batch_size']}", end="\r")

    return np.vstack(all_features), np.array(labels)

# ============================================================================
# MAIN
# ============================================================================

def main():
    sep = "=" * 60
    print(sep)
    print("LLAMASCOPE SAE - BASE MODEL ONLY")
    print("(Non-circular: testing if BASE Llama SAE captures AF)")
    print(sep)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Layers: {CONFIG['layers_to_test']}")
    print()

    train_samples, eval_samples = load_datasets()

    results = {
        "config": CONFIG,
        "timestamp": datetime.now().isoformat(),
        "layers": {},
    }

    # Load model ONCE
    print("\nLoading BASE model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("  Model loaded!")

    for layer in CONFIG["layers_to_test"]:
        print(f"\n{sep}\nLAYER {layer}\n{sep}")

        try:
            # Load SAE for this layer
            print(f"  Loading SAE for layer {layer}...")
            sae, _, _ = SAE.from_pretrained(
                release=CONFIG["sae_release"],
                sae_id=f"l{layer}m_32x",
                device=CONFIG["device"]
            )
            print(f"  SAE features: {sae.cfg.d_sae}")
        except Exception as e:
            print(f"  SAE failed: {e}")
            results["layers"][layer] = {"error": str(e)}
            continue

        # Extract features
        print("\n  Extracting train features...")
        X_train, y_train = extract_sae_features(model, tokenizer, sae, train_samples, layer)
        print(f"\n  Train shape: {X_train.shape}, Sparsity: {(X_train == 0).mean():.1%}")

        print("  Extracting eval features...")
        X_eval, y_eval = extract_sae_features(model, tokenizer, sae, eval_samples, layer)
        print(f"\n  Eval shape: {X_eval.shape}")

        # Preprocess
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_ev = scaler.transform(X_eval)

        # SVD for dimensionality reduction
        n_comp = min(CONFIG["svd_components"], X_tr.shape[0] - 1, X_tr.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_tr = svd.fit_transform(X_tr)
        X_ev = svd.transform(X_ev)
        var_exp = svd.explained_variance_ratio_.sum()
        print(f"  SVD: {n_comp} components, {var_exp:.1%} variance")

        # Train classifier
        print("  Training logistic regression...")
        clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")

        # Cross-validation
        cv_scores = cross_val_score(clf, X_tr, y_train, cv=CONFIG["n_folds"], scoring="roc_auc")
        print(f"  CV AUROC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Final evaluation on Gold-106
        clf.fit(X_tr, y_train)
        eval_prob = clf.predict_proba(X_ev)[:, 1]
        eval_pred = (eval_prob > 0.5).astype(int)
        eval_auroc = roc_auc_score(y_eval, eval_prob)
        eval_acc = accuracy_score(y_eval, eval_pred)

        print(f"  Gold-106 AUROC: {eval_auroc:.4f}")
        print(f"  Gold-106 Accuracy: {eval_acc:.1%}")

        results["layers"][layer] = {
            "cv_auroc_mean": float(cv_scores.mean()),
            "cv_auroc_std": float(cv_scores.std()),
            "eval_auroc": float(eval_auroc),
            "eval_accuracy": float(eval_acc),
            "n_sae_features": int(sae.cfg.d_sae),
            "sparsity": float((X_train == 0).mean()),
            "svd_variance_explained": float(var_exp),
        }

        # Clean up SAE
        del sae, X_train, y_train, X_eval, X_tr, X_ev
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{sep}\nSUMMARY\n{sep}")

    if results["layers"]:
        best_layer = max(
            [(l, v) for l, v in results["layers"].items() if "eval_auroc" in v],
            key=lambda x: x[1]["eval_auroc"],
            default=(None, None)
        )
        if best_layer[0] is not None:
            print(f"\nBest Layer: {best_layer[0]}")
            print(f"  AUROC: {best_layer[1]['eval_auroc']:.4f}")
            print(f"  Accuracy: {best_layer[1]['eval_accuracy']:.1%}")
            results["best_layer"] = best_layer[0]
            results["best_auroc"] = best_layer[1]["eval_auroc"]

    # Interpretation
    print(f"\n{sep}\nINTERPRETATION\n{sep}")
    if results.get("best_auroc", 0) > 0.7:
        print("\n-> SUCCESS: BASE model SAE features detect AF!")
        print("-> This is non-circular: SAE captures signal, not classifier artifacts")
    elif results.get("best_auroc", 0) > 0.6:
        print("\n-> WEAK: Some signal in BASE model SAE features")
        print("-> May be scenario encoding, not AF decision")
    else:
        print("\n-> FAILURE: BASE model SAE features don't detect AF")
        print("-> The 0.68 ceiling may have been from the trained model")

    # Save
    out_path = os.path.expanduser("~/llamascope_base_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{sep}")
    print(f"Saved: {out_path}")
    print(f"Done: {datetime.now().isoformat()}")
    print(sep)


if __name__ == "__main__":
    main()
