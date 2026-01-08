#!/usr/bin/env python3
"""
LlamaScope SAE Analysis - Last Token Strategy
==============================================
Use LAST token activation (not mean) for classification.
Mean pooling produces near-zero activations with ~99.99% SAE sparsity.
Last token position captures the classification-relevant state.

Memory: CPU offload strategy.
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
import tempfile
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

CONFIG = {
    "device": "cuda",
    "max_seq_len": 512,
    "batch_size": 4,
    "n_folds": 3,
    "svd_components": 200,
    "max_train_samples": 500,
    "layers_to_test": [8, 16, 24],
    "sae_release": "llama_scope_lxm_32x",
    "base_model": "meta-llama/Llama-3.1-8B",
}


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


def extract_activations_to_files(samples, layers, save_dir):
    """Extract LAST TOKEN activations from model."""
    print("\n[PHASE 1] Loading model and extracting activations...")

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("  Model loaded!")

    for layer in layers:
        print(f"\n  Layer {layer}...")
        all_acts = []

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

            batch_acts = []
            seq_lens = []

            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                # Store full sequence activations
                batch_acts.append(o.detach().cpu())

            target = model.model.layers[layer].mlp
            h = target.register_forward_hook(hook)

            with torch.no_grad():
                model(**inputs)
            h.remove()

            # Get attention_mask to find last non-padding token
            attn_mask = inputs["attention_mask"]
            seq_lengths = attn_mask.sum(dim=1) - 1  # Last non-padding position

            # Extract LAST token activation for each sample
            full_acts = batch_acts[0]  # (batch, seq, hidden)
            for j in range(full_acts.shape[0]):
                last_pos = seq_lengths[j].item()
                last_act = full_acts[j, last_pos, :]  # (hidden,)
                all_acts.append(last_act)

            del inputs, full_acts
            torch.cuda.empty_cache()

            if i % 100 == 0:
                print(f"    {i}/{len(samples)}", end="\r")

        # Stack and save
        all_acts = torch.stack(all_acts, dim=0)
        save_path = os.path.join(save_dir, f"layer_{layer}_acts.pt")
        torch.save(all_acts, save_path)
        print(f"  Saved: {save_path} (shape {all_acts.shape})")
        del all_acts

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("\n  Model unloaded!")


def process_with_sae(layer, acts_path):
    """Process activations with SAE."""
    print(f"\n[PHASE 2] Processing layer {layer} with SAE...")

    acts = torch.load(acts_path)
    print(f"  Loaded: {acts.shape}")
    print(f"  Act stats: min={acts.min():.4f}, max={acts.max():.4f}, mean={acts.mean():.4f}")

    sae = SAE.from_pretrained(
        release=CONFIG["sae_release"],
        sae_id=f"l{layer}m_32x",
        device=CONFIG["device"]
    )[0]
    print(f"  SAE features: {sae.cfg.d_sae}")

    all_features = []
    for i in range(0, acts.shape[0], CONFIG["batch_size"]):
        batch = acts[i:i + CONFIG["batch_size"]].to(CONFIG["device"]).float()
        with torch.no_grad():
            feats = sae.encode(batch)
        all_features.append(feats.cpu().numpy())
        del batch, feats
        torch.cuda.empty_cache()

    features = np.vstack(all_features)
    nonzero_per_sample = (features != 0).sum(axis=1).mean()
    sparsity = (features == 0).mean()
    print(f"  Features: {features.shape}")
    print(f"  Avg non-zero per sample: {nonzero_per_sample:.1f}")
    print(f"  Sparsity: {sparsity:.1%}")

    del sae, acts
    gc.collect()
    torch.cuda.empty_cache()

    return features, sparsity


def main():
    sep = "=" * 60
    print(sep)
    print("LLAMASCOPE SAE - LAST TOKEN STRATEGY")
    print("(Non-circular: BASE Llama SAE)")
    print(sep)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Layers: {CONFIG['layers_to_test']}")

    train_samples, eval_samples = load_datasets()
    labels_train = np.array([1 if s["label"] in ["potential_faking", "alignment_faking"] else 0 for s in train_samples])
    labels_eval = np.array([1 if s["label"] in ["potential_faking", "alignment_faking"] else 0 for s in eval_samples])

    results = {
        "config": CONFIG,
        "strategy": "last_token",
        "timestamp": datetime.now().isoformat(),
        "layers": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        all_samples = train_samples + eval_samples
        extract_activations_to_files(all_samples, CONFIG["layers_to_test"], tmpdir)

        for layer in CONFIG["layers_to_test"]:
            try:
                acts_path = os.path.join(tmpdir, f"layer_{layer}_acts.pt")
                features, sparsity = process_with_sae(layer, acts_path)

                X_train = features[:len(train_samples)]
                X_eval = features[len(train_samples):]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train)
                X_ev = scaler.transform(X_eval)

                n_comp = min(CONFIG["svd_components"], X_tr.shape[0] - 1, X_tr.shape[1])
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                X_tr = svd.fit_transform(X_tr)
                X_ev = svd.transform(X_ev)
                var_exp = svd.explained_variance_ratio_.sum()
                print(f"  SVD: {n_comp} components, {var_exp:.1%} variance")

                clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
                cv_scores = cross_val_score(clf, X_tr, labels_train, cv=CONFIG["n_folds"], scoring="roc_auc")
                print(f"  CV AUROC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

                clf.fit(X_tr, labels_train)
                eval_prob = clf.predict_proba(X_ev)[:, 1]
                eval_pred = (eval_prob > 0.5).astype(int)
                eval_auroc = roc_auc_score(labels_eval, eval_prob)
                eval_acc = accuracy_score(labels_eval, eval_pred)

                print(f"  Gold-106 AUROC: {eval_auroc:.4f}")
                print(f"  Gold-106 Accuracy: {eval_acc:.1%}")

                results["layers"][layer] = {
                    "cv_auroc_mean": float(cv_scores.mean()),
                    "cv_auroc_std": float(cv_scores.std()),
                    "eval_auroc": float(eval_auroc),
                    "eval_accuracy": float(eval_acc),
                    "avg_nonzero_features": float((features != 0).sum(axis=1).mean()),
                    "sparsity": float(sparsity),
                    "svd_variance_explained": float(var_exp),
                }

            except Exception as e:
                print(f"  Layer {layer} failed: {e}")
                import traceback
                traceback.print_exc()
                results["layers"][layer] = {"error": str(e)}

    # Summary
    print(f"\n{sep}\nSUMMARY\n{sep}")

    valid_layers = [(l, v) for l, v in results["layers"].items() if "eval_auroc" in v]
    if valid_layers:
        best = max(valid_layers, key=lambda x: x[1]["eval_auroc"])
        print(f"\nBest Layer: {best[0]}")
        print(f"  AUROC: {best[1]['eval_auroc']:.4f}")
        print(f"  Accuracy: {best[1]['eval_accuracy']:.1%}")
        print(f"  Avg non-zero features: {best[1]['avg_nonzero_features']:.0f}")
        results["best_layer"] = best[0]
        results["best_auroc"] = best[1]["eval_auroc"]

        print("\nAll Layers:")
        for l, v in sorted(valid_layers):
            print(f"  L{l}: AUROC={v['eval_auroc']:.4f}, Acc={v['eval_accuracy']:.1%}, NZ={v['avg_nonzero_features']:.0f}")

    # Interpretation
    print(f"\n{sep}\nINTERPRETATION\n{sep}")
    auroc = results.get("best_auroc", 0)
    if auroc > 0.7:
        print(f"\n-> SUCCESS: BASE model SAE features detect AF! (AUROC={auroc:.3f})")
        print("-> Non-circular: signal exists in base representations")
    elif auroc > 0.6:
        print(f"\n-> WEAK: Some signal in BASE model SAE (AUROC={auroc:.3f})")
    else:
        print(f"\n-> FAILURE: BASE model SAE doesn't detect AF (AUROC={auroc:.3f})")

    out_path = os.path.expanduser("~/llamascope_lasttoken_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{sep}")
    print(f"Saved: {out_path}")
    print(f"Done: {datetime.now().isoformat()}")
    print(sep)


if __name__ == "__main__":
    main()
