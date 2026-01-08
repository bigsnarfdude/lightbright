#!/usr/bin/env python3
"""
Systematic LlamaScope SAE Analysis v3 - Fast & Rigorous
=========================================================
Optimized for completion in reasonable time while maintaining rigor.

Changes from v2:
- Fast classifiers only (logreg, rf)
- SVD: 500 components (was 2000)
- 3-fold CV (was 5)
- 8 key layers (was 16)
"""

import os
if not os.environ.get("HF_TOKEN"):
    raise ValueError("HF_TOKEN environment variable required - get yours at huggingface.co/settings/tokens")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy import stats
from huggingface_hub import hf_hub_download, login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sae_lens import SAE

login(token=os.environ["HF_TOKEN"])

# ============================================================================
# CONFIGURATION - Optimized for speed
# ============================================================================

CONFIG = {
    "device": "cuda",
    "max_seq_len": 512,
    "batch_size": 16,  # Larger batch
    "n_folds": 3,      # Reduced from 5
    "n_seeds": 3,
    "svd_components": 500,  # Reduced from 2000
    "max_train_per_class": 1000,
    "layers_to_test": [0, 4, 8, 12, 16, 20, 24, 28],  # 8 key layers
    "sae_release": "llama_scope_lxm_32x",
    "base_model": "meta-llama/Llama-3.1-8B",
    "trained_model": "vincentoh/llama-8b-af-detector",
}

# Only fast classifiers
CLASSIFIERS = {
    "logreg": lambda: LogisticRegression(max_iter=300, C=1.0, solver="lbfgs", n_jobs=-1),
    "rf": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1),
}

# ============================================================================
# DATA & MODEL LOADING
# ============================================================================

def load_datasets():
    print("Loading datasets...")
    train_path = hf_hub_download(repo_id="vincentoh/alignment-faking-training", filename="training_data_final.json", repo_type="dataset")
    with open(train_path) as f:
        train_samples = json.load(f).get("samples", [])

    eval_path = hf_hub_download(repo_id="vincentoh/alignment-faking-training", filename="gold_106.json", repo_type="dataset")
    with open(eval_path) as f:
        eval_samples = json.load(f).get("samples", [])

    # Subsample
    np.random.seed(42)
    aligned = [s for s in train_samples if s["label"] == "aligned"]
    faking = [s for s in train_samples if s["label"] == "potential_faking"]
    max_n = CONFIG["max_train_per_class"]
    if len(aligned) > max_n: aligned = list(np.random.choice(aligned, max_n, replace=False))
    if len(faking) > max_n: faking = list(np.random.choice(faking, max_n, replace=False))
    train_samples = aligned + faking
    np.random.shuffle(train_samples)

    print(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}")
    return train_samples, eval_samples

def load_base_model():
    print("Loading BASE model...")
    model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def load_trained_model():
    print("Loading TRAINED model...")
    base = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, CONFIG["trained_model"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def load_sae(layer: int) -> SAE:
    sae, _, _ = SAE.from_pretrained(release=CONFIG["sae_release"], sae_id=f"l{layer}m_32x", device=CONFIG["device"])
    return sae

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_activations(model, tokenizer, texts, layer, is_peft=False):
    all_acts = []
    for i in range(0, len(texts), CONFIG["batch_size"]):
        batch = texts[i:i + CONFIG["batch_size"]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG["max_seq_len"]).to(CONFIG["device"])
        batch_acts = []
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            batch_acts.append(o.mean(dim=1).detach().cpu())
        target = model.base_model.model.model.layers[layer].mlp if is_peft else model.model.layers[layer].mlp
        h = target.register_forward_hook(hook)
        with torch.no_grad(): model(**inputs)
        h.remove()
        all_acts.append(batch_acts[0])
    return torch.cat(all_acts, dim=0)

def get_sae_features(model, tokenizer, samples, layer, sae, is_peft=False):
    texts = [s["text"][:2000] for s in samples]
    labels = np.array([1 if s["label"] == "potential_faking" else 0 for s in samples])
    acts = extract_activations(model, tokenizer, texts, layer, is_peft).to(CONFIG["device"]).float()
    with torch.no_grad():
        features = sae.encode(acts)
    return features.cpu().numpy(), labels

# ============================================================================
# EVALUATION
# ============================================================================

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

def run_cv(X_train, y_train, X_eval, y_eval, clf_name, seed):
    clf_fn = CLASSIFIERS[clf_name]

    # Scale + SVD
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ev = scaler.transform(X_eval)

    n_comp = min(CONFIG["svd_components"], X_tr.shape[0] - 1, X_tr.shape[1])
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    X_tr = svd.fit_transform(X_tr)
    X_ev = svd.transform(X_ev)

    # CV
    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=seed)
    cv_aurocs = []
    for tr_idx, val_idx in skf.split(X_tr, y_train):
        clf = clf_fn()
        clf.fit(X_tr[tr_idx], y_train[tr_idx])
        prob = clf.predict_proba(X_tr[val_idx])[:, 1]
        try: cv_aurocs.append(roc_auc_score(y_train[val_idx], prob))
        except: pass

    # Final eval
    clf = clf_fn()
    clf.fit(X_tr, y_train)
    eval_prob = clf.predict_proba(X_ev)[:, 1]
    eval_pred = (eval_prob > 0.5).astype(int)

    return {
        "cv_auroc": float(np.mean(cv_aurocs)) if cv_aurocs else 0.5,
        "cv_std": float(np.std(cv_aurocs)) if cv_aurocs else 0.0,
        "eval": compute_metrics(y_eval, eval_pred, eval_prob),
        "var_explained": float(svd.explained_variance_ratio_.sum()),
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    sep = "=" * 60
    print(sep)
    print("SYSTEMATIC LLAMASCOPE SAE v3 (Fast)")
    print(sep)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Layers: {CONFIG['layers_to_test']}")
    print(f"Classifiers: {list(CLASSIFIERS.keys())}")
    print(f"SVD: {CONFIG['svd_components']}, Folds: {CONFIG['n_folds']}, Seeds: {CONFIG['n_seeds']}")
    print()

    train_samples, eval_samples = load_datasets()

    results = {
        "config": {k: v for k, v in CONFIG.items() if k != "device"},
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }

    base_model, base_tok = load_base_model()
    trained_model, trained_tok = load_trained_model()
    models = {"base": (base_model, base_tok, False), "trained": (trained_model, trained_tok, True)}

    for layer in CONFIG["layers_to_test"]:
        print(f"\n{sep}\nLAYER {layer}\n{sep}")

        try:
            sae = load_sae(layer)
            print(f"  SAE: {sae.cfg.d_sae} features")
        except Exception as e:
            print(f"  SAE failed: {e}")
            continue

        for model_name, (model, tok, is_peft) in models.items():
            print(f"\n  {model_name.upper()}:")
            X_train, y_train = get_sae_features(model, tok, train_samples, layer, sae, is_peft)
            X_eval, y_eval = get_sae_features(model, tok, eval_samples, layer, sae, is_peft)
            sparsity = (X_train == 0).mean()
            print(f"    Shape: {X_train.shape}, Sparsity: {sparsity:.1%}")

            for clf_name in CLASSIFIERS:
                print(f"    {clf_name}: ", end="", flush=True)
                seed_res = []
                for seed in range(CONFIG["n_seeds"]):
                    try:
                        r = run_cv(X_train, y_train, X_eval, y_eval, clf_name, seed)
                        seed_res.append(r)
                    except Exception as e:
                        print(f"[err] ", end="")

                if not seed_res:
                    print("FAILED")
                    continue

                cv = [r["cv_auroc"] for r in seed_res]
                ev = [r["eval"]["auroc"] for r in seed_res]
                acc = [r["eval"]["accuracy"] for r in seed_res]
                f1 = [r["eval"]["f1"] for r in seed_res]

                key = f"l{layer}_{model_name}_{clf_name}"
                results["experiments"][key] = {
                    "layer": layer, "model": model_name, "clf": clf_name,
                    "cv_auroc": {"m": float(np.mean(cv)), "s": float(np.std(cv))},
                    "eval_auroc": {"m": float(np.mean(ev)), "s": float(np.std(ev))},
                    "eval_acc": {"m": float(np.mean(acc)), "s": float(np.std(acc))},
                    "eval_f1": {"m": float(np.mean(f1)), "s": float(np.std(f1))},
                }
                print(f"CV={np.mean(cv):.3f} Eval={np.mean(ev):.3f}")

        del sae
        torch.cuda.empty_cache()

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    print(f"\n{sep}\nBEST CONFIGURATIONS\n{sep}")

    for mname in ["base", "trained"]:
        mr = {k: v for k, v in results["experiments"].items() if v["model"] == mname}
        if mr:
            best_k = max(mr, key=lambda k: mr[k]["eval_auroc"]["m"])
            b = mr[best_k]
            print(f"\n{mname.upper()}: L{b['layer']} {b['clf']}")
            print(f"  AUROC: {b['eval_auroc']['m']:.4f} +/- {b['eval_auroc']['s']:.4f}")
            print(f"  Acc:   {b['eval_acc']['m']:.4f} +/- {b['eval_acc']['s']:.4f}")
            results[f"best_{mname}"] = b

    # Statistical comparison
    print(f"\n{sep}\nSTATISTICAL COMPARISON\n{sep}")

    comps = []
    for layer in CONFIG["layers_to_test"]:
        for clf in CLASSIFIERS:
            bk = f"l{layer}_base_{clf}"
            tk = f"l{layer}_trained_{clf}"
            if bk in results["experiments"] and tk in results["experiments"]:
                base_auc = results["experiments"][bk]["eval_auroc"]["m"]
                trained_auc = results["experiments"][tk]["eval_auroc"]["m"]
                comps.append({"layer": layer, "clf": clf, "base": base_auc, "trained": trained_auc, "diff": trained_auc - base_auc})

    if comps:
        diffs = [c["diff"] for c in comps]
        mean_d = np.mean(diffs)
        std_d = np.std(diffs)
        t_stat, t_p = stats.ttest_1samp(diffs, 0)
        w_stat, w_p = stats.wilcoxon(diffs) if len(diffs) >= 8 else (0, 1)
        cohens_d = mean_d / std_d if std_d > 0 else 0

        print(f"\nN={len(comps)} comparisons")
        print(f"Mean diff (trained-base): {mean_d:+.4f} +/- {std_d:.4f}")
        print(f"t-test p={t_p:.6f} {'*' if t_p < 0.05 else ''}")
        print(f"Wilcoxon p={w_p:.6f} {'*' if w_p < 0.05 else ''}")
        print(f"Cohen's d: {cohens_d:.3f}")

        # By layer
        print("\nBy Layer:")
        for layer in CONFIG["layers_to_test"]:
            ld = [c["diff"] for c in comps if c["layer"] == layer]
            if ld: print(f"  L{layer:2d}: {np.mean(ld):+.4f}")

        results["stats"] = {
            "mean_diff": float(mean_d), "std_diff": float(std_d),
            "t_p": float(t_p), "w_p": float(w_p), "cohens_d": float(cohens_d),
            "n": len(comps), "comparisons": comps,
        }

    # Layer profile
    print(f"\n{sep}\nLAYER PROFILE (best clf)\n{sep}")
    print(f"{'Layer':>6} {'Base':>8} {'Trained':>8} {'Diff':>8}")
    print("-" * 35)

    for layer in CONFIG["layers_to_test"]:
        base_r = {k: v for k, v in results["experiments"].items() if v["layer"] == layer and v["model"] == "base"}
        trained_r = {k: v for k, v in results["experiments"].items() if v["layer"] == layer and v["model"] == "trained"}
        if base_r and trained_r:
            bb = max(base_r.values(), key=lambda x: x["eval_auroc"]["m"])
            bt = max(trained_r.values(), key=lambda x: x["eval_auroc"]["m"])
            diff = bt["eval_auroc"]["m"] - bb["eval_auroc"]["m"]
            print(f"{layer:>6} {bb['eval_auroc']['m']:>8.4f} {bt['eval_auroc']['m']:>8.4f} {diff:>+8.4f}")

    # Save
    with open(os.path.expanduser("~/llamascope_v3_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{sep}")
    print(f"Saved: ~/llamascope_v3_results.json")
    print(f"Done: {datetime.now().isoformat()}")
    print(sep)

if __name__ == "__main__":
    main()
