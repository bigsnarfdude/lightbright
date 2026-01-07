#!/usr/bin/env python3
"""
Systematic LlamaScope SAE Analysis v2 - Optimized
==================================================
With dimensionality reduction for tractable computation.

Key optimizations:
- TruncatedSVD: 131K → 2000 features
- Efficient classifiers only (no slow L1/SVM)
- 3 seeds instead of 5
- Sampled training (max 1500 per class)
"""

import os
os.environ["HF_TOKEN"] = "***REMOVED***"

import torch
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
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
# CONFIGURATION
# ============================================================================

CONFIG = {
    "device": "cuda",
    "max_seq_len": 512,
    "batch_size": 8,
    "n_folds": 5,
    "n_seeds": 3,
    "svd_components": 2000,  # Reduce 131K → 2000
    "max_train_per_class": 1500,  # Sample training data
    "layers_to_test": list(range(0, 32, 2)),  # 16 layers
    "sae_release": "llama_scope_lxm_32x",
    "base_model": "meta-llama/Llama-3.1-8B",
    "trained_model": "vincentoh/llama-8b-af-detector",
}

CLASSIFIERS = {
    "logreg": lambda: LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=-1),
    "rf": lambda: RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1),
    "gbdt": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=300, early_stopping=True),
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_datasets() -> Tuple[List[dict], List[dict]]:
    """Load and optionally subsample datasets."""
    print("Loading datasets...")

    train_path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="training_data_final.json",
        repo_type="dataset"
    )
    with open(train_path) as f:
        train_data = json.load(f)
    train_samples = train_data.get("samples", train_data)

    eval_path = hf_hub_download(
        repo_id="vincentoh/alignment-faking-training",
        filename="gold_106.json",
        repo_type="dataset"
    )
    with open(eval_path) as f:
        eval_data = json.load(f)
    eval_samples = eval_data.get("samples", eval_data)

    # Subsample training data for faster classifier fitting
    np.random.seed(42)
    aligned = [s for s in train_samples if s["label"] == "aligned"]
    faking = [s for s in train_samples if s["label"] == "potential_faking"]

    max_per_class = CONFIG["max_train_per_class"]
    if len(aligned) > max_per_class:
        aligned = list(np.random.choice(aligned, max_per_class, replace=False))
    if len(faking) > max_per_class:
        faking = list(np.random.choice(faking, max_per_class, replace=False))

    train_samples = aligned + faking
    np.random.shuffle(train_samples)

    print(f"  Train: {len(train_samples)} (subsampled)")
    print(f"  Eval:  {len(eval_samples)}")
    print(f"  Train dist: aligned={len(aligned)}, faking={len(faking)}")

    return train_samples, eval_samples

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_base_model():
    print("Loading BASE model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def load_trained_model():
    print("Loading TRAINED model...")
    base = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, CONFIG["trained_model"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

# ============================================================================
# SAE FEATURE EXTRACTION
# ============================================================================

def load_sae(layer: int) -> SAE:
    sae, _, _ = SAE.from_pretrained(
        release=CONFIG["sae_release"],
        sae_id=f"l{layer}m_32x",
        device=CONFIG["device"]
    )
    return sae

def extract_activations(model, tokenizer, texts: List[str], layer: int, is_peft: bool = False) -> torch.Tensor:
    all_acts = []
    for i in range(0, len(texts), CONFIG["batch_size"]):
        batch_texts = texts[i:i + CONFIG["batch_size"]]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG["max_seq_len"]
        ).to(CONFIG["device"])

        batch_acts = []
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            batch_acts.append(output.mean(dim=1).detach().cpu())

        if is_peft:
            target = model.base_model.model.model.layers[layer].mlp
        else:
            target = model.model.layers[layer].mlp

        handle = target.register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        all_acts.append(batch_acts[0])

    return torch.cat(all_acts, dim=0)

def get_sae_features(model, tokenizer, samples: List[dict], layer: int, sae: SAE, is_peft: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    texts = [s["text"][:2000] for s in samples]
    labels = np.array([1 if s["label"] == "potential_faking" else 0 for s in samples])

    acts = extract_activations(model, tokenizer, texts, layer, is_peft)
    acts = acts.to(CONFIG["device"]).float()

    with torch.no_grad():
        features = sae.encode(acts)

    return features.cpu().numpy(), labels

# ============================================================================
# EVALUATION
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }

def run_cv_experiment(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, classifier_name: str, seed: int) -> Dict:
    clf_fn = CLASSIFIERS[classifier_name]

    # Standardize + SVD
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    # Dimensionality reduction
    n_components = min(CONFIG["svd_components"], X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    X_train_reduced = svd.fit_transform(X_train_scaled)
    X_eval_reduced = svd.transform(X_eval_scaled)

    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=seed)

    cv_scores = []
    for train_idx, val_idx in skf.split(X_train_reduced, y_train):
        clf = clf_fn()
        clf.fit(X_train_reduced[train_idx], y_train[train_idx])
        if hasattr(clf, "predict_proba"):
            val_prob = clf.predict_proba(X_train_reduced[val_idx])[:, 1]
        else:
            val_prob = clf.decision_function(X_train_reduced[val_idx])
        try:
            cv_scores.append(roc_auc_score(y_train[val_idx], val_prob))
        except:
            pass

    clf = clf_fn()
    clf.fit(X_train_reduced, y_train)
    if hasattr(clf, "predict_proba"):
        eval_prob = clf.predict_proba(X_eval_reduced)[:, 1]
    else:
        eval_prob = clf.decision_function(X_eval_reduced)
    eval_pred = (eval_prob > 0.5).astype(int)

    return {
        "cv_auroc_mean": float(np.mean(cv_scores)) if cv_scores else 0.5,
        "cv_auroc_std": float(np.std(cv_scores)) if cv_scores else 0.0,
        "eval_metrics": compute_metrics(y_eval, eval_pred, eval_prob),
        "svd_variance_explained": float(svd.explained_variance_ratio_.sum()),
    }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_systematic_experiment():
    sep = "=" * 70
    print(sep)
    print("SYSTEMATIC LLAMASCOPE SAE ANALYSIS v2 (Optimized)")
    print(sep)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Layers: {CONFIG['layers_to_test']}")
    print(f"Classifiers: {list(CLASSIFIERS.keys())}")
    print(f"SVD components: {CONFIG['svd_components']}")
    print(f"Seeds: {CONFIG['n_seeds']}, Folds: {CONFIG['n_folds']}")
    print()

    train_samples, eval_samples = load_datasets()

    results = {
        "config": {k: v for k, v in CONFIG.items() if k != "device"},
        "classifiers": list(CLASSIFIERS.keys()),
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }

    base_model, base_tok = load_base_model()
    trained_model, trained_tok = load_trained_model()
    models = {"base": (base_model, base_tok, False), "trained": (trained_model, trained_tok, True)}

    for layer in CONFIG["layers_to_test"]:
        print(f"\n{sep}")
        print(f"LAYER {layer}")
        print(sep)

        try:
            sae = load_sae(layer)
            print(f"  SAE: {sae.cfg.d_sae} features")
        except Exception as e:
            print(f"  SAE load failed: {e}")
            continue

        for model_name, (model, tok, is_peft) in models.items():
            print(f"\n  {model_name.upper()}:")
            X_train, y_train = get_sae_features(model, tok, train_samples, layer, sae, is_peft)
            X_eval, y_eval = get_sae_features(model, tok, eval_samples, layer, sae, is_peft)
            print(f"    Features: train={X_train.shape}, eval={X_eval.shape}")
            print(f"    Sparsity: {(X_train == 0).mean():.1%}")

            for clf_name in CLASSIFIERS.keys():
                print(f"    {clf_name}: ", end="", flush=True)
                seed_results = []
                for seed in range(CONFIG["n_seeds"]):
                    try:
                        res = run_cv_experiment(X_train, y_train, X_eval, y_eval, clf_name, seed)
                        seed_results.append(res)
                    except Exception as e:
                        print(f"[ERR:{seed}] ", end="")

                if not seed_results:
                    print("FAILED")
                    continue

                cv_aurocs = [r["cv_auroc_mean"] for r in seed_results]
                eval_aurocs = [r["eval_metrics"]["auroc"] for r in seed_results]
                eval_accs = [r["eval_metrics"]["accuracy"] for r in seed_results]
                eval_f1s = [r["eval_metrics"]["f1"] for r in seed_results]
                var_explained = np.mean([r["svd_variance_explained"] for r in seed_results])

                key = f"layer{layer}_{model_name}_{clf_name}"
                results["experiments"][key] = {
                    "layer": layer, "model": model_name, "classifier": clf_name,
                    "cv_auroc": {"mean": float(np.mean(cv_aurocs)), "std": float(np.std(cv_aurocs))},
                    "eval_auroc": {"mean": float(np.mean(eval_aurocs)), "std": float(np.std(eval_aurocs))},
                    "eval_accuracy": {"mean": float(np.mean(eval_accs)), "std": float(np.std(eval_accs))},
                    "eval_f1": {"mean": float(np.mean(eval_f1s)), "std": float(np.std(eval_f1s))},
                    "svd_variance_explained": float(var_explained),
                }
                print(f"CV={np.mean(cv_aurocs):.3f} Eval={np.mean(eval_aurocs):.3f}")

        del sae
        torch.cuda.empty_cache()

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    print(f"\n{sep}")
    print("BEST CONFIGURATIONS")
    print(sep)

    for model_name in ["base", "trained"]:
        model_results = {k: v for k, v in results["experiments"].items() if v["model"] == model_name}
        if model_results:
            best_key = max(model_results.keys(), key=lambda k: model_results[k]["eval_auroc"]["mean"])
            best = model_results[best_key]
            print(f"\n{model_name.upper()}:")
            print(f"  Best: Layer {best['layer']}, {best['classifier']}")
            print(f"  AUROC: {best['eval_auroc']['mean']:.4f} +/- {best['eval_auroc']['std']:.4f}")
            print(f"  Acc:   {best['eval_accuracy']['mean']:.4f} +/- {best['eval_accuracy']['std']:.4f}")
            print(f"  F1:    {best['eval_f1']['mean']:.4f} +/- {best['eval_f1']['std']:.4f}")
            results[f"best_{model_name}"] = best

    # ========================================================================
    # STATISTICAL COMPARISON
    # ========================================================================

    print(f"\n{sep}")
    print("STATISTICAL COMPARISON: TRAINED vs BASE")
    print(sep)

    comparisons = []
    for layer in CONFIG["layers_to_test"]:
        for clf_name in CLASSIFIERS.keys():
            base_key = f"layer{layer}_base_{clf_name}"
            trained_key = f"layer{layer}_trained_{clf_name}"
            if base_key in results["experiments"] and trained_key in results["experiments"]:
                base_auroc = results["experiments"][base_key]["eval_auroc"]["mean"]
                trained_auroc = results["experiments"][trained_key]["eval_auroc"]["mean"]
                comparisons.append({
                    "layer": layer, "classifier": clf_name,
                    "base": base_auroc, "trained": trained_auroc,
                    "diff": trained_auroc - base_auroc,
                })

    if comparisons:
        diffs = [c["diff"] for c in comparisons]
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        t_stat, p_value = stats.ttest_1samp(diffs, 0)
        w_stat, w_pvalue = stats.wilcoxon(diffs)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        print(f"\nN comparisons: {len(comparisons)}")
        print(f"Mean improvement: {mean_diff:+.4f} +/- {std_diff:.4f}")
        print(f"t-test p-value: {p_value:.6f} {'*' if p_value < 0.05 else ''}")
        print(f"Wilcoxon p-value: {w_pvalue:.6f} {'*' if w_pvalue < 0.05 else ''}")
        print(f"Cohen's d: {cohens_d:.4f}")

        # Layer-by-layer
        print("\nBy layer (mean diff across classifiers):")
        layer_diffs = {}
        for c in comparisons:
            if c["layer"] not in layer_diffs:
                layer_diffs[c["layer"]] = []
            layer_diffs[c["layer"]].append(c["diff"])
        for layer in sorted(layer_diffs.keys()):
            d = layer_diffs[layer]
            print(f"  L{layer:2d}: {np.mean(d):+.4f} +/- {np.std(d):.4f}")

        results["statistical_comparison"] = {
            "mean_diff": float(mean_diff), "std_diff": float(std_diff),
            "t_pvalue": float(p_value), "wilcoxon_pvalue": float(w_pvalue),
            "cohens_d": float(cohens_d), "n": len(comparisons),
            "comparisons": comparisons,
        }

    # ========================================================================
    # LAYER PROFILE
    # ========================================================================

    print(f"\n{sep}")
    print("LAYER PROFILE (best classifier per layer)")
    print(sep)

    print("\n{:>6} {:>10} {:>10} {:>10}".format("Layer", "Base", "Trained", "Diff"))
    print("-" * 40)

    for layer in CONFIG["layers_to_test"]:
        base_results = {k: v for k, v in results["experiments"].items()
                       if v["layer"] == layer and v["model"] == "base"}
        trained_results = {k: v for k, v in results["experiments"].items()
                         if v["layer"] == layer and v["model"] == "trained"}

        if base_results and trained_results:
            best_base = max(base_results.values(), key=lambda x: x["eval_auroc"]["mean"])
            best_trained = max(trained_results.values(), key=lambda x: x["eval_auroc"]["mean"])
            diff = best_trained["eval_auroc"]["mean"] - best_base["eval_auroc"]["mean"]
            print(f"{layer:>6} {best_base['eval_auroc']['mean']:>10.4f} {best_trained['eval_auroc']['mean']:>10.4f} {diff:>+10.4f}")

    # Save
    with open(os.path.expanduser("~/llamascope_systematic_v2.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{sep}")
    print(f"Saved: ~/llamascope_systematic_v2.json")
    print(f"Done: {datetime.now().isoformat()}")
    print(sep)

    return results

if __name__ == "__main__":
    run_systematic_experiment()
