#!/usr/bin/env python3
"""
Systematic LlamaScope SAE Analysis for Alignment Faking Detection
===================================================================
A rigorous experimental framework following DeepMind research standards.

Experiments:
1. Layer sweep (16 layers: 0,2,4,...,30)
2. Multiple classifiers (LogReg, SVM-RBF, RandomForest, MLP)
3. 5-fold stratified cross-validation
4. Multiple random seeds (5) for statistical significance
5. BASE vs TRAINED model comparison
6. Statistical significance testing

Author: Vincent Oh
Date: 2026-01-05
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
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
    "n_seeds": 5,
    "layers_to_test": list(range(0, 32, 2)),  # 16 layers: 0,2,4,...,30
    "sae_release": "llama_scope_lxm_32x",
    "base_model": "meta-llama/Llama-3.1-8B",
    "trained_model": "vincentoh/llama-8b-af-detector",
}

CLASSIFIERS = {
    "logreg": lambda: LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    "logreg_l1": lambda: LogisticRegression(max_iter=1000, C=1.0, solver="saga", penalty="l1"),
    "svm_rbf": lambda: SVC(kernel="rbf", probability=True, C=1.0),
    "svm_linear": lambda: SVC(kernel="linear", probability=True, C=1.0),
    "rf": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True),
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_datasets() -> Tuple[List[dict], List[dict]]:
    """Load training and evaluation datasets."""
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

    print(f"  Train: {len(train_samples)} samples")
    print(f"  Eval:  {len(eval_samples)} samples")

    train_labels = [s["label"] for s in train_samples]
    eval_labels = [s["label"] for s in eval_samples]
    print(f"  Train dist: aligned={train_labels.count('aligned')}, faking={train_labels.count('potential_faking')}")
    print(f"  Eval dist:  aligned={eval_labels.count('aligned')}, faking={eval_labels.count('potential_faking')}")

    return train_samples, eval_samples

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_base_model():
    """Load base Llama-3.1-8B."""
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
    """Load trained AF detector (LoRA)."""
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
    """Load LlamaScope SAE for given layer."""
    sae, _, _ = SAE.from_pretrained(
        release=CONFIG["sae_release"],
        sae_id=f"l{layer}m_32x",
        device=CONFIG["device"]
    )
    return sae

def extract_activations(model, tokenizer, texts: List[str], layer: int, is_peft: bool = False) -> torch.Tensor:
    """Extract MLP activations for given layer."""
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
    """Extract SAE features for all samples."""
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
    """Compute comprehensive metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }

def run_cv_experiment(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, classifier_name: str, seed: int) -> Dict:
    """Run cross-validation experiment with given classifier."""
    clf_fn = CLASSIFIERS[classifier_name]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=seed)

    cv_scores = []
    for train_idx, val_idx in skf.split(X_train_scaled, y_train):
        clf = clf_fn()
        clf.fit(X_train_scaled[train_idx], y_train[train_idx])

        if hasattr(clf, "predict_proba"):
            val_prob = clf.predict_proba(X_train_scaled[val_idx])[:, 1]
        else:
            val_prob = clf.decision_function(X_train_scaled[val_idx])

        cv_scores.append(roc_auc_score(y_train[val_idx], val_prob))

    clf = clf_fn()
    clf.fit(X_train_scaled, y_train)

    if hasattr(clf, "predict_proba"):
        eval_prob = clf.predict_proba(X_eval_scaled)[:, 1]
    else:
        eval_prob = clf.decision_function(X_eval_scaled)

    eval_pred = (eval_prob > 0.5).astype(int)

    return {
        "cv_auroc_mean": float(np.mean(cv_scores)),
        "cv_auroc_std": float(np.std(cv_scores)),
        "eval_metrics": compute_metrics(y_eval, eval_pred, eval_prob),
    }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_systematic_experiment():
    """Run the full systematic experiment."""
    sep = "=" * 70
    print(sep)
    print("SYSTEMATIC LLAMASCOPE SAE ANALYSIS")
    print(sep)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Layers: {CONFIG['layers_to_test']}")
    print(f"Classifiers: {list(CLASSIFIERS.keys())}")
    print(f"Seeds: {CONFIG['n_seeds']}, Folds: {CONFIG['n_folds']}")
    print()

    train_samples, eval_samples = load_datasets()

    results = {
        "config": {k: v for k, v in CONFIG.items() if k != "device"},
        "classifiers": list(CLASSIFIERS.keys()),
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }

    # Load models once
    base_model, base_tok = load_base_model()
    trained_model, trained_tok = load_trained_model()

    models = {
        "base": (base_model, base_tok, False),
        "trained": (trained_model, trained_tok, True),
    }

    # Main experiment loop
    for layer in CONFIG["layers_to_test"]:
        print(f"\n{sep}")
        print(f"LAYER {layer}")
        print(sep)

        try:
            sae = load_sae(layer)
            print(f"  SAE loaded: {sae.cfg.d_sae} features")
        except Exception as e:
            print(f"  Failed to load SAE: {e}")
            continue

        for model_name, (model, tok, is_peft) in models.items():
            print(f"\n  Model: {model_name.upper()}")
            print("    Extracting features...")

            X_train, y_train = get_sae_features(model, tok, train_samples, layer, sae, is_peft)
            X_eval, y_eval = get_sae_features(model, tok, eval_samples, layer, sae, is_peft)

            print(f"    Train: {X_train.shape}, Eval: {X_eval.shape}")
            print(f"    Feature sparsity: {(X_train == 0).mean():.2%}")

            for clf_name in CLASSIFIERS.keys():
                print(f"    {clf_name}:", end=" ", flush=True)

                seed_results = []
                for seed in range(CONFIG["n_seeds"]):
                    try:
                        res = run_cv_experiment(X_train, y_train, X_eval, y_eval, clf_name, seed)
                        seed_results.append(res)
                    except Exception as e:
                        print(f"[seed {seed} failed: {e}]", end=" ")
                        continue

                if not seed_results:
                    print("FAILED")
                    continue

                cv_aurocs = [r["cv_auroc_mean"] for r in seed_results]
                eval_aurocs = [r["eval_metrics"]["auroc"] for r in seed_results]
                eval_accs = [r["eval_metrics"]["accuracy"] for r in seed_results]
                eval_f1s = [r["eval_metrics"]["f1"] for r in seed_results]

                key = f"layer{layer}_{model_name}_{clf_name}"
                results["experiments"][key] = {
                    "layer": layer,
                    "model": model_name,
                    "classifier": clf_name,
                    "cv_auroc": {"mean": float(np.mean(cv_aurocs)), "std": float(np.std(cv_aurocs))},
                    "eval_auroc": {"mean": float(np.mean(eval_aurocs)), "std": float(np.std(eval_aurocs))},
                    "eval_accuracy": {"mean": float(np.mean(eval_accs)), "std": float(np.std(eval_accs))},
                    "eval_f1": {"mean": float(np.mean(eval_f1s)), "std": float(np.std(eval_f1s))},
                }

                print(f"CV={np.mean(cv_aurocs):.3f}, Eval={np.mean(eval_aurocs):.3f}")

        del sae
        torch.cuda.empty_cache()

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    print(f"\n{sep}")
    print("ANALYSIS: BEST CONFIGURATIONS")
    print(sep)

    for model_name in ["base", "trained"]:
        model_results = {k: v for k, v in results["experiments"].items() if v["model"] == model_name}
        if model_results:
            best_key = max(model_results.keys(), key=lambda k: model_results[k]["eval_auroc"]["mean"])
            best = model_results[best_key]
            print(f"\n{model_name.upper()} MODEL:")
            print(f"  Best config: Layer {best['layer']}, {best['classifier']}")
            print(f"  Eval AUROC: {best['eval_auroc']['mean']:.4f} +/- {best['eval_auroc']['std']:.4f}")
            print(f"  Eval Accuracy: {best['eval_accuracy']['mean']:.4f} +/- {best['eval_accuracy']['std']:.4f}")
            print(f"  Eval F1: {best['eval_f1']['mean']:.4f} +/- {best['eval_f1']['std']:.4f}")

            results[f"best_{model_name}"] = {
                "layer": best["layer"],
                "classifier": best["classifier"],
                "eval_auroc": best["eval_auroc"],
                "eval_accuracy": best["eval_accuracy"],
                "eval_f1": best["eval_f1"],
            }

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
                    "layer": layer,
                    "classifier": clf_name,
                    "base_auroc": base_auroc,
                    "trained_auroc": trained_auroc,
                    "improvement": trained_auroc - base_auroc,
                })

    if comparisons:
        improvements = [c["improvement"] for c in comparisons]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)

        # One-sample t-test: is mean improvement significantly different from 0?
        t_stat, p_value = stats.ttest_1samp(improvements, 0)

        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pvalue = stats.wilcoxon(improvements)

        print(f"\nN comparisons: {len(comparisons)}")
        print(f"Mean AUROC improvement (trained - base): {mean_imp:.4f} +/- {std_imp:.4f}")
        print(f"\nParametric test (t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant (p < 0.05): {p_value < 0.05}")
        print(f"\nNon-parametric test (Wilcoxon):")
        print(f"  W-statistic: {w_stat:.4f}")
        print(f"  p-value: {w_pvalue:.6f}")
        print(f"  Significant (p < 0.05): {w_pvalue < 0.05}")

        # Effect size (Cohen's d)
        cohens_d = mean_imp / std_imp if std_imp > 0 else 0
        print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")

        # By-layer analysis
        print("\nBy-layer improvement (averaged across classifiers):")
        layer_improvements = {}
        for c in comparisons:
            layer = c["layer"]
            if layer not in layer_improvements:
                layer_improvements[layer] = []
            layer_improvements[layer].append(c["improvement"])

        for layer in sorted(layer_improvements.keys()):
            imps = layer_improvements[layer]
            print(f"  Layer {layer:2d}: {np.mean(imps):+.4f} +/- {np.std(imps):.4f}")

        results["statistical_comparison"] = {
            "mean_improvement": float(mean_imp),
            "std_improvement": float(std_imp),
            "t_statistic": float(t_stat),
            "t_pvalue": float(p_value),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_pvalue": float(w_pvalue),
            "cohens_d": float(cohens_d),
            "n_comparisons": len(comparisons),
            "comparisons": comparisons,
        }

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    output_path = os.path.expanduser("~/llamascope_systematic_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{sep}")
    print(f"Results saved to: {output_path}")
    print(f"Completed: {datetime.now().isoformat()}")
    print(sep)

    return results


if __name__ == "__main__":
    run_systematic_experiment()
