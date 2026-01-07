#!/usr/bin/env python3
"""
Experiment 7: Full 50k Feature Sweep
=====================================
Sweep ALL features, not just top 8 by correlation.
Find the true needles in the haystack.

Time estimate: ~8-10 hours on H100
Cost estimate: ~$20-25 on cloud GPU rental

Usage:
    python sweep_all_features.py --layer 40          # One layer (~3 hrs)
    python sweep_all_features.py --all-layers        # All layers (~10 hrs)
    python sweep_all_features.py --resume results/   # Resume interrupted run
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    "model_name": "google/gemma-3-27b-it",
    "sae_release": "gemma-scope-2-27b-it-resid_post",
    # Gemma 3 27B has 62 layers. GemmaScope 2 provides SAEs at 25%, 50%, 65%, 85% depth
    "layers": [15, 31, 40, 53],
    "sae_width": "16k",
    "sae_l0": "medium",
    "n_features": 16384,  # Features per layer
    "device": "cuda",
    "max_length": 2048,
    "dataset_repo": "vincentoh/af-detection-benchmark",
    "checkpoint_every": 1000,  # Save progress every N features
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(cache_dir="./data"):
    """Load the 5-class benchmark."""
    print("📥 Loading dataset...")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    files = [
        "aligned.json",
        "alignment_faking.json",
        "genuine_compliance.json", 
        "hard_negative.json",
        "compliant_hard_negative.json"
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
    
    print(f"  ✓ Loaded {len(all_samples)} samples")
    return all_samples


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load Gemma-27B."""
    print("🔧 Loading Gemma-27B...")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("  ✓ Model loaded")
    return model, tokenizer


def load_sae(layer):
    """Load GemmaScope 2 SAE for one layer."""
    from sae_lens import SAE
    
    sae, _, _ = SAE.from_pretrained(
        release=CONFIG["sae_release"],
        sae_id=f"layer_{layer}_width_{CONFIG['sae_width']}_l0_{CONFIG['sae_l0']}",
        device=CONFIG["device"]
    )
    return sae


# ============================================================================
# FEATURE EXTRACTION (BATCHED FOR SPEED)
# ============================================================================

def extract_all_features_for_layer(model, tokenizer, samples, layer, sae):
    """
    Extract max activation for ALL 16k features for ALL samples.
    Returns: [n_samples, 16384] matrix
    """
    print(f"\n📊 Extracting all features for layer {layer}...")
    
    n_samples = len(samples)
    n_features = CONFIG["n_features"]
    
    # Store max activation per feature per sample
    all_max_activations = np.zeros((n_samples, n_features), dtype=np.float32)
    
    for idx, sample in enumerate(tqdm(samples, desc=f"Layer {layer}")):
        # Hook to capture activations
        activations = {}
        
        def hook(module, input, output):
            activations["out"] = output[0].detach()
        
        handle = model.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(hook)
        
        # Forward pass
        inputs = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"]
        ).to(CONFIG["device"])
        
        with torch.no_grad():
            model(**inputs)
        
        handle.remove()
        
        # Get SAE features
        act = activations["out"].squeeze(0)  # [seq_len, hidden_dim]
        
        with torch.no_grad():
            features = sae.encode(act)  # [seq_len, n_features]
        
        # Max across sequence
        max_per_feature = features.max(dim=0).values.cpu().numpy()
        all_max_activations[idx] = max_per_feature
        
        # Clear memory periodically
        if idx % 100 == 0:
            torch.cuda.empty_cache()
    
    return all_max_activations


# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

def analyze_all_features(activations, samples, output_dir, layer):
    """
    Compute discriminative power for ALL features.
    """
    print(f"\n🔬 Analyzing {activations.shape[1]} features...")
    
    # Labels
    y_af = np.array([s["is_af"] for s in samples])
    y_hard_neg = np.array([s["is_hard_neg"] for s in samples])
    
    # Masks
    af_mask = y_af == 1
    aligned_mask = y_af == 0
    hard_neg_mask = y_hard_neg == 1
    
    results = []
    
    for feat_idx in tqdm(range(activations.shape[1]), desc="Computing stats"):
        feat_vals = activations[:, feat_idx]
        
        # Skip dead features
        if feat_vals.max() == 0:
            continue
        
        # Basic stats
        af_mean = feat_vals[af_mask].mean()
        aligned_mean = feat_vals[aligned_mask].mean()
        hard_neg_mean = feat_vals[hard_neg_mask].mean()
        
        # Ratio (avoid div by zero)
        ratio_vs_aligned = af_mean / (aligned_mean + 1e-6)
        ratio_vs_hard_neg = af_mean / (hard_neg_mean + 1e-6)
        
        # AUROC for AF vs all others
        try:
            auroc_af = roc_auc_score(y_af, feat_vals)
        except:
            auroc_af = 0.5
        
        # AUROC for AF vs hard negatives only
        af_hn_mask = af_mask | hard_neg_mask
        if af_hn_mask.sum() > 10:
            try:
                auroc_af_hn = roc_auc_score(y_af[af_hn_mask], feat_vals[af_hn_mask])
            except:
                auroc_af_hn = 0.5
        else:
            auroc_af_hn = 0.5
        
        # T-test: AF vs hard negatives
        if hard_neg_mask.sum() > 5 and af_mask.sum() > 5:
            t_stat, p_val = stats.ttest_ind(feat_vals[af_mask], feat_vals[hard_neg_mask])
        else:
            t_stat, p_val = 0, 1
        
        results.append({
            "feature_idx": int(feat_idx),
            "af_mean": float(af_mean),
            "aligned_mean": float(aligned_mean),
            "hard_neg_mean": float(hard_neg_mean),
            "ratio_vs_aligned": float(ratio_vs_aligned),
            "ratio_vs_hard_neg": float(ratio_vs_hard_neg),
            "auroc_af_vs_all": float(auroc_af),
            "auroc_af_vs_hard_neg": float(auroc_af_hn),
            "ttest_pval": float(p_val),
        })
    
    # Sort by different criteria
    results_by_auroc = sorted(results, key=lambda x: x["auroc_af_vs_all"], reverse=True)
    results_by_auroc_hn = sorted(results, key=lambda x: x["auroc_af_vs_hard_neg"], reverse=True)
    results_by_ratio = sorted(results, key=lambda x: x["ratio_vs_aligned"], reverse=True)
    
    # Summary
    summary = {
        "layer": layer,
        "total_features": activations.shape[1],
        "active_features": len(results),
        "top_20_by_auroc_overall": results_by_auroc[:20],
        "top_20_by_auroc_vs_hard_neg": results_by_auroc_hn[:20],
        "top_20_by_ratio": results_by_ratio[:20],
        "all_features": results,  # Full data
    }
    
    # Print findings
    print(f"\n📈 Layer {layer} Results:")
    print(f"   Active features: {len(results)}/{activations.shape[1]}")
    
    print(f"\n   Top 5 by AUROC (AF vs ALL):")
    for f in results_by_auroc[:5]:
        print(f"     F{f['feature_idx']}: AUROC={f['auroc_af_vs_all']:.3f}, ratio={f['ratio_vs_aligned']:.1f}x")
    
    print(f"\n   Top 5 by AUROC (AF vs HARD NEGATIVES):")
    for f in results_by_auroc_hn[:5]:
        print(f"     F{f['feature_idx']}: AUROC={f['auroc_af_vs_hard_neg']:.3f}, ratio={f['ratio_vs_hard_neg']:.1f}x")
    
    # Save
    out_path = os.path.join(output_dir, f"sweep_layer_{layer}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   💾 Saved to {out_path}")
    
    return summary


# ============================================================================
# FIND TRUE NEEDLES
# ============================================================================

def find_needles(all_layer_results, output_dir):
    """
    Cross-reference all layers to find features that:
    1. Discriminate AF vs aligned (high AUROC overall)
    2. ALSO discriminate AF vs hard negatives (not just keywords)
    """
    print("\n" + "="*70)
    print("🎯 FINDING TRUE NEEDLES")
    print("="*70)
    
    needles = []
    
    for layer, results in all_layer_results.items():
        for feat in results["all_features"]:
            # Criteria for "true needle":
            # - AUROC AF vs all > 0.7
            # - AUROC AF vs hard neg > 0.65 (harder test!)
            # - p-value < 0.01
            
            if (feat["auroc_af_vs_all"] > 0.7 and 
                feat["auroc_af_vs_hard_neg"] > 0.65 and
                feat["ttest_pval"] < 0.01):
                
                needles.append({
                    "layer": layer,
                    **feat
                })
    
    # Sort by hard negative AUROC (the real test)
    needles = sorted(needles, key=lambda x: x["auroc_af_vs_hard_neg"], reverse=True)
    
    print(f"\n Found {len(needles)} true needles (pass both AUROC thresholds)")
    
    if needles:
        print(f"\n Top 10 TRUE NEEDLES (discriminate AF from hard negatives):")
        print("-"*70)
        for n in needles[:10]:
            print(f"  L{n['layer']}_F{n['feature_idx']:5d}: "
                  f"AUROC(all)={n['auroc_af_vs_all']:.3f}, "
                  f"AUROC(hn)={n['auroc_af_vs_hard_neg']:.3f}, "
                  f"p={n['ttest_pval']:.2e}")
    else:
        print("\n ⚠️ No features passed both thresholds!")
        print("   This suggests features may be confounded by surface patterns.")
    
    # Save
    out_path = os.path.join(output_dir, "true_needles.json")
    with open(out_path, "w") as f:
        json.dump({
            "total_needles": len(needles),
            "criteria": {
                "auroc_af_vs_all": "> 0.7",
                "auroc_af_vs_hard_neg": "> 0.65",
                "ttest_pval": "< 0.01"
            },
            "needles": needles
        }, f, indent=2)
    
    print(f"\n💾 Saved to {out_path}")
    
    return needles


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, help="Single layer to sweep")
    parser.add_argument("--all-layers", action="store_true", help="Sweep all layers")
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint directory")
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"sweep_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("🔬 FULL 50K FEATURE SWEEP")
    print("="*70)
    print(f"Output: {output_dir}")
    
    # Load data
    samples = load_dataset()
    
    # Load model
    model, tokenizer = load_model()
    
    # Determine layers to sweep
    if args.layer:
        layers = [args.layer]
    elif args.all_layers:
        layers = CONFIG["layers"]
    else:
        layers = [40]  # Default: just layer 40
        print(f"⚠️ No layer specified, defaulting to layer 40")
        print(f"   Use --all-layers for full sweep")
    
    # Sweep each layer
    all_results = {}
    
    for layer in layers:
        print(f"\n{'='*70}")
        print(f"LAYER {layer}")
        print("="*70)
        
        # Load SAE for this layer
        print(f"Loading SAE for layer {layer}...")
        sae = load_sae(layer)
        
        # Extract features
        activations = extract_all_features_for_layer(
            model, tokenizer, samples, layer, sae
        )
        
        # Save raw activations (for later analysis)
        np.save(os.path.join(output_dir, f"activations_layer_{layer}.npy"), activations)
        
        # Analyze
        results = analyze_all_features(activations, samples, output_dir, layer)
        all_results[layer] = results
        
        # Free memory
        del sae
        torch.cuda.empty_cache()
    
    # Find true needles across all layers
    if len(all_results) > 0:
        needles = find_needles(all_results, output_dir)
    
    print("\n" + "="*70)
    print("✅ SWEEP COMPLETE")
    print("="*70)
    print(f"Results: {output_dir}")
    print(f"\nKey files:")
    print(f"  - sweep_layer_*.json: Per-layer analysis")
    print(f"  - true_needles.json: Features that pass hard negative test")
    print(f"  - activations_layer_*.npy: Raw data for further analysis")


if __name__ == "__main__":
    main()
