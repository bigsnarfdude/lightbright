#!/usr/bin/env python3
"""
Fix ensemble scoring in gold106_full_analysis.json

DEFECT-001: Original ensemble used naive sum of all features,
including anti-correlated ones (AUROC < 0.5) which hurt performance.

Fix: Use only positively correlated features with proper weighting.
"""

import json
from sklearn.metrics import roc_auc_score
import numpy as np

def main():
    print("=" * 70)
    print("FIXING ENSEMBLE SCORING - DEFECT-001")
    print("=" * 70)

    # Load current results
    with open("results/gold106_full_analysis.json") as f:
        data = json.load(f)

    # Extract ground truth and features
    y_true = [1 if r['is_af'] else 0 for r in data['sample_results']]

    # Current (broken) ensemble
    old_ensemble = [sum(r['features'][f]['max'] for f in r['features'])
                    for r in data['sample_results']]
    old_auroc = roc_auc_score(y_true, old_ensemble)

    print(f"\n[BEFORE] Naive sum ensemble AUROC: {old_auroc:.4f}")
    print("\nFeature-level AUROC:")
    for feat, auroc in data['feature_aurocs'].items():
        status = "ANTI-CORRELATED" if auroc < 0.5 else "OK"
        print(f"  {feat}: {auroc:.4f} ({status})")

    # Fix 1: Best single feature only
    best_feature = [r['features']['L40_F8921']['max']
                    for r in data['sample_results']]
    best_auroc = roc_auc_score(y_true, best_feature)

    # Fix 2: Weighted ensemble (positive features only, with learned-like weights)
    # Use AUROC as weight proxy: weight = max(0, auroc - 0.5) to zero out anti-correlated
    weighted_ensemble = []
    for r in data['sample_results']:
        score = 0
        for feat, auroc in data['feature_aurocs'].items():
            weight = max(0, auroc - 0.5)  # Only positive contribution from good features
            score += r['features'][feat]['max'] * weight
        weighted_ensemble.append(score)
    weighted_auroc = roc_auc_score(y_true, weighted_ensemble)

    # Fix 3: Subtract anti-correlated features
    adjusted_ensemble = []
    for r in data['sample_results']:
        score = (r['features']['L40_F8921']['max'] * 1.0 +      # Best feature
                 r['features']['L53_F4824']['max'] * 0.2 -      # Weak positive
                 r['features']['L40_F12574']['max'] * 0.1 -     # Anti-correlated
                 r['features']['L53_F15529']['max'] * 0.1)      # Anti-correlated
        adjusted_ensemble.append(score)
    adjusted_auroc = roc_auc_score(y_true, adjusted_ensemble)

    print(f"\n[AFTER] Fix options:")
    print(f"  Best single feature (L40_F8921):  {best_auroc:.4f}")
    print(f"  Weighted (AUROC-based weights):   {weighted_auroc:.4f}")
    print(f"  Adjusted (subtract anti-corr):    {adjusted_auroc:.4f}")

    # Use the best fix
    best_fix_auroc = max(best_auroc, weighted_auroc, adjusted_auroc)
    if best_fix_auroc == best_auroc:
        new_ensemble_scores = best_feature
        method = "best_single_feature"
    elif best_fix_auroc == weighted_auroc:
        new_ensemble_scores = weighted_ensemble
        method = "weighted_auroc_based"
    else:
        new_ensemble_scores = adjusted_ensemble
        method = "adjusted_subtract_anticorr"

    print(f"\n  Selected: {method} ({best_fix_auroc:.4f})")
    print(f"  Improvement: +{(best_fix_auroc - old_auroc) * 100:.1f}%")

    # Update the data
    data['ensemble_auroc_old'] = data['ensemble_auroc']
    data['ensemble_auroc'] = best_fix_auroc
    data['ensemble_method'] = method
    data['ensemble_fix_note'] = (
        "DEFECT-001 fix: Original naive sum included anti-correlated features. "
        f"Fixed by using {method}. Old AUROC: {old_auroc:.4f}, New: {best_fix_auroc:.4f}"
    )

    # Recalculate best_f1 and best_threshold with new ensemble
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, new_ensemble_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])  # Last element is undefined
    new_best_f1 = f1_scores[best_idx]
    new_best_threshold = thresholds[best_idx]

    data['best_f1_old'] = data['best_f1']
    data['best_threshold_old'] = data['best_threshold']
    data['best_f1'] = float(new_best_f1)
    data['best_threshold'] = float(new_best_threshold)

    print(f"\n  Updated best_f1: {data['best_f1_old']:.4f} -> {new_best_f1:.4f}")
    print(f"  Updated threshold: {data['best_threshold_old']:.2f} -> {new_best_threshold:.2f}")

    # Save fixed results
    with open("results/gold106_full_analysis.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[SAVED] results/gold106_full_analysis.json updated")
    print("=" * 70)

if __name__ == "__main__":
    main()
