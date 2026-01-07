#!/usr/bin/env python3
"""Test probe ablation with synthetic and real data."""

import json
import numpy as np
import pickle

# Load probe and scaler
with open('probe_4b.pkl', 'rb') as f:
    data = pickle.load(f)

probe = data['probe']
scaler = data['scaler']

# Load results
with open('probe_4b_results.json') as f:
    results = json.load(f)

coef = np.array(results['probe_coef']).flatten()

print("=" * 60)
print("PROBE ABLATION TEST")
print("=" * 60)

# Generate a synthetic AF-like feature vector
X_af = scaler.mean_.copy()

# Add signal in direction of AF classification
for i in range(len(coef)):
    if coef[i] > 0:
        X_af[i] += scaler.scale_[i] * 2  # +2 std
    else:
        X_af[i] -= scaler.scale_[i] * 2  # -2 std

X_af_scaled = scaler.transform(X_af.reshape(1, -1))
baseline_pred = probe.predict(X_af_scaled)[0]
baseline_prob = probe.predict_proba(X_af_scaled)[0, 1]

print(f"\nSynthetic AF sample (2 std from mean):")
print(f"  Baseline: class={baseline_pred}, prob={baseline_prob:.4f}")

# Get top features
top_idx = np.argsort(np.abs(coef))[-50:][::-1]
layers = [9, 17, 22, 29]

print("\n--- Ablating top 10 features individually ---")
for idx in top_idx[:10]:
    X_ablated = X_af_scaled.copy()
    X_ablated[0, idx] = 0

    pred = probe.predict(X_ablated)[0]
    prob = probe.predict_proba(X_ablated)[0, 1]

    layer_idx = idx // 150
    layer = layers[layer_idx]
    delta = prob - baseline_prob
    flip = "FLIP!" if pred != baseline_pred else ""
    print(f"  idx {idx:3d} (L{layer:2d}): prob={prob:.4f} (delta={delta:+.4f}) {flip}")

# Ablate ALL top 20 features
X_ablated = X_af_scaled.copy()
for idx in top_idx[:20]:
    X_ablated[0, idx] = 0

pred = probe.predict(X_ablated)[0]
prob = probe.predict_proba(X_ablated)[0, 1]
delta = prob - baseline_prob
flip = "FLIP!" if pred != baseline_pred else ""
print(f"\n--- Ablate top 20 together ---")
print(f"  prob={prob:.4f} (delta={delta:+.4f}) {flip}")

# Ablate top 50
X_ablated = X_af_scaled.copy()
for idx in top_idx[:50]:
    X_ablated[0, idx] = 0

pred = probe.predict(X_ablated)[0]
prob = probe.predict_proba(X_ablated)[0, 1]
delta = prob - baseline_prob
flip = "FLIP!" if pred != baseline_pred else ""
print(f"\n--- Ablate top 50 together ---")
print(f"  prob={prob:.4f} (delta={delta:+.4f}) {flip}")

# Ablate ALL features
X_ablated = np.zeros_like(X_af_scaled)
pred = probe.predict(X_ablated)[0]
prob = probe.predict_proba(X_ablated)[0, 1]
flip = "FLIP!" if pred != baseline_pred else ""
print(f"\n--- Ablate ALL 600 features ---")
print(f"  prob={prob:.4f} {flip}")

# FLIP features to opposite direction (anti-steering)
print("\n--- FLIP top features to OPPOSE AF ---")

# Start with mean (scaled = 0)
X_flipped = np.zeros((1, 600))

# Set top features to OPPOSITE sign (if coef > 0, set feature negative)
for idx in top_idx[:20]:
    if coef[idx] > 0:
        X_flipped[0, idx] = -2  # Push negative (reduces AF prob)
    else:
        X_flipped[0, idx] = 2   # Push positive (reduces AF prob since coef is negative)

pred = probe.predict(X_flipped)[0]
prob = probe.predict_proba(X_flipped)[0, 1]
flip = "FLIP!" if pred == 0 else ""
print(f"  Top 20 flipped: prob={prob:.4f} {flip}")

# Flip more features
X_flipped = np.zeros((1, 600))
for idx in top_idx[:50]:
    if coef[idx] > 0:
        X_flipped[0, idx] = -2
    else:
        X_flipped[0, idx] = 2

pred = probe.predict(X_flipped)[0]
prob = probe.predict_proba(X_flipped)[0, 1]
flip = "FLIP!" if pred == 0 else ""
print(f"  Top 50 flipped: prob={prob:.4f} {flip}")

# Flip ALL to oppose
X_flipped = np.zeros((1, 600))
for i in range(600):
    if coef[i] > 0:
        X_flipped[0, i] = -2
    else:
        X_flipped[0, i] = 2

pred = probe.predict(X_flipped)[0]
prob = probe.predict_proba(X_flipped)[0, 1]
flip = "FLIP!" if pred == 0 else ""
print(f"  ALL 600 flipped: prob={prob:.4f} {flip}")

# Test with intercept only
print(f"\n--- Intercept only (no features) ---")
print(f"  Intercept = {probe.intercept_[0]:.4f}")
print(f"  Sigmoid(intercept) = {1/(1+np.exp(-probe.intercept_[0])):.4f}")
print(f"  This is the 'default' prediction when all features are at mean")
