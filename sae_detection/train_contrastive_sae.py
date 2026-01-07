#!/usr/bin/env python3
"""
Train Contrastive SAE on Real Minimal Pair Activations

1. Extract activations from minimal pairs using Gemma-27B
2. Train standard SAE (baseline)
3. Train contrastive SAE (with SupCon loss)
4. Compare separation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics.pairwise import cosine_similarity

MODEL_ID = "google/gemma-2-27b-it"  # Must match SAE (gemma-scope-2)
SAE_REPO = "google/gemma-scope-2-27b-it"
DEVICE = "cuda"

# Layer to extract activations from (best performing in probe)
TARGET_LAYER = 40


class Transcoder:
    def __init__(self, repo_id: str, layer: int):
        folder = f"transcoder/layer_{layer}_width_16k_l0_medium"
        config_path = hf_hub_download(repo_id, f"{folder}/config.json")
        weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
        with open(config_path) as f:
            self.config = json.load(f)
        weights = load_file(weights_path)
        self.w_enc = weights["w_enc"]
        self.b_enc = weights["b_enc"]
        self.threshold = weights["threshold"]

    def to(self, device, dtype=None):
        self.w_enc = self.w_enc.to(device, dtype=dtype)
        self.b_enc = self.b_enc.to(device, dtype=dtype)
        self.threshold = self.threshold.to(device, dtype=dtype)
        return self

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        return F.relu(pre_acts - self.threshold)


class ContrastiveSAE(nn.Module):
    """SAE with contrastive objective."""

    def __init__(self, input_dim, latent_dim, sparsity_weight=1e-3, contrast_weight=0.1, temperature=0.07):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight
        self.contrast_weight = contrast_weight
        self.temperature = temperature

        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(latent_dim))

    def encode(self, x):
        pre_acts = self.encoder(x)
        return F.relu(pre_acts - self.threshold)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def supervised_contrastive_loss(self, z, labels):
        batch_size = z.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)

        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.temperature

        labels = labels.view(-1, 1)
        mask_same = (labels == labels.T).float()
        mask_diag = torch.eye(batch_size, device=z.device)
        mask_same = mask_same * (1 - mask_diag)

        exp_sim = torch.exp(sim_matrix) * (1 - mask_diag)
        pos_sim = (exp_sim * mask_same).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)

        n_positives = mask_same.sum(dim=1).clamp(min=1)
        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
        loss = (loss * (n_positives > 0)).sum() / (n_positives > 0).sum().clamp(min=1)

        return loss

    def compute_loss(self, x, labels):
        x_hat, z = self.forward(x)

        l_recon = F.mse_loss(x_hat, x)
        l_sparse = z.abs().mean()
        l_contrast = self.supervised_contrastive_loss(z, labels)

        total = l_recon + self.sparsity_weight * l_sparse + self.contrast_weight * l_contrast

        return {
            'total': total,
            'reconstruction': l_recon,
            'sparsity': l_sparse,
            'contrastive': l_contrast,
        }


def extract_activations(model, tokenizer, text, layer):
    """Extract mean activation at specified layer."""
    activations = {}

    def hook(module, input, output):
        activations['out'] = output.detach()

    try:
        target = model.model.language_model.layers[layer]
    except AttributeError:
        target = model.base_model.model.model.language_model.layers[layer]

    handle = target.pre_feedforward_layernorm.register_forward_hook(hook)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        model(**inputs)

    handle.remove()

    # Mean over sequence positions
    acts = activations['out'].squeeze(0).float().mean(dim=0)
    return acts


def evaluate_separation(model, activations, labels):
    """Evaluate class separation in SAE feature space."""
    model.eval()
    with torch.no_grad():
        _, z = model(activations)
        z = z.cpu().numpy()

    af_mask = labels.cpu().numpy() == 1
    genuine_mask = ~af_mask

    z_af = z[af_mask]
    z_genuine = z[genuine_mask]

    # Mean vectors
    af_mean = z_af.mean(axis=0)
    genuine_mean = z_genuine.mean(axis=0)

    # Cosine similarity between means
    mean_sim = np.dot(af_mean, genuine_mean) / (
        np.linalg.norm(af_mean) * np.linalg.norm(genuine_mean) + 1e-8
    )

    # Within-pair similarity (for minimal pairs)
    pair_sims = []
    for i in range(len(z) // 2):
        af_idx = i * 2
        genuine_idx = i * 2 + 1
        sim = cosine_similarity(z[af_idx:af_idx+1], z[genuine_idx:genuine_idx+1])[0, 0]
        pair_sims.append(sim)

    # Separation metrics
    if len(z_af) > 1 and len(z_genuine) > 1:
        within_af = pdist(z_af).mean() if len(z_af) > 1 else 0
        within_genuine = pdist(z_genuine).mean() if len(z_genuine) > 1 else 0
        between = cdist(z_af, z_genuine).mean()
        separation_ratio = between / ((within_af + within_genuine) / 2 + 1e-8)
    else:
        separation_ratio = 0

    return {
        'mean_cosine_similarity': float(mean_sim),
        'mean_pair_similarity': float(np.mean(pair_sims)),
        'separation_ratio': float(separation_ratio),
    }


def train_sae(activations, labels, contrast_weight, n_epochs=200, lr=1e-3):
    """Train SAE with specified contrast weight."""
    input_dim = activations.shape[1]
    latent_dim = 4096  # Smaller than full 16k for training speed

    model = ContrastiveSAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        sparsity_weight=1e-4,
        contrast_weight=contrast_weight,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        losses = model.compute_loss(activations, labels)
        losses['total'].backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            metrics = evaluate_separation(model, activations, labels)
            print(f"  Epoch {epoch+1}: loss={losses['total']:.4f}, "
                  f"recon={losses['reconstruction']:.4f}, "
                  f"pair_sim={metrics['mean_pair_similarity']:.4f}")

    return model


def main():
    print("=" * 70)
    print("CONTRASTIVE SAE TRAINING ON REAL ACTIVATIONS")
    print("=" * 70)

    # Load minimal pairs
    print("\n[1/5] Loading minimal pairs...")
    pairs_path = os.path.expanduser("~/af-detection-benchmark/data/minimal_pairs_sae_draft.json")

    with open(pairs_path) as f:
        data = json.load(f)
    pairs = data['pairs']
    print(f"Loaded {len(pairs)} minimal pairs")

    # Load model
    print("\n[2/5] Loading Gemma-27B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Extract activations
    print(f"\n[3/5] Extracting activations from layer {TARGET_LAYER}...")
    all_activations = []
    all_labels = []

    for i, pair in enumerate(pairs):
        print(f"  Pair {i+1}/{len(pairs)}...")

        # AF sample
        af_text = pair['af_sample']['text']
        af_acts = extract_activations(model, tokenizer, af_text, TARGET_LAYER)
        all_activations.append(af_acts)
        all_labels.append(1)

        # Genuine sample
        genuine_text = pair['genuine_sample']['text']
        genuine_acts = extract_activations(model, tokenizer, genuine_text, TARGET_LAYER)
        all_activations.append(genuine_acts)
        all_labels.append(0)

    activations = torch.stack(all_activations).to(DEVICE)
    labels = torch.tensor(all_labels, dtype=torch.long).to(DEVICE)

    print(f"Activation shape: {activations.shape}")
    print(f"Labels: {labels.sum().item()} AF, {(1-labels).sum().item()} genuine")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Train standard SAE (no contrastive loss)
    print("\n[4/5] Training Standard SAE (contrast_weight=0)...")
    print("-" * 50)
    standard_sae = train_sae(activations, labels, contrast_weight=0.0, n_epochs=200)
    standard_metrics = evaluate_separation(standard_sae, activations, labels)

    # Train contrastive SAE
    print("\n[5/5] Training Contrastive SAE (contrast_weight=1.0)...")
    print("-" * 50)
    contrastive_sae = train_sae(activations, labels, contrast_weight=1.0, n_epochs=200)
    contrastive_metrics = evaluate_separation(contrastive_sae, activations, labels)

    # Also try higher contrast weight
    print("\n[Bonus] Training Contrastive SAE (contrast_weight=5.0)...")
    print("-" * 50)
    strong_contrastive_sae = train_sae(activations, labels, contrast_weight=5.0, n_epochs=200)
    strong_metrics = evaluate_separation(strong_contrastive_sae, activations, labels)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<30} {'Pair Sim':<12} {'Mean Sim':<12} {'Sep Ratio':<12}")
    print("-" * 66)
    print(f"{'Standard SAE (λ=0)':<30} {standard_metrics['mean_pair_similarity']:.4f}       "
          f"{standard_metrics['mean_cosine_similarity']:.4f}       "
          f"{standard_metrics['separation_ratio']:.4f}")
    print(f"{'Contrastive SAE (λ=1)':<30} {contrastive_metrics['mean_pair_similarity']:.4f}       "
          f"{contrastive_metrics['mean_cosine_similarity']:.4f}       "
          f"{contrastive_metrics['separation_ratio']:.4f}")
    print(f"{'Strong Contrastive (λ=5)':<30} {strong_metrics['mean_pair_similarity']:.4f}       "
          f"{strong_metrics['mean_cosine_similarity']:.4f}       "
          f"{strong_metrics['separation_ratio']:.4f}")

    # Interpretation
    print("\n--- Interpretation ---")
    baseline_sim = standard_metrics['mean_pair_similarity']
    best_sim = min(contrastive_metrics['mean_pair_similarity'], strong_metrics['mean_pair_similarity'])

    reduction = (baseline_sim - best_sim) / baseline_sim * 100

    if best_sim < 0.7:
        print(f"SUCCESS: Reduced within-pair similarity from {baseline_sim:.2f} to {best_sim:.2f}")
        print(f"  -> {reduction:.1f}% reduction in similarity")
        print(f"  -> Contrastive SAE now encodes DECISION, not just SCENARIO")
        print(f"  -> Next: Use these features for steering experiments")
    elif best_sim < baseline_sim - 0.05:
        print(f"PARTIAL SUCCESS: Reduced similarity from {baseline_sim:.2f} to {best_sim:.2f}")
        print(f"  -> {reduction:.1f}% reduction")
        print(f"  -> Try higher contrast weight or more training")
    else:
        print(f"NO IMPROVEMENT: Similarity remains at {best_sim:.2f}")
        print(f"  -> Decision may be encoded in a way contrastive loss can't separate")
        print(f"  -> Try: different layers, adversarial training, or circuit analysis")

    # Save results
    results = {
        'standard': standard_metrics,
        'contrastive_1': contrastive_metrics,
        'contrastive_5': strong_metrics,
        'baseline_pair_sim': baseline_sim,
        'best_pair_sim': best_sim,
        'reduction_pct': reduction,
    }

    with open("contrastive_sae_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to contrastive_sae_results.json")

    # Save best model
    best_model = strong_contrastive_sae if strong_metrics['mean_pair_similarity'] < contrastive_metrics['mean_pair_similarity'] else contrastive_sae
    torch.save(best_model.state_dict(), "contrastive_sae_best.pt")
    print("Best model saved to contrastive_sae_best.pt")


if __name__ == "__main__":
    main()
