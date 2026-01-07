#!/usr/bin/env python3
"""
Contrastive SAE Training for Alignment Faking Detection

This implements Section 1.2 of the AF Interpretability Research Plan:
"Add a contrastive objective that distinguishes AF from genuine samples"

Standard SAE loss: L = reconstruction + λ * sparsity
Contrastive SAE:   L = reconstruction + λ1 * sparsity + λ2 * contrast

The contrast term pushes AF and genuine samples apart in feature space,
forcing the SAE to encode DECISION (strategic vs genuine), not just SCENARIO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import os


class ContrastiveSAE(nn.Module):
    """
    Sparse Autoencoder with contrastive objective.

    Architecture matches GemmaScope transcoders for compatibility.
    """

    def __init__(
        self,
        input_dim: int = 3584,  # Gemma-27B hidden dim
        latent_dim: int = 16384,  # 16k features
        sparsity_weight: float = 1e-3,
        contrast_weight: float = 0.1,
        contrast_temperature: float = 0.07,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight
        self.contrast_weight = contrast_weight
        self.temperature = contrast_temperature

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Linear(latent_dim, input_dim)

        # Optional: learnable threshold (like GemmaScope)
        self.threshold = nn.Parameter(torch.zeros(latent_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features."""
        pre_acts = self.encoder[0](x)  # Linear
        return F.relu(pre_acts - self.threshold)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode features to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and features."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss."""
        return F.mse_loss(x_hat, x)

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """L1 sparsity loss on feature activations."""
        return z.abs().mean()

    def supervised_contrastive_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supervised contrastive loss (SupCon).

        Pulls together samples with same label, pushes apart different labels.

        Args:
            z: Feature vectors [batch_size, latent_dim]
            labels: Binary labels (1=AF, 0=genuine) [batch_size]
        """
        batch_size = z.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)

        # Normalize features
        z_norm = F.normalize(z, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.temperature

        # Create masks
        labels = labels.view(-1, 1)
        mask_same = (labels == labels.T).float()  # Same class
        mask_diff = 1 - mask_same  # Different class

        # Remove diagonal
        mask_diag = torch.eye(batch_size, device=z.device)
        mask_same = mask_same * (1 - mask_diag)

        # Compute contrastive loss
        # For each sample, maximize similarity to same-class, minimize to different-class
        exp_sim = torch.exp(sim_matrix) * (1 - mask_diag)

        # Positive pairs: same class
        pos_sim = (exp_sim * mask_same).sum(dim=1)
        # All pairs (for normalization)
        all_sim = exp_sim.sum(dim=1)

        # Avoid division by zero
        n_positives = mask_same.sum(dim=1).clamp(min=1)

        # SupCon loss
        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
        loss = (loss * (n_positives > 0)).sum() / (n_positives > 0).sum().clamp(min=1)

        return loss

    def triplet_loss(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        z_negative: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """
        Triplet loss variant.

        anchor: AF sample
        positive: Another AF sample
        negative: Genuine sample (same scenario)
        """
        dist_pos = F.pairwise_distance(z_anchor, z_positive)
        dist_neg = F.pairwise_distance(z_anchor, z_negative)

        loss = F.relu(dist_pos - dist_neg + margin)
        return loss.mean()

    def compute_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        contrastive_method: str = "supcon",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full loss with all components.

        Args:
            x: Input activations [batch_size, input_dim]
            labels: Binary labels (1=AF, 0=genuine)
            contrastive_method: "supcon" or "triplet"

        Returns:
            Dictionary of loss components and total loss
        """
        x_hat, z = self.forward(x)

        # Component losses
        l_recon = self.reconstruction_loss(x, x_hat)
        l_sparse = self.sparsity_loss(z)

        if contrastive_method == "supcon":
            l_contrast = self.supervised_contrastive_loss(z, labels)
        else:
            # For triplet, need restructured input (not implemented here)
            l_contrast = torch.tensor(0.0, device=x.device)

        # Total loss
        total = l_recon + self.sparsity_weight * l_sparse + self.contrast_weight * l_contrast

        return {
            'total': total,
            'reconstruction': l_recon,
            'sparsity': l_sparse,
            'contrastive': l_contrast,
        }


class MinimalPairDataset(Dataset):
    """Dataset of minimal pairs for contrastive training."""

    def __init__(self, pairs_file: str, activations_file: str):
        """
        Args:
            pairs_file: Path to minimal_pairs_sae_draft.json
            activations_file: Path to pre-extracted activations
        """
        with open(pairs_file) as f:
            self.pairs = json.load(f)['pairs']

        # Load pre-extracted activations
        if os.path.exists(activations_file):
            self.activations = torch.load(activations_file)
        else:
            raise FileNotFoundError(
                f"{activations_file} not found. "
                "Run extract_activations.py first."
            )

    def __len__(self):
        return len(self.pairs) * 2  # AF + genuine for each pair

    def __getitem__(self, idx):
        pair_idx = idx // 2
        is_af = idx % 2 == 0

        pair = self.pairs[pair_idx]
        key = f"pair_{pair_idx}_{'af' if is_af else 'genuine'}"

        return {
            'activation': self.activations[key],
            'label': torch.tensor(1 if is_af else 0, dtype=torch.long),
            'pair_id': pair_idx,
        }


def train_contrastive_sae(
    train_activations: torch.Tensor,
    train_labels: torch.Tensor,
    val_activations: torch.Tensor = None,
    val_labels: torch.Tensor = None,
    input_dim: int = 3584,
    latent_dim: int = 16384,
    sparsity_weight: float = 1e-3,
    contrast_weight: float = 0.1,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
) -> ContrastiveSAE:
    """
    Train a contrastive SAE.

    Args:
        train_activations: [n_samples, input_dim]
        train_labels: [n_samples] binary labels

    Returns:
        Trained ContrastiveSAE
    """
    model = ContrastiveSAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        sparsity_weight=sparsity_weight,
        contrast_weight=contrast_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    n_batches = (len(train_activations) + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = {'total': 0, 'reconstruction': 0, 'sparsity': 0, 'contrastive': 0}

        # Shuffle
        perm = torch.randperm(len(train_activations))
        train_activations = train_activations[perm]
        train_labels = train_labels[perm]

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(train_activations))

            x = train_activations[start:end].to(device)
            labels = train_labels[start:end].to(device)

            optimizer.zero_grad()
            losses = model.compute_loss(x, labels)
            losses['total'].backward()
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"total={epoch_losses['total']:.4f}, "
                  f"recon={epoch_losses['reconstruction']:.4f}, "
                  f"sparse={epoch_losses['sparsity']:.4f}, "
                  f"contrast={epoch_losses['contrastive']:.4f}")

        # Validation
        if val_activations is not None and (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_x = val_activations.to(device)
                val_y = val_labels.to(device)
                val_losses = model.compute_loss(val_x, val_y)
                print(f"  Val: total={val_losses['total']:.4f}")

    return model


def evaluate_sae_separation(
    model: ContrastiveSAE,
    activations: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate how well the SAE separates AF from genuine.

    Returns:
        Dictionary with separation metrics
    """
    model.eval()
    with torch.no_grad():
        x = activations.to(device)
        _, z = model(x)

        af_mask = labels == 1
        genuine_mask = labels == 0

        z_af = z[af_mask].cpu().numpy()
        z_genuine = z[genuine_mask].cpu().numpy()

        # Compute mean feature vectors
        af_mean = z_af.mean(axis=0)
        genuine_mean = z_genuine.mean(axis=0)

        # Cosine similarity between class means
        mean_sim = np.dot(af_mean, genuine_mean) / (
            np.linalg.norm(af_mean) * np.linalg.norm(genuine_mean) + 1e-8
        )

        # Average within-class vs between-class distances
        from scipy.spatial.distance import pdist, cdist

        if len(z_af) > 1 and len(z_genuine) > 1:
            within_af = pdist(z_af).mean()
            within_genuine = pdist(z_genuine).mean()
            between = cdist(z_af, z_genuine).mean()

            # Separation ratio: higher is better
            separation_ratio = between / ((within_af + within_genuine) / 2 + 1e-8)
        else:
            within_af = within_genuine = between = separation_ratio = 0

        return {
            'mean_cosine_similarity': float(mean_sim),
            'within_af_distance': float(within_af),
            'within_genuine_distance': float(within_genuine),
            'between_class_distance': float(between),
            'separation_ratio': float(separation_ratio),
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("CONTRASTIVE SAE TRAINING")
    print("=" * 70)

    # Synthetic example to demonstrate the approach
    print("\nRunning with synthetic data to demonstrate...")
    print("(For real training, use extract_activations.py first)")

    input_dim = 256  # Reduced for demo
    latent_dim = 1024
    n_samples = 100

    # Generate synthetic data with separable classes
    torch.manual_seed(42)

    # AF samples: higher activations in certain dimensions
    af_activations = torch.randn(n_samples // 2, input_dim) + 0.5
    af_activations[:, :50] += 1.0  # AF signal in first 50 dims

    # Genuine samples: different pattern
    genuine_activations = torch.randn(n_samples // 2, input_dim) + 0.5

    activations = torch.cat([af_activations, genuine_activations])
    labels = torch.cat([
        torch.ones(n_samples // 2),
        torch.zeros(n_samples // 2)
    ]).long()

    # Train standard SAE (no contrastive)
    print("\n--- Training Standard SAE ---")
    standard_sae = train_contrastive_sae(
        activations, labels,
        input_dim=input_dim,
        latent_dim=latent_dim,
        contrast_weight=0.0,  # No contrastive loss
        n_epochs=50,
        device="cpu",
    )

    standard_metrics = evaluate_sae_separation(standard_sae, activations, labels, device="cpu")
    print(f"Standard SAE separation: {standard_metrics['separation_ratio']:.4f}")
    print(f"Standard SAE mean similarity: {standard_metrics['mean_cosine_similarity']:.4f}")

    # Train contrastive SAE
    print("\n--- Training Contrastive SAE ---")
    contrastive_sae = train_contrastive_sae(
        activations, labels,
        input_dim=input_dim,
        latent_dim=latent_dim,
        contrast_weight=0.5,  # With contrastive loss
        n_epochs=50,
        device="cpu",
    )

    contrastive_metrics = evaluate_sae_separation(contrastive_sae, activations, labels, device="cpu")
    print(f"Contrastive SAE separation: {contrastive_metrics['separation_ratio']:.4f}")
    print(f"Contrastive SAE mean similarity: {contrastive_metrics['mean_cosine_similarity']:.4f}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Standard':<15} {'Contrastive':<15}")
    print("-" * 60)
    for key in standard_metrics:
        print(f"{key:<30} {standard_metrics[key]:<15.4f} {contrastive_metrics[key]:<15.4f}")

    improvement = contrastive_metrics['separation_ratio'] / (standard_metrics['separation_ratio'] + 1e-8)
    print(f"\nSeparation improvement: {improvement:.2f}x")

    if contrastive_metrics['mean_cosine_similarity'] < standard_metrics['mean_cosine_similarity']:
        print("SUCCESS: Contrastive training reduced similarity between AF and genuine")
        print("  -> Features now encode DECISION, not just SCENARIO")
    else:
        print("No improvement - may need more contrastive weight or epochs")
