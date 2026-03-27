#!/usr/bin/env python3
"""Generate PolarQuant constants for 4-bit KV cache quantization.

Outputs:
  Reader/Resources/polarquant_constants.json
    - rotation_matrix: [64*64] float  (orthogonal rotation matrix R)
    - quantizer_boundaries: [15] float (Lloyd-Max decision boundaries)
    - quantizer_centroids: [16] float  (Lloyd-Max reconstruction values)
"""
import numpy as np
import json
import os

HEAD_DIM = 64
BITS = 4
NUM_LEVELS = 2 ** BITS  # 16

# --- Random orthogonal rotation matrix ---
# Fixed seed for reproducibility across builds
rng = np.random.RandomState(42)
gaussian = rng.randn(HEAD_DIM, HEAD_DIM).astype(np.float64)
Q, _ = np.linalg.qr(gaussian)
rotation_matrix = Q.astype(np.float32)  # [64, 64] orthogonal

# Verify orthogonality: R @ R^T should be identity
identity_check = rotation_matrix @ rotation_matrix.T
max_off_diag = np.max(np.abs(identity_check - np.eye(HEAD_DIM)))
print(f"Rotation matrix orthogonality check: max |R@R^T - I| = {max_off_diag:.2e}")
assert max_off_diag < 1e-5, "Rotation matrix is not orthogonal!"

# --- Lloyd-Max codebook for approximately Gaussian distribution ---
# After rotation by a random orthogonal matrix, each coordinate of a
# high-dimensional vector converges to approximately N(0, sigma^2).
#
# For unit-normalized vectors in d=64 dimensions (L2 norm = 1.0):
#   E[x_i^2] = 1/d = 1/64
#   per-coordinate sigma = sqrt(1/64) = 0.125
#
# Using sigma=0.15 to provide a safety margin for distribution tails.
# Prior sigma=0.1 was 25% too narrow, causing ~5-6% of coordinates to
# clip to edge levels with ~45% relative quantization error.
# This mismatch caused value vector corruption (60% relative error) that
# accumulated over ~38 decode steps, producing token 0 repetition.
SIGMA = 0.15


def lloyd_max_gaussian(num_levels, sigma, iterations=200):
    """Compute Lloyd-Max optimal quantizer for Gaussian(0, sigma^2).
    
    Returns:
        boundaries: [num_levels-1] decision boundaries
        centroids: [num_levels] reconstruction values
    """
    from scipy.stats import norm
    
    # Initialize with uniform spacing over [-4sigma, 4sigma]
    boundaries = np.linspace(-4 * sigma, 4 * sigma, num_levels + 1)
    centroids = np.zeros(num_levels)
    
    for iteration in range(iterations):
        # Update centroids: E[X | boundary[i] < X < boundary[i+1]]
        for i in range(num_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            # Conditional expectation of N(0, sigma^2) in [lo, hi]
            num = sigma * (norm.pdf(lo / sigma) - norm.pdf(hi / sigma))
            den = norm.cdf(hi / sigma) - norm.cdf(lo / sigma)
            centroids[i] = num / max(den, 1e-15)
        
        # Update boundaries: midpoint between adjacent centroids
        for i in range(1, num_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0
    
    # Return inner boundaries (not the ±inf endpoints)
    return boundaries[1:-1].astype(np.float32), centroids.astype(np.float32)


boundaries, centroids = lloyd_max_gaussian(NUM_LEVELS, SIGMA)

print(f"\nLloyd-Max Quantizer (4-bit, sigma={SIGMA}):")
print(f"  Boundaries ({len(boundaries)}): {boundaries}")
print(f"  Centroids  ({len(centroids)}): {centroids}")

# Verify symmetry (Gaussian codebook should be symmetric around 0)
sym_error = np.max(np.abs(centroids + centroids[::-1]))
print(f"  Symmetry check: max |c[i] + c[N-1-i]| = {sym_error:.2e}")

# --- Save as JSON ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, "Reader", "Resources")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "polarquant_constants.json")

constants = {
    "head_dim": HEAD_DIM,
    "bits": BITS,
    "num_levels": NUM_LEVELS,
    "sigma": float(SIGMA),
    "rotation_matrix": rotation_matrix.flatten().tolist(),
    "quantizer_boundaries": boundaries.tolist(),
    "quantizer_centroids": centroids.tolist(),
}

with open(output_path, "w") as f:
    json.dump(constants, f, indent=2)

print(f"\nSaved to {output_path}")
print(f"  rotation_matrix: {len(constants['rotation_matrix'])} floats ({HEAD_DIM}x{HEAD_DIM})")
print(f"  boundaries: {len(constants['quantizer_boundaries'])} floats")
print(f"  centroids: {len(constants['quantizer_centroids'])} floats")
