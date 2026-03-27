#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// PolarQuant 4-bit KV Cache Quantization / Dequantization
// ─────────────────────────────────────────────────────────────────────────────
//
// Each thread processes ONE vector of HEAD_DIM=64 elements.
// Dispatch grid: [numHeads * seqLen] threads per kernel call.
//
// Quantize:  fp16 vector → rotate → scalar quantize → 4-bit packed + scale
// Dequantize: 4-bit packed + scale → centroid lookup → inverse rotate → fp16
// ─────────────────────────────────────────────────────────────────────────────

constant int HEAD_DIM = 64;
constant int PACKED_PER_VEC = 16;   // HEAD_DIM / 4 = 16 uint16s per vector

// ── Quantize: fp16 → 4-bit packed ──────────────────────────────────────────
//
// Buffers:
//   [0] input:      half[total_vecs * 64]     — source fp16 KV data
//   [1] output:     uint16[total_vecs * 16]   — packed 4-bit indices
//   [2] scales:     half[total_vecs]           — per-vector L2 norm
//   [3] rotation:   float[64 * 64]            — orthogonal rotation matrix R
//   [4] boundaries: float[15]                 — Lloyd-Max decision boundaries
//   [5] total_vecs: uint                      — number of vectors to process

kernel void polarquant_quantize(
    device const half*       input       [[buffer(0)]],
    device uint16_t*         output      [[buffer(1)]],
    device half*             scales      [[buffer(2)]],
    constant float*          rotation    [[buffer(3)]],
    constant float*          boundaries  [[buffer(4)]],
    constant uint&           total_vecs  [[buffer(5)]],
    uint                     gid         [[thread_position_in_grid]])
{
    if (gid >= total_vecs) return;

    // ── Load input vector and compute L2 norm ──
    float vec[HEAD_DIM];
    float norm_sq = 0.0f;
    const int base = gid * HEAD_DIM;

    for (int i = 0; i < HEAD_DIM; i++) {
        vec[i] = float(input[base + i]);
        norm_sq += vec[i] * vec[i];
    }

    const float norm = sqrt(max(norm_sq, 1e-12f));
    scales[gid] = half(norm);

    // ── Normalize to unit vector ──
    const float inv_norm = 1.0f / norm;
    for (int i = 0; i < HEAD_DIM; i++) {
        vec[i] *= inv_norm;
    }

    // ── Rotate: rotated = R @ vec ──
    float rotated[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        float sum = 0.0f;
        const int row = i * HEAD_DIM;
        for (int j = 0; j < HEAD_DIM; j++) {
            sum += rotation[row + j] * vec[j];
        }
        rotated[i] = sum;
    }

    // ── Scalar quantize each coordinate to 4-bit index ──
    // Pack 4 indices (4 bits each) into one uint16
    const int out_base = gid * PACKED_PER_VEC;

    for (int group = 0; group < PACKED_PER_VEC; group++) {
        uint16_t packed = 0;
        for (int k = 0; k < 4; k++) {
            const float val = rotated[group * 4 + k];

            // Linear scan through 15 boundaries (fast for 16 levels)
            uint16_t level = 0;
            for (int b = 0; b < 15; b++) {
                if (val > boundaries[b]) level = uint16_t(b + 1);
            }

            packed |= (level & 0xF) << (k * 4);
        }
        output[out_base + group] = packed;
    }
}

// ── Dequantize: 4-bit packed → fp16 ───────────────────────────────────────
//
// Buffers:
//   [0] input:     uint16[total_vecs * 16]   — packed 4-bit indices (APPEND-order)
//   [1] scales:    half[total_vecs]           — per-vector L2 norm (APPEND-order)
//   [2] output:    half[total_vecs * 64]      — reconstructed fp16 (ORT-order)
//   [3] rotation:  float[64 * 64]            — R^T (transposed rotation)
//   [4] centroids: float[16]                 — Lloyd-Max reconstruction values
//   [5] total_vecs: uint                     — number of vectors to process
//   [6] num_heads: uint                     — number of attention heads
//
// APPEND-order layout: [step0_h0, step0_h1, ..., step0_hN, step1_h0, ...]
// ORT-order layout:    [h0_s0, h0_s1, ..., h1_s0, h1_s1, ...]
//
// Kernel dispatch: 1D grid of [numHeads * seqLen] threads.
// Thread gid maps to ORT position: gid = head * seqLen + step.
// We must read from APPEND position:  step * numHeads + head.

kernel void polarquant_dequantize(
    device const uint16_t*   input       [[buffer(0)]],
    device const half*       scales_buf  [[buffer(1)]],
    device half*             output      [[buffer(2)]],
    constant float*          rotation_t  [[buffer(3)]],
    constant float*          centroids   [[buffer(4)]],
    constant uint&           total_vecs  [[buffer(5)]],
    constant uint&           num_heads   [[buffer(6)]],
    uint                     gid         [[thread_position_in_grid]])
{
    if (gid >= total_vecs) return;

    // Decode ORT position (gid = head * seqLen + step) to get head and step
    const uint head = gid / num_heads;
    const uint step = gid % num_heads;

    // ── Read from APPEND-order compressed store ──
    // APPEND position for this (step, head): step * numHeads + head
    float rotated[HEAD_DIM];
    const int in_base = (step * num_heads + head) * PACKED_PER_VEC;

    for (int group = 0; group < PACKED_PER_VEC; group++) {
        const uint16_t packed = input[in_base + group];
        for (int k = 0; k < 4; k++) {
            const uint16_t level = (packed >> (k * 4)) & 0xF;
            rotated[group * 4 + k] = centroids[level];
        }
    }

    // ── Inverse rotate: vec = R^T @ rotated ──
    float vec[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        float sum = 0.0f;
        const int row = i * HEAD_DIM;
        for (int j = 0; j < HEAD_DIM; j++) {
            sum += rotation_t[row + j] * rotated[j];
        }
        vec[i] = sum;
    }

    // ── Rescale and write fp16 output (already in ORT-order at gid) ──
    const float scale = float(scales_buf[step * num_heads + head]);
    const int out_base = gid * HEAD_DIM;
    for (int i = 0; i < HEAD_DIM; i++) {
        output[out_base + i] = half(vec[i] * scale);
    }
}
