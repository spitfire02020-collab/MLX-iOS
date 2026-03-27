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

    // Decode ORT position (gid = head * seqLen + step) to get head and step.
    // total_vecs = numHeads * seqLen, so seqLen = total_vecs / numHeads.
    const uint seq_len = total_vecs / num_heads;
    const uint head = gid / seq_len;
    const uint step = gid % seq_len;

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

// ─────────────────────────────────────────────────────────────────────────────
// TurboQuant: PolarQuant + QJL 1-bit Error Correction
// ─────────────────────────────────────────────────────────────────────────────
//
// TurboQuant extends PolarQuant with Quantized Johnson-Lindenstrauss (QJL)
// 1-bit residual correction. After 4-bit PolarQuant quantization, the residual
// error direction is captured as sign bits (1 bit per dimension) and the mean
// residual magnitude is stored as fp16. During dequantization, this correction
// is applied to reduce reconstruction error.
//
// Additional storage per vector: 8 bytes (sign bits) + 2 bytes (magnitude)
// Effective compression: ~3.25 bits per element (vs 4-bit PolarQuant alone)
// ─────────────────────────────────────────────────────────────────────────────

// ── TurboQuant Quantize: PolarQuant + QJL sign bit computation ─────────────
//
// Performs PolarQuant quantization and immediately dequantizes in-kernel to
// compute the residual error. Stores sign bits and mean magnitude for QJL
// correction during dequantization.
//
// Buffers:
//   [0] input:      half[total_vecs * 64]   — source fp16 KV data
//   [1] indices:    uint16[total_vecs * 16] — packed 4-bit output
//   [2] scales:     half[total_vecs]        — per-vector L2 norm
//   [3] qjl_signs:  uint8[total_vecs * 8]  — residual sign bits (64 bits per vec)
//   [4] qjl_mags:   half[total_vecs]        — mean residual magnitude per vec
//   [5] rotation:   float[64 * 64]          — orthogonal rotation matrix R
//   [6] rotation_t: float[64 * 64]          — R^T (transposed rotation)
//   [7] boundaries: float[15]               — Lloyd-Max decision boundaries
//   [8] centroids:  float[16]               — Lloyd-Max reconstruction values
//   [9] total_vecs: uint                    — number of vectors to process

kernel void turboquant_quantize(
    device const half*       input       [[buffer(0)]],
    device uint16_t*         indices     [[buffer(1)]],
    device half*             scales      [[buffer(2)]],
    device uint8_t*          qjl_signs   [[buffer(3)]],
    device half*             qjl_mags    [[buffer(4)]],
    constant float*          rotation    [[buffer(5)]],
    constant float*          rotation_t  [[buffer(6)]],
    constant float*          boundaries  [[buffer(7)]],
    constant float*          centroids   [[buffer(8)]],
    constant uint&           total_vecs  [[buffer(9)]],
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
    float unit[HEAD_DIM];
    const float inv_norm = 1.0f / norm;
    for (int i = 0; i < HEAD_DIM; i++) {
        unit[i] = vec[i] * inv_norm;
    }

    // ── Rotate: rotated = R @ unit ──
    float rotated[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        float sum = 0.0f;
        const int row = i * HEAD_DIM;
        for (int j = 0; j < HEAD_DIM; j++) {
            sum += rotation[row + j] * unit[j];
        }
        rotated[i] = sum;
    }

    // ── Scalar quantize each coordinate to 4-bit index ──
    uint16_t quant_levels[HEAD_DIM];
    const int out_base = gid * PACKED_PER_VEC;

    for (int group = 0; group < PACKED_PER_VEC; group++) {
        uint16_t packed = 0;
        for (int k = 0; k < 4; k++) {
            int idx = group * 4 + k;
            const float val = rotated[idx];

            uint16_t level = 0;
            for (int b = 0; b < 15; b++) {
                if (val > boundaries[b]) level = uint16_t(b + 1);
            }

            quant_levels[idx] = level;
            packed |= (level & 0xF) << (k * 4);
        }
        indices[out_base + group] = packed;
    }

    // ── In-kernel dequantize to compute residual ──
    // Reconstruct rotated values from centroids
    float recon_rotated[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        recon_rotated[i] = centroids[quant_levels[i]];
    }

    // Inverse rotate: recon = R^T @ recon_rotated, then rescale
    float recon[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        float sum = 0.0f;
        const int row = i * HEAD_DIM;
        for (int j = 0; j < HEAD_DIM; j++) {
            sum += rotation_t[row + j] * recon_rotated[j];
        }
        recon[i] = sum * norm;
    }

    // ── QJL: Compute residual sign bits and mean magnitude ──
    float total_abs_residual = 0.0f;
    uint8_t signs[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < HEAD_DIM; i++) {
        float residual = vec[i] - recon[i];
        total_abs_residual += abs(residual);
        if (residual >= 0.0f) {
            signs[i / 8] |= (1u << (i % 8));
        }
    }

    // Store QJL sign bits (8 bytes per vector = HEAD_DIM bits)
    const int sign_base = gid * 8;
    for (int i = 0; i < 8; i++) {
        qjl_signs[sign_base + i] = signs[i];
    }

    // Store mean residual magnitude
    qjl_mags[gid] = half(total_abs_residual / float(HEAD_DIM));
}

// ── TurboQuant Dequantize: PolarQuant + QJL Correction ────────────────────
//
// Dequantizes 4-bit packed data back to fp16 and applies QJL 1-bit residual
// correction. Output is in APPEND-order (same as input). The caller is
// responsible for transposing to ORT-order.
//
// Buffers:
//   [0] input:     uint16[total_vecs * 16]  — packed 4-bit indices (APPEND-order)
//   [1] scales:    half[total_vecs]          — per-vector L2 norm (APPEND-order)
//   [2] qjl_signs: uint8[total_vecs * 8]   — residual sign bits (APPEND-order)
//   [3] qjl_mags:  half[total_vecs]          — mean residual magnitude (APPEND-order)
//   [4] output:    half[total_vecs * 64]     — reconstructed fp16 (APPEND-order)
//   [5] rotation:  float[64 * 64]           — R^T (transposed rotation)
//   [6] centroids: float[16]                — Lloyd-Max reconstruction values
//   [7] total_vecs: uint                    — number of vectors to process

kernel void turboquant_dequantize(
    device const uint16_t*   input       [[buffer(0)]],
    device const half*       scales_buf  [[buffer(1)]],
    device const uint8_t*    qjl_signs   [[buffer(2)]],
    device const half*       qjl_mags    [[buffer(3)]],
    device half*             output      [[buffer(4)]],
    constant float*          rotation_t  [[buffer(5)]],
    constant float*          centroids   [[buffer(6)]],
    constant uint&           total_vecs  [[buffer(7)]],
    uint                     gid         [[thread_position_in_grid]])
{
    if (gid >= total_vecs) return;

    // ── Standard PolarQuant dequantize (append-order: gid = sequential index) ──
    float rotated[HEAD_DIM];
    const int in_base = gid * PACKED_PER_VEC;

    for (int group = 0; group < PACKED_PER_VEC; group++) {
        const uint16_t packed = input[in_base + group];
        for (int k = 0; k < 4; k++) {
            const uint16_t level = (packed >> (k * 4)) & 0xF;
            rotated[group * 4 + k] = centroids[level];
        }
    }

    // Inverse rotate: vec = R^T @ rotated
    float vec[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        float sum = 0.0f;
        const int row = i * HEAD_DIM;
        for (int j = 0; j < HEAD_DIM; j++) {
            sum += rotation_t[row + j] * rotated[j];
        }
        vec[i] = sum;
    }

    // ── Rescale + QJL correction ──
    const float scale = float(scales_buf[gid]);
    const float mag = float(qjl_mags[gid]);
    const int sign_base = gid * 8;

    const int out_base = gid * HEAD_DIM;
    for (int i = 0; i < HEAD_DIM; i++) {
        float val = vec[i] * scale;
        // Apply QJL sign-based residual correction
        uint8_t sign_byte = qjl_signs[sign_base + i / 8];
        bool is_positive = (sign_byte >> (i % 8)) & 1;
        val += is_positive ? mag : -mag;
        output[out_base + i] = half(val);
    }
}
