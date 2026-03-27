# PolarQuant KV Cache Compression — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce KV cache memory by 4x during autoregressive decode via 4-bit PolarQuant compression on Metal GPU, lowering memory pressure and improving token generation speed on iOS.

**Architecture:** Between each LM decode step, intercept the `present.{layer}.key/value` → `past_key_values.{layer}.key/value` handoff (ChatterboxEngine.swift:1154-1161). Quantize KV vectors to 4-bit using PolarQuant (random rotation + Lloyd-Max scalar quantization) via Metal compute shaders, store compressed. Dequantize back to fp16 ORTValues before the next ORT session.run() call. The rotation matrix and codebook are precomputed constants.

**Tech Stack:** Metal Shading Language, Swift, ONNX Runtime (existing), Accelerate framework

**KV Cache Specs:**
- Shape per layer: `[1, 16, seqLen, 64]` float16 (16 heads, 64 head_dim)
- 24 layers × 2 (key + value) = 48 tensors carried per step
- At step 500: ~49 MB fp16 → ~12 MB at 4-bit (4x reduction)

---

### Task 1: Precompute PolarQuant Constants (Python)

Generate the rotation matrix and Lloyd-Max codebook offline. These are baked into the app as constants.

**Files:**
- Create: `scripts/generate_polarquant_constants.py`

**Step 1: Write the Python script**

```python
#!/usr/bin/env python3
"""Generate PolarQuant constants for 4-bit KV cache quantization."""
import numpy as np
import json

HEAD_DIM = 64
BITS = 4
NUM_LEVELS = 2 ** BITS  # 16

# --- Random orthogonal rotation matrix ---
# Fixed seed for reproducibility across builds
rng = np.random.RandomState(42)
gaussian = rng.randn(HEAD_DIM, HEAD_DIM).astype(np.float32)
Q, _ = np.linalg.qr(gaussian)
rotation_matrix = Q  # [64, 64] orthogonal

# --- Lloyd-Max codebook for N(0, 1/sqrt(d)) ---
# After rotation, each coordinate ~ N(0, 1/d) for unit vectors.
# For our fp16 KV vectors, the distribution is data-dependent but
# approximately Gaussian. We use N(0, sigma) with sigma calibrated
# to typical KV cache value ranges.
sigma = 0.1  # Empirically good for transformer KV caches

def lloyd_max_gaussian(num_levels, sigma, iterations=100):
    """Compute Lloyd-Max quantizer for Gaussian(0, sigma^2)."""
    from scipy.stats import norm
    # Initialize with uniform spacing
    boundaries = np.linspace(-4*sigma, 4*sigma, num_levels + 1)
    centroids = np.zeros(num_levels)
    
    for _ in range(iterations):
        # Update centroids: E[X | boundary[i] < X < boundary[i+1]]
        for i in range(num_levels):
            lo, hi = boundaries[i], boundaries[i+1]
            # Conditional expectation of N(0, sigma^2) in [lo, hi]
            num = sigma * (norm.pdf(lo/sigma) - norm.pdf(hi/sigma))
            den = norm.cdf(hi/sigma) - norm.cdf(lo/sigma)
            centroids[i] = num / max(den, 1e-10)
        # Update boundaries: midpoint between adjacent centroids
        for i in range(1, num_levels):
            boundaries[i] = (centroids[i-1] + centroids[i]) / 2.0
    
    return boundaries[1:-1].astype(np.float32), centroids.astype(np.float32)

boundaries, centroids = lloyd_max_gaussian(NUM_LEVELS, sigma)

# Save as JSON for easy embedding in Swift
constants = {
    "head_dim": HEAD_DIM,
    "bits": BITS,
    "num_levels": NUM_LEVELS,
    "sigma": float(sigma),
    "rotation_matrix": rotation_matrix.flatten().tolist(),
    "quantizer_boundaries": boundaries.tolist(),  # 15 values
    "quantizer_centroids": centroids.tolist(),     # 16 values
}

with open("Reader/Resources/polarquant_constants.json", "w") as f:
    json.dump(constants, f, indent=2)

print(f"Rotation matrix: {rotation_matrix.shape}")
print(f"Boundaries: {boundaries.shape} = {boundaries}")
print(f"Centroids: {centroids.shape} = {centroids}")
print("Saved to Reader/Resources/polarquant_constants.json")
```

**Step 2: Run the script**

```bash
cd /Users/rockymoon/Downloads/Claude/Speaker
pip install scipy  # If not installed
python3 scripts/generate_polarquant_constants.py
```

Expected: Creates `Reader/Resources/polarquant_constants.json` with rotation matrix (4096 floats) and codebook.

**Step 3: Commit**

```bash
git add scripts/generate_polarquant_constants.py Reader/Resources/polarquant_constants.json
git commit -m "feat: add PolarQuant constant generation script and precomputed values"
```

---

### Task 2: Metal Compute Shaders for Quantize/Dequantize

Two Metal kernels: one to quantize fp16 → 4-bit packed, one to dequantize 4-bit packed → fp16.

**Files:**
- Create: `Reader/Services/Metal/PolarQuantKernels.metal`

**Step 1: Write the Metal shader file**

```metal
#include <metal_stdlib>
using namespace metal;

// PolarQuant 4-bit KV Cache Quantization
// Each thread processes one vector of HEAD_DIM=64 elements.
// Thread grid: [numHeads * seqLen] threads per layer call.

constant int HEAD_DIM = 64;
constant int NUM_LEVELS = 16;  // 4-bit

// Rotation matrix: [64, 64] stored row-major as float
// Quantizer boundaries: [15] floats (decision boundaries between 16 levels)
// Quantizer centroids: [16] floats (reconstruction values)

// --- Quantize kernel ---
// Input:  fp16 KV tensor [numHeads, seqLen, HEAD_DIM]
// Output: packed uint16 indices [numHeads, seqLen, HEAD_DIM/4]
//         (4 bits per element, 4 elements per uint16)
//         + per-vector scale (float16) [numHeads, seqLen]
kernel void polarquant_quantize(
    device const half*       input       [[buffer(0)]],  // [H * S * 64] fp16
    device uint16_t*         output      [[buffer(1)]],  // [H * S * 16] packed 4-bit
    device half*             scales      [[buffer(2)]],  // [H * S] per-vector norm
    constant float*          rotation    [[buffer(3)]],  // [64 * 64] rotation matrix
    constant float*          boundaries  [[buffer(4)]],  // [15] quantizer boundaries
    constant uint&           total_vecs  [[buffer(5)]],  // H * S
    uint                     gid         [[thread_position_in_grid]])
{
    if (gid >= total_vecs) return;
    
    // Load input vector and compute norm for scale
    float vec[HEAD_DIM];
    float norm_sq = 0.0;
    int base = gid * HEAD_DIM;
    
    for (int i = 0; i < HEAD_DIM; i++) {
        vec[i] = float(input[base + i]);
        norm_sq += vec[i] * vec[i];
    }
    
    float norm = sqrt(max(norm_sq, 1e-12));
    scales[gid] = half(norm);
    
    // Normalize
    float inv_norm = 1.0 / norm;
    for (int i = 0; i < HEAD_DIM; i++) {
        vec[i] *= inv_norm;
    }
    
    // Rotate: rotated = R @ vec
    float rotated[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        float sum = 0.0;
        for (int j = 0; j < HEAD_DIM; j++) {
            sum += rotation[i * HEAD_DIM + j] * vec[j];
        }
        rotated[i] = sum;
    }
    
    // Scalar quantize each rotated coordinate to 4-bit index
    // Pack 4 indices into each uint16 (4 bits each)
    int out_base = gid * (HEAD_DIM / 4);  // 16 uint16s per vector
    
    for (int group = 0; group < HEAD_DIM / 4; group++) {
        uint16_t packed = 0;
        for (int k = 0; k < 4; k++) {
            int idx = group * 4 + k;
            float val = rotated[idx];
            
            // Binary search for quantization level
            uint16_t level = 0;
            for (int b = 0; b < NUM_LEVELS - 1; b++) {
                if (val > boundaries[b]) level = b + 1;
            }
            
            packed |= (level & 0xF) << (k * 4);
        }
        output[out_base + group] = packed;
    }
}

// --- Dequantize kernel ---
// Input:  packed uint16 indices + scales
// Output: fp16 KV tensor
kernel void polarquant_dequantize(
    device const uint16_t*   input       [[buffer(0)]],  // [H * S * 16] packed
    device const half*       scales      [[buffer(1)]],  // [H * S] per-vector norm
    device half*             output      [[buffer(2)]],  // [H * S * 64] fp16
    constant float*          rotation    [[buffer(3)]],  // [64 * 64] rotation matrix (R^T)
    constant float*          centroids   [[buffer(4)]],  // [16] quantizer centroids
    constant uint&           total_vecs  [[buffer(5)]],  // H * S
    uint                     gid         [[thread_position_in_grid]])
{
    if (gid >= total_vecs) return;
    
    // Unpack 4-bit indices and look up centroids
    float rotated[HEAD_DIM];
    int in_base = gid * (HEAD_DIM / 4);
    
    for (int group = 0; group < HEAD_DIM / 4; group++) {
        uint16_t packed = input[in_base + group];
        for (int k = 0; k < 4; k++) {
            int idx = group * 4 + k;
            uint16_t level = (packed >> (k * 4)) & 0xF;
            rotated[idx] = centroids[level];
        }
    }
    
    // Inverse rotate: vec = R^T @ rotated
    // (R^T is passed as rotation buffer - transposed on CPU side)
    float vec[HEAD_DIM];
    for (int i = 0; i < HEAD_DIM; i++) {
        float sum = 0.0;
        for (int j = 0; j < HEAD_DIM; j++) {
            sum += rotation[i * HEAD_DIM + j] * rotated[j];
        }
        vec[i] = sum;
    }
    
    // Rescale and write fp16 output
    float scale = float(scales[gid]);
    int out_base = gid * HEAD_DIM;
    for (int i = 0; i < HEAD_DIM; i++) {
        output[out_base + i] = half(vec[i] * scale);
    }
}
```

**Step 2: Add the .metal file to the Xcode project**

The `.metal` file must be added to the Reader target's Compile Sources build phase in `project.pbxproj`. This will be done when integrating (Task 4).

**Step 3: Commit**

```bash
git add Reader/Services/Metal/PolarQuantKernels.metal
git commit -m "feat: add Metal compute shaders for PolarQuant 4-bit quantize/dequantize"
```

---

### Task 3: Swift KVCacheQuantizer Wrapper

Swift class that manages Metal pipeline state, buffers, and provides a clean API for the decode loop.

**Files:**
- Create: `Reader/Services/Metal/KVCacheQuantizer.swift`

**Step 1: Write the KVCacheQuantizer class**

```swift
import Foundation
import Metal
import OnnxRuntimeBindings
import os.log

private let kvqLogger = Logger(subsystem: "com.reader.app", category: "KVCacheQuantizer")

/// PolarQuant 4-bit KV cache compressor using Metal GPU.
///
/// Compresses KV cache tensors from fp16 to 4-bit between decode steps,
/// reducing memory by 4x. Reconstructs fp16 tensors for ORT input.
///
/// Usage:
///   1. Call `compress(present:)` after each LM step with the ORT output tensors
///   2. Call `decompress(layer:type:)` before next LM step to get fp16 ORTValues
///   3. Call `reset()` between synthesis chunks
final class KVCacheQuantizer {
    
    // Metal state
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let quantizePSO: MTLComputePipelineState
    private let dequantizePSO: MTLComputePipelineState
    
    // PolarQuant constants (loaded from JSON)
    private let rotationBuffer: MTLBuffer       // [64*64] float32, R for quantize
    private let rotationTBuffer: MTLBuffer      // [64*64] float32, R^T for dequantize
    private let boundariesBuffer: MTLBuffer     // [15] float32
    private let centroidsBuffer: MTLBuffer      // [16] float32
    
    // Config
    let numLayers: Int
    let numHeads: Int
    let headDim: Int
    
    // Compressed storage: [layer][keyOrValue] -> (packedIndices, scales)
    // packedIndices: [numHeads * seqLen * (headDim/4)] uint16
    // scales: [numHeads * seqLen] float16
    private var compressedKeys: [(indices: MTLBuffer, scales: MTLBuffer, numVecs: Int)]
    private var compressedValues: [(indices: MTLBuffer, scales: MTLBuffer, numVecs: Int)]
    
    /// Whether compression is enabled (can be toggled for A/B testing)
    var isEnabled: Bool = true
    
    init(numLayers: Int, numHeads: Int, headDim: Int) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw KVQError.metalUnavailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw KVQError.metalUnavailable
        }
        
        self.device = device
        self.commandQueue = queue
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.headDim = headDim
        
        // Load Metal shaders
        guard let library = device.makeDefaultLibrary() else {
            throw KVQError.shaderNotFound
        }
        guard let quantizeFn = library.makeFunction(name: "polarquant_quantize"),
              let dequantizeFn = library.makeFunction(name: "polarquant_dequantize") else {
            throw KVQError.shaderNotFound
        }
        
        self.quantizePSO = try device.makeComputePipelineState(function: quantizeFn)
        self.dequantizePSO = try device.makeComputePipelineState(function: dequantizeFn)
        
        // Load constants from JSON
        let constants = try Self.loadConstants()
        
        // Create rotation matrix buffer (R)
        let rotationData = constants.rotationMatrix.map { Float($0) }
        self.rotationBuffer = device.makeBuffer(
            bytes: rotationData, length: rotationData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        // Create R^T buffer (transpose for dequantize)
        var rotationT = [Float](repeating: 0, count: headDim * headDim)
        for i in 0..<headDim {
            for j in 0..<headDim {
                rotationT[i * headDim + j] = rotationData[j * headDim + i]
            }
        }
        self.rotationTBuffer = device.makeBuffer(
            bytes: rotationT, length: rotationT.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        // Boundaries and centroids
        let boundaryData = constants.boundaries.map { Float($0) }
        self.boundariesBuffer = device.makeBuffer(
            bytes: boundaryData, length: boundaryData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        let centroidData = constants.centroids.map { Float($0) }
        self.centroidsBuffer = device.makeBuffer(
            bytes: centroidData, length: centroidData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        // Initialize empty compressed storage
        self.compressedKeys = Array(repeating: (
            indices: device.makeBuffer(length: 1, options: .storageModeShared)!,
            scales: device.makeBuffer(length: 1, options: .storageModeShared)!,
            numVecs: 0
        ), count: numLayers)
        self.compressedValues = self.compressedKeys
        
        kvqLogger.info("KVCacheQuantizer initialized: \(numLayers) layers, \(numHeads) heads, \(headDim) headDim")
    }
    
    // MARK: - Public API
    
    /// Compress a single layer's KV cache output from ORT.
    /// Called after each LM step with `present.{layer}.key` and `present.{layer}.value`.
    func compress(layerIndex: Int, key: ORTValue, value: ORTValue) throws {
        guard isEnabled else { return }
        
        let keyInfo = try key.tensorTypeAndShapeInfo()
        let seqLen = keyInfo.shape[2].intValue
        let numVecs = numHeads * seqLen
        
        // Compress key
        let keyData = try key.tensorData() as Data
        compressedKeys[layerIndex] = try quantizeOnGPU(
            fp16Data: keyData, numVecs: numVecs
        )
        
        // Compress value
        let valData = try value.tensorData() as Data
        compressedValues[layerIndex] = try quantizeOnGPU(
            fp16Data: valData, numVecs: numVecs
        )
    }
    
    /// Decompress a layer's KV cache back to fp16 ORTValue for ORT input.
    func decompressKey(layerIndex: Int) throws -> ORTValue {
        guard isEnabled else {
            throw KVQError.notEnabled
        }
        let entry = compressedKeys[layerIndex]
        let fp16Data = try dequantizeOnGPU(entry: entry)
        let seqLen = entry.numVecs / numHeads
        return try createFloat16ORTValue(
            data: fp16Data,
            shape: [1, NSNumber(value: numHeads), NSNumber(value: seqLen), NSNumber(value: headDim)]
        )
    }
    
    func decompressValue(layerIndex: Int) throws -> ORTValue {
        guard isEnabled else {
            throw KVQError.notEnabled
        }
        let entry = compressedValues[layerIndex]
        let fp16Data = try dequantizeOnGPU(entry: entry)
        let seqLen = entry.numVecs / numHeads
        return try createFloat16ORTValue(
            data: fp16Data,
            shape: [1, NSNumber(value: numHeads), NSNumber(value: seqLen), NSNumber(value: headDim)]
        )
    }
    
    /// Reset compressed storage between chunks.
    func reset() {
        for i in 0..<numLayers {
            compressedKeys[i].numVecs = 0
            compressedValues[i].numVecs = 0
        }
    }
    
    /// Memory usage of compressed KV cache (bytes).
    var compressedMemoryBytes: Int {
        var total = 0
        for i in 0..<numLayers {
            let kVecs = compressedKeys[i].numVecs
            let vVecs = compressedValues[i].numVecs
            // Packed indices: numVecs * (headDim/4) * 2 bytes
            // Scales: numVecs * 2 bytes (fp16)
            total += (kVecs + vVecs) * ((headDim / 4) * 2 + 2)
        }
        return total
    }
    
    /// Equivalent fp16 memory (bytes) - for comparison logging.
    var uncompressedMemoryBytes: Int {
        var total = 0
        for i in 0..<numLayers {
            let kVecs = compressedKeys[i].numVecs
            let vVecs = compressedValues[i].numVecs
            total += (kVecs + vVecs) * headDim * 2  // fp16 = 2 bytes
        }
        return total
    }
    
    // MARK: - Metal GPU Operations
    
    private func quantizeOnGPU(
        fp16Data: Data, numVecs: Int
    ) throws -> (indices: MTLBuffer, scales: MTLBuffer, numVecs: Int) {
        let inputBuffer = device.makeBuffer(
            bytes: (fp16Data as NSData).bytes,
            length: fp16Data.count,
            options: .storageModeShared
        )!
        
        let packedSize = numVecs * (headDim / 4) * MemoryLayout<UInt16>.size
        let scalesSize = numVecs * MemoryLayout<UInt16>.size  // fp16
        
        let outputBuffer = device.makeBuffer(length: packedSize, options: .storageModeShared)!
        let scalesBuffer = device.makeBuffer(length: scalesSize, options: .storageModeShared)!
        
        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw KVQError.metalCommandFailed
        }
        
        encoder.setComputePipelineState(quantizePSO)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(scalesBuffer, offset: 0, index: 2)
        encoder.setBuffer(rotationBuffer, offset: 0, index: 3)
        encoder.setBuffer(boundariesBuffer, offset: 0, index: 4)
        
        var totalVecs = UInt32(numVecs)
        encoder.setBytes(&totalVecs, length: MemoryLayout<UInt32>.size, index: 5)
        
        let threadgroupSize = min(quantizePSO.maxTotalThreadsPerThreadgroup, 256)
        let gridSize = MTLSize(width: numVecs, height: 1, depth: 1)
        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        
        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        
        return (indices: outputBuffer, scales: scalesBuffer, numVecs: numVecs)
    }
    
    private func dequantizeOnGPU(
        entry: (indices: MTLBuffer, scales: MTLBuffer, numVecs: Int)
    ) throws -> Data {
        let numVecs = entry.numVecs
        guard numVecs > 0 else { return Data() }
        
        let outputSize = numVecs * headDim * MemoryLayout<UInt16>.size  // fp16
        let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared)!
        
        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw KVQError.metalCommandFailed
        }
        
        encoder.setComputePipelineState(dequantizePSO)
        encoder.setBuffer(entry.indices, offset: 0, index: 0)
        encoder.setBuffer(entry.scales, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBuffer(rotationTBuffer, offset: 0, index: 3)  // R^T
        encoder.setBuffer(centroidsBuffer, offset: 0, index: 4)
        
        var totalVecs = UInt32(numVecs)
        encoder.setBytes(&totalVecs, length: MemoryLayout<UInt32>.size, index: 5)
        
        let threadgroupSize = min(dequantizePSO.maxTotalThreadsPerThreadgroup, 256)
        let gridSize = MTLSize(width: numVecs, height: 1, depth: 1)
        let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        
        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        
        return Data(bytes: outputBuffer.contents(), count: outputSize)
    }
    
    // MARK: - Helpers
    
    private func createFloat16ORTValue(data: Data, shape: [NSNumber]) throws -> ORTValue {
        let mutableData = NSMutableData(data: data)
        return try ORTValue(tensorData: mutableData, elementType: .float16, shape: shape)
    }
    
    private struct PolarQuantConstants {
        let rotationMatrix: [Double]
        let boundaries: [Double]
        let centroids: [Double]
    }
    
    private static func loadConstants() throws -> PolarQuantConstants {
        guard let url = Bundle.main.url(forResource: "polarquant_constants", withExtension: "json") else {
            throw KVQError.constantsNotFound
        }
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        return PolarQuantConstants(
            rotationMatrix: json["rotation_matrix"] as! [Double],
            boundaries: json["quantizer_boundaries"] as! [Double],
            centroids: json["quantizer_centroids"] as! [Double]
        )
    }
}

// MARK: - Errors

enum KVQError: Error, LocalizedError {
    case metalUnavailable
    case shaderNotFound
    case metalCommandFailed
    case constantsNotFound
    case notEnabled
    
    var errorDescription: String? {
        switch self {
        case .metalUnavailable: return "Metal GPU not available"
        case .shaderNotFound: return "PolarQuant Metal shaders not found"
        case .metalCommandFailed: return "Metal command buffer failed"
        case .constantsNotFound: return "polarquant_constants.json not found in bundle"
        case .notEnabled: return "KV cache quantization not enabled"
        }
    }
}
```

**Step 2: Commit**

```bash
git add Reader/Services/Metal/KVCacheQuantizer.swift
git commit -m "feat: add KVCacheQuantizer Swift wrapper for PolarQuant Metal pipeline"
```

---

### Task 4: Integrate into ChatterboxEngine Decode Loop

Wire up the quantizer between decode steps. Add feature flag, timing, and fallback.

**Files:**
- Modify: `Reader/Services/ChatterboxEngine.swift`

**Step 1: Add quantizer property and initialization**

In `ChatterboxEngine`, add after `private var metalDevice: MTLDevice?` (line ~170):

```swift
/// PolarQuant KV cache compressor — reduces KV memory by 4x between decode steps.
/// nil if Metal is unavailable (graceful fallback to fp16 pass-through).
private var kvCacheQuantizer: KVCacheQuantizer?

/// Feature flag: enable PolarQuant KV cache compression.
/// When false, KV cache is passed through as fp16 (original behavior).
private let usePolarQuantKV: Bool = true
```

**Step 2: Initialize quantizer in loadModels()**

After line ~252 (`isLoaded = true`), add:

```swift
// Initialize PolarQuant KV cache quantizer
if usePolarQuantKV {
    do {
        kvCacheQuantizer = try KVCacheQuantizer(
            numLayers: numLayers,  // Detected from LM input names
            numHeads: config.numKVHeads,
            headDim: config.headDim
        )
        chatterboxLogger.info("PolarQuant KV cache quantizer initialized")
    } catch {
        chatterboxLogger.warning("PolarQuant init failed, using fp16 KV cache: \(error)")
        kvCacheQuantizer = nil
    }
}
```

> **Note:** `numLayers` is not available in `loadModels()` since it's inferred from LM input names inside `synthesizeChunk()`. Move the quantizer initialization to first use, or detect layer count during model load by inspecting `languageModelSession.inputNames()`.

**Step 3: Modify KV cache carry-forward in decode loop**

Replace lines 1151-1161 (the KV cache carry-forward block) with:

```swift
// Carry KV-cache forward, optionally compressing via PolarQuant
if let quantizer = self.kvCacheQuantizer, quantizer.isEnabled {
    // Compress this step's full KV cache
    for layer in 0..<numLayers {
        if let key = lmOutputs["present.\(layer).key"],
           let val = lmOutputs["present.\(layer).value"] {
            try quantizer.compress(layerIndex: layer, key: key, value: val)
        }
    }
    // Decompress for next step's input
    for layer in 0..<numLayers {
        nextStepInputs["past_key_values.\(layer).key"] = try quantizer.decompressKey(layerIndex: layer)
        nextStepInputs["past_key_values.\(layer).value"] = try quantizer.decompressValue(layerIndex: layer)
    }
} else {
    // Original fp16 pass-through
    for layer in 0..<numLayers {
        if let kv = lmOutputs["present.\(layer).key"] {
            nextStepInputs["past_key_values.\(layer).key"] = kv
        }
        if let kv = lmOutputs["present.\(layer).value"] {
            nextStepInputs["past_key_values.\(layer).value"] = kv
        }
    }
}

// Log compression stats periodically
if step % 100 == 0, let q = self.kvCacheQuantizer {
    let compressed = q.compressedMemoryBytes
    let original = q.uncompressedMemoryBytes
    let ratio = original > 0 ? Double(original) / Double(compressed) : 0
    chatterboxLogger.info("KV cache step \(step): \(compressed/1024)KB compressed (\(String(format: "%.1f", ratio))x ratio)")
}
```

**Step 4: Add quantizer reset at chunk start**

At the beginning of `synthesizeChunk()` (after the guard statements, ~line 870), add:

```swift
// Reset compressed KV cache for this chunk
kvCacheQuantizer?.reset()
```

**Step 5: Add files to Xcode project**

Update `project.pbxproj` to include:
- `PolarQuantKernels.metal` in Compile Sources
- `KVCacheQuantizer.swift` in Compile Sources  
- `polarquant_constants.json` in Copy Bundle Resources

**Step 6: Commit**

```bash
git add Reader/Services/ChatterboxEngine.swift Reader.xcodeproj/project.pbxproj
git commit -m "feat: integrate PolarQuant KV cache compression into decode loop"
```

---

### Task 5: Validation and Quality Testing

Verify the quantization doesn't degrade TTS quality.

**Step 1: Build and run with PolarQuant enabled**

```bash
xcodebuild -project Reader.xcodeproj -scheme Reader \
  -destination 'generic/platform=iOS Simulator' \
  -sdk iphonesimulator build 2>&1 | tail -20
```

Expected: BUILD SUCCEEDED

**Step 2: Run on simulator and check logs**

Look for these log lines:
- `PolarQuant KV cache quantizer initialized` — init success
- `KV cache step 100: XXX KB compressed (3.8x ratio)` — compression working
- No crashes or garbled audio

**Step 3: A/B quality test**

Toggle `usePolarQuantKV` between `true` and `false`, synthesize the same text with the same voice and seed, and compare:
- Audio sounds intelligible (not garbled)
- No new repetition patterns
- Timing improvement (log total synthesis time)

**Step 4: If quality degrades, adjust sigma**

The `sigma` parameter in `generate_polarquant_constants.py` controls the quantizer's dynamic range. If audio is garbled:
1. Capture actual KV cache statistics by logging `norm` values from the quantize kernel
2. Regenerate constants with adjusted sigma
3. Rebuild

**Step 5: Commit final tuned version**

```bash
git add -A
git commit -m "feat: PolarQuant KV cache compression validated and tuned"
```

---

## Memory Impact Summary

| Step | fp16 (original) | 4-bit PolarQuant | Reduction |
|------|-----------------|------------------|-----------|
| 100  | 9.8 MB         | 2.7 MB           | 3.6x      |
| 300  | 29.5 MB        | 8.1 MB           | 3.6x      |
| 500  | 49.2 MB        | 13.5 MB          | 3.6x      |
| 1000 | 98.3 MB        | 27.0 MB          | 3.6x      |

> Compression ratio is ~3.6x (not full 4x) due to the per-vector scale stored as fp16.

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Audio quality degradation at d=64 | Feature flag for instant rollback; A/B test |
| Metal shader compilation failure on older devices | Graceful fallback to fp16 (nil quantizer) |
| Quantize/dequantize overhead exceeds memory savings | Log timing per step; disable if overhead > 1ms |
| Sigma mismatch with actual KV distributions | Log runtime stats, tune sigma empirically |
