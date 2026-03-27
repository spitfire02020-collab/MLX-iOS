import Foundation
import Metal
import OnnxRuntimeBindings
import os.log

private let kvqLogger = Logger(subsystem: "com.reader.app", category: "KVCacheQuantizer")

// MARK: - Errors

enum KVQError: Error, LocalizedError {
    case metalUnavailable
    case shaderNotFound
    case metalCommandFailed
    case constantsNotFound
    case notEnabled

    var errorDescription: String? {
        switch self {
        case .metalUnavailable:    return "Metal GPU not available"
        case .shaderNotFound:      return "PolarQuant Metal shaders not found in default library"
        case .metalCommandFailed:  return "Metal command buffer execution failed"
        case .constantsNotFound:   return "polarquant_constants.json not found in bundle"
        case .notEnabled:          return "KV cache quantization not enabled"
        }
    }
}

// MARK: - KVCacheQuantizer

/// PolarQuant 4-bit KV cache compressor using Metal GPU.
///
/// Compresses KV cache tensors from fp16 → 4-bit between decode steps,
/// reducing memory by ~3.6x. Reconstructs fp16 ORTValues for ORT input.
///
/// Usage in decode loop:
/// ```
/// // After LM step — compress
/// for layer in 0..<numLayers {
///     try quantizer.compress(layerIndex: layer,
///                            key: outputs["present.\(layer).key"]!,
///                            value: outputs["present.\(layer).value"]!)
/// }
/// // Before next step — decompress
/// for layer in 0..<numLayers {
///     inputs["past_key_values.\(layer).key"]   = try quantizer.decompressKey(layerIndex: layer)
///     inputs["past_key_values.\(layer).value"] = try quantizer.decompressValue(layerIndex: layer)
/// }
/// ```
final class KVCacheQuantizer {

    // MARK: - Metal State

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let quantizePSO: MTLComputePipelineState
    private let dequantizePSO: MTLComputePipelineState

    // MARK: - PolarQuant Constants (GPU-resident)

    private let rotationBuffer: MTLBuffer       // [64*64] float32  — R for quantize
    private let rotationTBuffer: MTLBuffer      // [64*64] float32  — R^T for dequantize
    private let boundariesBuffer: MTLBuffer     // [15] float32
    private let centroidsBuffer: MTLBuffer      // [16] float32

    // MARK: - Config

    let numLayers: Int
    let numHeads: Int
    let headDim: Int

    /// Toggle for A/B testing. When false, compress/decompress are no-ops.
    var isEnabled: Bool = true

    // MARK: - Compressed Storage

    /// Per-layer compressed key/value data.
    private struct CompressedEntry {
        var indices: MTLBuffer   // packed 4-bit uint16  [numVecs * (headDim/4)]
        var scales: MTLBuffer    // per-vector norm fp16  [numVecs]
        var numVecs: Int         // numHeads * seqLen
    }

    private var compressedKeys: [CompressedEntry]
    private var compressedValues: [CompressedEntry]

    // MARK: - Init

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

        // ── Load Metal shaders ──
        guard let library = device.makeDefaultLibrary(),
              let quantizeFn = library.makeFunction(name: "polarquant_quantize"),
              let dequantizeFn = library.makeFunction(name: "polarquant_dequantize") else {
            throw KVQError.shaderNotFound
        }

        self.quantizePSO = try device.makeComputePipelineState(function: quantizeFn)
        self.dequantizePSO = try device.makeComputePipelineState(function: dequantizeFn)

        // ── Load constants from JSON ──
        let constants = try Self.loadConstants()

        // Rotation matrix R (row-major)
        let rotationData = constants.rotationMatrix
        self.rotationBuffer = device.makeBuffer(
            bytes: rotationData,
            length: rotationData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        // R^T (transpose for dequantize)
        var rotationT = [Float](repeating: 0, count: headDim * headDim)
        for i in 0..<headDim {
            for j in 0..<headDim {
                rotationT[i * headDim + j] = rotationData[j * headDim + i]
            }
        }
        self.rotationTBuffer = device.makeBuffer(
            bytes: rotationT,
            length: rotationT.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        // Boundaries [15]
        let boundaryData = constants.boundaries
        self.boundariesBuffer = device.makeBuffer(
            bytes: boundaryData,
            length: boundaryData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        // Centroids [16]
        let centroidData = constants.centroids
        self.centroidsBuffer = device.makeBuffer(
            bytes: centroidData,
            length: centroidData.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!

        // ── Empty storage ──
        let emptyEntry = CompressedEntry(
            indices: device.makeBuffer(length: 4, options: .storageModeShared)!,
            scales: device.makeBuffer(length: 4, options: .storageModeShared)!,
            numVecs: 0
        )
        self.compressedKeys = Array(repeating: emptyEntry, count: numLayers)
        self.compressedValues = Array(repeating: emptyEntry, count: numLayers)

        kvqLogger.info("KVCacheQuantizer ready: \(numLayers) layers, \(numHeads) heads, \(headDim)d, 4-bit PolarQuant")
    }

    // MARK: - Public API

    /// Compress a layer's KV cache from ORT `present` outputs.
    func compress(layerIndex: Int, key: ORTValue, value: ORTValue) throws {
        guard isEnabled else { return }

        let keyInfo = try key.tensorTypeAndShapeInfo()
        let seqLen = keyInfo.shape[2].intValue
        let numVecs = numHeads * seqLen

        let keyData = try key.tensorData() as Data
        compressedKeys[layerIndex] = try quantizeOnGPU(fp16Data: keyData, numVecs: numVecs)

        let valData = try value.tensorData() as Data
        compressedValues[layerIndex] = try quantizeOnGPU(fp16Data: valData, numVecs: numVecs)
    }

    /// Decompress a layer's key cache to fp16 ORTValue.
    func decompressKey(layerIndex: Int) throws -> ORTValue {
        let entry = compressedKeys[layerIndex]
        let fp16Data = try dequantizeOnGPU(entry: entry)
        let seqLen = entry.numVecs / numHeads
        return try makeFloat16ORTValue(
            data: fp16Data,
            shape: [1, NSNumber(value: numHeads), NSNumber(value: seqLen), NSNumber(value: headDim)]
        )
    }

    /// Decompress a layer's value cache to fp16 ORTValue.
    func decompressValue(layerIndex: Int) throws -> ORTValue {
        let entry = compressedValues[layerIndex]
        let fp16Data = try dequantizeOnGPU(entry: entry)
        let seqLen = entry.numVecs / numHeads
        return try makeFloat16ORTValue(
            data: fp16Data,
            shape: [1, NSNumber(value: numHeads), NSNumber(value: seqLen), NSNumber(value: headDim)]
        )
    }

    /// Reset all compressed storage (call between synthesis chunks).
    func reset() {
        for i in 0..<numLayers {
            compressedKeys[i].numVecs = 0
            compressedValues[i].numVecs = 0
        }
        kvqLogger.debug("KV cache quantizer reset")
    }

    // MARK: - Memory Stats

    /// Current compressed KV cache memory (bytes).
    var compressedMemoryBytes: Int {
        var total = 0
        for i in 0..<numLayers {
            // packed indices: numVecs * (headDim/4) * sizeof(uint16) = numVecs * 32
            // scales: numVecs * sizeof(fp16) = numVecs * 2
            let kVecs = compressedKeys[i].numVecs
            let vVecs = compressedValues[i].numVecs
            total += (kVecs + vVecs) * ((headDim / 4) * 2 + 2)
        }
        return total
    }

    /// Equivalent fp16 memory without compression (bytes).
    var uncompressedEquivalentBytes: Int {
        var total = 0
        for i in 0..<numLayers {
            let kVecs = compressedKeys[i].numVecs
            let vVecs = compressedValues[i].numVecs
            total += (kVecs + vVecs) * headDim * 2
        }
        return total
    }

    /// Compression ratio (e.g. 3.6).
    var compressionRatio: Double {
        let compressed = compressedMemoryBytes
        guard compressed > 0 else { return 0 }
        return Double(uncompressedEquivalentBytes) / Double(compressed)
    }

    // MARK: - Metal GPU Operations

    private func quantizeOnGPU(fp16Data: Data, numVecs: Int) throws -> CompressedEntry {
        let inputBuffer = device.makeBuffer(
            bytes: (fp16Data as NSData).bytes,
            length: fp16Data.count,
            options: .storageModeShared
        )!

        let packedSize = numVecs * (headDim / 4) * MemoryLayout<UInt16>.size
        let scalesSize = numVecs * MemoryLayout<UInt16>.size  // fp16

        let outputBuffer = device.makeBuffer(length: max(packedSize, 4), options: .storageModeShared)!
        let scalesBuffer = device.makeBuffer(length: max(scalesSize, 4), options: .storageModeShared)!

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

        let tgWidth = min(quantizePSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: numVecs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
        )

        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if let error = cmdBuffer.error {
            kvqLogger.error("Quantize GPU error: \(error.localizedDescription)")
            throw KVQError.metalCommandFailed
        }

        return CompressedEntry(indices: outputBuffer, scales: scalesBuffer, numVecs: numVecs)
    }

    private func dequantizeOnGPU(entry: CompressedEntry) throws -> Data {
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

        let tgWidth = min(dequantizePSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: numVecs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
        )

        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if let error = cmdBuffer.error {
            kvqLogger.error("Dequantize GPU error: \(error.localizedDescription)")
            throw KVQError.metalCommandFailed
        }

        return Data(bytes: outputBuffer.contents(), count: outputSize)
    }

    // MARK: - Helpers

    private func makeFloat16ORTValue(data: Data, shape: [NSNumber]) throws -> ORTValue {
        let mutableData = NSMutableData(data: data)
        return try ORTValue(tensorData: mutableData, elementType: .float16, shape: shape)
    }

    // MARK: - Constants Loading

    private struct PolarQuantConstants {
        let rotationMatrix: [Float]
        let boundaries: [Float]
        let centroids: [Float]
    }

    private static func loadConstants() throws -> PolarQuantConstants {
        guard let url = Bundle.main.url(forResource: "polarquant_constants", withExtension: "json") else {
            throw KVQError.constantsNotFound
        }
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw KVQError.constantsNotFound
        }

        // Convert Double arrays from JSON to Float arrays
        guard let rotMatrix = json["rotation_matrix"] as? [Double],
              let bounds = json["quantizer_boundaries"] as? [Double],
              let cents = json["quantizer_centroids"] as? [Double] else {
            throw KVQError.constantsNotFound
        }

        return PolarQuantConstants(
            rotationMatrix: rotMatrix.map { Float($0) },
            boundaries: bounds.map { Float($0) },
            centroids: cents.map { Float($0) }
        )
    }
}
