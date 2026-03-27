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
/// **Incremental** quantization: each KV position is quantized exactly ONCE when
/// it first appears (as the last position in `present.{layer}.key/value`).
/// Subsequent steps dequantize from this single-quantized store, avoiding
/// compounding quantization error.
///
/// Memory layout per layer:
///   - packed indices: growing buffer of 4-bit packed uint16  [numHeads * seqLen * (headDim/4)]
///   - scales:         growing buffer of fp16 per-vector norms [numHeads * seqLen]
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
    private let packedPerVec: Int   // headDim / 4

    /// Toggle for A/B testing. When false, compress/decompress are no-ops.
    var isEnabled: Bool = true

    // MARK: - Incremental Compressed Storage

    /// Per-layer growing compressed store.
    /// Each step appends `numHeads` new vectors (one per head for the new token position).
    private struct CompressedStore {
        var indicesData: Data    // packed 4-bit uint16, grows by numHeads * packedPerVec * 2 bytes per step
        var scalesData: Data     // per-vector norm fp16, grows by numHeads * 2 bytes per step
        var seqLen: Int          // number of token positions stored
    }

    private var compressedKeys: [CompressedStore]
    private var compressedValues: [CompressedStore]

    // Pre-allocated scratch buffer for quantizing one step's new vectors
    // (numHeads vectors per layer)
    private var scratchInputBuffer: MTLBuffer
    private var scratchIndicesBuffer: MTLBuffer
    private var scratchScalesBuffer: MTLBuffer

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
        self.packedPerVec = headDim / 4

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

        // ── Pre-allocate scratch buffers for one step (numHeads vectors) ──
        let bytesPerVecFP16 = headDim * 2                       // input: fp16
        let bytesPerVecPacked = (headDim / 4) * 2               // output: packed uint16
        let bytesPerVecScale = 2                                 // output: fp16 norm

        self.scratchInputBuffer = device.makeBuffer(
            length: numHeads * bytesPerVecFP16, options: .storageModeShared
        )!
        self.scratchIndicesBuffer = device.makeBuffer(
            length: numHeads * bytesPerVecPacked, options: .storageModeShared
        )!
        self.scratchScalesBuffer = device.makeBuffer(
            length: numHeads * bytesPerVecScale, options: .storageModeShared
        )!

        // ── Empty storage ──
        let emptyStore = CompressedStore(indicesData: Data(), scalesData: Data(), seqLen: 0)
        self.compressedKeys = Array(repeating: emptyStore, count: numLayers)
        self.compressedValues = Array(repeating: emptyStore, count: numLayers)

        kvqLogger.info("KVCacheQuantizer ready: \(numLayers) layers, \(numHeads) heads, \(headDim)d, 4-bit incremental PolarQuant")
    }

    // MARK: - Public API

    /// Extract and compress ONLY the new token position from this step's `present` output.
    ///
    /// The `present` tensor has shape [1, numHeads, seqLen, headDim] in fp16.
    /// We extract the LAST position (index seqLen-1) across all heads, quantize those
    /// `numHeads` vectors, and append to our growing compressed store.
    ///
    /// This ensures each position is quantized exactly ONCE — no compounding error.
    func compressNewPosition(layerIndex: Int, key: ORTValue, value: ORTValue) throws {
        guard isEnabled else { return }

        let keyInfo = try key.tensorTypeAndShapeInfo()
        let seqLen = keyInfo.shape[2].intValue

        // Extract last position's vectors from key and value
        let keyData = try key.tensorData() as Data
        let valData = try value.tensorData() as Data

        let newKeyVecs = extractLastPosition(from: keyData, seqLen: seqLen)
        let newValVecs = extractLastPosition(from: valData, seqLen: seqLen)

        // Quantize the new vectors on GPU
        let compressedKey = try quantizeVectors(newKeyVecs)
        let compressedVal = try quantizeVectors(newValVecs)

        // Append to growing store
        compressedKeys[layerIndex].indicesData.append(compressedKey.indices)
        compressedKeys[layerIndex].scalesData.append(compressedKey.scales)
        compressedKeys[layerIndex].seqLen = seqLen

        compressedValues[layerIndex].indicesData.append(compressedVal.indices)
        compressedValues[layerIndex].scalesData.append(compressedVal.scales)
        compressedValues[layerIndex].seqLen = seqLen
    }

    /// Decompress a layer's full key cache to fp16 ORTValue.
    func decompressKey(layerIndex: Int) throws -> ORTValue {
        let store = compressedKeys[layerIndex]
        guard store.seqLen > 0 else {
            return try makeEmptyKVTensor()
        }
        let fp16Data = try dequantizeStore(store: store)
        return try makeFloat16ORTValue(
            data: fp16Data,
            shape: [1, NSNumber(value: numHeads), NSNumber(value: store.seqLen), NSNumber(value: headDim)]
        )
    }

    /// Decompress a layer's full value cache to fp16 ORTValue.
    func decompressValue(layerIndex: Int) throws -> ORTValue {
        let store = compressedValues[layerIndex]
        guard store.seqLen > 0 else {
            return try makeEmptyKVTensor()
        }
        let fp16Data = try dequantizeStore(store: store)
        return try makeFloat16ORTValue(
            data: fp16Data,
            shape: [1, NSNumber(value: numHeads), NSNumber(value: store.seqLen), NSNumber(value: headDim)]
        )
    }

    /// Reset all compressed storage (call between synthesis chunks).
    func reset() {
        for i in 0..<numLayers {
            compressedKeys[i] = CompressedStore(indicesData: Data(), scalesData: Data(), seqLen: 0)
            compressedValues[i] = CompressedStore(indicesData: Data(), scalesData: Data(), seqLen: 0)
        }
        kvqLogger.debug("KV cache quantizer reset")
    }

    // MARK: - Memory Stats

    /// Current compressed KV cache memory (bytes).
    var compressedMemoryBytes: Int {
        var total = 0
        for i in 0..<numLayers {
            total += compressedKeys[i].indicesData.count + compressedKeys[i].scalesData.count
            total += compressedValues[i].indicesData.count + compressedValues[i].scalesData.count
        }
        return total
    }

    /// Equivalent fp16 memory without compression (bytes).
    var uncompressedEquivalentBytes: Int {
        var total = 0
        for i in 0..<numLayers {
            let kVecs = numHeads * compressedKeys[i].seqLen
            let vVecs = numHeads * compressedValues[i].seqLen
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

    // MARK: - Internal: Extract Last Position

    /// Extract the last position (seqLen-1) from a KV tensor [1, numHeads, seqLen, headDim] fp16.
    /// Returns `numHeads` vectors of `headDim` fp16 elements, laid out contiguously.
    private func extractLastPosition(from tensorData: Data, seqLen: Int) -> Data {
        // Tensor layout: [1, numHeads, seqLen, headDim]  in fp16
        // For head h, position p: offset = (h * seqLen + p) * headDim * 2
        let bytesPerVec = headDim * 2  // fp16
        let lastPos = seqLen - 1
        var result = Data(capacity: numHeads * bytesPerVec)

        tensorData.withUnsafeBytes { rawPtr in
            let base = rawPtr.baseAddress!
            for h in 0..<numHeads {
                let offset = (h * seqLen + lastPos) * bytesPerVec
                result.append(base.advanced(by: offset).assumingMemoryBound(to: UInt8.self), count: bytesPerVec)
            }
        }

        return result
    }

    // MARK: - Internal: GPU Quantize (small batch)

    private struct QuantizedChunk {
        let indices: Data   // packed 4-bit
        let scales: Data    // fp16 norms
    }

    /// Quantize `numHeads` vectors on Metal GPU.
    private func quantizeVectors(_ fp16Data: Data) throws -> QuantizedChunk {
        let numVecs = numHeads

        // Copy input to scratch buffer
        fp16Data.withUnsafeBytes { ptr in
            memcpy(scratchInputBuffer.contents(), ptr.baseAddress!, fp16Data.count)
        }

        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw KVQError.metalCommandFailed
        }

        encoder.setComputePipelineState(quantizePSO)
        encoder.setBuffer(scratchInputBuffer, offset: 0, index: 0)
        encoder.setBuffer(scratchIndicesBuffer, offset: 0, index: 1)
        encoder.setBuffer(scratchScalesBuffer, offset: 0, index: 2)
        encoder.setBuffer(rotationBuffer, offset: 0, index: 3)
        encoder.setBuffer(boundariesBuffer, offset: 0, index: 4)

        var totalVecs = UInt32(numVecs)
        encoder.setBytes(&totalVecs, length: MemoryLayout<UInt32>.size, index: 5)

        let tgWidth = min(quantizePSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: numVecs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(tgWidth, numVecs), height: 1, depth: 1)
        )

        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if let error = cmdBuffer.error {
            kvqLogger.error("Quantize GPU error: \(error.localizedDescription)")
            throw KVQError.metalCommandFailed
        }

        // Read results
        let indicesSize = numVecs * packedPerVec * MemoryLayout<UInt16>.size
        let scalesSize = numVecs * MemoryLayout<UInt16>.size

        let indicesData = Data(bytes: scratchIndicesBuffer.contents(), count: indicesSize)
        let scalesData = Data(bytes: scratchScalesBuffer.contents(), count: scalesSize)

        return QuantizedChunk(indices: indicesData, scales: scalesData)
    }

    // MARK: - Internal: GPU Dequantize (full store)

    /// Dequantize the entire compressed store back to fp16.
    private func dequantizeStore(store: CompressedStore) throws -> Data {
        let numVecs = numHeads * store.seqLen
        guard numVecs > 0 else { return Data() }

        // Create GPU buffers from compressed data
        let indicesBuffer = device.makeBuffer(
            bytes: (store.indicesData as NSData).bytes,
            length: store.indicesData.count,
            options: .storageModeShared
        )!
        let scalesBuffer = device.makeBuffer(
            bytes: (store.scalesData as NSData).bytes,
            length: store.scalesData.count,
            options: .storageModeShared
        )!

        let outputSize = numVecs * headDim * MemoryLayout<UInt16>.size  // fp16
        let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared)!

        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw KVQError.metalCommandFailed
        }

        encoder.setComputePipelineState(dequantizePSO)
        encoder.setBuffer(indicesBuffer, offset: 0, index: 0)
        encoder.setBuffer(scalesBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBuffer(rotationTBuffer, offset: 0, index: 3)
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

        // The output is [numHeads * seqLen, headDim] in fp16, laid out as
        // head0_pos0, head0_pos1, ..., head0_posN, head1_pos0, ...
        // But ORT expects [1, numHeads, seqLen, headDim] which is the same layout.
        // Wait — our compressed store appends per-step: [h0_p0, h1_p0, ..., h0_p1, h1_p1, ...]
        // But ORT expects: [h0_p0, h0_p1, ..., h1_p0, h1_p1, ...]
        // We need to transpose! Let's do it on CPU (it's just memory copies).

        return transposeToORTLayout(
            from: Data(bytes: outputBuffer.contents(), count: outputSize),
            numHeads: numHeads,
            seqLen: store.seqLen,
            headDim: headDim
        )
    }

    /// Transpose from append-order [step, head, headDim] to ORT order [head, step, headDim].
    ///
    /// Append order (how we stored it):  step0_head0, step0_head1, ..., step1_head0, ...
    /// ORT order:                        head0_step0, head0_step1, ..., head1_step0, ...
    private func transposeToORTLayout(from data: Data, numHeads: Int, seqLen: Int, headDim: Int) -> Data {
        let bytesPerVec = headDim * 2  // fp16
        var result = Data(count: data.count)

        data.withUnsafeBytes { srcPtr in
            result.withUnsafeMutableBytes { dstPtr in
                let src = srcPtr.baseAddress!
                let dst = dstPtr.baseAddress!

                for step in 0..<seqLen {
                    for head in 0..<numHeads {
                        // Source: step * numHeads + head
                        let srcOffset = (step * numHeads + head) * bytesPerVec
                        // Dest: head * seqLen + step
                        let dstOffset = (head * seqLen + step) * bytesPerVec
                        memcpy(dst.advanced(by: dstOffset), src.advanced(by: srcOffset), bytesPerVec)
                    }
                }
            }
        }

        return result
    }

    // MARK: - Helpers

    private func makeFloat16ORTValue(data: Data, shape: [NSNumber]) throws -> ORTValue {
        let mutableData = NSMutableData(data: data)
        return try ORTValue(tensorData: mutableData, elementType: .float16, shape: shape)
    }

    private func makeEmptyKVTensor() throws -> ORTValue {
        let emptyData = NSMutableData()
        return try ORTValue(
            tensorData: emptyData,
            elementType: .float16,
            shape: [1, NSNumber(value: numHeads), 0, NSNumber(value: headDim)]
        )
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
