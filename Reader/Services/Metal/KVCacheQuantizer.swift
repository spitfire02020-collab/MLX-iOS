import Foundation
import Metal
import OnnxRuntimeBindings
import os.log

private let kvqLogger = Logger(subsystem: "com.reader.app", category: "KVCacheQuantizer")

// MARK: - TurboQuant Scheduler

/// Manages TurboQuant delayed quantization scheduling.
///
/// Tracks decode steps and determines whether each position should be
/// stored in fp16 (during warmup) or quantized (after warmup).
/// This is the first stage of Google's TurboQuant algorithm, which delays
/// quantization for the first N decode steps to preserve precision for
/// early, heavily-attended KV cache positions.
struct TurboQuantScheduler {

    /// Number of initial decode steps to keep in full fp16 precision.
    let warmupSteps: Int

    /// Current decode step (0-indexed, incremented once per decode step).
    private(set) var currentDecodeStep: Int = 0

    init(warmupSteps: Int = 8) {
        self.warmupSteps = max(0, warmupSteps)
    }

    /// Returns true if the current step is within the warmup period (full fp16).
    var isInWarmup: Bool {
        currentDecodeStep < warmupSteps
    }

    /// Returns true if quantization should be applied for the current step.
    var shouldQuantize: Bool {
        currentDecodeStep >= warmupSteps
    }

    /// Returns true if this step is the first quantized step (transition point).
    var isTransitionStep: Bool {
        currentDecodeStep == warmupSteps
    }

    /// Advance to the next decode step.
    mutating func advanceStep() {
        currentDecodeStep += 1
    }

    /// Reset the scheduler (e.g., between synthesis chunks).
    mutating func reset() {
        currentDecodeStep = 0
    }

    /// Number of warmup positions stored in fp16 so far.
    var warmupPositionsStored: Int {
        min(currentDecodeStep, warmupSteps)
    }

    /// Number of quantized positions stored so far.
    var quantizedPositionsStored: Int {
        max(0, currentDecodeStep - warmupSteps)
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
        case .metalUnavailable:    return "Metal GPU not available"
        case .shaderNotFound:      return "PolarQuant Metal shaders not found in default library"
        case .metalCommandFailed:  return "Metal command buffer execution failed"
        case .constantsNotFound:   return "polarquant_constants.json not found in bundle"
        case .notEnabled:          return "KV cache quantization not enabled"
        }
    }
}

// MARK: - KVCacheQuantizer

/// TurboQuant KV cache compressor using Metal GPU.
///
/// Implements Google's TurboQuant two-stage approach:
/// 1. **PolarQuant**: Random rotation + 4-bit scalar quantization
/// 2. **QJL**: 1-bit residual error correction (sign bits + mean magnitude)
///
/// **Delayed quantization**: The first `warmupSteps` decode steps are stored in
/// full fp16 precision. Subsequent steps are quantized using PolarQuant + QJL.
/// This preserves accuracy for early, heavily-attended KV positions.
///
/// **Incremental** quantization: each KV position is quantized exactly ONCE when
/// it first appears, avoiding compounding quantization error.
///
/// Memory layout per layer (after warmup transition):
///   - packed indices:  growing buffer of 4-bit packed uint16 (append-order)
///   - scales:          growing buffer of fp16 per-vector norms (append-order)
///   - QJL sign bits:   1-bit residual direction per dimension (8 bytes/vec)
///   - QJL magnitudes:  fp16 mean residual magnitude per vector
final class KVCacheQuantizer {

    // MARK: - Metal State

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let quantizePSO: MTLComputePipelineState
    private let dequantizePSO: MTLComputePipelineState
    private let turboQuantizePSO: MTLComputePipelineState
    private let turboDequantizePSO: MTLComputePipelineState

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

    /// Number of initial decode steps to keep in full fp16 precision.
    let warmupSteps: Int

    /// Toggle for A/B testing. When false, compress/decompress are no-ops.
    var isEnabled: Bool = true

    /// TurboQuant delay scheduler. Tracks warmup/quantized state transitions.
    var scheduler: TurboQuantScheduler

    // MARK: - Compressed Storage

    /// Per-layer compressed store using TurboQuant (PolarQuant + QJL).
    ///
    /// During warmup, the engine passes present tensors directly (fp16 pass-through)
    /// without involving the quantizer. At the warmup-to-quantized transition, the
    /// engine calls `bootstrapFromFullPresent` to quantize ALL accumulated positions.
    /// Subsequent steps call `compressNewPosition` to add one position at a time.
    private struct CompressedStore {
        // Quantized: PolarQuant + QJL compressed (append-order)
        var indicesData: Data        // packed 4-bit uint16
        var scalesData: Data         // per-vector norm fp16
        var qjlSignBits: Data        // 1-bit residual signs [quantizedVecCount * 8 bytes]
        var qjlMagnitudes: Data      // fp16 mean residual magnitude
        var quantizedPositions: Int  // Number of sequence positions quantized

        /// Total number of positions per head in this store.
        var totalSeqLen: Int { quantizedPositions }
    }

    private var compressedKeys: [CompressedStore]
    private var compressedValues: [CompressedStore]

    // Pre-allocated scratch buffers for quantizing one step's new vectors
    // (numHeads vectors per layer)
    private var scratchInputBuffer: MTLBuffer
    private var scratchIndicesBuffer: MTLBuffer
    private var scratchScalesBuffer: MTLBuffer
    private var scratchQJLSignsBuffer: MTLBuffer
    private var scratchQJLMagsBuffer: MTLBuffer

    // MARK: - Init

    init(numLayers: Int, numHeads: Int, headDim: Int, warmupSteps: Int = 8) throws {
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
        self.warmupSteps = max(0, warmupSteps)
        self.scheduler = TurboQuantScheduler(warmupSteps: max(0, warmupSteps))

        // ── Load Metal shaders ──
        guard let library = device.makeDefaultLibrary(),
              let quantizeFn = library.makeFunction(name: "polarquant_quantize"),
              let dequantizeFn = library.makeFunction(name: "polarquant_dequantize"),
              let turboQuantizeFn = library.makeFunction(name: "turboquant_quantize"),
              let turboDequantizeFn = library.makeFunction(name: "turboquant_dequantize") else {
            throw KVQError.shaderNotFound
        }

        self.quantizePSO = try device.makeComputePipelineState(function: quantizeFn)
        self.dequantizePSO = try device.makeComputePipelineState(function: dequantizeFn)
        self.turboQuantizePSO = try device.makeComputePipelineState(function: turboQuantizeFn)
        self.turboDequantizePSO = try device.makeComputePipelineState(function: turboDequantizeFn)

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
        let bytesPerVecQJLSigns = 8                              // output: 8 bytes (64 sign bits)
        let bytesPerVecQJLMag = 2                                // output: fp16 magnitude

        self.scratchInputBuffer = device.makeBuffer(
            length: numHeads * bytesPerVecFP16, options: .storageModeShared
        )!
        self.scratchIndicesBuffer = device.makeBuffer(
            length: numHeads * bytesPerVecPacked, options: .storageModeShared
        )!
        self.scratchScalesBuffer = device.makeBuffer(
            length: numHeads * bytesPerVecScale, options: .storageModeShared
        )!
        self.scratchQJLSignsBuffer = device.makeBuffer(
            length: numHeads * bytesPerVecQJLSigns, options: .storageModeShared
        )!
        self.scratchQJLMagsBuffer = device.makeBuffer(
            length: numHeads * bytesPerVecQJLMag, options: .storageModeShared
        )!

        // ── Empty storage ──
        let emptyStore = CompressedStore(
            indicesData: Data(), scalesData: Data(),
            qjlSignBits: Data(), qjlMagnitudes: Data(),
            quantizedPositions: 0
        )
        self.compressedKeys = Array(repeating: emptyStore, count: numLayers)
        self.compressedValues = Array(repeating: emptyStore, count: numLayers)

        kvqLogger.info("KVCacheQuantizer ready: \(numLayers) layers, \(numHeads) heads, \(headDim)d, TurboQuant (PolarQuant 4-bit + QJL 1-bit), warmup=\(warmupSteps) steps")
    }

    // MARK: - Public API

    // Track step count for diagnostic logging (per-layer calls, not decode steps)
    private var stepCount: Int = 0

    /// Bootstrap the quantizer by quantizing ALL positions from a full `present` tensor.
    ///
    /// Called at the transition from warmup to quantized mode. The engine has been
    /// passing present tensors directly (fp16) during warmup. At the transition point,
    /// this method quantizes every position in the current full KV cache tensor using
    /// TurboQuant (PolarQuant + QJL) and stores them. After this call, subsequent
    /// `compressNewPosition` calls add one position at a time.
    func bootstrapFromFullPresent(layerIndex: Int, key: ORTValue, value: ORTValue) throws {
        guard isEnabled else { return }

        let keyInfo = try key.tensorTypeAndShapeInfo()
        let seqLen = keyInfo.shape[2].intValue

        let keyData = try key.tensorData() as Data
        let valData = try value.tensorData() as Data

        kvqLogger.info("TurboQuant bootstrap layer \(layerIndex): quantizing \(seqLen) positions")

        // Quantize each position individually using the scratch buffer
        for pos in 0..<seqLen {
            let keyVecs = extractPosition(from: keyData, position: pos, seqLen: seqLen)
            let valVecs = extractPosition(from: valData, position: pos, seqLen: seqLen)

            let compressedKey = try turboQuantizeVectors(keyVecs)
            let compressedVal = try turboQuantizeVectors(valVecs)

            compressedKeys[layerIndex].indicesData.append(compressedKey.indices)
            compressedKeys[layerIndex].scalesData.append(compressedKey.scales)
            compressedKeys[layerIndex].qjlSignBits.append(compressedKey.qjlSigns)
            compressedKeys[layerIndex].qjlMagnitudes.append(compressedKey.qjlMags)

            compressedValues[layerIndex].indicesData.append(compressedVal.indices)
            compressedValues[layerIndex].scalesData.append(compressedVal.scales)
            compressedValues[layerIndex].qjlSignBits.append(compressedVal.qjlSigns)
            compressedValues[layerIndex].qjlMagnitudes.append(compressedVal.qjlMags)
        }

        compressedKeys[layerIndex].quantizedPositions = seqLen
        compressedValues[layerIndex].quantizedPositions = seqLen
    }

    /// Extract and compress ONLY the new token position from this step's `present` output.
    ///
    /// The `present` tensor has shape [1, numHeads, seqLen, headDim] in fp16.
    /// We extract the LAST position (index seqLen-1) across all heads, quantize those
    /// `numHeads` vectors using TurboQuant (PolarQuant + QJL), and append to our
    /// growing compressed store.
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

        // TurboQuant: quantize using PolarQuant + QJL
        let compressedKey = try turboQuantizeVectors(newKeyVecs)
        let compressedVal = try turboQuantizeVectors(newValVecs)

        // Append to growing quantized store
        compressedKeys[layerIndex].indicesData.append(compressedKey.indices)
        compressedKeys[layerIndex].scalesData.append(compressedKey.scales)
        compressedKeys[layerIndex].qjlSignBits.append(compressedKey.qjlSigns)
        compressedKeys[layerIndex].qjlMagnitudes.append(compressedKey.qjlMags)
        compressedKeys[layerIndex].quantizedPositions += 1

        compressedValues[layerIndex].indicesData.append(compressedVal.indices)
        compressedValues[layerIndex].scalesData.append(compressedVal.scales)
        compressedValues[layerIndex].qjlSignBits.append(compressedVal.qjlSigns)
        compressedValues[layerIndex].qjlMagnitudes.append(compressedVal.qjlMags)
        compressedValues[layerIndex].quantizedPositions += 1

        stepCount += 1
    }

    /// Decompress a layer's full key cache to fp16 ORTValue.
    ///
    /// Combines warmup fp16 data (if any) with dequantized TurboQuant data,
    /// producing a tensor in ORT-order [1, numHeads, totalSeqLen, headDim].
    func decompressKey(layerIndex: Int) throws -> ORTValue {
        let store = compressedKeys[layerIndex]
        guard store.totalSeqLen > 0 else {
            return try makeEmptyKVTensor()
        }
        let fp16Data = try decompressStore(store: store)
        return try makeFloat16ORTValue(
            data: fp16Data,
            shape: [1, NSNumber(value: numHeads), NSNumber(value: store.totalSeqLen), NSNumber(value: headDim)]
        )
    }

    /// Decompress a layer's full value cache to fp16 ORTValue.
    func decompressValue(layerIndex: Int) throws -> ORTValue {
        let store = compressedValues[layerIndex]
        guard store.totalSeqLen > 0 else {
            return try makeEmptyKVTensor()
        }
        let fp16Data = try decompressStore(store: store)
        return try makeFloat16ORTValue(
            data: fp16Data,
            shape: [1, NSNumber(value: numHeads), NSNumber(value: store.totalSeqLen), NSNumber(value: headDim)]
        )
    }

    /// Reset all compressed storage (call between synthesis chunks).
    func reset() {
        let emptyStore = CompressedStore(
            indicesData: Data(), scalesData: Data(),
            qjlSignBits: Data(), qjlMagnitudes: Data(),
            quantizedPositions: 0
        )
        for i in 0..<numLayers {
            compressedKeys[i] = emptyStore
            compressedValues[i] = emptyStore
        }
        scheduler.reset()
        stepCount = 0
        kvqLogger.debug("TurboQuant KV cache quantizer reset")
    }

    // MARK: - Memory Stats

    /// Current compressed KV cache memory (bytes).
    var compressedMemoryBytes: Int {
        var total = 0
        for i in 0..<numLayers {
            // Quantized (PolarQuant indices + scales + QJL signs + magnitudes)
            total += compressedKeys[i].indicesData.count + compressedKeys[i].scalesData.count
            total += compressedKeys[i].qjlSignBits.count + compressedKeys[i].qjlMagnitudes.count
            total += compressedValues[i].indicesData.count + compressedValues[i].scalesData.count
            total += compressedValues[i].qjlSignBits.count + compressedValues[i].qjlMagnitudes.count
        }
        return total
    }

    /// Equivalent fp16 memory without compression (bytes).
    var uncompressedEquivalentBytes: Int {
        var total = 0
        for i in 0..<numLayers {
            let kTotal = compressedKeys[i].totalSeqLen
            let vTotal = compressedValues[i].totalSeqLen
            total += (numHeads * kTotal + numHeads * vTotal) * headDim * 2
        }
        return total
    }

    /// Compression ratio (e.g. 3.6).
    var compressionRatio: Double {
        let compressed = compressedMemoryBytes
        guard compressed > 0 else { return 0 }
        return Double(uncompressedEquivalentBytes) / Double(compressed)
    }

    // MARK: - Internal: Extract Positions

    /// Extract the last position (seqLen-1) from a KV tensor [1, numHeads, seqLen, headDim] fp16.
    /// Returns `numHeads` vectors of `headDim` fp16 elements in append-order.
    private func extractLastPosition(from tensorData: Data, seqLen: Int) -> Data {
        return extractPosition(from: tensorData, position: seqLen - 1, seqLen: seqLen)
    }

    /// Extract a specific position from a KV tensor [1, numHeads, seqLen, headDim] fp16.
    /// Returns `numHeads` vectors of `headDim` fp16 elements in append-order.
    private func extractPosition(from tensorData: Data, position: Int, seqLen: Int) -> Data {
        // Tensor layout: [1, numHeads, seqLen, headDim]  in fp16
        // For head h, position p: offset = (h * seqLen + p) * headDim * 2
        let bytesPerVec = headDim * 2  // fp16
        var result = Data(capacity: numHeads * bytesPerVec)

        tensorData.withUnsafeBytes { rawPtr in
            let base = rawPtr.baseAddress!
            for h in 0..<numHeads {
                let offset = (h * seqLen + position) * bytesPerVec
                result.append(base.advanced(by: offset).assumingMemoryBound(to: UInt8.self), count: bytesPerVec)
            }
        }

        return result
    }

    // MARK: - Internal: Append-to-ORT Transpose

    /// Transpose data from append-order to ORT-order.
    /// Append-order: [s0_h0, s0_h1, ..., s1_h0, ...]  (step-major)
    /// ORT-order:    [h0_s0, h0_s1, ..., h1_s0, ...]  (head-major)
    private func transposeAppendToORT(appendData: Data, numPositions: Int) -> Data {
        let bytesPerVec = headDim * 2  // fp16
        let totalBytes = numPositions * numHeads * bytesPerVec
        var result = Data(count: totalBytes)

        appendData.withUnsafeBytes { src in
            result.withUnsafeMutableBytes { dst in
                for h in 0..<numHeads {
                    for s in 0..<numPositions {
                        let appendOffset = (s * numHeads + h) * bytesPerVec
                        let ortOffset = (h * numPositions + s) * bytesPerVec
                        memcpy(dst.baseAddress!.advanced(by: ortOffset),
                               src.baseAddress!.advanced(by: appendOffset),
                               bytesPerVec)
                    }
                }
            }
        }

        return result
    }

    // MARK: - Internal: GPU TurboQuant (PolarQuant + QJL)

    private struct TurboQuantizedChunk {
        let indices: Data   // packed 4-bit
        let scales: Data    // fp16 norms
        let qjlSigns: Data  // 1-bit residual signs
        let qjlMags: Data   // fp16 mean residual magnitude
    }

    /// Quantize `numHeads` vectors using TurboQuant (PolarQuant + QJL) on Metal GPU.
    private func turboQuantizeVectors(_ fp16Data: Data) throws -> TurboQuantizedChunk {
        let numVecs = numHeads

        // Copy input to scratch buffer
        _ = fp16Data.withUnsafeBytes { ptr in
            memcpy(scratchInputBuffer.contents(), ptr.baseAddress!, fp16Data.count)
        }

        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw KVQError.metalCommandFailed
        }

        encoder.setComputePipelineState(turboQuantizePSO)
        encoder.setBuffer(scratchInputBuffer, offset: 0, index: 0)
        encoder.setBuffer(scratchIndicesBuffer, offset: 0, index: 1)
        encoder.setBuffer(scratchScalesBuffer, offset: 0, index: 2)
        encoder.setBuffer(scratchQJLSignsBuffer, offset: 0, index: 3)
        encoder.setBuffer(scratchQJLMagsBuffer, offset: 0, index: 4)
        encoder.setBuffer(rotationBuffer, offset: 0, index: 5)
        encoder.setBuffer(rotationTBuffer, offset: 0, index: 6)
        encoder.setBuffer(boundariesBuffer, offset: 0, index: 7)
        encoder.setBuffer(centroidsBuffer, offset: 0, index: 8)

        var totalVecs = UInt32(numVecs)
        encoder.setBytes(&totalVecs, length: MemoryLayout<UInt32>.size, index: 9)

        let tgWidth = min(turboQuantizePSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: numVecs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(tgWidth, numVecs), height: 1, depth: 1)
        )

        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if let error = cmdBuffer.error {
            kvqLogger.error("TurboQuant GPU error: \(error.localizedDescription)")
            throw KVQError.metalCommandFailed
        }

        // Read results
        let indicesSize = numVecs * packedPerVec * MemoryLayout<UInt16>.size
        let scalesSize = numVecs * MemoryLayout<UInt16>.size
        let qjlSignsSize = numVecs * 8  // 8 bytes per vector (64 sign bits)
        let qjlMagsSize = numVecs * MemoryLayout<UInt16>.size

        let indicesData = Data(bytes: scratchIndicesBuffer.contents(), count: indicesSize)
        let scalesData = Data(bytes: scratchScalesBuffer.contents(), count: scalesSize)
        let qjlSignsData = Data(bytes: scratchQJLSignsBuffer.contents(), count: qjlSignsSize)
        let qjlMagsData = Data(bytes: scratchQJLMagsBuffer.contents(), count: qjlMagsSize)

        return TurboQuantizedChunk(
            indices: indicesData,
            scales: scalesData,
            qjlSigns: qjlSignsData,
            qjlMags: qjlMagsData
        )
    }

    // MARK: - Internal: GPU Dequantize (TurboQuant with QJL correction)

    /// Dequantize the quantized portion of a store using TurboQuant (PolarQuant + QJL).
    /// Returns fp16 data in append-order.
    private func turboDequantizeStore(store: CompressedStore) throws -> Data {
        let numVecs = numHeads * store.quantizedPositions
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
        let qjlSignsBuffer = device.makeBuffer(
            bytes: (store.qjlSignBits as NSData).bytes,
            length: store.qjlSignBits.count,
            options: .storageModeShared
        )!
        let qjlMagsBuffer = device.makeBuffer(
            bytes: (store.qjlMagnitudes as NSData).bytes,
            length: store.qjlMagnitudes.count,
            options: .storageModeShared
        )!

        let outputSize = numVecs * headDim * MemoryLayout<UInt16>.size  // fp16
        let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared)!

        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw KVQError.metalCommandFailed
        }

        encoder.setComputePipelineState(turboDequantizePSO)
        encoder.setBuffer(indicesBuffer, offset: 0, index: 0)
        encoder.setBuffer(scalesBuffer, offset: 0, index: 1)
        encoder.setBuffer(qjlSignsBuffer, offset: 0, index: 2)
        encoder.setBuffer(qjlMagsBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer, offset: 0, index: 4)
        encoder.setBuffer(rotationTBuffer, offset: 0, index: 5)
        encoder.setBuffer(centroidsBuffer, offset: 0, index: 6)

        var totalVecs = UInt32(numVecs)
        encoder.setBytes(&totalVecs, length: MemoryLayout<UInt32>.size, index: 7)

        let tgWidth = min(turboDequantizePSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: numVecs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
        )

        encoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if let error = cmdBuffer.error {
            kvqLogger.error("TurboDequantize GPU error: \(error.localizedDescription)")
            throw KVQError.metalCommandFailed
        }

        return Data(bytes: outputBuffer.contents(), count: outputSize)
    }

    // MARK: - Internal: Decompression

    /// Decompress a store by dequantizing TurboQuant data and transposing to ORT-order.
    /// Returns fp16 data in ORT-order [numHeads, totalSeqLen, headDim].
    private func decompressStore(store: CompressedStore) throws -> Data {
        guard store.quantizedPositions > 0 else { return Data() }

        let appendOrderData = try turboDequantizeStore(store: store)
        return transposeAppendToORT(appendData: appendOrderData, numPositions: store.quantizedPositions)
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
