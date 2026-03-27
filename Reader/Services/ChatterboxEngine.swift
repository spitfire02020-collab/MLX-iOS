import Foundation
@preconcurrency import AVFoundation
import AudioToolbox
import Accelerate
import Metal
import OnnxRuntimeBindings
import OnnxRuntimeExtensions
import os.log

// MARK: - LanguageModelBackend Protocol

/// Unified protocol for language model inference backends.
/// Allows swapping between ONNX Runtime and Metal GPU implementations.
public protocol LanguageModelBackend: Sendable {
    /// Initialize the backend with model configuration.
    func initialize(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) async throws

    /// Run one forward pass.
    /// - Parameters:
    ///   - inputsEmbds: Input embeddings [1, seqLen, hiddenDim] half
    ///   - kvWriteOffset: Position in KV cache to write this step's keys/values
    ///   - kvReadLength: Total positions to attend over (kvWriteOffset + 1)
    ///   - commandBuffer: Metal command buffer to encode into
    /// - Returns: Logits buffer [vocabSize] float32 at the last position
    func forward(
        inputsEmbds: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer

    /// Reset all KV cache state.
    func reset() async
}

/// Errors for language model backend operations.
public enum LMBackendError: Error, LocalizedError, Sendable {
    case notInitialized
    case metalDeviceUnavailable
    case kernelNotFound(String)
    case commandBufferFailed
    case invalidInputShape

    public var errorDescription: String? {
        switch self {
        case .notInitialized:        return "Backend not initialized"
        case .metalDeviceUnavailable: return "Metal device unavailable"
        case .kernelNotFound(let n): return "Metal kernel not found: \(n)"
        case .commandBufferFailed:   return "Metal command buffer failed"
        case .invalidInputShape:     return "Invalid input shape"
        }
    }
}

// MARK: - Logger
private let chatterboxLogger = Logger(subsystem: "com.reader.app", category: "ChatterboxEngine")

// MARK: - Thread-Safe Audio Results Collector
/// Actor for thread-safe collection of audio results from concurrent synthesis tasks
actor AudioResultsCollector {
    private var results: [(Int, [Float])] = []

    func append(_ result: (Int, [Float])) {
        results.append(result)
    }

    func getSortedResults() -> [(Int, [Float])] {
        results.sorted { $0.0 < $1.0 }
    }

    func clear() {
        results.removeAll()
    }
}

// MARK: - Chatterbox TTS Engine
// On-device text-to-speech using Chatterbox Turbo ONNX models
// Pipeline: Text -> Tokenize -> Embed -> Language Model -> Decode Audio

// MARK: - Engine Configuration

struct ChatterboxConfig {
    let sampleRate: Int = 24000
    let startSpeechToken: Int = 6561
    let stopSpeechToken: Int = 6562
    let silenceToken: Int = 4299
    let numKVHeads: Int = 16
    let headDim: Int = 64
    let maxNewTokens: Int = 1500  // Sufficient for longer sentence chunks
    let repetitionPenalty: Float = 1.2  // Match Python reference exactly

    // Generation parameters (matching server API)
    var seed: Int = 0                          // 0 = random, non-zero = reproducible
    var exaggeration: Float = 0.5              // 0.25-2.0, controls expressiveness
    var cfgWeight: Float = 0.5                  // 0.2-1.0, classifier-free guidance weight
    var speedFactor: Float = 1.0                // 0.25-4.0, playback speed

    // Decoding: greedy (argmax) + repetition penalty — matches reference exactly.
    // No temperature scaling or top-k sampling.
    let modelVariant: ModelVariant

    static let `default` = ChatterboxConfig(modelVariant: .q4f16)
}

// MARK: - Synthesized Chunk

/// Represents a single synthesized audio chunk with metadata
struct SynthesizedChunk: Sendable {
    let url: URL
    let progress: Double
    let isFirst: Bool
    let isLast: Bool
}

// MARK: - Chatterbox Engine

final class ChatterboxEngine: ObservableObject {
    /// Shared singleton instance - ensures cancellation works across all synthesis tasks
    static let shared = ChatterboxEngine()

    @Published var isLoaded = false
    private let chunkProcessQueue = DispatchQueue(label: "com.reader.chatterbox.chunk", qos: .userInitiated)
    private var isSynthesizing = false
    private var isCancelled = false


    @Published var synthesisProgress: Double = 0
    @Published var errorMessage: String?

    // Static cancellation token - cancelled when new synthesis starts
    // This ensures previous synthesis stops when switching articles
    private static var currentSynthesisTask: Task<Void, Never>?

    private let baseConfig: ChatterboxConfig
    private var currentGenerationConfig: ChatterboxConfig  // Set during synthesis
    private let tokenizer = TokenizerService()

    // Convenience accessor for current config
    private var config: ChatterboxConfig {
        currentGenerationConfig
    }

    // ONNX Runtime sessions
    private var ortEnv: ORTEnv?
    private var speechEncoderSession: ORTSession?
    private var embedTokensSession: ORTSession?
    private var languageModelSession: ORTSession?
    private var conditionalDecoderSession: ORTSession?
    // private var conditionalDecoderSession: ORTSession? // Removed: created dynamically per chunk

    // === Metal LM Backend (experimental) ===
    /// Feature flag: use Metal GPU for language model instead of ONNX Runtime.
    /// When false (default), uses the existing ONNX decode path.
    /// When true, uses MetalLMBackend + MetalPipeline for GPU-accelerated inference.
    /// Requires Task 10 (Xcode build config) to be completed for Metal types to be available.
    private let useMetalLM: Bool = false

    /// Metal pipeline for GPU-accelerated LM inference.
    /// Set during model loading when useMetalLM = true.
    /// TODO(Task 10): Initialize with MetalLMBackend after Xcode build config is set up.
    private var metalPipeline: (any LanguageModelBackend)?

    /// Metal device for GPU inference. Cached for reuse across pipeline calls.
    private var metalDevice: MTLDevice?

    // === PolarQuant KV Cache Compression ===
    /// Feature flag: compress KV cache to 4-bit between decode steps.
    /// Reduces KV memory ~3.6x via Metal GPU. Falls back to fp16 if Metal unavailable.
    private let usePolarQuantKV: Bool = true

    /// PolarQuant KV cache compressor — nil until first use (lazy init needs numLayers).
    private var kvCacheQuantizer: KVCacheQuantizer?

    // Holds all relevant speech encoder outputs.
    // audio_features is used as a prefix in inputs_embeds for the language model
    // to condition voice style. speaker_embeddings + speaker_features go to the
    // conditional decoder. audio_tokens (codec tokens of the reference audio) are
    // prepended to generated speech_tokens before the decoder — they provide the
    // decoder with voice context, without which the output is noise.
    private struct SpeakerContext {
        let audioFeatures: ORTValue  // "audio_features"      – [1, S, 1024], LM prefix
        let audioTokens: ORTValue    // "audio_tokens"        – [1, T], int64, decoder prefix
        let embeddings: ORTValue     // "speaker_embeddings"  – [1, 192], decoder input
        let features: ORTValue       // "speaker_features"    – [1, F, 80], decoder input
    }

    init(config: ChatterboxConfig = .default) {
        self.baseConfig = config
        self.currentGenerationConfig = config
    }

    // MARK: - Model Loading

    /// Load all ONNX model components into memory
    @MainActor func loadModels() async throws {
        let downloadService = ModelDownloadService.shared
        guard downloadService.isModelReady else {
            throw ChatterboxError.modelsNotDownloaded
        }

        // Enable CoreML with NeuralNetwork format (NOT MLProgram).
        // MLProgram uses Apple's E5RT/MIL compiler which rejects dynamic shapes for
        // Slice/Pad operators (common in variable-length audio models).
        // NeuralNetwork format uses the older CoreML compiler which is more permissive.
        let useCoreML = true

        chatterboxLogger.info("loadModels: loading tokenizer")
        try tokenizer.load(from: downloadService.tokenizerPath)

        chatterboxLogger.info("loadModels: creating ORT env")
        let env = try ORTEnv(loggingLevel: .warning)
        self.ortEnv = env

        // Use CoreML for all models except the dynamically recompiled one (conditional_decoder is handled separately)
        let coreMLOptions = try createSessionOptions(useCoreML: true)

        // Disable CoreML specifically for the conditional_decoder. CoreML graph partitioning breaks
        // the ConstantOfShape node when the input sequence length varies between consecutive chunks,
        // which causes a fatal 38GB+ memory allocation crash.
        let cpuOptions = try createSessionOptions(useCoreML: false)

        // Use dynamic ONNX models
        let modelPathFn: (ModelComponent) -> URL = { modelComponent in
            downloadService.modelPath(for: modelComponent, variant: self.config.modelVariant)
        }

        chatterboxLogger.info("loadModels: loading speechEncoder from: \(modelPathFn(.speechEncoder).lastPathComponent)")
        speechEncoderSession = try ORTSession(
            env: env,
            modelPath: modelPathFn(.speechEncoder).path,
            sessionOptions: coreMLOptions
        )
        chatterboxLogger.info("loadModels: loading embedTokens from: \(modelPathFn(.embedTokens).lastPathComponent)")
        embedTokensSession = try ORTSession(
            env: env,
            modelPath: modelPathFn(.embedTokens).path,
            sessionOptions: coreMLOptions
        )
        chatterboxLogger.info("loadModels: loading languageModel from: \(modelPathFn(.languageModel).lastPathComponent)")
        languageModelSession = try ORTSession(
            env: env,
            modelPath: modelPathFn(.languageModel).path,
            sessionOptions: coreMLOptions
        )
        print("[ChatterboxEngine] Loaded Language Model")

        chatterboxLogger.info("loadModels: loading conditionalDecoder from: \(modelPathFn(.conditionalDecoder).lastPathComponent)")
        conditionalDecoderSession = try ORTSession(
            env: env,
            modelPath: modelPathFn(.conditionalDecoder).path,
            sessionOptions: cpuOptions
        )

        isLoaded = true
        chatterboxLogger.info("loadModels: all models loaded")

        // ── Initialize Metal LM backend (experimental) ───────────────────────────
        // Metal LM is initialized after ONNX models so that if Metal initialization
        // fails, the ONNX path still works (useMetalLM=false remains the default).
        // TODO(Task 10): Uncomment and wire up once Xcode build config adds
        // chatterbox-metal-lm as a dependency with MetalLMBackend and MetalPipeline types.
        //
        // if useMetalLM {
        //     guard let device = MTLCreateSystemDefaultDevice() else {
        //         chatterboxLogger.warning("Metal GPU not available, falling back to ONNX")
        //     } else {
        //         self.metalDevice = device
        //         let weightsDir = downloadService.modelDirectory
        //             .appendingPathComponent("metal_weights")
        //         let pipeline = try MetalPipeline(
        //             device: device,
        //             weightsDir: weightsDir,
        //             maxNewTokens: config.maxNewTokens,
        //             repetitionPenalty: config.repetitionPenalty
        //         )
        //         try await pipeline.initialize(
        //             numLayers: 24,
        //             numKVHeads: config.numKVHeads,
        //             headDim: config.headDim,
        //             maxSeqLen: config.maxNewTokens,
        //             device: device
        //         )
        //         self.metalPipeline = pipeline
        //         chatterboxLogger.info("Metal LM pipeline initialized")
        //     }
        // }
    }

    private func createSessionOptions(useCoreML: Bool = true) throws -> ORTSessionOptions {
        let options = try ORTSessionOptions()

        // Register ONNX Runtime Extensions custom ops via function pointer.
        // The q4f16 models use `GatherBlockQuantized` (com.microsoft domain),
        // which lives in the separate onnxruntime_extensions library.
        // enableOrtExtensionsCustomOps() only works when ORT is built with
        // --use_extensions; with a separate library we must use the pointer API.
        let fnPtr = OrtExt.getRegisterCustomOpsFunctionPointer()
        try options.registerCustomOps(functionPointer: fnPtr)

        // Graph optimization: .all for maximum performance after model is loaded
        try options.setGraphOptimizationLevel(.all)

        // CPU performance tuning
        // intra_op_num_threads: parallelize within a single operator (matmul, attention).
        // On Apple Silicon M-series, num_performance_cores is typically 4 (or 8 on Pro Max).
        // 4 is a good default for audio workloads which are latency-sensitive.
        // Note: execution_mode and arena_extend_strategy are not available in SPM ORT.
        try options.setIntraOpNumThreads(4)

        // CoreML with NeuralNetwork format (NOT MLProgram).
        // MLProgram uses Apple's E5RT/MIL compiler which strictly rejects dynamic
        // shapes for Slice/Pad operators — crashes at session creation for our
        // variable-length audio models. NeuralNetwork format uses the older CoreML
        // compiler which is more permissive and supports dynamic shapes.
        if useCoreML {
            var coremlEnabled = false

            // Strategy: cascade through CoreML configuration approaches from most capable
            // to least, landing on whichever works for the installed ORT version.
            // This handles the full range from ancient SPM cached versions (pre-1.24)
            // to the latest 1.24.2+ where V2 dict options are fully recognized.

            // 1. Try modern V2 dictionary API (ORT 1.24.x): MLComputeUnits + NeuralNetwork + EnableOnSubgraphs.
            //    EnableOnSubgraphs lets CoreML claim entire compatible subgraphs, dramatically
            //    increasing the number of nodes handled by CoreML vs individual-node assignment.
            //    Only available in ORT 1.24.x — safe to use since CI now uses 1.24.2+.
            do {
                let coremlOptionsV2: [AnyHashable: Any] = [
                    "kCoremlProviderOption_MLComputeUnits": "All",
                    "kCoremlProviderOption_ModelFormat": "NeuralNetwork",
                    "kCoremlProviderOption_EnableOnSubgraphs": "1"
                ]
                try options.appendCoreMLExecutionProvider(withOptionsV2: coremlOptionsV2)
                coremlEnabled = true
                chatterboxLogger.info("CoreML (NeuralNetwork, EnableOnSubgraphs) execution provider enabled")
            } catch {
                // 2. V2 failed with EnableOnSubgraphs — try without it (older ORT compatibility).
                do {
                    let coremlOptionsV2: [AnyHashable: Any] = [
                        "kCoremlProviderOption_MLComputeUnits": "All",
                        "kCoremlProviderOption_ModelFormat": "NeuralNetwork"
                    ]
                    try options.appendCoreMLExecutionProvider(withOptionsV2: coremlOptionsV2)
                    coremlEnabled = true
                    chatterboxLogger.info("CoreML (NeuralNetwork, V2) execution provider enabled")
                } catch {
                    // 3. V2 failed — try non-V2 object API which has existed since early ORT CoreML support.
                    do {
                        let opts = ORTCoreMLExecutionProviderOptions()
                        try options.appendCoreMLExecutionProvider(with: opts)
                        coremlEnabled = true
                        chatterboxLogger.info("CoreML (all compute units) execution provider enabled via non-V2 API")
                    } catch {
                        // All CoreML approaches exhausted — degrade gracefully to CPU.
                        chatterboxLogger.warning("CoreML not available, falling back to CPU: \(error.localizedDescription)")
                    }
                }
            }

            if !coremlEnabled {
                chatterboxLogger.info("Using CPU execution provider")
            }
        } else {
            chatterboxLogger.info("Using CPU execution provider")
        }

        return options
    }

    // loadAllSessions inlined into loadModels() with step logging

    // MARK: - Speech Synthesis Pipeline

    /// Strip quotation marks from text for TTS synthesis.
    /// Keeps the text for display but removes quotes before sending to the model.
    private func stripQuotes(_ text: String) -> String {
        var result = text
        // Remove double quotes
        result = result.replacingOccurrences(of: "\"", with: "")
        // Remove smart quotes
        result = result.replacingOccurrences(of: "\u{201C}", with: "")  // "
        result = result.replacingOccurrences(of: "\u{201D}", with: "")  // "
        return result
    }

    /// Synthesize speech from text, writing WAV audio to the output URL.
    ///
    /// Text is automatically split into chunks for progressive streaming. Each chunk's
    /// audio is written to a separate file and `onChunkReady` is called immediately,
    /// allowing playback to start before the entire synthesis completes.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - referenceAudioURL: Optional reference audio for voice cloning.
    ///   - outputURL: Path for the final concatenated WAV file.
    ///   - onChunkReady: Called after each chunk is synthesized with the chunk's URL.
    ///   - onProgress: Fraction complete (0…1) after each chunk.
    ///   - seed: Seed for reproducible output (0 = random).
    ///   - exaggeration: Expressiveness level (0.25-2.0).
    ///   - cfgWeight: Classifier-free guidance weight (0.2-1.0).
    ///   - speedFactor: Playback speed (0.25-4.0).
    ///   - preChunkedText: Optional pre-chunked text array. If provided, the engine will use these
    ///                     exact chunks instead of rechunking the text. This ensures chunk boundaries
    ///                     match between display and audio.
    func synthesize(
        text: String,
        preChunkedText: [String]? = nil,
        referenceAudioURL: URL? = nil,
        outputURL: URL,
        onChunkReady: ((URL) -> Void)? = nil,
        onProgress: ((Double) -> Void)? = nil,
        seed: Int = 0,
        exaggeration: Float = 0.5,
        cfgWeight: Float = 0.5,
        speedFactor: Float = 1.0
    ) async throws {
        chatterboxLogger.info("synthesize() called, onChunkReady=\(onChunkReady != nil)")
        guard isLoaded else { throw ChatterboxError.modelNotLoaded }

        // Cancel any previous synthesis from a different article
        ChatterboxEngine.currentSynthesisTask?.cancel()
        chatterboxLogger.info("Cancelled any previous synthesis")

        // Wrap synthesis in a trackable task
        let synthesisTask = Task {
            do {
                try await synthesizeInternal(
                    text: text,
                    preChunkedText: preChunkedText,
                    referenceAudioURL: referenceAudioURL,
                    outputURL: outputURL,
                    onChunkReady: onChunkReady,
                    onProgress: onProgress,
                    seed: seed,
                    exaggeration: exaggeration,
                    cfgWeight: cfgWeight,
                    speedFactor: speedFactor
                )
            } catch {
                if !Task.isCancelled {
                    chatterboxLogger.error("Synthesis failed: \(error.localizedDescription)")
                }
            }
        }

        ChatterboxEngine.currentSynthesisTask = synthesisTask

        // Wait for completion (but allow cancellation)
        do {
            try await synthesisTask.value
        } catch {
            // Cancellation is expected
        }

        // Clear the task reference after completion
        ChatterboxEngine.currentSynthesisTask = nil
    }

    // Internal synthesis implementation
    private func synthesizeInternal(
        text: String,
        preChunkedText: [String]? = nil,
        referenceAudioURL: URL? = nil,
        outputURL: URL,
        onChunkReady: ((URL) -> Void)? = nil,
        onProgress: ((Double) -> Void)? = nil,
        seed: Int = 0,
        exaggeration: Float = 0.5,
        cfgWeight: Float = 0.5,
        speedFactor: Float = 1.0
    ) async throws {
        isSynthesizing = true
        synthesisProgress = 0
        defer {
            isSynthesizing = false
            // Note: currentSynthesisTask is cleared in synthesize() after task completes
            // to avoid race condition where it gets cleared while still running
        }

        // Track this synthesis task for cancellation
        // Note: We can't easily wrap the entire function, so we rely on the
        // Task.checkCancellation() calls in the decode loop and between batches

        // Apply generation parameters to config
        var generationConfig = config
        generationConfig.seed = seed
        generationConfig.exaggeration = exaggeration
        generationConfig.cfgWeight = cfgWeight
        generationConfig.speedFactor = speedFactor

        // Apply seed for reproducibility
        if seed != 0 {
            chatterboxLogger.info("Setting seed to \(seed) for reproducible generation")
            srand48(seed)
        }

        // Step 1: Chunk text for streaming - use preChunkedText if provided, otherwise use TextChunker
        let chunks: [String]
        if let preChunked = preChunkedText, !preChunked.isEmpty {
            chatterboxLogger.info("synthesize: using \(preChunked.count) pre-chunked texts (no rechunking)")
            chunks = preChunked
        } else {
            chunks = TextChunker.chunkText(text)
        }
        guard !chunks.isEmpty else { throw ChatterboxError.emptyText }

        chatterboxLogger.info("synthesize: split into \(chunks.count) chunks for streaming")

        // Determine output directory and base name for chunk files
        let stem = outputURL.deletingPathExtension().lastPathComponent
        let audioDir = outputURL.deletingLastPathComponent()

        // Step 2: Encode reference voice ONCE for all chunks
        let speakerContext: SpeakerContext
        chatterboxLogger.info("synthesize: encoding speaker voice")
        if let refURL = referenceAudioURL {
            speakerContext = try await encodeSpeakerVoice(from: refURL)
        } else {
            let downloadService = ModelDownloadService.shared // Local reference
            speakerContext = try await createDefaultSpeakerContext(downloadService: downloadService)
        }
        chatterboxLogger.info("synthesize: speaker encoded, processing \(chunks.count) chunk(s)")

        // Step 3: Process chunks in PARALLEL with LIMITED CONCURRENCY
        // Running 173 tasks in parallel causes OOM - use batch size of 4
        // Callbacks fire in SEQUENTIAL ORDER for proper audio playback
        let speakerCtx = speakerContext
        let chunkCount = chunks.count
        let configCopy = generationConfig
        let maxConcurrent = 1  // Sequential processing to prevent OOM and ensure stable synthesis

        // Use Actor for thread-safe collection of audio results
        let audioResultsCollector = AudioResultsCollector()

        // Track completed chunks and next expected index for sequential callback ordering
        var completedChunks: [Int: (audio: [Float], url: URL)] = [:]
        var nextExpectedIndex = 0
        var failedChunks: [(index: Int, text: String)] = []

        // Prepare non-empty chunks with their indices
        var chunkTasks: [(index: Int, text: String)] = []
        for (index, chunk) in chunks.enumerated() {
            let trimmedChunk = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
            let chunkForTTS = stripQuotes(trimmedChunk)
            if !chunkForTTS.isEmpty {
                chunkTasks.append((index, chunkForTTS))
            }
        }

        // Process in batches of maxConcurrent
        let totalBatches = (chunkTasks.count + maxConcurrent - 1) / maxConcurrent

        for batchIdx in 0..<totalBatches {
            try Task.checkCancellation()  // Stop if cancelled before starting new batch

            let startIdx = batchIdx * maxConcurrent
            let endIdx = min(startIdx + maxConcurrent, chunkTasks.count)
            let batch = Array(chunkTasks[startIdx..<endIdx])

            chatterboxLogger.info("Processing batch \(batchIdx + 1)/\(totalBatches) with \(batch.count) tasks")

            // Process this batch in parallel
            await withTaskGroup(of: (Int, [Float]?).self) { group in
                for (index, chunkText) in batch {
                    group.addTask { [self] in
                        chatterboxLogger.info("synthesizing chunk \(index+1)/\(chunkCount) (\(chunkText.count) chars)")
                        do {
                            let audio = try await self.synthesizeChunk(chunkText, speakerContext: speakerCtx, generationConfig: configCopy)
                            chatterboxLogger.info("synthesizeChunk SUCCESS chunk \(index+1), samples: \(audio.count)")
                            return (index, audio)
                        } catch {
                            chatterboxLogger.error("chunk \(index) FAILED: \(error.localizedDescription)")
                            return (index, nil)
                        }
                    }
                }

                // Process results as they complete (not after all complete)
                for await (index, audio) in group {
                    if let audio = audio {
                        // Write chunk file when this chunk completes
                        let chunkURL = audioDir.appendingPathComponent("\(stem)_part\(index).wav")
                        do {
                            try writeWAV(samples: audio, to: chunkURL, sampleRate: config.sampleRate)
                            chatterboxLogger.info("Chunk \(index) file written: \(chunkURL.lastPathComponent)")

                            // Store in completedChunks for ordered callback
                            completedChunks[index] = (audio, chunkURL)

                            // Also store in audioResults for final concatenated WAV
                            await audioResultsCollector.append((index, audio))
                        } catch {
                            chatterboxLogger.error("Failed to write chunk \(index): \(error.localizedDescription)")
                        }
                    } else {
                        // Track failed chunk for retry
                        if let chunkTask = chunkTasks.first(where: { $0.index == index }) {
                            failedChunks.append(chunkTask)
                            chatterboxLogger.info("Chunk \(index) queued for retry")
                        }
                    }
                }
            }

            // NOW fire callbacks in SEQUENTIAL ORDER
            // This ensures audio plays in correct order even if chunks complete out of order
            while let nextChunk = completedChunks.removeValue(forKey: nextExpectedIndex) {
                if let onChunkReady = onChunkReady {
                    chatterboxLogger.info("Calling onChunkReady for chunk \(nextExpectedIndex+1)/\(chunkCount) (sequential)")
                    await MainActor.run {
                        onChunkReady(nextChunk.url)
                    }
                }
                nextExpectedIndex += 1
            }

            // Retry failed chunks from this batch (up to 2 retries)
            let maxRetries = 2
            var retriesRemaining = maxRetries

            while !failedChunks.isEmpty && retriesRemaining > 0 {
                chatterboxLogger.info("Retrying \(failedChunks.count) failed chunks (attempt \(maxRetries - retriesRemaining + 1)/\(maxRetries))")
                let chunksToRetry = failedChunks
                failedChunks = []

                for chunkTask in chunksToRetry {
                    do {
                        let audio = try await self.synthesizeChunk(chunkTask.text, speakerContext: speakerCtx, generationConfig: configCopy)
                        chatterboxLogger.info("Retry chunk \(chunkTask.index) SUCCESS, samples: \(audio.count)")

                        let chunkURL = audioDir.appendingPathComponent("\(stem)_part\(chunkTask.index).wav")
                        try writeWAV(samples: audio, to: chunkURL, sampleRate: config.sampleRate)

                        completedChunks[chunkTask.index] = (audio, chunkURL)
                        await audioResultsCollector.append((chunkTask.index, audio))

                        // Fire callback if this is the next expected chunk
                        while let nextChunk = completedChunks.removeValue(forKey: nextExpectedIndex) {
                            if let onChunkReady = onChunkReady {
                                chatterboxLogger.info("Calling onChunkReady for retry chunk \(nextExpectedIndex+1)")
                                await MainActor.run {
                                    onChunkReady(nextChunk.url)
                                }
                            }
                            nextExpectedIndex += 1
                        }
                    } catch {
                        chatterboxLogger.error("Retry chunk \(chunkTask.index) FAILED: \(error.localizedDescription)")
                        failedChunks.append(chunkTask)
                    }
                }

                retriesRemaining -= 1
            }

            // If still failed after retries, log error
            if !failedChunks.isEmpty {
                chatterboxLogger.warning("\(failedChunks.count) chunks still failed after \(maxRetries) retries")
                failedChunks = []  // Clear to avoid infinite loop
            }

            // Update progress after batch completes
            let completedCount = completedChunks.count + nextExpectedIndex
            let progress = Double(completedCount) / Double(chunkCount)
            if let onProgress = onProgress {
                await MainActor.run {
                    onProgress(progress)
                }
            }
        }

        // All chunks have been processed
        chatterboxLogger.info("All \(chunks.count) chunks complete - callback fired sequentially")

        if let onProgress = onProgress {
            await MainActor.run {
                onProgress(1.0)
            }
        }

        // Step 4: Write the final concatenated WAV (used on subsequent opens).
        // Note: individual chunk files were already written in the TaskGroup loop above
        // Get sorted results from actor (thread-safe)
        let sortedResults = await audioResultsCollector.getSortedResults()
        let allAudioSamples = sortedResults.flatMap { $0.1 }
        try writeWAV(samples: allAudioSamples, to: outputURL, sampleRate: config.sampleRate)

        synthesisProgress = 1.0
    }

    // MARK: - Async Stream Synthesis (for proper backpressure)

    /// Synthesize text and yield chunk URLs as an AsyncStream.
    /// This provides proper backpressure: playback can await each chunk before the next is synthesized.
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - referenceAudioURL: Optional reference audio for voice cloning.
    ///   - outputURL: Path for the final concatenated WAV file.
    ///   - seed: Seed for reproducible output (0 = random).
    ///   - exaggeration: Expressiveness level (0.25-2.0).
    ///   - cfgWeight: Classifier-free guidance weight (0.2-1.0).
    ///   - speedFactor: Playback speed (0.25-4.0).
    /// - Returns: An AsyncStream of chunk URLs, plus progress updates.
    func synthesizeStream(
        text: String,
        referenceAudioURL: URL? = nil,
        outputURL: URL,
        seed: Int = 0,
        exaggeration: Float = 0.5,
        cfgWeight: Float = 0.5,
        speedFactor: Float = 1.0
    ) -> AsyncThrowingStream<SynthesizedChunk, Error> {
        // Cancel any previous synthesis
        ChatterboxEngine.currentSynthesisTask?.cancel()

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    guard self.isLoaded else {
                        throw ChatterboxError.modelNotLoaded
                    }

                    // Apply generation parameters
                    var generationConfig = self.config
                    generationConfig.seed = seed
                    generationConfig.exaggeration = exaggeration
                    generationConfig.cfgWeight = cfgWeight
                    generationConfig.speedFactor = speedFactor

                    // Apply seed for reproducibility
                    if seed != 0 {
                        chatterboxLogger.info("Setting seed to \(seed) for reproducible generation")
                        srand48(seed)
                    }

                    // Chunk text
                    let chunks = TextChunker.chunkText(text)
                    chatterboxLogger.info("synthesizeStream: text length=\(text.count), chunks count=\(chunks.count)")
                    guard !chunks.isEmpty else {
                        throw ChatterboxError.emptyText
                    }

                    // Encode speaker once
                    let speakerContext: SpeakerContext
                    if let refURL = referenceAudioURL {
                        speakerContext = try await self.encodeSpeakerVoice(from: refURL)
                    } else {
                        let downloadService = ModelDownloadService.shared // Local reference
                        speakerContext = try await self.createDefaultSpeakerContext(downloadService: downloadService)
                    }

                    let stem = outputURL.deletingPathExtension().lastPathComponent
                    let audioDir = outputURL.deletingLastPathComponent()
                    var allAudioSamples: [Float] = []

                    // Process each chunk
                    for (index, chunk) in chunks.enumerated() {
                        // Check for cancellation
                        try Task.checkCancellation()

                        // Strip quotes for TTS
                        let chunkForTTS = stripQuotes(chunk.trimmingCharacters(in: .whitespacesAndNewlines))
                        guard !chunkForTTS.isEmpty else {
                            continue
                        }

                        let chunkAudio = try await self.synthesizeChunk(chunkForTTS, speakerContext: speakerContext, generationConfig: generationConfig)
                        allAudioSamples.append(contentsOf: chunkAudio)

                        let chunkURL = audioDir.appendingPathComponent("\(stem)_part\(index).wav")
                        try self.writeWAV(samples: chunkAudio, to: chunkURL, sampleRate: self.config.sampleRate)

                        let progress = Double(index + 1) / Double(chunks.count)

                        // Yield the chunk with its URL and progress
                        let synthesizedChunk = SynthesizedChunk(
                            url: chunkURL,
                            progress: progress,
                            isFirst: index == 0,
                            isLast: index == chunks.count - 1
                        )
                        continuation.yield(synthesizedChunk)
                    }

                    // Write final concatenated file
                    try self.writeWAV(samples: allAudioSamples, to: outputURL, sampleRate: self.config.sampleRate)

                    continuation.finish()
                } catch {
                    chatterboxLogger.error("synthesizeStream error: \(error.localizedDescription)")
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Core Inference Steps

    /// Encode a reference audio file and return a SpeakerContext with both encoder outputs.
    private func encodeSpeakerVoice(from audioURL: URL) async throws -> SpeakerContext {
        let samples = try loadAudioAsFloat(from: audioURL, targetSampleRate: config.sampleRate)
        return try encodeSpeakerVoiceFromSamples(samples)
    }

    /// Build a default SpeakerContext using the downloaded default_voice.wav, or
    /// falling back to 0.5s of silence if the file is not yet available.
    @MainActor private func createDefaultSpeakerContext(downloadService: ModelDownloadService) async throws -> SpeakerContext {
        let voicePath = downloadService.defaultVoicePath
        chatterboxLogger.info("createDefaultSpeakerContext: path=\(voicePath.path)")
        chatterboxLogger.info("createDefaultSpeakerContext: exists=\(FileManager.default.fileExists(atPath: voicePath.path))")
        if FileManager.default.fileExists(atPath: voicePath.path) {
            chatterboxLogger.info("createDefaultSpeakerContext: loading voice from \(voicePath.path)")
            return try await encodeSpeakerVoice(from: voicePath)
        }
        // Silence fallback — produces lower quality but avoids a crash.
        chatterboxLogger.warning("createDefaultSpeakerContext: FILE NOT FOUND, using silence fallback")
        let silenceSamples = Array(repeating: Float(0), count: config.sampleRate / 2)
        return try encodeSpeakerVoiceFromSamples(silenceSamples)
    }

    /// Run the speech encoder on raw float samples and return speaker_embeddings + speaker_features.
    private func encodeSpeakerVoiceFromSamples(_ samples: [Float]) throws -> SpeakerContext {
        guard let session = speechEncoderSession else {
            throw ChatterboxError.modelNotLoaded
        }

        // Input: [1, audio_length] — use float32 (q4f16 quantized models still use float32 for inputs)
        let inputTensor = try createFloatTensor(samples, shape: [1, NSNumber(value: samples.count)])

        let inputNames  = try session.inputNames()  as [String]
        let outputNames = try session.outputNames() as [String]
        let audioInputName = inputNames.first ?? "audio_values"

        let outputs = try session.run(
            withInputs: [audioInputName: inputTensor],
            outputNames: Set(outputNames),
            runOptions: nil
        )

        // The speech encoder produces: audio_features, audio_tokens,
        // speaker_embeddings, speaker_features.
        // audio_features is used as LM prefix; embeddings+features go to decoder.
        guard let audioFeaturesValue = outputs["audio_features"] else {
            throw ChatterboxError.inferenceError("Speech encoder missing 'audio_features' output")
        }
        guard let audioTokensValue = outputs["audio_tokens"] else {
            throw ChatterboxError.inferenceError("Speech encoder missing 'audio_tokens' output")
        }
        guard let embeddingsValue = outputs["speaker_embeddings"] else {
            throw ChatterboxError.inferenceError("Speech encoder missing 'speaker_embeddings' output")
        }
        guard let featuresValue = outputs["speaker_features"] else {
            throw ChatterboxError.inferenceError("Speech encoder missing 'speaker_features' output")
        }

        return SpeakerContext(audioFeatures: audioFeaturesValue,
                              audioTokens: audioTokensValue,
                              embeddings: embeddingsValue,
                              features: featuresValue)
    }

    /// Synthesize a single text chunk to audio samples
    private func synthesizeChunk(
        _ text: String,
        speakerContext: SpeakerContext,
        generationConfig: ChatterboxConfig
    ) async throws -> [Float] {
        guard let embedSession = embedTokensSession,
              let decoderSession = conditionalDecoderSession,
              let lmSession = languageModelSession else {
            throw ChatterboxError.modelNotLoaded
        }

        // Reset PolarQuant compressed KV cache for this chunk
        kvCacheQuantizer?.reset()

        // Apply exaggeration + cfg_weight: scale audio features ───────────────
        // Combine both parameters into a single scaling factor:
        // - exaggeration: controls emotional intensity (0.25-2.0)
        // - cfg_weight: controls conditioning influence (0.2-1.0)
        // Formula: factor = exaggeration * (cfg_weight / 0.5) to preserve default behavior
        let combinedFactor = generationConfig.exaggeration * (generationConfig.cfgWeight / 0.5)
        chatterboxLogger.info("Applying combined factor: \(combinedFactor) (exag=\(generationConfig.exaggeration), cfg=\(generationConfig.cfgWeight))")

        let scaledAudioFeatures: ORTValue
        if combinedFactor != 1.0 {  // Only scale if not default (0.5 * 1.0 = 0.5)
            scaledAudioFeatures = try scaleAudioFeatures(
                speakerContext.audioFeatures,
                factor: combinedFactor
            )
        } else {
            scaledAudioFeatures = speakerContext.audioFeatures
        }

        // ── Step 1: Tokenize text ──────────────────────────────────────────────
        // Debug: Log truncated text to identify which chunks fail
        let truncatedText = text.prefix(100)
        let commaCount = text.filter { $0 == "," }.count
        let periodCount = text.filter { $0 == "." }.count
        chatterboxLogger.debug("CHUNK TEXT: \"\(truncatedText)\"... (commas=\(commaCount), periods=\(periodCount), len=\(text.count))")

        let tokenIDs = tokenizer.encode(text)
        guard !tokenIDs.isEmpty else { return [] }
        let textSeqLen = tokenIDs.count
        chatterboxLogger.debug("tokenIDs (\(textSeqLen)): \(tokenIDs.prefix(30)) ...")

        // ── Step 2: Embed text tokens ──────────────────────────────────────────
        let int64Tokens = tokenIDs.map { Int64($0) }
        let inputIDsTensor = try createInt64Tensor(
            int64Tokens, shape: [1, NSNumber(value: textSeqLen)]
        )

        let embedInputNames  = try embedSession.inputNames()  as [String]
        let embedOutputNames = try embedSession.outputNames() as [String]
        let embedInputName   = embedInputNames.first  ?? "input_ids"
        let embedOutputName  = embedOutputNames.first ?? "inputs_embeds"

        let embedOutputs = try embedSession.run(
            withInputs: [embedInputName: inputIDsTensor],
            outputNames: Set([embedOutputName]),
            runOptions: nil
        )
        guard let textEmbeddings = embedOutputs[embedOutputName] else {
            throw ChatterboxError.inferenceError("Token embedding produced no output")
        }

        // ── Step 2b: Prepend audio_features as voice prefix ───────────────────
        // The language model is a decoder-only Llama. Voice conditioning must come
        // through inputs_embeds. We prepend the audio_features from the speech encoder
        // (reference voice representation) before the text embeddings so the model
        // generates speech in the correct voice style.
        //
        // inputs_embeds = concat([audio_features, text_embeds], axis=1)
        //   shape: [1, audioSeqLen + textSeqLen, 1024]
        // Log element types to catch dtype mismatches before concat
        let afInfo = try speakerContext.audioFeatures.tensorTypeAndShapeInfo()
        let teInfo = try textEmbeddings.tensorTypeAndShapeInfo()
        chatterboxLogger.debug("audioFeatures type=\(afInfo.elementType.rawValue) shape=\(afInfo.shape)")
        chatterboxLogger.debug("textEmbeddings type=\(teInfo.elementType.rawValue) shape=\(teInfo.shape)")

        // Use scaled audio features if exaggeration != 0.5
        let audioFeaturesToUse = scaledAudioFeatures
        let prefixEmbeds = try concatEmbeddings(audioFeaturesToUse, textEmbeddings)
        let prefixInfo   = try prefixEmbeds.tensorTypeAndShapeInfo()
        let prefixShape  = prefixInfo.shape
        let totalSeqLen  = prefixShape[1].intValue  // audioSeqLen + textSeqLen

        // ── Step 3: Autoregressive language model generation ───────────────────
        //
        // The language model (LM) is a standard decoder-only transformer.
        // Required inputs every step:
        //   • inputs_embeds     – [1, seqLen, hiddenDim]  (prefix on step 0, 1 token after)
        //   • attention_mask    – [1, totalLen]            all-ones int64
        //   • position_ids      – [1, seqLen]              current token positions int64
        //   • past_key_values.N.key/value  – [1, heads, pastLen, headDim]  per-layer KV cache
        //
        // KV cache naming:
        //   Input  names: past_key_values.{layer}.key / .value
        //   Output names: present.{layer}.key          / .value
        //   After each step map present→past_key_values for next step.

        // Track all generated tokens for repetition penalty, starting with START_SPEECH.
        // Matches reference: generate_tokens = [[START_SPEECH_TOKEN]] initially.
        var generateTokens: [Int] = [config.startSpeechToken]
        var speechTokens: [Int] = []  // tokens between START_SPEECH and STOP_SPEECH

        let lmInputNames  = try lmSession.inputNames()  as [String]
        let lmOutputNames = try lmSession.outputNames() as [String]

        // Locate logits output (exact name preferred)
        let logitsOutputName = lmOutputNames.first(where: { $0 == "logits" })
                            ?? lmOutputNames.first(where: { $0.contains("logit") })
                            ?? lmOutputNames[0]

        // Infer number of KV-cache layers from input names
        let numLayers = lmInputNames.filter {
            $0.hasPrefix("past_key_values.") && $0.hasSuffix(".key")
        }.count

        // ── Lazy-init PolarQuant KV cache quantizer (needs numLayers) ─────────
        if usePolarQuantKV && kvCacheQuantizer == nil {
            do {
                kvCacheQuantizer = try KVCacheQuantizer(
                    numLayers: numLayers,
                    numHeads: config.numKVHeads,
                    headDim: config.headDim
                )
                chatterboxLogger.info("PolarQuant KV cache quantizer initialized (\(numLayers) layers)")
            } catch {
                chatterboxLogger.warning("PolarQuant init failed, using fp16 pass-through: \(error)")
            }
        }

        // ── Build prefill inputs ───────────────────────────────────────────────
        var lmInputs: [String: ORTValue] = [:]

        // Pre-allocate reusable buffers for the decode loop to avoid per-step heap allocations.
        // These are reset and reused every step instead of creating new ORTValue objects.
        let reusableNextTokenBuffer = UnsafeMutableBufferPointer<Int64>.allocate(capacity: 1)
        defer { reusableNextTokenBuffer.deallocate() }
        let reusablePositionIdBuffer = UnsafeMutableBufferPointer<Int64>.allocate(capacity: 1)
        defer { reusablePositionIdBuffer.deallocate() }

        // Initialize all KV-cache with empty tensors: [1, numKVHeads, 0, headDim]
        for layer in 0..<numLayers {
            lmInputs["past_key_values.\(layer).key"]   = try createEmptyFloatTensor(
                shape: [1, config.numKVHeads, 0, config.headDim]
            )
            lmInputs["past_key_values.\(layer).value"] = try createEmptyFloatTensor(
                shape: [1, config.numKVHeads, 0, config.headDim]
            )
        }

        // Prefill: audio_features + text embeddings concatenated [1, totalSeqLen, 1024]
        lmInputs["inputs_embeds"] = prefixEmbeds

        // position_ids for prefill: [0, 1, …, totalSeqLen-1]
        let prefillPositions = (0..<totalSeqLen).map { Int64($0) }
        lmInputs["position_ids"] = try createInt64Tensor(
            prefillPositions, shape: [1, NSNumber(value: totalSeqLen)]
        )

        // attention_mask for prefill: all ones [1, totalSeqLen]
        let prefillMask = Array(repeating: Int64(1), count: totalSeqLen)
        lmInputs["attention_mask"] = try createInt64Tensor(
            prefillMask, shape: [1, NSNumber(value: totalSeqLen)]
        )

        let lmOutputNameSet = Set(lmOutputNames)

        // ── Decode loop ────────────────────────────────────────────────────────
        // Each iteration is wrapped in autoreleasepool so ORT's ObjC-autoreleased
        // tensors (logits, intermediate activations) are freed immediately after
        // each step rather than accumulating for all 400 steps (~23 GB without this).
        chatterboxLogger.debug("decode loop start, totalSeqLen=\(totalSeqLen), maxTokens=\(self.config.maxNewTokens)")

        var shouldStopDecoding = false

        // Phase 1 Optimization: Pre-allocate buffers outside the tight autoregressive loop
        // to prevent allocating 500+ Swift Arrays per sentence (which triggers ARC thrashing).
        let dynamicMaxTokens = self.config.maxNewTokens
        let maxPossibleMaskLen = totalSeqLen + dynamicMaxTokens + 1

        // Pre-allocate the mask buffer as a raw pointer to avoid Swift Array COW overhead
        let maskBuffer = UnsafeMutableBufferPointer<Int64>.allocate(capacity: maxPossibleMaskLen)
        defer { maskBuffer.deallocate() }
        // Fill with ones (attention mask is all-ones)
        for i in 0..<maxPossibleMaskLen { maskBuffer[i] = 1 }

        for step in 0..<dynamicMaxTokens {
            try Task.checkCancellation()  // Stop if cancelled
            chatterboxLogger.debug("decode step \(step): maxToken=\(self.config.maxNewTokens)")

            var nextStepInputs: [String: ORTValue] = [:]
            var generatedToken = config.stopSpeechToken

            try autoreleasepool {
                let lmOutputs = try lmSession.run(
                    withInputs: lmInputs,
                    outputNames: lmOutputNameSet,
                    runOptions: nil
                )

                guard let logitsValue = lmOutputs[logitsOutputName] else {
                    throw ChatterboxError.inferenceError("Language model produced no logits at step \(step)")
                }

                // Logits shape: [1, curSeqLen, vocabSize]
                // Step 0 (prefill): curSeqLen = totalSeqLen → take last position's logits.
                // Step k≥1 (decode): curSeqLen = 1          → take the only position.
                let allLogits  = try extractFloatArray(from: logitsValue)
                let curSeqLen  = (step == 0) ? totalSeqLen : 1
                let vocabSize  = allLogits.count / max(curSeqLen, 1)
                guard vocabSize > 0 else {
                    throw ChatterboxError.inferenceError("Invalid logits shape at step \(step)")
                }
                let lastStart  = (curSeqLen - 1) * vocabSize
                let lastEnd    = lastStart + vocabSize
                guard lastEnd <= allLogits.count else {
                    throw ChatterboxError.inferenceError("Logits slice out of bounds at step \(step)")
                }
                let lastPosLogits = Array(allLogits[lastStart..<lastEnd])

                // Greedy decode: argmax + repetition penalty (matches reference exactly).
                // No temperature scaling, no top-k sampling, no logit masking.
                generatedToken = greedyNextToken(lastPosLogits, previous: generateTokens)
                generateTokens.append(generatedToken)

                // Log tokens for debugging repetition patterns
                // Log more frequently in middle/end (steps 200-800) where issue occurs
                let shouldLog = step < 10 || step % 20 == 0 || (step > 200 && step < 800)
                if shouldLog {
                    let logPos = totalSeqLen + speechTokens.count - 1
                    chatterboxLogger.debug("step \(step): tok=\(generatedToken), pos=\(logPos)")
                }

                // Detect token repetition patterns (3-6 word sequences = ~6-18 tokens)
                if speechTokens.count >= 6 {
                    let recentTokens = Array(speechTokens.suffix(12))
                    // Check for 3-token repeat (6 tokens back)
                    if recentTokens.count >= 6 {
                        let threeBack = speechTokens[speechTokens.count - 6]
                        if generatedToken == threeBack {
                            chatterboxLogger.warning("REPETITION DETECTED: token \(generatedToken) repeats 6 positions back at step \(step)")
                        }
                    }
                    // Check for 6-token repeat pattern
                    if recentTokens.count >= 12 {
                        let last6 = Array(speechTokens.suffix(6))
                        let prev6 = Array(speechTokens.dropLast(6).suffix(6))
                        if last6 == prev6 {
                            chatterboxLogger.warning("REPETITION PATTERN: 6-token sequence repeats at step \(step): \(last6)")
                        }
                    }
                }

                // Check for STOP_SPEECH BEFORE adding to speechTokens
                // Matches Python: speech_tokens = generate_tokens[:, 1:-1] strips START and STOP
                if generatedToken == config.stopSpeechToken {
                    chatterboxLogger.info("STOP_SPEECH detected at step \(step)")
                    shouldStopDecoding = true
                    // Don't return here - let code flow to line 498 which breaks the loop
                } else {
                    speechTokens.append(generatedToken)
                }

                // ── Prepare next-step inputs ───────────────────────────────────
                // Embed the newly generated token: [1, 1] → [1, 1, hiddenDim]
                // CPU Opt: Reuse pre-allocated buffer for the token ID tensor.
                reusableNextTokenBuffer[0] = Int64(generatedToken)
                let nextTokenTensor = try createInt64TensorFromBuffer(
                    reusableNextTokenBuffer, count: 1, shape: [1, 1]
                )
                let nextEmbedOutputs = try embedSession.run(
                    withInputs: [embedInputName: nextTokenTensor],
                    outputNames: Set([embedOutputName]),
                    runOptions: nil
                )
                guard let nextEmbed = nextEmbedOutputs[embedOutputName] else {
                    shouldStopDecoding = true
                    return
                }

                // Position of the next token to be generated
                // After prefill  (step=0, |speechTokens|=1): next pos = totalSeqLen
                // After decode k (step=k, |speechTokens|=k+1): next pos = totalSeqLen + k
                let nextPos     = Int64(totalSeqLen + speechTokens.count)
                let nextMaskLen = totalSeqLen + speechTokens.count

                // CPU Opt: Reuse pre-allocated buffers instead of creating new tensors each step.
                reusablePositionIdBuffer[0] = nextPos
                let nextPosTensor = try createInt64TensorFromBuffer(
                    reusablePositionIdBuffer, count: 1, shape: [1, 1]
                )

                // Debug: Log position IDs at key steps to verify they're correct at step 500+
                if step == 100 || step == 300 || step == 500 || step == 700 || step == 900 {
                    chatterboxLogger.debug("POSITION CHECK step \(step): nextPos=\(nextPos), maskLen=\(nextMaskLen), totalSeqLen=\(totalSeqLen)")
                }

                nextStepInputs = [
                    "inputs_embeds":  nextEmbed,
                    "position_ids":   nextPosTensor,
                    // Reuse pre-allocated mask buffer instead of allocating new array each step
                    "attention_mask": try fillInt64Tensor(
                        buffer: maskBuffer, count: nextMaskLen, shape: [1, NSNumber(value: nextMaskLen)]
                    )
                ]

                // Carry KV-cache forward, optionally compressing via PolarQuant.
                // When enabled, KV cache is stored at 4-bit between steps (~3.6x smaller).
                if let quantizer = self.kvCacheQuantizer, quantizer.isEnabled {
                    // Compress this step's full KV cache on Metal GPU
                    for layer in 0..<numLayers {
                        if let key = lmOutputs["present.\(layer).key"],
                           let val = lmOutputs["present.\(layer).value"] {
                            try quantizer.compress(layerIndex: layer, key: key, value: val)
                        }
                    }
                    // Decompress for next step's input (fp16 ORTValues)
                    for layer in 0..<numLayers {
                        nextStepInputs["past_key_values.\(layer).key"] = try quantizer.decompressKey(layerIndex: layer)
                        nextStepInputs["past_key_values.\(layer).value"] = try quantizer.decompressValue(layerIndex: layer)
                    }

                    // Log compression stats every 100 steps
                    if step % 100 == 0 {
                        let ratio = quantizer.compressionRatio
                        let compressedKB = quantizer.compressedMemoryBytes / 1024
                        chatterboxLogger.info("KV step \(step): \(compressedKB)KB (\(String(format: "%.1f", ratio))x compression)")
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
                // lmOutputs goes out of scope here; autoreleasepool drains ORT's
                // autoreleased logits and activation tensors immediately.
            }

            if generatedToken == config.stopSpeechToken || shouldStopDecoding { break }
            // Assigning nextStepInputs releases the old lmInputs dict (previous KV cache).
            lmInputs = nextStepInputs
        }

        // All tokens generated between START_SPEECH and STOP_SPEECH are valid codec tokens.
        // The turbo model's conditional decoder accepts any token in the 0-6562 range.
        // Matches reference: speech_tokens = generate_tokens[:, 1:-1] (no range filtering).
        let validSpeechTokens = speechTokens
        chatterboxLogger.debug("speech tokens: \(speechTokens.count) generated, first 10: \(speechTokens.prefix(10))")

        // Debug: Analyze token distribution for repetition patterns
        var tokenCounts: [Int: Int] = [:]
        for tok in speechTokens {
            tokenCounts[tok, default: 0] += 1
        }
        // Find most common tokens
        let sortedTokens = tokenCounts.sorted { $0.value > $1.value }
        if let topToken = sortedTokens.first, topToken.value > 5 {
            chatterboxLogger.warning("REPEATED TOKEN: tok=\(topToken.key) appears \(topToken.value) times")
        }
        // Log token diversity (unique tokens / total tokens)
        let diversity = Double(tokenCounts.count) / Double(max(speechTokens.count, 1))
        chatterboxLogger.debug("token diversity: \(String(format: "%.2f", diversity)), unique=\(tokenCounts.count)/\(speechTokens.count)")

        // The f0_predictor/condnet uses Conv1d with "valid" (no-padding) convolutions.
        // With kernel_size k and input length N, output length = N - k + 1; if N < k the
        // output is 0, and ORT raises "Invalid input shape: {0}".  A safe minimum of 10
        // tokens (~150 ms of audio) avoids this for any reasonable Conv kernel size.
        let minSpeechTokens = 10
        guard validSpeechTokens.count >= minSpeechTokens else {
            chatterboxLogger.warning("only \(validSpeechTokens.count) valid tokens (< \(minSpeechTokens) minimum) – returning silence")
            let estimatedSamples = Int(TextChunker.estimateDuration(for: text) * Double(config.sampleRate))
            return Array(repeating: 0, count: max(estimatedSamples, config.sampleRate / 2))
        }

        // ── Step 4: Decode speech tokens to audio waveform ────────────────────
        //
        // Conditional decoder inputs (exact model names):
        //   • speech_tokens      – int64  [1, numSpeechTokens]  (only codec range 704–6560)
        //   • speaker_embeddings – float  from speech encoder
        //   • speaker_features   – float  from speech encoder
        //
        // Output:
        //   • waveform           – float  [1, audioSamples]

        // Matches Python test EXACTLY:
        //   decoder_input = np.concatenate([prompt_token, speech_tokens, silence_tokens], axis=1)
        // where prompt_token = audioTokens from speech encoder
        let int64SpeechTokens = validSpeechTokens.map { Int64($0) }
        let silenceTail = [Int64(config.silenceToken), Int64(config.silenceToken), Int64(config.silenceToken)]
        let speechAndSilence = int64SpeechTokens + silenceTail
        let speechTokensTensor = try createInt64Tensor(speechAndSilence, shape: [1, NSNumber(value: speechAndSilence.count)])

        // Always prepend audioTokens (matching Python reference pipeline)
        let speechTokenTensor = try concatInt64Tensors(
            speakerContext.audioTokens,
            speechTokensTensor
        )

        // DEBUG: Print decoder input shapes
        let embInfo = try speakerContext.embeddings.tensorTypeAndShapeInfo()
        let featInfo = try speakerContext.features.tensorTypeAndShapeInfo()
        let speechTokInfo = try speechTokenTensor.tensorTypeAndShapeInfo()
        let audioTokInfo = try speakerContext.audioTokens.tensorTypeAndShapeInfo()
        chatterboxLogger.debug("decoder input: speechTokens=\(speechTokInfo.shape), embeddings=\(embInfo.shape), features=\(featInfo.shape)")
        chatterboxLogger.debug("audioTokens (prompt): \(audioTokInfo.shape)")

        // Pass speaker_features directly to decoder - matching reference implementation exactly
        let decoderInputs: [String: ORTValue] = [
            "speech_tokens":      speechTokenTensor,
            "speaker_embeddings": speakerContext.embeddings,
            "speaker_features":   speakerContext.features
        ]

        let decoderOutputNames = try decoderSession.outputNames() as [String]
        let audioOutputName = decoderOutputNames.first(where: { $0 == "waveform" })
                           ?? decoderOutputNames.first(where: { $0.contains("audio") || $0.contains("wav") || $0.contains("wave") })
                           ?? decoderOutputNames[0]

        let decoderOutputs = try decoderSession.run(
            withInputs: decoderInputs,
            outputNames: [audioOutputName],
            runOptions: nil
        )

        guard let audioOutput = decoderOutputs[audioOutputName] else {
            throw ChatterboxError.inferenceError("Decoder produced no audio output")
        }

        let outInfo = try audioOutput.tensorTypeAndShapeInfo()
        chatterboxLogger.debug("decoder output: \(outInfo.shape)")

        var samples = try extractFloatArray(from: audioOutput)

        // Debug: Check for audio repetition patterns in samples
        let checkInterval = 4800  // Check every ~0.2 seconds
        if samples.count > checkInterval * 2 {
            let firstSegment = Array(samples[0..<checkInterval])
            let secondSegment = Array(samples[checkInterval..<checkInterval*2])
            // Simple comparison - count matching samples
            var matches = 0
            for i in 0..<min(firstSegment.count, secondSegment.count) {
                if abs(firstSegment[i] - secondSegment[i]) < 0.001 { matches += 1 }
            }
            let matchRatio = Double(matches) / Double(checkInterval)
            if matchRatio > 0.9 {
                chatterboxLogger.warning("POTENTIAL AUDIO REPETITION: first 0.2s matches second 0.2s at \(matchRatio*100)%")
            }
        }

        chatterboxLogger.debug("CHUNK COMPLETE: \(validSpeechTokens.count) speech tokens generated, \(samples.count) audio samples")
        // Clip to [-1, 1] to prevent distortion — matches Python: np.clip(..., -1.0, 1.0)
        for i in samples.indices { samples[i] = max(-1, min(1, samples[i])) }
        return samples
    }

    // MARK: - Tensor Helpers

    /// Concatenate two int64 tensors along axis 1 (sequence dimension).
    /// Both tensors must have shape [1, S_i]. Returns [1, S1+S2].
    private func concatInt64Tensors(_ a: ORTValue, _ b: ORTValue) throws -> ORTValue {
        let aInfo  = try a.tensorTypeAndShapeInfo()
        let bInfo  = try b.tensorTypeAndShapeInfo()
        let seqA   = aInfo.shape[1].intValue
        let seqB   = bInfo.shape[1].intValue

        let aData  = try a.tensorData() as Data
        let bData  = try b.tensorData() as Data

        var combined = Data(capacity: (seqA + seqB) * MemoryLayout<Int64>.size)
        combined.append(aData)
        combined.append(bData)

        let mutableData = NSMutableData(data: combined)
        return try ORTValue(
            tensorData: mutableData,
            elementType: .int64,
            shape: [1, NSNumber(value: seqA + seqB)]
        )
    }

    /// Concatenate two embedding tensors along axis 1 (sequence dimension).
    /// Both tensors must have shape [1, S_i, D].
    /// If their element types differ, converts b to match a's type before concatenation.
    /// Returns a tensor of shape [1, S1+S2, D] with a's element type.
    private func concatEmbeddings(_ a: ORTValue, _ b: ORTValue) throws -> ORTValue {
        let aInfo  = try a.tensorTypeAndShapeInfo()
        let bInfo  = try b.tensorTypeAndShapeInfo()
        let aShape = aInfo.shape

        let batch = aShape[0].intValue
        let seqA  = aShape[1].intValue
        let seqB  = bInfo.shape[1].intValue
        let dim   = aShape[2].intValue

        let elementType = aInfo.elementType
        let bytesPerElement: Int = (elementType == .float16) ? 2 : 4

        chatterboxLogger.debug("concatEmbeddings: a=\(elementType.rawValue) seqA=\(seqA), b=\(bInfo.elementType.rawValue) seqB=\(seqB), dim=\(dim)")

        let aData = try a.tensorData() as Data

        // If types match, use b's raw bytes directly.
        // If they differ, convert b to match a's type.
        let bData: Data
        if bInfo.elementType == elementType {
            bData = try b.tensorData() as Data
        } else {
            // Types differ - convert b to match a's type
            let bFloats = try extractFloatArray(from: b)
            if elementType == .float16 {
                // Convert Float array to Float16 bytes
                let bFloat16 = bFloats.map { float32ToFloat16($0) }
                bData = bFloat16.withUnsafeBytes { Data($0) }
            } else {
                bData = bFloats.withUnsafeBytes { Data($0) }
            }
            chatterboxLogger.debug("concatEmbeddings: converted b to match a's type")
        }

        var combined = Data(capacity: (seqA + seqB) * dim * bytesPerElement)
        combined.append(aData)
        combined.append(bData)

        let mutableData = NSMutableData(data: combined)
        let outShape: [NSNumber] = [
            NSNumber(value: batch),
            NSNumber(value: seqA + seqB),
            NSNumber(value: dim)
        ]
        return try ORTValue(tensorData: mutableData, elementType: elementType, shape: outShape)
    }

    private func createInt64Tensor(_ data: [Int64], shape: [NSNumber]) throws -> ORTValue {
        let byteCount = data.count * MemoryLayout<Int64>.size
        let mutableData = data.withUnsafeBytes {
            NSMutableData(bytes: $0.baseAddress!, length: byteCount)
        }
        return try ORTValue(
            tensorData: mutableData,
            elementType: .int64,
            shape: shape
        )
    }

    /// Create an int64 tensor by wrapping an existing pre-allocated buffer.
    /// The buffer is NOT copied — the ORTValue references the original memory.
    /// The caller must keep the buffer alive for the duration of the ORT run.
    private func createInt64TensorFromBuffer(
        _ buffer: UnsafeMutableBufferPointer<Int64>,
        count: Int,
        shape: [NSNumber]
    ) throws -> ORTValue {
        let mutableData = NSMutableData(
            bytes: buffer.baseAddress!,
            length: count * MemoryLayout<Int64>.size
        )
        return try ORTValue(
            tensorData: mutableData,
            elementType: .int64,
            shape: shape
        )
    }

    /// Create an int64 tensor by filling a pre-allocated buffer.
    /// The buffer must have at least `count` elements available.
    /// This avoids repeated heap allocation inside the decode loop.
    private func fillInt64Tensor(
        buffer: UnsafeMutableBufferPointer<Int64>,
        count: Int,
        shape: [NSNumber]
    ) throws -> ORTValue {
        precondition(count <= buffer.count, "Buffer too small: \(buffer.count) < \(count)")
        // Fill with ones (attention mask pattern)
        for i in 0..<count {
            buffer[i] = 1
        }
        let mutableData = NSMutableData(
            bytes: buffer.baseAddress!,
            length: count * MemoryLayout<Int64>.size
        )
        return try ORTValue(
            tensorData: mutableData,
            elementType: .int64,
            shape: shape
        )
    }

    /// Create a zero-element float32 tensor with one dimension set to 0.
    /// Create a zero-element float16 tensor with one dimension set to 0.
    /// Used to initialise the KV-cache before the first LM inference step.
    /// q4f16 models expect float16 for KV cache.
    private func createEmptyFloatTensor(shape: [Int]) throws -> ORTValue {
        let nsShape = shape.map { NSNumber(value: $0) }
        let emptyData = NSMutableData()   // 0 bytes – the 0-dim product makes this correct
        return try ORTValue(
            tensorData: emptyData,
            elementType: .float16,
            shape: nsShape
        )
    }

    /// Scale audio features by a factor (for exaggeration control)
    /// Audio features shape: [1, seqLen, hiddenDim]
    /// Factor: >1.0 = more expressive, <1.0 = less expressive
    private func scaleAudioFeatures(_ audioFeatures: ORTValue, factor: Float) throws -> ORTValue {
        let info = try audioFeatures.tensorTypeAndShapeInfo()
        let shape = info.shape
        let elementType = info.elementType

        chatterboxLogger.debug("scaleAudioFeatures: factor=\(factor), type=\(elementType.rawValue)")

        // Extract, scale, and recreate tensor
        if elementType == .float16 {
            // Handle float16
            let floatData = try extractFloatArrayFromFloat16(from: audioFeatures)
            let scaledData = floatData.map { $0 * factor }
            let nsShape = shape.map { NSNumber(value: $0.intValue) }
            return try createFloat16Tensor(scaledData, shape: nsShape)
        } else {
            // Handle float32
            let floatData = try extractFloatArray(from: audioFeatures)
            let scaledData = floatData.map { $0 * factor }
            let nsShape = shape.map { NSNumber(value: $0.intValue) }
            return try createFloatTensor(scaledData, shape: nsShape)
        }
    }

    /// Convert a Float32 array to an ORTValue tensor with float16 element type.
    /// Required for q4f16 quantized models.
    private func createFloat16Tensor(_ data: [Float], shape: [NSNumber]) throws -> ORTValue {
        // Convert Float32 to Float16
        let float16Data = data.withUnsafeBufferPointer { buffer -> [UInt16] in
            var result = [UInt16](repeating: 0, count: buffer.count)
            for i in 0..<buffer.count {
                result[i] = float32ToFloat16(buffer[i])
            }
            return result
        }

        let byteCount = float16Data.count * MemoryLayout<UInt16>.size
        let mutableData = float16Data.withUnsafeBytes {
            NSMutableData(bytes: $0.baseAddress!, length: byteCount)
        }
        return try ORTValue(
            tensorData: mutableData,
            elementType: .float16,
            shape: shape
        )
    }

    /// Convert a Float32 array to an ORTValue tensor.
    /// Note: q4f16 quantized models still expect float32 inputs - only weights are quantized.
    private func createFloatTensor(_ data: [Float], shape: [NSNumber]) throws -> ORTValue {
        // q4f16 models expect float32 input tensors
        let byteCount = data.count * MemoryLayout<Float>.size
        let mutableData = data.withUnsafeBytes {
            NSMutableData(bytes: $0.baseAddress!, length: byteCount)
        }
        return try ORTValue(
            tensorData: mutableData,
            elementType: .float,
            shape: shape
        )
    }

    /// Extract Float16 array from Float16 tensor
    private func extractFloatArrayFromFloat16(from value: ORTValue) throws -> [Float] {
        let rawData = try value.tensorData() as Data
        let count = rawData.count / MemoryLayout<UInt16>.size
        return rawData.withUnsafeBytes { ptr in
            let buffer = ptr.bindMemory(to: UInt16.self)
            return buffer.prefix(count).map { float16ToFloat32($0) }
        }
    }

    /// Convert Float32 to Float16 (IEEE-754 half-precision)
    private func float32ToFloat16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt32((bits >> 16) & 0x8000)
        let exp = Int((bits >> 23) & 0xFF) - 127 + 15
        let mantissa = bits & 0x7FFFFF

        if exp <= 0 { // Subnormal or zero
            return UInt16(sign)
        } else if exp >= 31 { // Inf or NaN
            return UInt16(sign | 0x7C00 | (mantissa != 0 ? (mantissa >> 13) : 0))
        } else {
            return UInt16(sign | UInt32(exp << 10) | (mantissa >> 13))
        }
    }

    /// Convert Float16 to Float32
    private func float16ToFloat32(_ value: UInt16) -> Float {
        let sign: UInt32 = (value & 0x8000) != 0 ? 0x80000000 : 0
        let exp = Int((value >> 10) & 0x1F)
        var mantissa = value & 0x3FF

        if exp == 0 {
            if mantissa == 0 {
                return Float(bitPattern: sign)
            } else { // Denormal
                var result: UInt32 = sign
                var e = exp
                while e > 0 {
                    mantissa <<= 1
                    e -= 1
                }
                result |= UInt32((0x7F - e) << 23) | (UInt32(mantissa) << 13)
                return Float(bitPattern: result)
            }
        } else if exp == 31 { // Inf or NaN
            return Float(bitPattern: sign | 0x7F800000 | (UInt32(mantissa) << 13))
        } else {
            return Float(bitPattern: sign | (UInt32(exp + (127 - 15)) << 23) | (UInt32(mantissa) << 13))
        }
    }

    /// Extract a Float32 array from an ORTValue.
    /// Note: Swift ONNX Runtime only supports Float32
    private func extractFloatArray(from value: ORTValue) throws -> [Float] {
        let rawData = try value.tensorData() as Data
        let count = rawData.count / MemoryLayout<Float>.size
        return rawData.withUnsafeBytes { ptr in
            let buffer = ptr.bindMemory(to: Float.self)
            return Array(buffer.prefix(count))
        }
    }

    // MARK: - Token Sampling

    /// Greedy decode with repetition penalty — matches the reference pipeline exactly.
    ///
    /// Reference (ResembleAI/chatterbox-turbo-ONNX README):
    ///   next_token_logits = rep_penalty(generate_tokens, logits)
    ///   input_ids = argmax(next_token_logits)
    ///
    /// No temperature scaling, no top-k, no logit masking.
    /// `previous` = all tokens generated so far (including initial START_SPEECH=6561).
    private func greedyNextToken(_ logits: [Float], previous: [Int]) -> Int {
        guard !logits.isEmpty else { return config.stopSpeechToken }

        // Apply repetition penalty: reduces score of previously generated tokens.
        // For positive scores: divide by penalty (makes them less likely).
        // For negative scores: multiply by penalty (makes them more negative).
        var adj = logits
        for token in previous {
            guard token < adj.count else { continue }
            if adj[token] > 0 {
                adj[token] /= config.repetitionPenalty
            } else {
                adj[token] *= config.repetitionPenalty
            }
        }

        // Greedy: argmax
        var bestIdx = 0
        var bestVal = adj[0]
        for i in 1..<adj.count {
            if adj[i] > bestVal {
                bestVal = adj[i]
                bestIdx = i
            }
        }
        return bestIdx
    }

    // MARK: - Audio I/O Utilities

    /// Load audio file and convert to Float array at target sample rate
    private func loadAudioAsFloat(from url: URL, targetSampleRate: Int) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(targetSampleRate),
            channels: 1,
            interleaved: false
        )!

        let frameCount = AVAudioFrameCount(audioFile.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameCount) else {
            throw ChatterboxError.audioLoadFailed
        }
        try audioFile.read(into: buffer)

        // Resample if needed
        if Int(audioFile.processingFormat.sampleRate) != targetSampleRate {
            guard let converter = AVAudioConverter(from: audioFile.processingFormat, to: format) else {
                throw ChatterboxError.audioLoadFailed
            }
            let ratio = Double(targetSampleRate) / audioFile.processingFormat.sampleRate
            let outFrameCount = AVAudioFrameCount(Double(frameCount) * ratio)
            guard let outBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: outFrameCount) else {
                throw ChatterboxError.audioLoadFailed
            }
            var error: NSError?
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            converter.convert(to: outBuffer, error: &error, withInputFrom: inputBlock)
            if let error { throw error }

            let ptr = outBuffer.floatChannelData![0]
            return Array(UnsafeBufferPointer(start: ptr, count: Int(outBuffer.frameLength)))
        }

        let ptr = buffer.floatChannelData![0]
        return Array(UnsafeBufferPointer(start: ptr, count: Int(buffer.frameLength)))
    }

    /// Trim leading and trailing silence from audio samples.
    /// Uses energy threshold to detect silence.
    /// - Parameters:
    ///   - samples: Input audio samples
    ///   - threshold: Energy threshold (0-1), samples below this are considered silence
    ///   - padding: Number of non-silent samples to keep at boundaries
    /// - Returns: Trimmed audio samples
    private func trimSilence(_ samples: [Float], threshold: Float = 0.01, padding: Int = 1200) -> [Float] {
        guard samples.count > padding * 2 else { return samples }

        // Calculate frame energies (using 100-sample frames)
        let frameSize = 100
        var frameEnergies: [Float] = []

        for frameStart in stride(from: 0, to: samples.count - frameSize, by: frameSize) {
            var energy: Float = 0
            for i in frameStart..<(frameStart + frameSize) {
                energy += samples[i] * samples[i]
            }
            energy = sqrt(energy / Float(frameSize))
            frameEnergies.append(energy)
        }

        // Find first and last non-silent frames
        var firstNonSilent = 0
        for (i, energy) in frameEnergies.enumerated() {
            if energy > threshold {
                firstNonSilent = i
                break
            }
        }

        var lastNonSilent = frameEnergies.count - 1
        for i in stride(from: frameEnergies.count - 1, through: 0, by: -1) {
            if frameEnergies[i] > threshold {
                lastNonSilent = i
                break
            }
        }

        // Convert frame indices back to sample indices
        let startSample = max(0, firstNonSilent * frameSize - padding)
        let endSample = min(samples.count, (lastNonSilent + 1) * frameSize + padding)

        if startSample >= endSample {
            return samples
        }

        return Array(samples[startSample..<endSample])
    }

    /// Write Float audio samples to a standard 16-bit WAV file using CoreAudio
    private func writeWAV(samples: [Float], to url: URL, sampleRate: Int, shouldTrimSilence: Bool = false) throws {
        // Handle empty samples
        guard !samples.isEmpty else {
            // Write a short silent WAV file
            let silentSamples = [Float](repeating: 0, count: sampleRate / 10) // 100ms silence
            return try writeWAV(samples: silentSamples, to: url, sampleRate: sampleRate, shouldTrimSilence: false)
        }

        // Trim silence if enabled
        let processedSamples = shouldTrimSilence ? trimSilence(samples) : samples

        // Convert Float samples to Int16 using a raw pointer approach
        let sampleCount = processedSamples.count
        let bufferByteSize = sampleCount * MemoryLayout<Int16>.size

        // Allocate buffer for int16 samples
        let int16Buffer = UnsafeMutablePointer<Int16>.allocate(capacity: sampleCount)
        defer { int16Buffer.deallocate() }

        // Convert samples
        for i in 0..<sampleCount {
            let clamped = max(-1.0, min(1.0, processedSamples[i]))
            int16Buffer[i] = Int16(clamped * 32767.0)
        }

        // Create WAV file using ExtAudioFile
        var audioFileID: ExtAudioFileRef?

        var asbd = AudioStreamBasicDescription(
            mSampleRate: Double(sampleRate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 2,
            mFramesPerPacket: 1,
            mBytesPerFrame: 2,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 16,
            mReserved: 0
        )

        var status = ExtAudioFileCreateWithURL(
            url as CFURL,
            kAudioFileWAVEType,
            &asbd,
            nil,
            AudioFileFlags.eraseFile.rawValue,
            &audioFileID
        )

        guard status == noErr, let fileID = audioFileID else {
            throw ChatterboxError.audioWriteFailed
        }

        defer {
            ExtAudioFileDispose(fileID)
        }

        // Write the audio data using AudioBufferList
        var audioBufferList = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(
                mNumberChannels: 1,
                mDataByteSize: UInt32(bufferByteSize),
                mData: int16Buffer
            )
        )

        let framesToWrite = UInt32(sampleCount)
        status = ExtAudioFileWrite(fileID, framesToWrite, &audioBufferList)

        guard status == noErr else {
            throw ChatterboxError.audioWriteFailed
        }

        chatterboxLogger.debug("writeWAV: wrote \(samples.count) samples to \(url.lastPathComponent)")
    }
}

// MARK: - Errors

enum ChatterboxError: LocalizedError {
    case modelsNotDownloaded
    case modelNotLoaded
    case emptyText
    case audioLoadFailed
    case audioWriteFailed
    case inferenceError(String)

    var errorDescription: String? {
        switch self {
        case .modelsNotDownloaded: return "Models have not been downloaded yet."
        case .modelNotLoaded: return "Models are not loaded into memory."
        case .emptyText: return "No text to synthesize."
        case .audioLoadFailed: return "Failed to load audio file."
        case .audioWriteFailed: return "Failed to write audio file."
        case .inferenceError(let msg): return "Inference error: \(msg)"
        }
    }
}
