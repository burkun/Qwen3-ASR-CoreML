import Foundation
import CoreML
import AVFoundation

/// Qwen3-ASR CoreML model for speech recognition
/// Supports iOS 17+ and macOS 14+ with ANE acceleration
public final class Qwen3ASRModel {
    
    // MARK: - Properties
    
    private let audioEncoder: AudioEncoder
    private let textDecoder: TextDecoder
    private let audioProcessor: AudioProcessor
    private let tokenizer: Qwen2Tokenizer
    private let textGenerator: TextGenerator
    
    // MARK: - Configuration
    
    public struct Configuration {
        public var computeUnits: MLComputeUnits
        public var maxTokens: Int
        public var language: String?
        
        public init(
            computeUnits: MLComputeUnits = .all,
            maxTokens: Int = 448,
            language: String? = nil
        ) {
            self.computeUnits = computeUnits
            self.maxTokens = maxTokens
            self.language = language
        }
    }
    
    // MARK: - Initialization
    
    public init(
        modelPath: URL,
        tokenizerPath: URL,
        configuration: Configuration = Configuration()
    ) throws {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = configuration.computeUnits
        
        // Load audio encoder
        let audioEncoderURL = modelPath.appendingPathComponent("AudioEncoder.mlmodelc")
        self.audioEncoder = try AudioEncoder(modelURL: audioEncoderURL, configuration: mlConfig)
        
        // Load text decoder
        let textDecoderURL = modelPath.appendingPathComponent("TextDecoder.mlmodelc")
        self.textDecoder = try TextDecoder(modelURL: textDecoderURL, configuration: mlConfig)
        
        // Initialize audio processor
        self.audioProcessor = AudioProcessor()
        
        // Load tokenizer
        self.tokenizer = try Qwen2Tokenizer(path: tokenizerPath)
        
        // Initialize text generator
        self.textGenerator = TextGenerator(
            decoder: textDecoder,
            tokenizer: tokenizer,
            maxTokens: configuration.maxTokens
        )
    }
    
    // MARK: - Transcription
    
    /// Transcribe audio file to text
    /// - Parameter audioURL: URL to the audio file (WAV, MP3, M4A supported)
    /// - Returns: Transcribed text
    public func transcribe(audioURL: URL) async throws -> String {
        // 1. Load and process audio
        let audioData = try audioProcessor.loadAudio(from: audioURL)
        
        // 2. Compute mel spectrogram
        let melSpectrogram = try audioProcessor.computeMelSpectrogram(audio: audioData)
        
        // 3. Encode audio features
        let audioFeatures = try audioEncoder.encode(melSpectrogram: melSpectrogram)
        
        // 4. Generate text
        let tokens = try await textGenerator.generate(audioFeatures: audioFeatures)
        
        // 5. Decode tokens to text
        let text = tokenizer.decode(tokens: tokens)
        
        return text
    }
    
    /// Transcribe raw audio samples
    /// - Parameters:
    ///   - samples: Audio samples (mono, 16kHz)
    ///   - sampleRate: Sample rate of the input audio
    /// - Returns: Transcribed text
    public func transcribe(samples: [Float], sampleRate: Int = 16000) async throws -> String {
        // Resample if needed
        let processedSamples: [Float]
        if sampleRate != 16000 {
            processedSamples = audioProcessor.resample(samples, from: sampleRate, to: 16000)
        } else {
            processedSamples = samples
        }
        
        // Compute mel spectrogram
        let melSpectrogram = try audioProcessor.computeMelSpectrogram(audio: processedSamples)
        
        // Encode audio features
        let audioFeatures = try audioEncoder.encode(melSpectrogram: melSpectrogram)
        
        // Generate text
        let tokens = try await textGenerator.generate(audioFeatures: audioFeatures)
        
        // Decode tokens to text
        let text = tokenizer.decode(tokens: tokens)
        
        return text
    }
}
