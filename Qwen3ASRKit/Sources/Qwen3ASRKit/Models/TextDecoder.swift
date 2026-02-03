import Foundation
import CoreML

/// Qwen3-ASR Text Decoder
/// Generates text tokens from audio features
public final class TextDecoder {
    
    // MARK: - Properties
    
    private let model: MLModel
    
    /// Vocabulary size
    public static let vocabSize = 151936
    
    /// Hidden dimension
    public static let hiddenSize = 4096
    
    // MARK: - Initialization
    
    public init(modelURL: URL, configuration: MLModelConfiguration) throws {
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }
    
    // MARK: - Forward Pass
    
    /// Run forward pass
    /// - Parameters:
    ///   - inputIds: Token IDs [1, seq_len]
    ///   - audioEmbeds: Audio embeddings [audio_len, 3584] (optional, only for first forward)
    ///   - positionIds: Position IDs [3, 1, seq_len] for MRoPE
    /// - Returns: Logits [1, seq_len, vocab_size]
    public func forward(
        inputIds: MLMultiArray,
        audioEmbeds: MLMultiArray?,
        positionIds: MLMultiArray
    ) throws -> MLMultiArray {
        var inputDict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "position_ids": MLFeatureValue(multiArray: positionIds)
        ]
        
        // Add audio embeddings if provided
        if let audioEmbeds = audioEmbeds {
            inputDict["audio_embeds"] = MLFeatureValue(multiArray: audioEmbeds)
            inputDict["has_audio"] = MLFeatureValue(int64: 1)
        } else {
            inputDict["has_audio"] = MLFeatureValue(int64: 0)
        }
        
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: inputDict)
        
        // Run inference
        let output = try model.prediction(from: inputFeatures)
        
        // Get logits
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw TextDecoderError.invalidOutput
        }
        
        return logits
    }
    
    /// Get the last token logits from output
    /// - Parameter logits: Full logits [1, seq_len, vocab_size]
    /// - Returns: Last token logits [vocab_size]
    public func getLastTokenLogits(_ logits: MLMultiArray) -> [Float] {
        let seqLen = logits.shape[1].intValue
        let vocabSize = logits.shape[2].intValue
        
        var lastLogits = [Float](repeating: 0, count: vocabSize)
        let lastIdx = seqLen - 1
        
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        let offset = lastIdx * vocabSize
        
        for i in 0..<vocabSize {
            lastLogits[i] = ptr[offset + i]
        }
        
        return lastLogits
    }
}

// MARK: - Errors

public enum TextDecoderError: Error, LocalizedError {
    case invalidOutput
    case invalidInput
    
    public var errorDescription: String? {
        switch self {
        case .invalidOutput:
            return "Failed to get logits from decoder output"
        case .invalidInput:
            return "Invalid decoder input"
        }
    }
}
