import Foundation
import CoreML

/// Qwen3-ASR Audio Encoder
/// Converts mel spectrogram to audio features
public final class AudioEncoder {
    
    // MARK: - Properties
    
    private let model: MLModel
    
    /// Output feature dimension
    public static let outputDim = 3584
    
    // MARK: - Initialization
    
    public init(modelURL: URL, configuration: MLModelConfiguration) throws {
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }
    
    // MARK: - Encoding
    
    /// Encode mel spectrogram to audio features
    /// - Parameter melSpectrogram: Input mel spectrogram [1, 128, T]
    /// - Returns: Audio features [L, 3584]
    public func encode(melSpectrogram: MLMultiArray) throws -> MLMultiArray {
        let inputName = "mel_spectrogram"
        let featureLensName = "feature_lens"
        
        // Get sequence length
        let seqLength = melSpectrogram.shape[2].intValue
        
        // Create feature lens input
        let featureLens = try MLMultiArray(shape: [1], dataType: .int32)
        featureLens[0] = NSNumber(value: seqLength)
        
        // Create input features
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: melSpectrogram),
            featureLensName: MLFeatureValue(multiArray: featureLens)
        ])
        
        // Run inference
        let output = try model.prediction(from: inputFeatures)
        
        // Get output
        guard let audioFeatures = output.featureValue(for: "audio_features")?.multiArrayValue else {
            throw AudioEncoderError.invalidOutput
        }
        
        return audioFeatures
    }
    
    /// Calculate output sequence length from input length
    /// - Parameter inputLength: Input mel spectrogram length
    /// - Returns: Output sequence length
    public static func outputLength(for inputLength: Int) -> Int {
        // Based on Qwen3-ASR _get_feat_extract_output_lengths
        let inputLengthsLeave = inputLength % 100
        var featLengths = (inputLengthsLeave - 1) / 2 + 1
        featLengths = (featLengths - 1) / 2 + 1
        let outputLengths = (featLengths - 1) / 2 + 1 + (inputLength / 100) * 13
        return outputLengths
    }
}

// MARK: - Errors

public enum AudioEncoderError: Error, LocalizedError {
    case invalidOutput
    case invalidInput
    
    public var errorDescription: String? {
        switch self {
        case .invalidOutput:
            return "Failed to get audio features from encoder output"
        case .invalidInput:
            return "Invalid mel spectrogram input"
        }
    }
}
