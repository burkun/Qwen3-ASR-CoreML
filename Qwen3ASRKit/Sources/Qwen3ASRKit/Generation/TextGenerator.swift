import Foundation
import CoreML

/// Text generator for Qwen3-ASR
/// Performs autoregressive text generation from audio features
public final class TextGenerator {
    
    // MARK: - Properties
    
    private let decoder: TextDecoder
    private let tokenizer: Qwen2Tokenizer
    private let mrope: MRoPE
    private let maxTokens: Int
    
    // MARK: - Initialization
    
    public init(decoder: TextDecoder, tokenizer: Qwen2Tokenizer, maxTokens: Int = 448) {
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.mrope = MRoPE()
        self.maxTokens = maxTokens
    }
    
    // MARK: - Generation
    
    /// Generate text tokens from audio features
    /// - Parameter audioFeatures: Encoded audio features from AudioEncoder
    /// - Returns: Array of generated token IDs
    public func generate(audioFeatures: MLMultiArray) async throws -> [Int] {
        // Get audio sequence length
        let audioLength = audioFeatures.shape[0].intValue
        
        // Initialize with prompt tokens
        // Format: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio:<|audio_bos|><|audio|>...<|audio_eos|><|im_end|>\n<|im_start|>assistant\n
        var inputIds = buildInitialPrompt(audioLength: audioLength)
        
        // Generate tokens
        var generatedTokens = [Int]()
        var isFirstForward = true
        
        for _ in 0..<maxTokens {
            // Compute position IDs
            let positionIds = try mrope.computePositionIds(
                inputIds: inputIds,
                audioLength: audioLength
            )
            
            // Create input_ids MLMultiArray
            let inputIdsArray = try createInputIdsArray(inputIds)
            
            // Forward pass
            let logits: MLMultiArray
            if isFirstForward {
                logits = try decoder.forward(
                    inputIds: inputIdsArray,
                    audioEmbeds: audioFeatures,
                    positionIds: positionIds
                )
                isFirstForward = false
            } else {
                // For subsequent passes, only use the last token
                let lastTokenArray = try createInputIdsArray([inputIds.last!])
                let lastPositionIds = try mrope.computePositionIds(
                    inputIds: [inputIds.last!],
                    audioLength: audioLength,
                    offset: inputIds.count - 1
                )
                logits = try decoder.forward(
                    inputIds: lastTokenArray,
                    audioEmbeds: nil,
                    positionIds: lastPositionIds
                )
            }
            
            // Get next token (greedy decoding)
            let lastLogits = decoder.getLastTokenLogits(logits)
            let nextToken = argmax(lastLogits)
            
            // Check for EOS
            if nextToken == Qwen2Tokenizer.eosTokenId {
                break
            }
            
            generatedTokens.append(nextToken)
            inputIds.append(nextToken)
            
            // Check cancellation
            try Task.checkCancellation()
        }
        
        return generatedTokens
    }
    
    // MARK: - Private Methods
    
    private func buildInitialPrompt(audioLength: Int) -> [Int] {
        var ids = [Int]()
        
        // Add audio placeholder tokens
        ids.append(Qwen2Tokenizer.audioBosTokenId)
        ids.append(contentsOf: [Int](repeating: Qwen2Tokenizer.audioTokenId, count: audioLength))
        ids.append(Qwen2Tokenizer.audioEosTokenId)
        
        return ids
    }
    
    private func createInputIdsArray(_ ids: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, ids.count as NSNumber], dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: ids.count)
        
        for (i, id) in ids.enumerated() {
            ptr[i] = Int32(id)
        }
        
        return array
    }
    
    private func argmax(_ array: [Float]) -> Int {
        var maxIdx = 0
        var maxVal = array[0]
        
        for (i, val) in array.enumerated() {
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }
        
        return maxIdx
    }
}

// MARK: - Generation Configuration

public struct GenerationConfig {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var doSample: Bool
    
    public init(
        maxTokens: Int = 448,
        temperature: Float = 1.0,
        topP: Float = 1.0,
        doSample: Bool = false
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.doSample = doSample
    }
}
