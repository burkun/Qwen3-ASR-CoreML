import Foundation
import CoreML

/// Multi-dimensional Rotary Position Embedding (MRoPE) for Qwen3-ASR
/// Computes 3D position IDs for temporal, height, and width dimensions
public final class MRoPE {
    
    // MARK: - Constants
    
    /// MRoPE section sizes: [temporal, height, width]
    /// These define how the head dimension is split across 3 positional dimensions
    public static let mropeSection = [24, 20, 20]  // Total: 64 (half of head_dim=128)
    
    /// Head dimension
    public static let headDim = 128
    
    /// RoPE base frequency
    public static let ropeTheta: Float = 5_000_000.0
    
    // MARK: - Initialization
    
    public init() {}
    
    // MARK: - Position ID Computation
    
    /// Compute position IDs for MRoPE
    /// - Parameters:
    ///   - inputIds: Current input token IDs
    ///   - audioLength: Length of audio features sequence
    ///   - offset: Position offset for incremental generation
    /// - Returns: Position IDs MLMultiArray with shape [3, 1, seq_len]
    public func computePositionIds(
        inputIds: [Int],
        audioLength: Int,
        offset: Int = 0
    ) throws -> MLMultiArray {
        let seqLen = inputIds.count
        
        // Create position IDs array [3, 1, seq_len]
        let positionIds = try MLMultiArray(shape: [3, 1, seqLen as NSNumber], dataType: .int32)
        let ptr = positionIds.dataPointer.bindMemory(to: Int32.self, capacity: 3 * seqLen)
        
        // Find audio token positions
        let audioBosIdx = inputIds.firstIndex(of: Qwen2Tokenizer.audioBosTokenId) ?? 0
        let audioEosIdx = inputIds.lastIndex(of: Qwen2Tokenizer.audioEosTokenId) ?? seqLen
        
        // Compute positions for each dimension
        for i in 0..<seqLen {
            let globalPos = offset + i
            
            // Dimension 0: Temporal position (increments for all tokens)
            ptr[0 * seqLen + i] = Int32(globalPos)
            
            // Dimension 1: Height position
            // Audio tokens have special handling
            if i > audioBosIdx && i <= audioEosIdx {
                // Within audio region: use 0 for height
                ptr[1 * seqLen + i] = 0
            } else {
                // Outside audio region: same as temporal
                ptr[1 * seqLen + i] = Int32(globalPos)
            }
            
            // Dimension 2: Width position
            // Similar to height for now
            if i > audioBosIdx && i <= audioEosIdx {
                ptr[2 * seqLen + i] = 0
            } else {
                ptr[2 * seqLen + i] = Int32(globalPos)
            }
        }
        
        return positionIds
    }
    
    /// Compute position IDs for a single new token (incremental generation)
    /// - Parameters:
    ///   - currentLength: Current sequence length (before adding new token)
    ///   - audioLength: Length of audio features sequence
    /// - Returns: Position IDs for the new token [3, 1, 1]
    public func computeIncrementalPositionIds(
        currentLength: Int,
        audioLength: Int
    ) throws -> MLMultiArray {
        let positionIds = try MLMultiArray(shape: [3, 1, 1], dataType: .int32)
        let ptr = positionIds.dataPointer.bindMemory(to: Int32.self, capacity: 3)
        
        // For text generation after audio, all dimensions use the same position
        let pos = Int32(currentLength)
        ptr[0] = pos  // Temporal
        ptr[1] = pos  // Height
        ptr[2] = pos  // Width
        
        return positionIds
    }
    
    // MARK: - RoPE Computation (for reference)
    
    /// Compute RoPE frequencies
    /// - Parameter maxSeqLen: Maximum sequence length
    /// - Returns: Tuple of (cos, sin) frequency tensors
    public func computeFrequencies(maxSeqLen: Int) -> (cos: [[Float]], sin: [[Float]]) {
        let halfDim = Self.headDim / 2
        var frequencies = [[Float]](repeating: [Float](repeating: 0, count: halfDim), count: maxSeqLen)
        
        // Compute inverse frequencies based on MRoPE sections
        var invFreq = [Float](repeating: 0, count: halfDim)
        var offset = 0
        
        for (sectionIdx, sectionSize) in Self.mropeSection.enumerated() {
            for i in 0..<sectionSize {
                let freq = 1.0 / pow(Self.ropeTheta, Float(2 * i) / Float(2 * sectionSize))
                invFreq[offset + i] = freq
            }
            offset += sectionSize
        }
        
        // Compute cos and sin for each position
        var cosFreq = [[Float]](repeating: [Float](repeating: 0, count: halfDim), count: maxSeqLen)
        var sinFreq = [[Float]](repeating: [Float](repeating: 0, count: halfDim), count: maxSeqLen)
        
        for pos in 0..<maxSeqLen {
            for i in 0..<halfDim {
                let angle = Float(pos) * invFreq[i]
                cosFreq[pos][i] = cos(angle)
                sinFreq[pos][i] = sin(angle)
            }
        }
        
        return (cosFreq, sinFreq)
    }
}
