import Foundation

/// Qwen2 BPE Tokenizer for Qwen3-ASR
public final class Qwen2Tokenizer {
    
    // MARK: - Special Tokens
    
    public static let audioTokenId = 151646
    public static let audioBosTokenId = 151647
    public static let audioEosTokenId = 151648
    public static let bosTokenId = 151644
    public static let eosTokenId = 151645
    public static let padTokenId = 151643
    
    // MARK: - Properties
    
    private let vocab: [String: Int]
    private let reverseVocab: [Int: String]
    private let merges: [(String, String)]
    private let byteEncoder: ByteEncoder
    private let specialTokens: Set<Int>
    
    // MARK: - Initialization
    
    public init(path: URL) throws {
        let vocabURL = path.appendingPathComponent("vocab.json")
        let mergesURL = path.appendingPathComponent("merges.txt")
        
        // Load vocab
        let vocabData = try Data(contentsOf: vocabURL)
        self.vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
        
        // Create reverse vocab
        self.reverseVocab = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
        
        // Load merges
        let mergesContent = try String(contentsOf: mergesURL, encoding: .utf8)
        self.merges = Self.parseMerges(mergesContent)
        
        // Initialize byte encoder
        self.byteEncoder = ByteEncoder()
        
        // Define special tokens
        self.specialTokens = Set([
            Self.audioTokenId,
            Self.audioBosTokenId,
            Self.audioEosTokenId,
            Self.bosTokenId,
            Self.eosTokenId,
            Self.padTokenId
        ])
    }
    
    /// Initialize with vocab and merges directly
    public init(vocab: [String: Int], merges: [(String, String)]) {
        self.vocab = vocab
        self.reverseVocab = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
        self.merges = merges
        self.byteEncoder = ByteEncoder()
        self.specialTokens = Set([
            Self.audioTokenId,
            Self.audioBosTokenId,
            Self.audioEosTokenId,
            Self.bosTokenId,
            Self.eosTokenId,
            Self.padTokenId
        ])
    }
    
    // MARK: - Encoding
    
    /// Encode text to token IDs
    /// - Parameter text: Input text
    /// - Returns: Array of token IDs
    public func encode(_ text: String) -> [Int] {
        // Normalize text
        let normalized = text.precomposedStringWithCanonicalMapping
        
        // Convert to byte-level tokens
        var tokens = byteEncoder.encode(normalized)
        
        // Apply BPE merges
        tokens = applyBPE(tokens)
        
        // Convert to IDs
        return tokens.compactMap { vocab[$0] }
    }
    
    /// Encode text with special tokens for ASR
    /// - Parameter text: Input text (usually empty for ASR)
    /// - Returns: Array of token IDs with special tokens
    public func encodeForASR(_ text: String = "") -> [Int] {
        var ids = [Self.bosTokenId]
        
        if !text.isEmpty {
            ids.append(contentsOf: encode(text))
        }
        
        return ids
    }
    
    // MARK: - Decoding
    
    /// Decode token IDs to text
    /// - Parameters:
    ///   - tokens: Array of token IDs
    ///   - skipSpecialTokens: Whether to skip special tokens
    /// - Returns: Decoded text
    public func decode(tokens: [Int], skipSpecialTokens: Bool = true) -> String {
        var result = ""
        
        for token in tokens {
            if skipSpecialTokens && specialTokens.contains(token) {
                continue
            }
            
            if let tokenStr = reverseVocab[token] {
                result += tokenStr
            }
        }
        
        // Decode byte-level encoding
        return byteEncoder.decode(result)
    }
    
    // MARK: - Vocabulary Access
    
    /// Get vocabulary size
    public var vocabSize: Int {
        return vocab.count
    }
    
    /// Convert token to ID
    public func tokenToId(_ token: String) -> Int? {
        return vocab[token]
    }
    
    /// Convert ID to token
    public func idToToken(_ id: Int) -> String? {
        return reverseVocab[id]
    }
    
    // MARK: - Private Methods
    
    private static func parseMerges(_ content: String) -> [(String, String)] {
        var merges = [(String, String)]()
        let lines = content.components(separatedBy: .newlines)
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                continue
            }
            
            let parts = trimmed.split(separator: " ", maxSplits: 1)
            if parts.count == 2 {
                merges.append((String(parts[0]), String(parts[1])))
            }
        }
        
        return merges
    }
    
    private func applyBPE(_ tokens: [String]) -> [String] {
        var result = tokens
        
        // Create merge priority map using string key
        var mergePriority = [String: Int]()
        for (index, merge) in merges.enumerated() {
            let key = "\(merge.0)\t\(merge.1)"
            mergePriority[key] = index
        }
        
        while result.count > 1 {
            // Find the best merge
            var bestPriority = Int.max
            var bestIndex = -1
            
            for i in 0..<(result.count - 1) {
                let key = "\(result[i])\t\(result[i + 1])"
                if let priority = mergePriority[key], priority < bestPriority {
                    bestPriority = priority
                    bestIndex = i
                }
            }
            
            guard bestIndex >= 0 else { break }
            
            // Apply the merge
            let merged = result[bestIndex] + result[bestIndex + 1]
            result[bestIndex] = merged
            result.remove(at: bestIndex + 1)
        }
        
        return result
    }
}

// MARK: - Errors

public enum TokenizerError: Error, LocalizedError {
    case failedToLoadVocab
    case failedToLoadMerges
    case invalidFormat
    
    public var errorDescription: String? {
        switch self {
        case .failedToLoadVocab:
            return "Failed to load vocabulary file"
        case .failedToLoadMerges:
            return "Failed to load merges file"
        case .invalidFormat:
            return "Invalid tokenizer format"
        }
    }
}
