import Foundation

/// Byte-level encoder for BPE tokenization
/// Maps bytes to unicode characters to avoid whitespace/control character issues
public final class ByteEncoder {
    
    // MARK: - Properties
    
    private let byteToChar: [UInt8: Character]
    private let charToByte: [Character: UInt8]
    
    // MARK: - Initialization
    
    public init() {
        var byteToChar = [UInt8: Character]()
        var charToByte = [Character: UInt8]()
        
        // Create mapping from bytes to unicode characters
        // Based on GPT-2 / Qwen byte encoding
        var n = 0
        
        // Printable ASCII characters map to themselves
        for b in UInt8(33)...UInt8(126) {  // '!' to '~'
            let char = Character(UnicodeScalar(b))
            byteToChar[b] = char
            charToByte[char] = b
        }
        
        // Extended characters
        for b in UInt8(161)...UInt8(172) {  // '¡' to '¬'
            let char = Character(UnicodeScalar(b))
            byteToChar[b] = char
            charToByte[char] = b
        }
        
        for b in UInt8(174)...UInt8(255) {  // '®' to 'ÿ'
            let char = Character(UnicodeScalar(b))
            byteToChar[b] = char
            charToByte[char] = b
        }
        
        // Map remaining bytes to characters starting at 256
        n = 0
        for b: UInt8 in 0...255 {
            if byteToChar[b] == nil {
                let scalar = UnicodeScalar(256 + n)!
                let char = Character(scalar)
                byteToChar[b] = char
                charToByte[char] = b
                n += 1
            }
        }
        
        self.byteToChar = byteToChar
        self.charToByte = charToByte
    }
    
    // MARK: - Encoding
    
    /// Encode a string to byte-level tokens
    /// - Parameter text: Input string
    /// - Returns: Array of byte-level token strings
    public func encode(_ text: String) -> [String] {
        let bytes = Array(text.utf8)
        var result = [String]()
        
        for byte in bytes {
            if let char = byteToChar[byte] {
                result.append(String(char))
            }
        }
        
        return result
    }
    
    /// Encode a string to a single byte-level string
    /// - Parameter text: Input string
    /// - Returns: Byte-level encoded string
    public func encodeToString(_ text: String) -> String {
        let bytes = Array(text.utf8)
        var result = ""
        
        for byte in bytes {
            if let char = byteToChar[byte] {
                result.append(char)
            }
        }
        
        return result
    }
    
    // MARK: - Decoding
    
    /// Decode a byte-level string back to original string
    /// - Parameter encoded: Byte-level encoded string
    /// - Returns: Decoded string
    public func decode(_ encoded: String) -> String {
        var bytes = [UInt8]()
        
        for char in encoded {
            if let byte = charToByte[char] {
                bytes.append(byte)
            }
        }
        
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }
    
    /// Decode an array of byte-level tokens to original string
    /// - Parameter tokens: Array of byte-level token strings
    /// - Returns: Decoded string
    public func decode(_ tokens: [String]) -> String {
        return decode(tokens.joined())
    }
}
