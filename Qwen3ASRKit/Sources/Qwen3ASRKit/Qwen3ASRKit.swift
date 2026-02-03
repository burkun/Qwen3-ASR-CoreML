// Qwen3ASRKit - CoreML implementation for Qwen3-ASR speech recognition
// Supports iOS 17+ and macOS 14+ with ANE acceleration

@_exported import CoreML
@_exported import Accelerate
@_exported import AVFoundation

// Re-export public types
public typealias ASRModel = Qwen3ASRModel
