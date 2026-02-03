import Foundation
import AVFoundation
import Accelerate

/// Audio processor for Qwen3-ASR
/// Handles audio loading, resampling, and mel spectrogram computation
public final class AudioProcessor {
    
    // MARK: - Constants
    
    /// Target sample rate for Qwen3-ASR (16kHz)
    public static let sampleRate: Int = 16000
    
    /// Number of mel frequency bins
    public static let nMels: Int = 128
    
    /// FFT window size
    public static let nFFT: Int = 400
    
    /// Hop length between frames
    public static let hopLength: Int = 160
    
    // MARK: - Properties
    
    private let melSpectrogram: MelSpectrogram
    
    // MARK: - Initialization
    
    public init() {
        self.melSpectrogram = MelSpectrogram(
            sampleRate: Self.sampleRate,
            nFFT: Self.nFFT,
            hopLength: Self.hopLength,
            nMels: Self.nMels
        )
    }
    
    // MARK: - Audio Loading
    
    /// Load audio from a file URL
    /// - Parameter url: URL to the audio file
    /// - Returns: Audio samples as Float array (mono, 16kHz)
    public func loadAudio(from url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioProcessorError.failedToCreateBuffer
        }
        
        try audioFile.read(into: buffer)
        
        // Convert to mono Float array
        var samples = convertToMonoFloat(buffer: buffer)
        
        // Resample to 16kHz if needed
        let inputSampleRate = Int(format.sampleRate)
        if inputSampleRate != Self.sampleRate {
            samples = resample(samples, from: inputSampleRate, to: Self.sampleRate)
        }
        
        return samples
    }
    
    /// Load audio from raw data
    /// - Parameters:
    ///   - data: Raw audio data
    ///   - format: Audio format description
    /// - Returns: Audio samples as Float array
    public func loadAudio(from data: Data, sampleRate: Int = 16000) throws -> [Float] {
        // Assuming 16-bit PCM
        let samples = data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
            let int16Ptr = ptr.bindMemory(to: Int16.self)
            return int16Ptr.map { Float($0) / Float(Int16.max) }
        }
        
        if sampleRate != Self.sampleRate {
            return resample(samples, from: sampleRate, to: Self.sampleRate)
        }
        
        return samples
    }
    
    // MARK: - Mel Spectrogram
    
    /// Compute mel spectrogram from audio samples
    /// - Parameter audio: Audio samples (mono, 16kHz)
    /// - Returns: MLMultiArray with shape [1, 128, T]
    public func computeMelSpectrogram(audio: [Float]) throws -> MLMultiArray {
        let melSpec = melSpectrogram.compute(audio: audio)
        
        // Convert to MLMultiArray [1, nMels, T]
        let numFrames = melSpec.count
        let shape = [1, Self.nMels, numFrames] as [NSNumber]
        
        let mlArray = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = mlArray.dataPointer.bindMemory(to: Float.self, capacity: mlArray.count)
        
        for t in 0..<numFrames {
            for m in 0..<Self.nMels {
                ptr[m * numFrames + t] = melSpec[t][m]
            }
        }
        
        return mlArray
    }
    
    // MARK: - Resampling
    
    /// Resample audio from one sample rate to another
    /// - Parameters:
    ///   - samples: Input audio samples
    ///   - fromRate: Source sample rate
    ///   - toRate: Target sample rate
    /// - Returns: Resampled audio samples
    public func resample(_ samples: [Float], from fromRate: Int, to toRate: Int) -> [Float] {
        guard fromRate != toRate else { return samples }
        
        let ratio = Double(toRate) / Double(fromRate)
        let outputLength = Int(Double(samples.count) * ratio)
        var output = [Float](repeating: 0, count: outputLength)
        
        // Use vDSP for linear interpolation resampling
        var control = vDSP_Length(0)
        var filterStride = vDSP_Stride(1)
        
        // Simple linear interpolation
        for i in 0..<outputLength {
            let srcIdx = Double(i) / ratio
            let srcIdxInt = Int(srcIdx)
            let frac = Float(srcIdx - Double(srcIdxInt))
            
            if srcIdxInt + 1 < samples.count {
                output[i] = samples[srcIdxInt] * (1 - frac) + samples[srcIdxInt + 1] * frac
            } else if srcIdxInt < samples.count {
                output[i] = samples[srcIdxInt]
            }
        }
        
        return output
    }
    
    // MARK: - Private Methods
    
    private func convertToMonoFloat(buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else {
            return []
        }
        
        let frameLength = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        
        if channelCount == 1 {
            // Already mono
            return Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
        }
        
        // Mix channels to mono
        var monoSamples = [Float](repeating: 0, count: frameLength)
        let scale = 1.0 / Float(channelCount)
        
        for channel in 0..<channelCount {
            let channelPtr = channelData[channel]
            vDSP_vsma(channelPtr, 1, [scale], monoSamples, 1, &monoSamples, 1, vDSP_Length(frameLength))
        }
        
        return monoSamples
    }
}

// MARK: - Errors

public enum AudioProcessorError: Error, LocalizedError {
    case failedToCreateBuffer
    case failedToReadAudio
    case invalidFormat
    case unsupportedSampleRate
    
    public var errorDescription: String? {
        switch self {
        case .failedToCreateBuffer:
            return "Failed to create audio buffer"
        case .failedToReadAudio:
            return "Failed to read audio file"
        case .invalidFormat:
            return "Invalid audio format"
        case .unsupportedSampleRate:
            return "Unsupported sample rate"
        }
    }
}
