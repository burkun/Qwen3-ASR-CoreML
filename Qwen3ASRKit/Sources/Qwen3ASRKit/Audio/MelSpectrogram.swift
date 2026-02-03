import Foundation
import Accelerate

/// Mel spectrogram computation using Accelerate framework
/// Based on librosa/WhisperKit implementation
public final class MelSpectrogram {
    
    // MARK: - Properties
    
    private let sampleRate: Int
    private let nFFT: Int
    private let hopLength: Int
    private let nMels: Int
    
    /// Precomputed mel filter bank [nMels, nFFT/2 + 1]
    private let melFilters: [[Float]]
    
    /// Precomputed Hann window
    private let hannWindow: [Float]
    
    /// FFT setup
    private let fftSetup: vDSP_DFT_Setup
    
    // MARK: - Initialization
    
    public init(sampleRate: Int = 16000, nFFT: Int = 400, hopLength: Int = 160, nMels: Int = 128) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.nMels = nMels
        
        // Create Hann window
        self.hannWindow = Self.createHannWindow(length: nFFT)
        
        // Create mel filter bank
        self.melFilters = Self.createMelFilterBank(
            sampleRate: sampleRate,
            nFFT: nFFT,
            nMels: nMels
        )
        
        // Setup FFT
        self.fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(nFFT),
            .FORWARD
        )!
    }
    
    deinit {
        vDSP_DFT_DestroySetup(fftSetup)
    }
    
    // MARK: - Computation
    
    /// Compute log mel spectrogram from audio samples
    /// - Parameter audio: Audio samples (mono, expected sample rate)
    /// - Returns: Log mel spectrogram [T, nMels]
    public func compute(audio: [Float]) -> [[Float]] {
        // Pad audio to ensure we get at least one frame
        let paddedAudio = padAudio(audio)
        
        // Calculate number of frames
        let numFrames = (paddedAudio.count - nFFT) / hopLength + 1
        guard numFrames > 0 else { return [] }
        
        var melSpectrogram = [[Float]]()
        melSpectrogram.reserveCapacity(numFrames)
        
        // Process each frame
        for frameIdx in 0..<numFrames {
            let startIdx = frameIdx * hopLength
            let frame = Array(paddedAudio[startIdx..<(startIdx + nFFT)])
            
            // Apply window
            let windowedFrame = applyWindow(frame)
            
            // Compute FFT magnitude
            let magnitude = computeFFTMagnitude(windowedFrame)
            
            // Apply mel filter bank
            let melFrame = applyMelFilters(magnitude)
            
            // Apply log scaling
            let logMelFrame = applyLogScale(melFrame)
            
            melSpectrogram.append(logMelFrame)
        }
        
        return melSpectrogram
    }
    
    // MARK: - Private Methods
    
    private func padAudio(_ audio: [Float]) -> [Float] {
        // Pad to ensure at least one frame
        let minLength = nFFT
        if audio.count >= minLength {
            return audio
        }
        
        var padded = audio
        padded.append(contentsOf: [Float](repeating: 0, count: minLength - audio.count))
        return padded
    }
    
    private func applyWindow(_ frame: [Float]) -> [Float] {
        var windowed = [Float](repeating: 0, count: frame.count)
        vDSP_vmul(frame, 1, hannWindow, 1, &windowed, 1, vDSP_Length(frame.count))
        return windowed
    }
    
    private func computeFFTMagnitude(_ frame: [Float]) -> [Float] {
        let n = frame.count
        let halfN = n / 2 + 1
        
        // Prepare input for DFT (real to complex)
        var realInput = frame
        var imagInput = [Float](repeating: 0, count: n)
        var realOutput = [Float](repeating: 0, count: n)
        var imagOutput = [Float](repeating: 0, count: n)
        
        // Execute DFT
        vDSP_DFT_Execute(fftSetup, &realInput, &imagInput, &realOutput, &imagOutput)
        
        // Compute magnitude: sqrt(real^2 + imag^2)
        var magnitude = [Float](repeating: 0, count: halfN)
        for i in 0..<halfN {
            let real = realOutput[i]
            let imag = imagOutput[i]
            magnitude[i] = sqrtf(real * real + imag * imag)
        }
        
        // Square the magnitude to get power spectrum
        vDSP_vsq(magnitude, 1, &magnitude, 1, vDSP_Length(halfN))
        
        return magnitude
    }
    
    private func applyMelFilters(_ magnitude: [Float]) -> [Float] {
        var melFrame = [Float](repeating: 0, count: nMels)
        
        for m in 0..<nMels {
            var sum: Float = 0
            let filter = melFilters[m]
            vDSP_dotpr(magnitude, 1, filter, 1, &sum, vDSP_Length(min(magnitude.count, filter.count)))
            melFrame[m] = sum
        }
        
        return melFrame
    }
    
    private func applyLogScale(_ frame: [Float]) -> [Float] {
        // log10(max(frame, 1e-10))
        var logFrame = [Float](repeating: 0, count: frame.count)
        
        for i in 0..<frame.count {
            logFrame[i] = log10f(max(frame[i], 1e-10))
        }
        
        return logFrame
    }
    
    // MARK: - Static Methods
    
    /// Create Hann window
    private static func createHannWindow(length: Int) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        vDSP_hann_window(&window, vDSP_Length(length), Int32(vDSP_HANN_NORM))
        return window
    }
    
    /// Create mel filter bank
    /// - Parameters:
    ///   - sampleRate: Audio sample rate
    ///   - nFFT: FFT size
    ///   - nMels: Number of mel bands
    /// - Returns: Mel filter bank [nMels, nFFT/2 + 1]
    private static func createMelFilterBank(sampleRate: Int, nFFT: Int, nMels: Int) -> [[Float]] {
        let nFreqs = nFFT / 2 + 1
        let fMin: Float = 0.0
        let fMax = Float(sampleRate) / 2.0
        
        // Convert Hz to Mel
        func hzToMel(_ hz: Float) -> Float {
            return 2595.0 * log10f(1.0 + hz / 700.0)
        }
        
        // Convert Mel to Hz
        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (powf(10.0, mel / 2595.0) - 1.0)
        }
        
        // Create mel points
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let melPoints = (0...(nMels + 1)).map { i in
            melToHz(melMin + Float(i) * (melMax - melMin) / Float(nMels + 1))
        }
        
        // Convert to FFT bin indices
        let binPoints = melPoints.map { hz -> Int in
            Int(floorf(Float(nFFT + 1) * hz / Float(sampleRate)))
        }
        
        // Create filter bank
        var filterBank = [[Float]]()
        filterBank.reserveCapacity(nMels)
        
        for m in 0..<nMels {
            var filter = [Float](repeating: 0, count: nFreqs)
            
            let left = binPoints[m]
            let center = binPoints[m + 1]
            let right = binPoints[m + 2]
            
            // Rising edge
            for k in left..<center {
                if k >= 0 && k < nFreqs && center > left {
                    filter[k] = Float(k - left) / Float(center - left)
                }
            }
            
            // Falling edge
            for k in center..<right {
                if k >= 0 && k < nFreqs && right > center {
                    filter[k] = Float(right - k) / Float(right - center)
                }
            }
            
            filterBank.append(filter)
        }
        
        return filterBank
    }
}
