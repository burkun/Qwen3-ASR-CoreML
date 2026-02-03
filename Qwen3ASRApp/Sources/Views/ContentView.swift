import SwiftUI
import UniformTypeIdentifiers
import Qwen3ASRKit

struct ContentView: View {
    @State private var selectedFileURL: URL?
    @State private var transcriptionResult: String = ""
    @State private var isTranscribing: Bool = false
    @State private var errorMessage: String?
    @State private var isDragOver: Bool = false
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            Text("Qwen3-ASR")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("CoreML Speech Recognition")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Divider()
            
            // File Selection Area
            VStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(isDragOver ? Color.accentColor : Color.gray.opacity(0.3), lineWidth: 2)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(isDragOver ? Color.accentColor.opacity(0.1) : Color.gray.opacity(0.05))
                        )
                        .frame(height: 120)
                    
                    VStack(spacing: 8) {
                        Image(systemName: "waveform")
                            .font(.system(size: 32))
                            .foregroundColor(.accentColor)
                        
                        if let url = selectedFileURL {
                            Text(url.lastPathComponent)
                                .font(.headline)
                            Text(url.path)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        } else {
                            Text("Drop audio file here or click to select")
                                .font(.headline)
                            Text("Supports WAV, MP3, M4A")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .onDrop(of: [.audio, .fileURL], isTargeted: $isDragOver) { providers in
                    handleDrop(providers: providers)
                }
                .onTapGesture {
                    selectFile()
                }
                
                // Transcribe Button
                Button(action: transcribe) {
                    HStack {
                        if isTranscribing {
                            ProgressView()
                                .scaleEffect(0.8)
                                .frame(width: 16, height: 16)
                        } else {
                            Image(systemName: "text.bubble")
                        }
                        Text(isTranscribing ? "Transcribing..." : "Transcribe")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .disabled(selectedFileURL == nil || isTranscribing)
            }
            
            // Error Message
            if let error = errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.orange)
                }
                .padding(.horizontal)
            }
            
            Divider()
            
            // Result Area
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Transcription Result")
                        .font(.headline)
                    Spacer()
                    if !transcriptionResult.isEmpty {
                        Button(action: copyResult) {
                            Image(systemName: "doc.on.doc")
                        }
                        .buttonStyle(.borderless)
                        .help("Copy to clipboard")
                    }
                }
                
                ScrollView {
                    Text(transcriptionResult.isEmpty ? "Transcription will appear here..." : transcriptionResult)
                        .font(.body)
                        .foregroundColor(transcriptionResult.isEmpty ? .secondary : .primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 100, maxHeight: 200)
                .padding(8)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)
            }
            
            Spacer()
            
            // Footer
            HStack {
                Text("Model: Qwen3-ASR-1.7B")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Powered by CoreML")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(24)
        .frame(width: 500, height: 520)
    }
    
    // MARK: - Actions
    
    private func selectFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.audio, .wav, .mp3, .mpeg4Audio]
        
        if panel.runModal() == .OK {
            selectedFileURL = panel.url
            errorMessage = nil
        }
    }
    
    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        
        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, error in
                if let data = item as? Data,
                   let url = URL(dataRepresentation: data, relativeTo: nil) {
                    DispatchQueue.main.async {
                        selectedFileURL = url
                        errorMessage = nil
                    }
                }
            }
            return true
        }
        return false
    }
    
    private func transcribe() {
        guard let url = selectedFileURL else { return }
        
        isTranscribing = true
        errorMessage = nil
        transcriptionResult = ""
        
        Task {
            do {
                // TODO: Replace with actual model inference when models are ready
                // For now, simulate transcription
                try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
                
                await MainActor.run {
                    // Placeholder result
                    transcriptionResult = "[Model not loaded]\n\nTo use this app:\n1. Run the model conversion scripts\n2. Place the CoreML models in Resources/Models/\n3. Place tokenizer files in Resources/Tokenizer/"
                    isTranscribing = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isTranscribing = false
                }
            }
        }
    }
    
    private func copyResult() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(transcriptionResult, forType: .string)
    }
}

#Preview {
    ContentView()
}
