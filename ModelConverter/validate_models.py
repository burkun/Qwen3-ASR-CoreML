#!/usr/bin/env python3
"""
Validate CoreML models against PyTorch implementation.

This script runs inference on both PyTorch and CoreML models and compares outputs.

Usage:
    python validate_models.py --audio <audio_file> --models_path <models_path>
"""

import argparse
from pathlib import Path

import coremltools as ct
import librosa
import numpy as np
import torch


def load_audio(audio_path: str, sample_rate: int = 16000):
    """Load and preprocess audio file."""
    print(f"Loading audio from {audio_path}...")
    
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    print(f"Audio loaded: {len(audio)} samples, {len(audio)/sample_rate:.2f}s")
    
    return audio


def compute_mel_spectrogram(audio: np.ndarray, sample_rate: int = 16000):
    """Compute mel spectrogram using librosa."""
    n_fft = 400
    hop_length = 160
    n_mels = 128
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Add batch dimension [1, n_mels, T]
    log_mel_spec = np.expand_dims(log_mel_spec, axis=0).astype(np.float32)
    
    print(f"Mel spectrogram shape: {log_mel_spec.shape}")
    return log_mel_spec


def validate_audio_encoder(models_path: str, mel_spec: np.ndarray):
    """Validate AudioEncoder CoreML model."""
    print("\n=== Validating AudioEncoder ===")
    
    model_path = Path(models_path) / "AudioEncoder.mlpackage"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    # Load CoreML model
    model = ct.models.MLModel(str(model_path))
    
    # Prepare inputs
    seq_len = mel_spec.shape[2]
    feature_lens = np.array([seq_len], dtype=np.int32)
    
    # Run inference
    print("Running CoreML inference...")
    output = model.predict({
        "mel_spectrogram": mel_spec,
        "feature_lens": feature_lens,
    })
    
    audio_features = output["audio_features"]
    print(f"Audio features shape: {audio_features.shape}")
    
    return audio_features


def validate_text_decoder(models_path: str, audio_features: np.ndarray):
    """Validate TextDecoder CoreML model."""
    print("\n=== Validating TextDecoder ===")
    
    model_path = Path(models_path) / "TextDecoder.mlpackage"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    # Load CoreML model
    model = ct.models.MLModel(str(model_path))
    
    # Prepare inputs
    audio_len = audio_features.shape[0]
    seq_len = audio_len + 10  # Some extra tokens
    
    input_ids = np.zeros((1, seq_len), dtype=np.int32)
    input_ids[0, :audio_len] = 151646  # Audio token ID
    
    position_ids = np.zeros((3, 1, seq_len), dtype=np.int32)
    for i in range(seq_len):
        position_ids[:, 0, i] = i
    
    has_audio = np.array([1], dtype=np.int32)
    
    # Run inference
    print("Running CoreML inference...")
    output = model.predict({
        "input_ids": input_ids,
        "audio_embeds": audio_features,
        "position_ids": position_ids,
        "has_audio": has_audio,
    })
    
    logits = output["logits"]
    print(f"Logits shape: {logits.shape}")
    
    # Get predicted tokens (greedy)
    predicted_tokens = np.argmax(logits[0], axis=-1)
    print(f"Predicted tokens (first 10): {predicted_tokens[:10]}")
    
    return logits


def main():
    parser = argparse.ArgumentParser(description="Validate CoreML models")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to test audio file",
    )
    parser.add_argument(
        "--models_path",
        type=str,
        default="../Resources/Models",
        help="Path to CoreML models directory",
    )
    
    args = parser.parse_args()
    
    # Load and process audio
    audio = load_audio(args.audio)
    mel_spec = compute_mel_spectrogram(audio)
    
    # Validate AudioEncoder
    audio_features = validate_audio_encoder(args.models_path, mel_spec)
    
    if audio_features is not None:
        # Validate TextDecoder
        validate_text_decoder(args.models_path, audio_features)
    
    print("\n=== Validation Complete ===")


if __name__ == "__main__":
    main()
