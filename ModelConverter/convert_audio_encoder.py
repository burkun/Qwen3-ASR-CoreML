#!/usr/bin/env python3
"""
Convert Qwen3-ASR Audio Encoder to CoreML format.

This script extracts the audio encoder from the Qwen3-ASR model and converts it
to CoreML format with ANE optimization.

Usage:
    python convert_audio_encoder.py --model_path <path_to_model> --output_path <output_path>
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Qwen3-ASR-Code"))

from transformers import AutoModel, AutoConfig


class AudioEncoderWrapper(nn.Module):
    """Wrapper for Qwen3-ASR Audio Encoder to make it traceable."""
    
    def __init__(self, audio_tower):
        super().__init__()
        self.audio_tower = audio_tower
    
    def forward(self, input_features: torch.Tensor, feature_lens: torch.Tensor):
        """
        Forward pass for audio encoding.
        
        Args:
            input_features: Mel spectrogram [batch, n_mels, seq_len]
            feature_lens: Sequence lengths [batch]
            
        Returns:
            Audio features [batch, output_seq_len, hidden_dim]
        """
        # The audio tower expects input_features and attention_mask
        outputs = self.audio_tower(
            input_features=input_features,
            feature_lens=feature_lens,
        )
        return outputs.last_hidden_state


def load_model(model_path: str):
    """Load Qwen3-ASR model and extract audio encoder."""
    print(f"Loading model from {model_path}...")
    
    # Load the full model
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    
    # Extract audio tower
    if hasattr(model, 'thinker'):
        audio_tower = model.thinker.audio_tower
    elif hasattr(model, 'audio_tower'):
        audio_tower = model.audio_tower
    else:
        raise ValueError("Could not find audio_tower in model")
    
    return audio_tower


def trace_model(audio_tower, example_seq_len: int = 3000):
    """Trace the audio encoder for conversion."""
    print("Tracing audio encoder...")
    
    wrapper = AudioEncoderWrapper(audio_tower)
    wrapper.eval()
    
    # Create example inputs
    batch_size = 1
    n_mels = 128
    
    example_mel = torch.randn(batch_size, n_mels, example_seq_len)
    example_lens = torch.tensor([example_seq_len], dtype=torch.long)
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapper,
            (example_mel, example_lens),
            strict=False
        )
    
    return traced_model


def convert_to_coreml(
    traced_model,
    output_path: str,
    min_seq_len: int = 100,
    max_seq_len: int = 30000,
):
    """Convert traced model to CoreML format."""
    print("Converting to CoreML...")
    
    # Define input shapes with flexible sequence length
    inputs = [
        ct.TensorType(
            name="mel_spectrogram",
            shape=(1, 128, ct.RangeDim(min_seq_len, max_seq_len, default=3000)),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="feature_lens",
            shape=(1,),
            dtype=np.int32,
        ),
    ]
    
    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=[ct.TensorType(name="audio_features", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,  # Enable ANE
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    
    # Add metadata
    coreml_model.author = "Qwen3-ASR CoreML"
    coreml_model.short_description = "Qwen3-ASR Audio Encoder for speech recognition"
    coreml_model.version = "1.0"
    
    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(str(output_path))
    
    print(f"Model saved to {output_path}")
    return coreml_model


def validate_model(coreml_model, audio_tower, seq_len: int = 1000):
    """Validate CoreML model output matches PyTorch."""
    print("Validating model...")
    
    # Create test input
    test_mel = np.random.randn(1, 128, seq_len).astype(np.float32)
    test_lens = np.array([seq_len], dtype=np.int32)
    
    # PyTorch inference
    wrapper = AudioEncoderWrapper(audio_tower)
    wrapper.eval()
    with torch.no_grad():
        torch_output = wrapper(
            torch.from_numpy(test_mel),
            torch.from_numpy(test_lens).long()
        ).numpy()
    
    # CoreML inference
    coreml_output = coreml_model.predict({
        "mel_spectrogram": test_mel,
        "feature_lens": test_lens,
    })["audio_features"]
    
    # Compare outputs
    max_diff = np.max(np.abs(torch_output - coreml_output))
    mean_diff = np.mean(np.abs(torch_output - coreml_output))
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if max_diff < 0.01:
        print("Validation PASSED!")
        return True
    else:
        print("Validation FAILED - outputs differ significantly")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-ASR Audio Encoder to CoreML")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-ASR",
        help="Path to Qwen3-ASR model or HuggingFace model ID",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../Resources/Models/AudioEncoder.mlpackage",
        help="Output path for CoreML model",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=100,
        help="Minimum sequence length",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=30000,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate converted model",
    )
    
    args = parser.parse_args()
    
    # Load model
    audio_tower = load_model(args.model_path)
    
    # Trace model
    traced_model = trace_model(audio_tower)
    
    # Convert to CoreML
    coreml_model = convert_to_coreml(
        traced_model,
        args.output_path,
        args.min_seq_len,
        args.max_seq_len,
    )
    
    # Validate if requested
    if args.validate:
        validate_model(coreml_model, audio_tower)
    
    print("Done!")


if __name__ == "__main__":
    main()
