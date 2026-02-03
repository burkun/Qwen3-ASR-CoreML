#!/usr/bin/env python3
"""
Convert Qwen3-ASR Text Decoder to CoreML format.

This script extracts the text decoder from the Qwen3-ASR model and converts it
to CoreML format with ANE optimization.

Usage:
    python convert_text_decoder.py --model_path <path_to_model> --output_path <output_path>
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


class TextDecoderWrapper(nn.Module):
    """Wrapper for Qwen3-ASR Text Decoder to make it traceable."""
    
    def __init__(self, text_model, lm_head, embed_tokens, audio_token_id: int = 151646):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.embed_tokens = embed_tokens
        self.audio_token_id = audio_token_id
    
    def forward(
        self,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        has_audio: torch.Tensor,
    ):
        """
        Forward pass for text decoding.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            audio_embeds: Audio embeddings [audio_len, hidden_dim]
            position_ids: Position IDs [3, batch, seq_len] for MRoPE
            has_audio: Boolean tensor indicating if audio is present
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Get text embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Fuse audio embeddings if present
        if has_audio.item() > 0:
            audio_mask = (input_ids == self.audio_token_id)
            # Expand audio_embeds to match batch dimension
            batch_size = input_ids.shape[0]
            audio_embeds_expanded = audio_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Replace audio token embeddings with audio features
            audio_positions = audio_mask.nonzero(as_tuple=True)
            if len(audio_positions[0]) > 0:
                inputs_embeds[audio_mask] = audio_embeds_expanded.reshape(-1, audio_embeds.shape[-1])[:audio_mask.sum()]
        
        # Forward through text model
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=False,
        )
        
        # Apply LM head
        logits = self.lm_head(outputs.last_hidden_state)
        
        return logits


def load_model(model_path: str):
    """Load Qwen3-ASR model and extract text decoder components."""
    print(f"Loading model from {model_path}...")
    
    # Load the full model
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    
    # Extract components
    if hasattr(model, 'thinker'):
        text_model = model.thinker.text_model
        lm_head = model.thinker.lm_head
        embed_tokens = model.thinker.text_model.embed_tokens
    else:
        text_model = model.text_model
        lm_head = model.lm_head
        embed_tokens = model.text_model.embed_tokens
    
    return text_model, lm_head, embed_tokens


def trace_model(text_model, lm_head, embed_tokens, example_seq_len: int = 100):
    """Trace the text decoder for conversion."""
    print("Tracing text decoder...")
    
    wrapper = TextDecoderWrapper(text_model, lm_head, embed_tokens)
    wrapper.eval()
    
    # Create example inputs
    batch_size = 1
    audio_len = 50
    hidden_dim = 3584
    
    example_input_ids = torch.randint(0, 1000, (batch_size, example_seq_len))
    example_audio_embeds = torch.randn(audio_len, hidden_dim)
    example_position_ids = torch.arange(example_seq_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
    example_has_audio = torch.tensor([1])
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapper,
            (example_input_ids, example_audio_embeds, example_position_ids, example_has_audio),
            strict=False
        )
    
    return traced_model


def convert_to_coreml(
    traced_model,
    output_path: str,
    min_seq_len: int = 1,
    max_seq_len: int = 2048,
    vocab_size: int = 151936,
):
    """Convert traced model to CoreML format."""
    print("Converting to CoreML...")
    
    hidden_dim = 3584
    
    # Define input shapes
    inputs = [
        ct.TensorType(
            name="input_ids",
            shape=(1, ct.RangeDim(min_seq_len, max_seq_len, default=100)),
            dtype=np.int32,
        ),
        ct.TensorType(
            name="audio_embeds",
            shape=(ct.RangeDim(1, 1000, default=50), hidden_dim),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="position_ids",
            shape=(3, 1, ct.RangeDim(min_seq_len, max_seq_len, default=100)),
            dtype=np.int32,
        ),
        ct.TensorType(
            name="has_audio",
            shape=(1,),
            dtype=np.int32,
        ),
    ]
    
    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,  # Enable ANE
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    
    # Add metadata
    coreml_model.author = "Qwen3-ASR CoreML"
    coreml_model.short_description = "Qwen3-ASR Text Decoder for speech recognition"
    coreml_model.version = "1.0"
    
    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(str(output_path))
    
    print(f"Model saved to {output_path}")
    return coreml_model


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-ASR Text Decoder to CoreML")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-ASR",
        help="Path to Qwen3-ASR model or HuggingFace model ID",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../Resources/Models/TextDecoder.mlpackage",
        help="Output path for CoreML model",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=1,
        help="Minimum sequence length",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    
    args = parser.parse_args()
    
    # Load model
    text_model, lm_head, embed_tokens = load_model(args.model_path)
    
    # Trace model
    traced_model = trace_model(text_model, lm_head, embed_tokens)
    
    # Convert to CoreML
    convert_to_coreml(
        traced_model,
        args.output_path,
        args.min_seq_len,
        args.max_seq_len,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
