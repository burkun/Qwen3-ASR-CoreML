#!/usr/bin/env python3
"""
Export Qwen3-ASR Tokenizer files for Swift implementation.

This script exports the tokenizer vocabulary and merges files
in a format that can be loaded by the Swift Qwen2Tokenizer.

Usage:
    python export_tokenizer.py --model_path <path_to_model> --output_path <output_path>
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def export_tokenizer(model_path: str, output_path: str):
    """Export tokenizer files."""
    print(f"Loading tokenizer from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export vocabulary
    vocab_path = output_dir / "vocab.json"
    print(f"Exporting vocabulary to {vocab_path}...")
    
    vocab = tokenizer.get_vocab()
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Export merges (for BPE tokenizers)
    if hasattr(tokenizer, "bpe_ranks") or hasattr(tokenizer, "_tokenizer"):
        merges_path = output_dir / "merges.txt"
        print(f"Exporting merges to {merges_path}...")
        
        # Try to get merges from the tokenizer
        merges = []
        if hasattr(tokenizer, "_tokenizer"):
            # Fast tokenizer
            model = tokenizer._tokenizer.model
            if hasattr(model, "merges"):
                merges = model.merges
        
        if merges:
            with open(merges_path, "w", encoding="utf-8") as f:
                f.write("#version: 0.2\n")
                for merge in merges:
                    if isinstance(merge, tuple):
                        f.write(f"{merge[0]} {merge[1]}\n")
                    else:
                        f.write(f"{merge}\n")
            print(f"Exported {len(merges)} merges")
        else:
            print("Warning: Could not extract merges from tokenizer")
    
    # Export tokenizer config
    config_path = output_dir / "tokenizer_config.json"
    print(f"Exporting config to {config_path}...")
    
    config = {
        "vocab_size": len(vocab),
        "model_type": "qwen2",
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "special_tokens": {
            "audio_token": "<|audio_pad|>",
            "audio_token_id": 151646,
            "audio_bos_token": "<|audio_bos|>",
            "audio_bos_token_id": 151647,
            "audio_eos_token": "<|audio_eos|>",
            "audio_eos_token_id": 151648,
        }
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    # Print special token info
    print("\nSpecial tokens:")
    print(f"  BOS token ID: {tokenizer.bos_token_id}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    print(f"  PAD token ID: {tokenizer.pad_token_id}")
    
    print(f"\nTokenizer files exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR Tokenizer")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-ASR",
        help="Path to Qwen3-ASR model or HuggingFace model ID",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../Resources/Tokenizer",
        help="Output directory for tokenizer files",
    )
    
    args = parser.parse_args()
    export_tokenizer(args.model_path, args.output_path)


if __name__ == "__main__":
    main()
