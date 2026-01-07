#!/usr/bin/env python3
"""
Numpy-only CoreML Inference Demo for VibeVoice Streaming Model.

This script demonstrates the pure numpy inference pipeline for VibeVoice,
eliminating PyTorch dependency while leveraging CoreML models for neural
network computations on Apple Silicon.

Key Benefits:
- No PyTorch dependency for inference
- Pure numpy operations throughout
- Leverages CoreML for Neural Engine acceleration
- Compatible with ONNX Runtime for cross-platform deployment

Usage:
    python demo/numpy_realtime_inference.py \
        --model_path microsoft/VibeVoice-Realtime-0.5B \
        --txt_path demo/text_examples/1p_vibevoice.txt \
        --speaker_name Wayne \
        --output_dir ./outputs
"""

import argparse
import os
import sys
import time
import copy
from typing import List, Tuple, Union, Dict, Any, Optional

import numpy as np

# Import CoreML tools
import coremltools as ct

# Import VibeVoice components
# from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from vibevoice.modular.configuration_vibevoice_streaming import VibeVoiceStreamingConfig

# Import the numpy-only inference module
from coreml.wrapped_inference import (
    WrappedInferenceModel,
    NumPyStreamingGenerator,
    NumPyDPMSolverMultistepScheduler,
    NumPyGenerationOutput,
    torch_to_numpy,
)


class VoiceMapper:
    """Maps speaker names to voice file paths."""
    
    def __init__(self):
        self.setup_voice_presets()

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices/streaming_model")
        
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        self.voice_presets = {}
        pt_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.pt') and os.path.isfile(os.path.join(voices_dir, f))]
        
        for pt_file in pt_files:
            name = os.path.splitext(pt_file)[0]
            full_path = os.path.join(voices_dir, pt_file)
            self.voice_presets[name] = full_path
        
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name."""
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path
        
        default_voice = list(self.voice_presets.values())[0]
        print(f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}")
        return default_voice


def parse_args():
    parser = argparse.ArgumentParser(description="Numpy-only VibeVoice Streaming Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-Realtime-0.5B",
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--txt_path",
        type=str,
        default="demo/text_examples/1p_vibevoice copy.txt",
        help="Path to the txt file containing the script",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="Wayne",
        help="Speaker name for voice cloning",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save output audio files",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="CFG scale for generation (default: 1.5)",
    )
    parser.add_argument(
        "--use_numpy",
        action="store_true",
        default=True,
        help="Use numpy-only generation (default: True)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def load_model_configuration(model_path: str) -> VibeVoiceStreamingConfig:
    """
    Load model configuration from the specified path.
    """
    config = VibeVoiceStreamingConfig.from_pretrained(model_path)
    return config


def extract_model_weights() -> Dict[str, np.ndarray]:
    """
    Extract model weights as numpy arrays for numpy-only inference.
    
    This extracts:
    - Embedding table
    - EOS classifier (2-layer MLP with fc1, fc2)
    - Speech connector (fc1, norm, fc2)
    """
    weights = {}
    
    # Embedding table
    # if hasattr(model, 'get_input_embeddings'):
    #     embed_tokens = model.get_input_embeddings().weight.detach().cpu().numpy()
    #     weights['embed_tokens'] = embed_tokens
    
    # # TTS EOS classifier (2-layer MLP)
    # if hasattr(model, 'tts_eos_classifier'):
    #     eos_classifier = model.tts_eos_classifier
    #     weights['tts_eos_classifier'] = {
    #         'fc1_weight': eos_classifier.fc1.weight.detach().cpu().numpy(),
    #         'fc1_bias': eos_classifier.fc1.bias.detach().cpu().numpy() if eos_classifier.fc1.bias is not None else None,
    #         'fc2_weight': eos_classifier.fc2.weight.detach().cpu().numpy(),
    #         'fc2_bias': eos_classifier.fc2.bias.detach().cpu().numpy() if eos_classifier.fc2.bias is not None else None,
    #     }
    
    # # Speech connector (fc1, norm, fc2)
    # if hasattr(model, 'model') and hasattr(model.model, 'acoustic_connector'):
    #     connector = model.model.acoustic_connector
    #     weights['speech_connector'] = {
    #         'fc1_weight': connector.fc1.weight.detach().cpu().numpy(),
    #         'fc1_bias': connector.fc1.bias.detach().cpu().numpy() if connector.fc1.bias is not None else None,
    #         'fc2_weight': connector.fc2.weight.detach().cpu().numpy(),
    #         'fc2_bias': connector.fc2.bias.detach().cpu().numpy() if connector.fc2.bias is not None else None,
    #         'norm_weight': connector.norm.weight.detach().cpu().numpy() if hasattr(connector.norm, 'weight') else None,
    #         'norm_bias': connector.norm.bias.detach().cpu().numpy() if hasattr(connector.norm, 'bias') else None,
    #     }
    
    # Speech scaling factors
    # if hasattr(model, 'speech_scaling_factor'):
    #     weights['speech_scaling_factor'] = float(model.speech_scaling_factor.item())
    # if hasattr(model, 'speech_bias_factor'):
    #     weights['speech_bias_factor'] = float(model.speech_bias_factor.item())

    weights['speech_bias_factor'] = -0.0703125 # hardcode
    weights['speech_scaling_factor'] = 0.2333984375 # hardcode

    
    return weights


def create_numpy_generator(
    model_path: str,
    lm_mlmodel_path: str,
    tts_lm_mlmodel_path: str,
    diffusion_head_mlmodel_path: Optional[str] = None,
    acoustic_detokenizer_mlmodel_path: Optional[str] = None,
    speech_connector_mlmodel_path: Optional[str] = None,
    eos_classifier_mlmodel_path: Optional[str] = None,
    embed_tokens_path: Optional[str] = None,
    weights: Optional[Dict[str, np.ndarray]] = None,
    config=None,
) -> NumPyStreamingGenerator:
    """
    Create a numpy-only streaming generator.
    
    Args:
        model_path: Path to the HuggingFace model.
        lm_mlmodel_path: Path to the language model CoreML package.
        tts_lm_mlmodel_path: Path to the TTS LM CoreML package.
        diffusion_head_mlmodel_path: Optional path to diffusion head CoreML package.
        acoustic_detokenizer_mlmodel_path: Optional path to acoustic detokenizer CoreML package.
        speech_connector_mlmodel_path: Optional path to speech connector CoreML package.
        eos_classifier_mlmodel_path: Optional path to EOS classifier CoreML package.
        weights: Dictionary of numpy weights extracted from PyTorch model.
        config: Model configuration.
        
    Returns:
        NumPyStreamingGenerator ready for inference.
    """
    # Create the numpy generator
    numpy_gen = NumPyStreamingGenerator(
        config=config,
        lm_mlmodel_path=lm_mlmodel_path,
        tts_lm_mlmodel_path=tts_lm_mlmodel_path,
        diffusion_head_mlmodel_path=diffusion_head_mlmodel_path,
        acoustic_detokenizer_mlmodel_path=acoustic_detokenizer_mlmodel_path,
        speech_connector_mlmodel_path=speech_connector_mlmodel_path,
        eos_classifier_mlmodel_path=eos_classifier_mlmodel_path,
        embed_tokens_path=embed_tokens_path,
        speech_scaling_factor=weights.get('speech_scaling_factor', 1.0),
        speech_bias_factor=weights.get('speech_bias_factor', 0.0),
        acoustic_vae_dim=config.acoustic_vae_dim if config else 64,
        hidden_size=config.decoder_config.hidden_size if config else 896,
        ddpm_num_inference_steps=5,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    
    return numpy_gen


def prepare_inputs_numpy(
    processor: VibeVoiceStreamingProcessor,
    text: str,
    all_prefilled_outputs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare numpy inputs for inference.
    
    Args:
        processor: VibeVoiceStreamingProcessor.
        text: Input text to synthesize.
        all_prefilled_outputs: Cached prefilled outputs from voice prompt.
        
    Returns:
        Tuple of (tts_text_ids, tts_lm_input_ids).
    """
    # Process input
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Convert to numpy
    tts_text_ids = inputs['tts_text_ids'].cpu().numpy()
    tts_lm_input_ids = inputs['tts_lm_input_ids'].cpu().numpy()
    
    return tts_text_ids, tts_lm_input_ids


def save_audio_numpy(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 24000,
):
    """
    Save numpy audio array to WAV file.
    
    Args:
        audio: Audio waveform as numpy array.
        output_path: Path to save the WAV file.
        sample_rate: Sample rate of the audio.
    """
    import wave
    
    # Ensure audio is in the correct format
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save as WAV
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"Saved audio to: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("VibeVoice Numpy-Only CoreML Inference Demo")
    print("=" * 60)
    
    # Normalize device
    import torch
    device = "cpu"  # Use CPU for weight extraction
    
    # Validate txt file
    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        return
    
    # Read script
    with open(args.txt_path, 'r', encoding='utf-8') as f:
        scripts = f.read().strip()
    
    if not scripts:
        print("Error: No valid scripts found in the txt file")
        return
    
    full_script = scripts.replace("'", "'").replace('"', '"').replace('"', '"')
    
    print(f"\nLoading processor from {args.model_path}")
    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_path)
    
    # Load voice
    voice_mapper = VoiceMapper()
    voice_sample = voice_mapper.get_voice_path(args.speaker_name)
    all_prefilled_outputs = torch.load(voice_sample, map_location=device, weights_only=False)
    
    print(f"Using voice: {voice_sample}")
    
    # Load PyTorch model only for weight extraction
    print("\nLoading PyTorch model for weight extraction...")
    config = load_model_configuration(args.model_path)
    
    # Extract weights
    print("Extracting model weights as numpy arrays...")
    weights = extract_model_weights()

    # Create numpy generator
    print("\nCreating numpy-only generator...")
    numpy_gen = NumPyStreamingGenerator(
        config=config,
        # model_path=args.model_path,
        lm_mlmodel_path="vibe_voice_lm_model_seqlen_32.mlpackage",
        tts_lm_mlmodel_path="vibe_voice_tts_lm_model_seqlen_8.mlpackage",
        diffusion_head_mlmodel_path="diffusion_head_model.mlpackage",
        # acoustic_detokenizer_mlmodel_path="decoder_coreml_1_f16.mlpackage",
        acoustic_detokenizer_mlmodel_path="decoder_coreml_12_ne.mlpackage",
        speech_connector_mlmodel_path="acoustic_connector.mlpackage",
        eos_classifier_mlmodel_path="tts_eos_classifier.mlpackage",
        embed_tokens_path="vibevoice_embeddings.npy",
        tts_input_types_path="tts_input_types.npy",
        speech_scaling_factor=weights.get('speech_scaling_factor', 1.0),
        speech_bias_factor=weights.get('speech_bias_factor', 0.0),
        acoustic_vae_dim=config.acoustic_vae_dim if config else 64,
        hidden_size=config.decoder_config.hidden_size if config else 896,
        ddpm_num_inference_steps=5,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    
    # Prepare inputs
    print("\nPreparing inputs...")
    tts_text_ids, tts_lm_input_ids = prepare_inputs_numpy(
        processor, full_script, all_prefilled_outputs
    )
    print(f"Text tokens shape: {tts_text_ids.shape}")
    print(f"TTS LM input shape: {tts_lm_input_ids.shape}")
    
    # Generate
    print(f"\nStarting numpy generation with cfg_scale={args.cfg_scale}...")
    start_time = time.time()
    
    outputs = numpy_gen.generate(
        tts_text_ids=tts_text_ids,
        tts_lm_input_ids=tts_lm_input_ids,
        all_prefilled_outputs=all_prefilled_outputs,
        cfg_scale=args.cfg_scale,
        max_length=config.decoder_config.max_position_embeddings,
        return_speech=True,
        verbose=args.verbose,
    )
    
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")
    
    # Process output
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        sample_rate = 24000
        audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        
        print(f"\nGenerated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
        
        # Save output
        os.makedirs(args.output_dir, exist_ok=True)
        txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
        output_path = os.path.join(args.output_dir, f"{txt_filename}_numpy_generated.wav")
        save_audio_numpy(outputs.speech_outputs[0], output_path, sample_rate)
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY (Numpy-Only)")
    print("=" * 60)
    print(f"Input file: {args.txt_path}")
    print(f"Output file: {output_path}")
    print(f"Speaker: {args.speaker_name}")
    print(f"Generation time: {generation_time:.2f} seconds")
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        print(f"Audio duration: {audio_duration:.2f} seconds")
        print(f"RTF: {rtf:.2f}x")
    print(f"Reach max step sample: {outputs.reach_max_step_sample}")
    print("=" * 60)
    
    print("\nNumpy-only generation completed successfully!")


if __name__ == "__main__":
    main()
