import os
import shutil
from typing import Optional, Tuple, Union, List, Dict, Any, Callable
from dataclasses import dataclass

import coremltools as ct
import numpy as np

from tqdm import tqdm


TTS_TEXT_WINDOW_SIZE = 5
TTS_SPEECH_WINDOW_SIZE = 6

# ============================================================================
# NUMPY-BASED DPM SOLVER (No PyTorch dependency)
# ============================================================================

class NumPyDPMSolverMultistepScheduler:
    """
    Pure numpy implementation of the DPM-Solver++ scheduler.
    This eliminates PyTorch dependency for the diffusion sampling process.
    
    Based on the diffusers DPMSolverMultistepScheduler but converted to numpy.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        final_sigmas_type: str = "zero",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.final_sigmas_type = final_sigmas_type
        
        if trained_betas is not None:
            self.betas = trained_betas.astype(np.float32)
        elif beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=np.float32) ** 2
        elif beta_schedule == "cosine" or beta_schedule == "squaredcos_cap_v2":
            self.betas = self._betas_for_alpha_bar_cosine(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented")
        
        if rescale_betas_zero_snr:
            self.betas = self._rescale_zero_terminal_snr(self.betas)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, dtype=np.float32)
        
        # Handle very small values at the end
        if rescale_betas_zero_snr:
            self.alphas_cumprod[-1] = 2 ** -24
        
        self.alpha_t = np.sqrt(self.alphas_cumprod)
        self.sigma_t = np.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = np.log(self.alpha_t) - np.log(self.sigma_t)
        self.sigmas = np.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)
        
        self.timesteps = None
        self.sigmas_schedule = None
        self.num_inference_steps = None
        self.step_index = None
        self.model_outputs = None
        
    def _betas_for_alpha_bar_cosine(self, num_diffusion_timesteps: int, max_beta: float = 0.999) -> np.ndarray:
        """Create cosine beta schedule."""
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            alpha_bar_fn = np.cos((t1 + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar_fn_next = np.cos((t2 + 0.008) / 1.008 * np.pi / 2) ** 2
            beta = min(1 - alpha_bar_fn_next / alpha_bar_fn, max_beta)
            betas.append(beta)
        return np.array(betas, dtype=np.float32)
    
    def _rescale_zero_terminal_snr(self, betas: np.ndarray) -> np.ndarray:
        """Rescale betas to have zero terminal SNR."""
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, dtype=np.float32)
        alphas_bar_sqrt = np.sqrt(alphas_cumprod)
        
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0]
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1]
        
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
        
        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = np.concatenate([alphas_bar[:1], alphas])
        betas = 1 - alphas
        return betas.astype(np.float32)
    
    def set_timesteps(self, num_inference_steps: int, timesteps: Optional[List[int]] = None):
        """Set the discrete timesteps for diffusion chain."""
        if timesteps is not None:
            timesteps = np.array(timesteps, dtype=np.int64)
        else:
            if self.timestep_spacing == "linspace":
                timesteps = np.linspace(
                    0, self.num_train_timesteps - 1, num_inference_steps + 1
                )[::-1][:-1].round().astype(np.int64)
            elif self.timestep_spacing == "leading":
                step_ratio = self.num_train_timesteps // (num_inference_steps + 1)
                timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].astype(np.int64)
                timesteps += self.steps_offset
            elif self.timestep_spacing == "trailing":
                step_ratio = self.num_train_timesteps / num_inference_steps
                timesteps = np.arange(self.num_train_timesteps, 0, -step_ratio).round().astype(np.int64) - 1
            else:
                raise ValueError(f"Unknown timestep_spacing: {self.timestep_spacing}")
        
        sigmas = np.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)
        sigmas = np.interp(timesteps, np.arange(len(sigmas)), sigmas)
        
        if self.final_sigmas_type == "zero":
            sigma_last = 0.0
        elif self.final_sigmas_type == "sigma_min":
            sigma_last = np.sqrt((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0])
        else:
            raise ValueError(f"Unknown final_sigmas_type: {self.final_sigmas_type}")
        
        self.sigmas_schedule = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        self.timesteps = timesteps
        self.num_inference_steps = len(timesteps)
        
        self.model_outputs = [None] * self.solver_order
        self.step_index = 0
        
    def _sigma_to_alpha_sigma_t(self, sigma: float) -> Tuple[float, float]:
        """Convert sigma to alpha and sigma_t."""
        alpha_t = 1.0 / np.sqrt(sigma ** 2 + 1)
        sigma_t = sigma * alpha_t
        return alpha_t, sigma_t
    
    def _convert_model_output(self, model_output: np.ndarray, sample: np.ndarray) -> np.ndarray:
        """Convert model output based on prediction type."""
        if self.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.prediction_type == "epsilon":
                sigma = self.sigmas_schedule[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == "sample":
                x0_pred = model_output
            elif self.prediction_type == "v_prediction":
                sigma = self.sigmas_schedule[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
            return x0_pred
        else:
            return model_output
    
    def _dpm_solver_first_order_update(
        self,
        model_output: np.ndarray,
        sample: np.ndarray,
        noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """First-order DPM-Solver update."""
        sigma_t = self.sigmas_schedule[self.step_index + 1]
        sigma_s = self.sigmas_schedule[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        
        lambda_t = np.log(alpha_t) - np.log(sigma_t)
        lambda_s = np.log(alpha_s) - np.log(sigma_s)
        h = lambda_t - lambda_s
        
        if self.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (np.exp(-h) - 1.0)) * model_output
        elif self.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (np.exp(h) - 1.0)) * model_output
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * np.exp(-h)) * sample
                + (alpha_t * (1 - np.exp(-2.0 * h))) * model_output
                + sigma_t * np.sqrt(1.0 - np.exp(-2 * h)) * noise
            )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (np.exp(h) - 1.0)) * model_output
                + sigma_t * np.sqrt(np.exp(2 * h) - 1.0) * noise
            )
        return x_t
    
    def _multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[np.ndarray],
        sample: np.ndarray,
        noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Second-order multistep DPM-Solver update."""
        sigma_t = self.sigmas_schedule[self.step_index + 1]
        sigma_s0 = self.sigmas_schedule[self.step_index]
        sigma_s1 = self.sigmas_schedule[self.step_index - 1]
        
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        
        lambda_t = np.log(alpha_t) - np.log(sigma_t)
        lambda_s0 = np.log(alpha_s0) - np.log(sigma_s0)
        lambda_s1 = np.log(alpha_s1) - np.log(sigma_s1)
        
        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (np.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (np.exp(-h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (np.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((np.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (np.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (np.exp(h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (np.exp(h) - 1.0)) * D0
                    - (sigma_t * ((np.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * np.exp(-h)) * sample
                    + (alpha_t * (1 - np.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - np.exp(-2.0 * h))) * D1
                    + sigma_t * np.sqrt(1.0 - np.exp(-2 * h)) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * np.exp(-h)) * sample
                    + (alpha_t * (1 - np.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - np.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * np.sqrt(1.0 - np.exp(-2 * h)) * noise
                )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (np.exp(h) - 1.0)) * D0
                    - (sigma_t * (np.exp(h) - 1.0)) * D1
                    + sigma_t * np.sqrt(np.exp(2 * h) - 1.0) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (np.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((np.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * np.sqrt(np.exp(2 * h) - 1.0) * noise
                )
        return x_t
    
    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        generator: Optional[np.random.Generator] = None,
        return_dict: bool = True,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict the sample from the previous timestep by reversing the SDE.
        
        Args:
            model_output: The direct output from learned diffusion model.
            timestep: The current discrete timestep.
            sample: A current instance of a sample created by diffusion process.
            generator: Random number generator for SDE variants.
            return_dict: Whether to return a dict or tuple.
            
        Returns:
            If return_dict is True, returns {'prev_sample': np.ndarray}.
            Otherwise returns the prev_sample directly.
        """
        if self.step_index is None or self.sigmas_schedule is None:
            raise ValueError("Must call set_timesteps before step")
        
        model_output = self._convert_model_output(model_output, sample)
        
        # Update model outputs history
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output
        
        # Determine if using lower order
        lower_order_final = (
            self.step_index == len(self.sigmas_schedule) - 2
            and len(self.sigmas_schedule) < 15
        )
        
        noise = None
        if self.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            shape = sample.shape
            if generator is not None:
                noise = generator.standard_normal(size=shape, dtype=np.float32)
            else:
                noise = np.random.randn(*shape).astype(np.float32)
        
        if self.solver_order == 1 or self.step_index < 1 or lower_order_final:
            prev_sample = self._dpm_solver_first_order_update(model_output, sample, noise)
        else:
            prev_sample = self._multistep_dpm_solver_second_order_update(self.model_outputs, sample, noise)
        
        self.step_index += 1
        
        if return_dict:
            return {"prev_sample": prev_sample}
        return prev_sample
    
    def add_noise(
        self,
        original_samples: np.ndarray,
        noise: np.ndarray,
        timesteps: np.ndarray
    ) -> np.ndarray:
        """Add noise to samples at specified timesteps."""
        alpha_t = self.alpha_t[timesteps].flatten()
        sigma_t = self.sigma_t[timesteps].flatten()
        
        # Broadcast to match sample shape
        while len(alpha_t.shape) < len(original_samples.shape):
            alpha_t = alpha_t[:, np.newaxis]
        while len(sigma_t.shape) < len(original_samples.shape):
            sigma_t = sigma_t[:, np.newaxis]
        
        return alpha_t * original_samples + sigma_t * noise


# ============================================================================
# NUMPY UTILITY FUNCTIONS
# ============================================================================

def torch_to_numpy(tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    if hasattr(tensor, 'cpu'):
        return tensor.detach().cpu().float().numpy()
    return np.array(tensor)

def numpy_to_torch(array: np.ndarray, device=None, dtype=None):
    """Convert numpy array to PyTorch tensor."""
    import torch
    result = torch.from_numpy(array)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if device is not None:
        result = result.to(device=device)
    return result

def squeeze_0(x):
    """Remove first dimension if size 1."""
    if x.shape[0] == 1:
        return x[0]
    return x


# ============================================================================
# CORE ML LOADING HELPERS
# ============================================================================

def try_loading_or_compile_mlmodel(mlmodel_path: str, compute_units=ct.ComputeUnit.CPU_AND_NE):
    maybe_mlmodelc_path = mlmodel_path.replace(".mlpackage", ".mlmodelc")
    if os.path.isdir(maybe_mlmodelc_path):
        return ct.models.CompiledMLModel(maybe_mlmodelc_path, compute_units=compute_units)
    else:
        mlmodel = ct.models.MLModel(mlmodel_path, compute_units=compute_units)
        mlmodelc_path = mlmodel.get_compiled_model_path()
        shutil.copytree(mlmodelc_path, maybe_mlmodelc_path)
        return mlmodel


# ============================================================================
# NUMPY-ONLY GENERATION CLASS
# ============================================================================

@dataclass
class NumPyGenerationOutput:
    """Output dataclass for numpy-only generation."""
    sequences: np.ndarray  # Token IDs
    speech_outputs: Optional[List[np.ndarray]] = None  # Audio chunks
    reach_max_step_sample: bool = False


class NumPyStreamingGenerator:
    """
    Pure numpy-based streaming speech generator.
    
    This class provides a complete numpy-only inference pipeline for VibeVoice,
    eliminating PyTorch dependency while still leveraging CoreML models.
    """
    
    TTS_TEXT_WINDOW_SIZE = 5
    TTS_SPEECH_WINDOW_SIZE = 6
    
    def __init__(
        self,
        config: Any,
        lm_mlmodel_path: str,
        tts_lm_mlmodel_path: str,
        diffusion_head_mlmodel_path: Optional[str] = None,
        acoustic_detokenizer_mlmodel_path: Optional[str] = None,
        speech_connector_mlmodel_path: Optional[str] = None,
        eos_classifier_mlmodel_path: Optional[str] = None,
        embed_tokens_path: Optional[str] = None,
        tts_input_types_path: Optional[str] = None,
        speech_scaling_factor: float = 1.0,
        speech_bias_factor: float = 0.0,
        acoustic_vae_dim: int = 64,
        hidden_size: int = 896,
        ddpm_num_inference_steps: int = 5,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_GPU,
    ):
        """
        Initialize the numpy-based streaming generator.
        
        Args:
            config: Model configuration object.
            lm_mlmodel_path: Path to the language model CoreML package.
            tts_lm_mlmodel_path: Path to the TTS language model CoreML package.
            diffusion_head_mlmodel_path: Optional path to diffusion head CoreML package.
            acoustic_detokenizer_mlmodel_path: Optional path to acoustic detokenizer CoreML package.
            speech_connector_mlmodel_path: Optional path to speech connector CoreML package.
            eos_classifier_mlmodel_path: Optional path to EOS classifier CoreML package.
            speech_scaling_factor: Scaling factor for speech latent decoding.
            speech_bias_factor: Bias factor for speech latent decoding.
            acoustic_vae_dim: Dimension of acoustic VAE latent space (default: 64).
            hidden_size: Hidden dimension of the model (default: 896).
            ddpm_num_inference_steps: Number of diffusion steps.
            compute_units: CoreML compute units.
        """
        self.config = config
        self.speech_scaling_factor = speech_scaling_factor
        self.speech_bias_factor = speech_bias_factor
        self.acoustic_vae_dim = acoustic_vae_dim
        self.hidden_size = hidden_size
        self.ddpm_num_inference_steps = ddpm_num_inference_steps
        
        # Load CoreML models
        self.lm_mlmodel = try_loading_or_compile_mlmodel(lm_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.tts_lm_mlmodel = try_loading_or_compile_mlmodel(tts_lm_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        
        # Diffusion head (optional)
        self.diffusion_head_mlmodel = None
        if diffusion_head_mlmodel_path is not None:
            self.diffusion_head_mlmodel = try_loading_or_compile_mlmodel(
                diffusion_head_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE
            )
        
        # Acoustic detokenizer (optional)
        self.acoustic_detokenizer = None
        if acoustic_detokenizer_mlmodel_path is not None:
            self.acoustic_detokenizer = NumPyAcousticDetokenizerCoreMLModel(acoustic_detokenizer_mlmodel_path)
        
        # Speech connector (optional - CoreML or numpy)
        self.speech_connector_mlmodel = None
        if speech_connector_mlmodel_path is not None:
            self.speech_connector_mlmodel = try_loading_or_compile_mlmodel(
                speech_connector_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE
            )
        self.speech_connector_weights = None
        
        # EOS classifier (optional - CoreML or numpy)
        self.eos_classifier_mlmodel = None
        if eos_classifier_mlmodel_path is not None:
            self.eos_classifier_mlmodel = try_loading_or_compile_mlmodel(
                eos_classifier_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE
            )
        
        # Initialize numpy DPM scheduler
        self.noise_scheduler = NumPyDPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type,
        )
        self.noise_scheduler.set_timesteps(ddpm_num_inference_steps)
        
        # CoreML states for KV cache
        self.lm_state = None
        self.tts_lm_state = None
        self.tts_lm_state_negative = None
        
        # Embedding table (will be loaded from config/model)
        self.embed_tokens = None
        
        # TTS EOS classifier weights (will be loaded from model)
        # The classifier is a 2-layer MLP: hidden -> hidden (ReLU) -> 1
        self.tts_eos_classifier_weights = None  # Can be dict with fc1_weight, fc2_weight, fc1_bias, fc2_bias
        self.tts_eos_classifier_bias = None

        self.set_embed_tokens(np.load(embed_tokens_path))
        self.tts_input_types = np.load(tts_input_types_path)
        
    def set_embed_tokens(self, embed_tokens: np.ndarray):
        """Set the token embedding table."""
        self.embed_tokens = embed_tokens.astype(np.float16)
    
    def set_tts_eos_classifier(self, weights: Dict[str, np.ndarray]):
        """
        Set the TTS EOS classifier weights.
        
        The classifier is a 2-layer MLP: hidden -> hidden (ReLU) -> 1
        
        Args:
            weights: Dictionary with keys 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias'
        """
        self.tts_eos_classifier_weights = weights
    
    def set_speech_connector(self, weights: Dict[str, np.ndarray]):
        """
        Set the speech connector weights.
        
        SpeechConnector:
            x = fc1(features)
            x = norm(x)
            x = fc2(x)
        
        Args:
            weights: Dictionary with keys 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias',
                     'norm_weight', 'norm_bias'
        """
        self.speech_connector_weights = weights
    
    def _apply_speech_connector(self, features: np.ndarray) -> np.ndarray:
        """Apply speech connector to features.
        
        SpeechConnector:
            x = fc1(features)
            x = norm(x)
            x = fc2(x)
        
        Args:
            features: Input features in channels-first format (C,) or (1, C, S).
            
        Returns:
            Output in same format as input.
        """
        # Use CoreML model if available
        if self.speech_connector_mlmodel is not None:
            # Ensure channels-first format: (1, channels, sequence_length)
            squeeze_output = False
            if features.ndim == 1:
                features = features[np.newaxis, :, np.newaxis]  # (C,) -> (1, C, 1)
                squeeze_output = True
            elif features.ndim == 2:
                features = features[..., np.newaxis]  # (C, S) -> (1, C, S)
            
            output = self.speech_connector_mlmodel.predict({
                'input_x': features,
            })
            result = output['output']
            
            if squeeze_output:
                result = result[0, :, 0]  # (1, C, 1) -> (C,)
            
            return result
        
        # Fallback to numpy-based computation using weights
        elif self.speech_connector_weights is not None:
            # Save original shape for restoration
            original_shape = features.shape
            squeeze_output = False
            
            # Ensure 2D input: (batch, channels)
            if features.ndim == 1:
                features = features[np.newaxis, :]  # (C,) -> (1, C)
                squeeze_output = True
            elif features.ndim == 3:
                features = features.reshape(-1, features.shape[1])  # (1, C, S) -> (S, C)
                squeeze_output = True
            
            # fc1: acoustic_vae_dim -> hidden_size
            fc1_weight = self.speech_connector_weights['fc1_weight']
            fc1_bias = self.speech_connector_weights['fc1_bias']
            x = np.dot(features, fc1_weight.T)
            if fc1_bias is not None:
                x += fc1_bias
            
            # Layer normalization
            norm_weight = self.speech_connector_weights['norm_weight']
            norm_bias = self.speech_connector_weights['norm_bias']
            if norm_weight is not None:
                # Layer norm: (x - mean) / sqrt(var + eps) * weight + bias
                eps = 1e-5
                mean = np.mean(x, axis=-1, keepdims=True)
                var = np.var(x, axis=-1, keepdims=True)
                x = (x - mean) / np.sqrt(var + eps)
                x = x * norm_weight + norm_bias
            
            # fc2: hidden_size -> hidden_size
            fc2_weight = self.speech_connector_weights['fc2_weight']
            fc2_bias = self.speech_connector_weights['fc2_bias']
            x = np.dot(x, fc2_weight.T)
            if fc2_bias is not None:
                x += fc2_bias
            
            # Restore original shape if needed
            if squeeze_output and len(original_shape) == 1:
                x = x[0, :]  # (1, H) -> (H,)
            
            return x
        
        else:
            raise ValueError("Either speech_connector_mlmodel or speech_connector_weights must be set")

    def _compute_eos_logits(self, hidden_states: np.ndarray) -> np.ndarray:
        """Compute EOS classification logits using 2-layer MLP.
        
        BinaryClassifier:
            x = ReLU(fc1(x))
            logits = fc2(x)
            
        Args:
            hidden_states: Input in channels-first format (C,) or (1, C, S).
            
        Returns:
            Logits tensor.
        """
        # Use CoreML model if available
        if self.eos_classifier_mlmodel is not None:
            # Ensure correct shape for CoreML model: (batch, hidden_dim, sequence_length)
            squeeze_output = False
            if hidden_states.ndim == 1:
                hidden_states = hidden_states[np.newaxis, :, np.newaxis]  # (C,) -> (1, C, 1)
                squeeze_output = True
            elif hidden_states.ndim == 2:
                # (B, H) -> (B, H, 1)
                hidden_states = hidden_states[:, :, np.newaxis]
            
            output = self.eos_classifier_mlmodel.predict({
                'input_x': hidden_states,
            })
            result = output['output']
            
            if squeeze_output:
                result = result[0, 0, 0]  # (1, 1, 1) -> scalar
            
            return result
        
        # Fallback to numpy-based computation using weights
        elif self.tts_eos_classifier_weights is not None:
            # Ensure 2D input: (batch, hidden_dim)
            if hidden_states.ndim == 1:
                hidden_states = hidden_states[np.newaxis, :]
            elif hidden_states.ndim == 3:
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[1])
            
            # fc1: hidden -> hidden with ReLU
            fc1_weight = self.tts_eos_classifier_weights['fc1_weight']
            fc1_bias = self.tts_eos_classifier_weights['fc1_bias']
            x = np.dot(hidden_states, fc1_weight.T)
            if fc1_bias is not None:
                x += fc1_bias
            x = np.maximum(x, 0)  # ReLU
            
            # fc2: hidden -> 1
            fc2_weight = self.tts_eos_classifier_weights['fc2_weight']
            fc2_bias = self.tts_eos_classifier_weights['fc2_bias']
            logits = np.dot(x, fc2_weight.T)
            if fc2_bias is not None:
                logits += fc2_bias
            
            # Return scalar if batch size is 1
            if logits.shape[0] == 1:
                return logits[0, 0]
            return logits[:, 0]
        
        else:
            raise ValueError("Either eos_classifier_mlmodel or tts_eos_classifier_weights must be set")

    def _get_embeddings(self, input_ids: np.ndarray) -> np.ndarray:
        """Get embeddings for input token IDs."""
        if self.embed_tokens is None:
            raise ValueError("embed_tokens not set. Call set_embed_tokens first.")
        return self.embed_tokens[input_ids]
    
    def _forward_lm_numpy(
        self,
        inputs_ids: np.ndarray,
        cache_position: np.ndarray,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Any]:
        """
        Forward pass through language model using CoreML.
        
        Args:
            inputs_embeds: Input embeddings of shape (batch, seq_len, hidden_dim).
            cache_position: Current cache position.
            past_key_values: KV cache state.
            
        Returns:
            Tuple of (last_hidden_state, new_kv_state).
        """
        input_length = inputs_ids.shape[1]
        coreml_inputs_embeds = self.embed_tokens[inputs_ids].transpose(0, 2, 1)[:, :, None, :]
        

        # LM CoreML model expects max 32 tokens of input
        LMMODEL_INPUT_LENGTH = 32
        
        # Pad to LMMODEL_INPUT_LENGTH tokens if needed
        if input_length < LMMODEL_INPUT_LENGTH:
            padding = np.zeros((1, coreml_inputs_embeds.shape[1], 1, LMMODEL_INPUT_LENGTH - input_length), dtype=np.float16)
            coreml_inputs_embeds = np.concatenate([coreml_inputs_embeds, padding], axis=-1)
        
        # Create state if needed
        if self.lm_state is None or not self.lm_state.is_initialized:
            self.lm_state = self.lm_mlmodel.make_state()
            self.lm_state.is_initialized = True
        
        # CoreML inference
        coreml_outputs = self.lm_mlmodel.predict({
            "inputs_embeds": coreml_inputs_embeds,
            "position_id": cache_position
        }, self.lm_state)
        
        # Extract output and reshape - output shape is (B, H, S)
        lm_output = coreml_outputs["output"][..., :input_length]  # (B, H, S)
        return lm_output
    
    def _forward_tts_lm_numpy(
        self,
        inputs_embeds: np.ndarray,
        cache_position: np.ndarray,
        past_key_values: Optional[Any] = None,
        tts_text_masks: Optional[np.ndarray] = None,
        use_negative_state: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Forward pass through TTS language model using CoreML.
        
        Args:
            inputs_embeds: Input embeddings.
            cache_position: Current cache position.
            past_key_values: KV cache state.
            tts_text_masks: Mask indicating text (1) vs speech (0) tokens.
            use_negative_state: If True, use separate negative TTS LM state.
            
        Returns:
            Tuple of (last_hidden_state, logits, new_kv_state).
        """
        input_length = inputs_embeds.shape[-1]
        coreml_inputs_embeds = inputs_embeds + self.tts_input_types[tts_text_masks].transpose(0, 2, 1)[:, :, None, :]
        
        MLMODEL_INPUT_LENGTH = 8
        # Pad to MLMODEL_INPUT_LENGTH tokens if needed
        if input_length < MLMODEL_INPUT_LENGTH:
            padding = np.zeros((1, inputs_embeds.shape[1], 1, MLMODEL_INPUT_LENGTH - input_length), dtype=np.float16)
            coreml_inputs_embeds = np.concatenate([coreml_inputs_embeds, padding], axis=-1)
        
        # Create or reuse state (separate states for positive and negative)
        if use_negative_state:
            if self.tts_lm_state_negative is None or not self.tts_lm_state_negative.is_initialized:
                self.tts_lm_state_negative = self.tts_lm_mlmodel.make_state()
                self.tts_lm_state_negative.is_initialized = True
            state = self.tts_lm_state_negative
        else:
            if self.tts_lm_state is None or not self.tts_lm_state.is_initialized:
                self.tts_lm_state = self.tts_lm_mlmodel.make_state()
                self.tts_lm_state.is_initialized = True
            state = self.tts_lm_state
        
        # CoreML inference (same model, different state)
        coreml_outputs = self.tts_lm_mlmodel.predict({
            "inputs_embeds": coreml_inputs_embeds,
            "position_id": cache_position,
        }, state)
        
        # Extract output and reshape - output shape is (B, H, 1, S)
        lm_output = coreml_outputs["output"][..., :input_length]  # (B, H, S)
        
        # Compute EOS logits from last hidden state
        # last_hidden = lm_output[..., -1]
        # logits = self._compute_eos_logits(last_hidden)
        
        # return lm_output, logits
        return lm_output
    
    def _forward_tts_lm_negative_numpy(
        self,
        inputs_embeds: np.ndarray,
        cache_position: np.ndarray,
    ) -> Tuple[np.ndarray, Any]:
        """Forward pass through negative TTS LM for classifier-free guidance."""
        input_dtype = inputs_embeds.dtype
        input_length = inputs_embeds.shape[1]
        
        # TTS LM CoreML model expects max 8 tokens of input
        MLMODEL_INPUT_LENGTH = 8
        if input_length > MLMODEL_INPUT_LENGTH:
            inputs_embeds = inputs_embeds[:, -MLMODEL_INPUT_LENGTH:, :]
            input_length = MLMODEL_INPUT_LENGTH
        
        # Prepare input: (B, S, H) -> (B, H, 1, S)
        coreml_inputs_embeds = np.transpose(inputs_embeds, (0, 2, 1))[:, :, np.newaxis, :]
        
        if input_length < MLMODEL_INPUT_LENGTH:
            padding = np.zeros((1, inputs_embeds.shape[2], 1, MLMODEL_INPUT_LENGTH - input_length), dtype=np.float32)
            coreml_inputs_embeds = np.concatenate([coreml_inputs_embeds, padding], axis=3)
        
        if self.tts_lm_state_negative is None or not self.tts_lm_state_negative.is_initialized:
            self.tts_lm_state_negative = self.tts_lm_mlmodel.make_state()
            self.tts_lm_state_negative.is_initialized = True
        
        coreml_outputs = self.tts_lm_mlmodel.predict({
            "inputs_embeds": coreml_inputs_embeds,
            "position_id": cache_position[-1:].astype(np.int32),
        }, self.tts_lm_state_negative)
        
        lm_output = coreml_outputs["output"][:, :, 0, :input_length]
        lm_output = np.transpose(lm_output, (0, 2, 1))
        lm_output = lm_output.astype(input_dtype)
        
        return lm_output, self.tts_lm_state_negative
    
    def _sample_speech_tokens_numpy(
        self,
        positive_condition: np.ndarray,
        negative_condition: np.ndarray,
        cfg_scale: float = 3.0,
    ) -> np.ndarray:
        """
        Sample speech tokens using diffusion.
        
        Args:
            positive_condition: Positive conditioning from TTS LM.
            negative_condition: Negative conditioning for CFG.
            cfg_scale: Classifier-free guidance scale.
            
        Returns:
            Sampled speech latents.
        """
        # Concatenate conditions for CFG
        condition = np.concatenate([positive_condition, negative_condition], axis=0)
        # condition should have shape (2, hidden_size)
        batch_size = condition.shape[0]
        speech = np.random.randn(batch_size, self.acoustic_vae_dim).astype(np.float32)
        # speech = np.zeros((batch_size, self.acoustic_vae_dim), dtype=np.float32)

        self.noise_scheduler.set_timesteps(self.ddpm_num_inference_steps)
        # Get timesteps from scheduler
        timesteps = self.noise_scheduler.timesteps
        
        for t in timesteps:
            half = speech[: len(speech) // 2]
            combined = np.concatenate([half, half], axis=0)
            
            # Prepare timestep embedding
            t_scalar = float(t)
            t_embed = self._timestep_embedding(t_scalar, 256)
            
            # CoreML diffusion head inference
            # Reshape to match CoreML model's expected input shape
            batch_size = combined.shape[0]
            eps_coreml = self.diffusion_head_mlmodel.predict({
                "noisy_images": combined[..., None, None],
                "timesteps": t_embed.reshape(-1, 1, 1),
                "condition": condition,
            })
            eps = eps_coreml["predicted_noise"].squeeze()
            
            # Classifier-free guidance
            cond_eps = eps[: len(eps) // 2]
            uncond_eps = eps[len(eps) // 2:]
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            # Reconstruct full batch with proper shape
            eps = np.concatenate([half_eps, half_eps], axis=0)
            
            # Scheduler step
            result = self.noise_scheduler.step(eps, int(t), speech)
            speech = result["prev_sample"]
        
        return speech[: len(speech) // 2]
    
    def _timestep_embedding(self, timestep: float, dim: int, max_period: int = 10000) -> np.ndarray:
        """Create sinusoidal timestep embedding."""
        half = dim // 2
        freqs = np.exp(
            -np.log(max_period) * np.arange(half) / half
        ).astype(np.float32)
        args = timestep * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        return embedding
    
    def _decode_acoustic_to_audio(
        self,
        speech_latent: np.ndarray,
        sample_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Decode acoustic latent to audio waveform.
        
        Args:
            speech_latent: Acoustic latent tensor.
            sample_indices: Indices for streaming cache.
            
        Returns:
            Audio waveform as numpy array.
        """
        if self.acoustic_detokenizer is not None:
            return self.acoustic_detokenizer.decode(
                speech_latent,
                sample_indices=sample_indices,
            )
        else:
            # Fallback: return raw latent scaled
            scaled = speech_latent / self.speech_scaling_factor - self.speech_bias_factor
            return scaled
    
    def _load_past_key_values_into_coreml_state(
        self,
        past_key_values: Any,
        mlmodel: Any,
        state: Any,
    ):
        """
        Load PyTorch past_key_values into CoreML MLState.
        
        Args:
            past_key_values: PyTorch past_key_values tuple structure.
            mlmodel: The CoreML model.
            state: The CoreML MLState to write to.
        """
        if past_key_values is None:
            return
        
        # Extract key and value caches from past_key_values
        # past_key_values is a tuple of tuples: ((layer0_k, layer0_v), (layer1_k, layer1_v), ...)
        key_cache_list = []
        value_cache_list = []
        
        for layer_kv in past_key_values:
            key, value = layer_kv
            key_cache_list.append(key)
            value_cache_list.append(value)
        
        # Concatenate all layers
        key_cache = torch.concatenate(key_cache_list, dim=0)
        value_cache = torch.concatenate(value_cache_list, dim=0)
        
        # Convert to numpy
        key_cache_np = key_cache.cpu().float().numpy()
        value_cache_np = value_cache.cpu().float().numpy()
        
        # Get state buffer shapes and pad if necessary
        key_state_shape = state.read_state("key_cache").shape
        value_state_shape = state.read_state("value_cache").shape
        
        # Pad key_cache if needed
        if key_cache_np.shape[2] < key_state_shape[2]:
            pad_width = ((0, 0), (0, 0), (0, key_state_shape[2] - key_cache_np.shape[2]), (0, 0))
            key_cache_np = np.pad(key_cache_np, pad_width, mode='constant')
        
        # Pad value_cache if needed
        if value_cache_np.shape[2] < value_state_shape[2]:
            pad_width = ((0, 0), (0, 0), (0, value_state_shape[2] - value_cache_np.shape[2]), (0, 0))
            value_cache_np = np.pad(value_cache_np, pad_width, mode='constant')
        
        # Write to state
        state.write_state("key_cache", key_cache_np)
        state.write_state("value_cache", value_cache_np)
    
    def _prepare_prefilled_outputs(
        self,
        all_prefilled_outputs: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare prefilled outputs from cached prompt and load KV cache into CoreML states.
        
        This method:
        1. Extracts the last hidden states from prefilled outputs
        2. Loads the past_key_values into CoreML MLState for each model
        
        Returns:
            Tuple of (lm_output, tts_lm_output, negative_tts_lm_output, lm_state).
        """
        # Extract prefilled outputs - last hidden states
        # lm_output = torch_to_numpy(all_prefilled_outputs.get("lm", {}).get("last_hidden_state"))
        # tts_lm_output = torch_to_numpy(all_prefilled_outputs.get("tts_lm", {}).get("last_hidden_state"))
        tts_lm_negative_output = torch_to_numpy(all_prefilled_outputs.get("neg_tts_lm", {}).get("last_hidden_state"))
        tts_lm_negative_output = tts_lm_negative_output[0, :, :, None, None]
        
        # Load KV cache into CoreML states
        # 1. Load LM cache
        lm_past_key_values = all_prefilled_outputs.get("lm", {}).get("past_key_values")
        if lm_past_key_values is not None:
            lm_cache_position = lm_past_key_values._seen_tokens
            if self.lm_state is None or not self.lm_state.is_initialized:
                self.lm_state = self.lm_mlmodel.make_state()
                self.lm_state.is_initialized = True
            self._load_past_key_values_into_coreml_state(
                lm_past_key_values, self.lm_mlmodel, self.lm_state
            )
        else:
            lm_cache_position = 0
        
        # 2. Load TTS LM cache
        tts_lm_past_key_values = all_prefilled_outputs.get("tts_lm", {}).get("past_key_values")
        if tts_lm_past_key_values is not None:
            tts_lm_cache_position = tts_lm_past_key_values._seen_tokens
            if self.tts_lm_state is None or not self.tts_lm_state.is_initialized:
                self.tts_lm_state = self.tts_lm_mlmodel.make_state()
                self.tts_lm_state.is_initialized = True
            self._load_past_key_values_into_coreml_state(
                tts_lm_past_key_values, self.tts_lm_mlmodel, self.tts_lm_state
            )
        else:
            tts_lm_cache_position = 0
        
        # 3. Load negative TTS LM cache
        tts_lm_negative_past_key_values = all_prefilled_outputs.get("neg_tts_lm", {}).get("past_key_values")
        if tts_lm_negative_past_key_values is not None:
            tts_lm_negative_cache_position = tts_lm_negative_past_key_values._seen_tokens
            if self.tts_lm_state_negative is None or not self.tts_lm_state_negative.is_initialized:
                self.tts_lm_state_negative = self.tts_lm_mlmodel.make_state()
                self.tts_lm_state_negative.is_initialized = True
            self._load_past_key_values_into_coreml_state(
                tts_lm_negative_past_key_values, self.tts_lm_mlmodel, self.tts_lm_state_negative
            )
        else:
            tts_lm_negative_cache_position = 0

        return (
            # lm_output,
            # tts_lm_output,
            tts_lm_negative_output,
            np.array([lm_cache_position], dtype=np.int32),
            np.array([tts_lm_cache_position], dtype=np.int32),
            np.array([tts_lm_negative_cache_position], dtype=np.int32),
        )

    def generate(
        self,
        tts_text_ids: np.ndarray,
        tts_lm_input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        all_prefilled_outputs: Optional[Dict[str, Any]] = None,
        cfg_scale: float = 3.0,
        max_length: int = 2048,
        return_speech: bool = True,
        verbose: bool = False,
        audio_streamer: Optional[Any] = None,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        **kwargs,
    ) -> NumPyGenerationOutput:
        """
        Main generation function - pure numpy implementation.
        
        Args:
            tts_text_ids: Text token IDs to generate speech for.
            tts_lm_input_ids: TTS LM input token IDs.
            attention_mask: Attention mask for input.
            all_prefilled_outputs: Cached prefilled outputs from prompt.
            cfg_scale: Classifier-free guidance scale.
            max_length: Maximum generation length.
            return_speech: Whether to return audio.
            verbose: Whether to print progress.
            audio_streamer: Optional audio streamer for real-time output.
            stop_check_fn: Optional function to check for external stop signal.
            
        Returns:
            NumPyGenerationOutput with sequences and speech outputs.
        """
        batch_size = 1
        assert batch_size == 1, "Currently only supports batch size == 1"
        
        # Initialize audio chunks
        audio_chunks = [[] for _ in range(batch_size)]
        speech_latent_buffer = []
        acoustic_cache = []  # Simple list for cache
        
        # Prepare prefilled outputs
        if all_prefilled_outputs is not None:
            (
                # lm_output,
                # tts_lm_output,
                tts_lm_negative_output,
                lm_cache_position,
                tts_lm_cache_position,
                tts_lm_negative_cache_position,
            ) = self._prepare_prefilled_outputs(
                all_prefilled_outputs
            )
        else:
            lm_output = None
            tts_lm_output = None
            tts_lm_negative_output = None
        
        # Initialize sequence tracking
        tts_lm_seq = tts_lm_input_ids.copy()
        step = tts_lm_input_ids.shape[1]
        tts_text_window_index = 0
        finished = np.zeros(batch_size, dtype=np.bool_)
        reach_max_step_sample = np.zeros(batch_size, dtype=np.bool_)
        
        # Cache position tracking
        # cache_position = np.arange(step, dtype=np.int64)
        
        # Get text window sizes
        first_text_window_size = min(self.TTS_TEXT_WINDOW_SIZE, tts_text_ids.shape[1])
        
        # Set timesteps for diffusion
        self.noise_scheduler.set_timesteps(self.ddpm_num_inference_steps)
        
        if verbose:
            print(f"Starting numpy generation: step={step}, max_length={max_length}")
        
        step = tts_lm_input_ids.shape[1]
        total_generated_speech_tokens = 0
        total_prefilled_text_tokens = 0
        if kwargs.get("show_progress_bar", True):
            progress_bar = tqdm(
                total=max_length,
                desc=f"Prefilled {step} tokens, current step ({step} / {max_length})",
                initial=step,
                leave=False
            )
        else:
            progress_bar = None

        finished = False

        while not finished:
            if finished:
                break
            # Check for external stop signal
            if stop_check_fn is not None and stop_check_fn():
                if verbose:
                    print(f"Generation stopped externally at step {step + 1}")
                # End the audio streamer if it exists
                if audio_streamer is not None:
                    audio_streamer.end()
                break
            
            # # Check if audio_streamer has been ended (stopped externally)
            # if audio_streamer is not None and hasattr(audio_streamer, 'finished_flags'):
            #     if any(audio_streamer.finished_flags):
            #         if verbose:
            #             print(f"Audio generation stopped externally at step {step + 1}")
            #         break
            
            # if finished_tags.all():
            #     if hasattr(progress_bar, 'set_description'):
            #         progress_bar.set_description("Generation complete")
            #     break

            cur_input_tts_text_ids = tts_text_ids[:, tts_text_window_index*TTS_TEXT_WINDOW_SIZE:(tts_text_window_index+1)*TTS_TEXT_WINDOW_SIZE]
            next_text_window_size = tts_text_ids[:, (tts_text_window_index+1)*TTS_TEXT_WINDOW_SIZE:(tts_text_window_index+2)*TTS_TEXT_WINDOW_SIZE].shape[1]
            tts_text_window_index += 1

            if cur_input_tts_text_ids.shape[1] > 0:
                input_ids = cur_input_tts_text_ids
                tts_lm_input_ids = cur_input_tts_text_ids

                if tts_lm_input_ids.shape[1] > max_length:
                    if verbose:
                        print(f"Reached maximum generation length {max_length}, stopped it.")
                    # reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                    # if reached_samples.numel() > 0:
                    #     reach_max_step_sample[reached_samples] = True
                    finished = True
                    break
                
                step += cur_input_tts_text_ids.shape[1]
                total_prefilled_text_tokens += cur_input_tts_text_ids.shape[1]
                if progress_bar is not None:
                    progress_bar.update(cur_input_tts_text_ids.shape[1])
                    progress_bar.set_description(f"Prefilled {total_prefilled_text_tokens} text tokens, generated {total_generated_speech_tokens} speech tokens, current step ({step} / {max_length})")

                # Forward pass through the model
                outputs = self._forward_lm_numpy(
                    input_ids,
                    cache_position=lm_cache_position,
                )
                lm_cache_position += input_ids.shape[1]

                # Forward pass through the model
                tts_lm_outputs = self._forward_tts_lm_numpy(
                    inputs_embeds=outputs,
                    cache_position=tts_lm_cache_position,
                    tts_text_masks=np.ones_like(input_ids),
                    use_negative_state=False,
                )
                tts_lm_cache_position += input_ids.shape[1]

            for cur_speech_index in range(TTS_SPEECH_WINDOW_SIZE):
                positive_condition = tts_lm_outputs[..., -1:]
                negative_condition = tts_lm_negative_output
                
                speech_latent = self._sample_speech_tokens_numpy(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                ) # .unsqueeze(1)
                
                # Buffer speech latents
                speech_latent_buffer.append(speech_latent)
                
                SPEECH_DECODING_BATCH_SIZE = 12

                # Decode if buffer is full
                if len(speech_latent_buffer) >= SPEECH_DECODING_BATCH_SIZE:
                    # Concatenate latents
                    # speech_latent is (1, dim), so stack/cat to (Batch, dim)
                    batch_latents = np.stack(speech_latent_buffer, axis=-1) # (B, dim)
                    
                    # Decode acoustic latent to audio using acoustic streaming cache
                    scaled_latents = batch_latents / self.speech_scaling_factor - self.speech_bias_factor
                    audio_chunk = self.acoustic_detokenizer.decode(
                        scaled_latents,
                    )
                    # audio_chunk might be (B, Samples) or (B*Samples) depending on squeeze.
                    # Ensure it is flat for appending
                    if audio_chunk.ndim > 1:
                        audio_chunk = audio_chunk.flatten()
                        
                    audio_chunks[0].append(audio_chunk)

                    # Add streaming support here
                    if audio_streamer is not None:
                        # Stream the audio chunks immediately
                        audio_streamer.put(audio_chunk, 0)
                    
                    # Clear buffer
                    speech_latent_buffer = []

                acoustic_embed = self._apply_speech_connector(speech_latent)[..., None]

                if tts_lm_input_ids.shape[1] > max_length:
                    finished = True
                    break
                
                step += 1
                total_generated_speech_tokens += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_description(f"Prefilled {total_prefilled_text_tokens} text tokens, generated {total_generated_speech_tokens} speech tokens, current step ({step} / {max_length})")

                tts_lm_outputs = self._forward_tts_lm_numpy(
                    inputs_embeds=acoustic_embed,
                    cache_position=tts_lm_cache_position,
                    tts_text_masks=np.zeros((1, 1), dtype=np.int32),
                    use_negative_state=False,
                )
                tts_lm_cache_position += 1

                tts_lm_negative_output = self._forward_tts_lm_numpy(
                    inputs_embeds=acoustic_embed,
                    cache_position=tts_lm_negative_cache_position,
                    tts_text_masks=np.zeros((1, 1), dtype=np.int32),
                    use_negative_state=True,
                )
                tts_lm_negative_cache_position += 1

                tts_eos_logits = self._compute_eos_logits(tts_lm_outputs[..., -1])

                if tts_eos_logits.item() > 0:
                    # If EOS token is predicted, we can stop generation for this sample
                    # finished_tags[0] = True
                    if audio_streamer is not None:
                        audio_streamer.end(0)
                    print("FINISHED")
                    finished = True
                    break
            
        # Flush remaining tokens in buffer if finished or loop ends (though loop is per window, we must check buffer emptiness)
        # Actually, the outer loop continues. We only need to flush if 'finished' is True.
        # But wait, if we exit inner loop but not finished, buffer remains for next outer loop? 
        # Yes, that's fine.
        if len(speech_latent_buffer) > 0:
            batch_latents = np.stack(speech_latent_buffer, axis=-1)
            scaled_latents = batch_latents / self.speech_scaling_factor - self.speech_bias_factor
            audio_chunk = self.acoustic_detokenizer.decode(scaled_latents)
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.flatten()
            audio_chunks[0].append(audio_chunk)
            if audio_streamer is not None:
                audio_streamer.put(audio_chunk, 0)
            speech_latent_buffer = []

        print("FINISHED")

        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                # Concatenate all chunks along the time dimension (assumed to be the last dimension)
                concatenated_audio = np.concatenate(sample_chunks)
                final_audio_outputs.append(concatenated_audio)
            else:
                # If no audio was generated for this sample, append None
                final_audio_outputs.append(None)
        
        if reach_max_step_sample is not None and reach_max_step_sample.any():
            print(f"Reached maximum generation length {max_length}, stopped it.")

        return VibeVoiceGenerationOutput(
            sequences=tts_lm_input_ids,
            speech_outputs=final_audio_outputs if return_speech else None,
            reach_max_step_sample=reach_max_step_sample,
        )
        


import torch
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceGenerationOutput,
    VibeVoiceStreamingForConditionalGenerationInference,
    BaseModelOutputWithPast,
    VibeVoiceCausalLMOutputWithPast
)

# ============================================================================
# NUMPY ACOUSTIC DETOKENIZER
# ============================================================================

class NumPyAcousticDetokenizerCoreMLModel:
    """
    Numpy-based acoustic detokenizer using CoreML.
    """
    
    stages_caches = [
        (1, 64, 6),
        (1, 2048, 6 * 8 + 15),
        (1, 1024, 6 * 3 + 9),
        (1, 512, 6 * 3 + 9),
        (1, 256, 6 * 3 + 7),
        (1, 128, 6 * 3 + 3),
        (1, 64, 6 * 3 + 3),
        (1, 32, 6 * 3 + 6),
    ]
    
    def __init__(self, mlmodel_path: str):
        self.mlmodel = try_loading_or_compile_mlmodel(mlmodel_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        self.caches = [
            np.zeros(shape, dtype=np.float32) for shape in self.stages_caches
        ]
    
    def decode(
        self,
        latents: np.ndarray,
        cache: Optional[List[np.ndarray]] = None,
        sample_indices: Optional[np.ndarray] = None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> np.ndarray:
        """Decode acoustic latents to audio."""
        input_x = latents
        if latents.ndim == 2:
            # (B, H) -> (B, H, 1)
            input_x = latents[..., None]
        if latents.shape[-1] != 12:
            # pad
            pad_size = 12 - latents.shape[-1]
            input_x = np.pad(latents, ((0, 0), (0, 0), (0, pad_size)))
        
        output = self.mlmodel.predict({
            "input_x": input_x,
            "cache_0": self.caches[0],
            "cache_1": self.caches[1],
            "cache_2": self.caches[2],
            "cache_3": self.caches[3],
            "cache_4": self.caches[4],
            "cache_5": self.caches[5],
            "cache_6": self.caches[6],
            "cache_7": self.caches[7],
        })
        
        for i in range(len(self.stages_caches)):
            # since padding is probably added only on the latest decoding
            # step it shouldn't be necessary to slice the cache
            self.caches[i][:] = output[f"new_cache_{i}"]
        
        # if latents.shape[-1] != 12:
        #     output = output["output"][..., :3200 * latents.shape[0]]
        # else:
        #     output = output["output"]

        return output["output"].squeeze()


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

@dataclass
class CoreMLState:
    """State container for CoreML model states."""
    is_initialized: bool = False
    state: Optional[Any] = None


def write_state(new_state: np.ndarray, state, key: str):
    """Write state to CoreML state container."""
    new_state = new_state.cpu().float().numpy()
    shape = state.read_state(key).shape
    padded = np.pad(new_state, ((0, 0), (0, 0), (0, shape[2] - new_state.shape[2]), (0, 0)))
    state.write_state(key, padded)


def load_past_key_values_into_state(past_key_values, state):
    """Load PyTorch past_key_values into CoreML state."""
    # write_state(np.concatenate(past_key_values.value_cache), state, "value_cache")
    write_state(torch.concatenate(past_key_values.value_cache), state, "value_cache")
    # write_state(np.concatenate(past_key_values.key_cache), state, "key_cache")
    write_state(torch.concatenate(past_key_values.key_cache), state, "key_cache")


# ============================================================================
# ORIGINAL WRAPPED INFERENCE CLASS (Enhanced with numpy methods)
# ============================================================================

class State:
    is_initialized: bool = False
    state = None

class AcousticDetokenizerCoreMLModel:
    stages_caches = [
        (1, 64, 6),
        (1, 2048, 6 * 8 + 15),
        (1, 1024, 6 * 3 + 9),
        (1, 512, 6 * 3 + 9),
        (1, 256, 6 * 3 + 7),
        (1, 128, 6 * 3 + 3),
        (1, 64, 6 * 3 + 3),
        (1, 32, 6 * 3 + 6),
    ]

    def __init__(self, mlmodel_path: str):
        # self.mlmodel = try_loading_or_compile_mlmodel(mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_NE)        
        self.mlmodel = try_loading_or_compile_mlmodel(mlmodel_path, compute_units=ct.ComputeUnit.CPU_ONLY)        
        self.caches = [
            np.zeros(shape, dtype=np.float32) for shape in self.stages_caches
        ]

    def decode(
        self,
        latents, cache=None, sample_indices=None, use_cache=False, debug=False,
    ):
        input_dtype = latents.dtype
        input_device = latents.device
        input_x = latents.permute(0, 2, 1).detach().cpu().float().numpy()

        output = self.mlmodel.predict({
            "input_x": input_x,
            "cache_0": self.caches[0],
            "cache_1": self.caches[1],
            "cache_2": self.caches[2],
            "cache_3": self.caches[3],
            "cache_4": self.caches[4],
            "cache_5": self.caches[5],
            "cache_6": self.caches[6],
            "cache_7": self.caches[7],
        })
        for i in range(len(self.stages_caches)):
            self.caches[i] = output[f"new_cache_{i}"]

        output = torch.from_numpy(output["output"]).to(device=input_device, dtype=input_dtype)
        return output
    


class WrappedInferenceModel:
    def __init__(
            self,
            model: VibeVoiceStreamingForConditionalGenerationInference,
            lm_mlmodel_path: str,
            tts_lm_mlmodel_path: str,
            diffusion_head_mlmodel_path: Optional[str] = None,
            acoustic_detokenizer_mlmodel_path: Optional[str] = None,
        ):
        self.model = model
        
        # self.lm_mlmodel = try_loading_or_compile_mlmodel(lm_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        # self.tts_lm_mlmodel = try_loading_or_compile_mlmodel(tts_lm_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

        # Load diffusion head CoreML model if provided
        self.diffusion_head_mlmodel = None
        # if diffusion_head_mlmodel_path is not None:
        #     self.diffusion_head_mlmodel = try_loading_or_compile_mlmodel(diffusion_head_mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
            
            # Test the diffusion head model
            # print("Testing diffusion head model...")
            # test_result = self.diffusion_head_mlmodel.predict({
            #     "noisy_images": np.random.rand(1, 64).astype(np.float32),
            #     "timesteps": np.array([500], dtype=np.int64),
            #     "condition": np.random.rand(1, 896).astype(np.float32),
            # })
            # print("Diffusion head test successful:", test_result["predicted_noise"].shape)

        self.states = {}

        self.lm_state = State()
        self.tts_lm_state = State()
        self.tts_lm_state_negative = State()

        # print(self.lm_mlmodel.predict({
        #     "inputs_embeds": np.random.rand(1, 896, 1, 32).astype(np.float32),
        #     "position_id": np.array([0], dtype=np.int32),
        # }, self.lm_mlmodel.make_state()))
        # print(self.tts_lm_mlmodel.predict({
        #     "inputs_embeds": np.random.rand(1, 896, 1, 8).astype(np.float32),
        #     "position_id": np.array([0], dtype=np.int32),
        # }, self.tts_lm_mlmodel.make_state()))

        # self.original_forward_lm = self.model.forward_lm
        # self.model.forward_lm = self.forward_lm
        # self.original_forward_tts_lm = self.model.forward_tts_lm
        # self.model.forward_tts_lm = self.forward_tts_lm
        
        # Patch sample_speech_tokens if diffusion head model is available
        # if self.diffusion_head_mlmodel is not None:
        #     self.original_sample_speech_tokens = self.model.sample_speech_tokens
        #     self.model.sample_speech_tokens = self.sample_speech_tokens

        if acoustic_detokenizer_mlmodel_path is not None:
            self.model.acoustic_tokenizer.decode = AcousticDetokenizerCoreMLModel(acoustic_detokenizer_mlmodel_path).decode

    def set_ddpm_inference_steps(self, num_inference_steps: int):
        return self.model.set_ddpm_inference_steps(num_inference_steps)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def get_coreml_state(self, past_key_values, mlmodel):
        if past_key_values not in self.states:
            new_state = mlmodel.make_state()
            load_past_key_values_into_state(past_key_values, new_state)
            self.states[past_key_values] = new_state
        return self.states[past_key_values]

    def forward_lm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        state = self.get_coreml_state(past_key_values, self.lm_mlmodel)
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        input_dtype = inputs_embeds.dtype
        input_device = inputs_embeds.device
        input_length = inputs_embeds.shape[1]
        coreml_inputs_embeds = torch.transpose(inputs_embeds, 1, 2).unsqueeze(-2)
        coreml_inputs_embeds = torch.nn.functional.pad(
            coreml_inputs_embeds,
            (0, 32 - input_length)
        ).cpu().float().numpy()

        # print(self.original_forward_lm(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     cache_position=cache_position,
        #     **kwargs,
        # ))

        print("ane", cache_position[:1].cpu().numpy().astype(np.int32))
        coreml_outputs = self.lm_mlmodel.predict({
            "inputs_embeds": coreml_inputs_embeds,
            "position_id": cache_position[:1].cpu().numpy().astype(np.int32),
            # "position_id": cache_position[:1].cpu().numpy().astype(np.int32) if position_ids is not None else np.array([0], dtype=np.int32),
            # "position_id": np.array([0], dtype=np.int32),
        }, state)
        lm_output = torch.from_numpy(coreml_outputs["output"][:, :, :, :input_length].squeeze(-2)).transpose(1, 2).to(device=input_device, dtype=input_dtype)
        print(lm_output)

        return BaseModelOutputWithPast(
            past_key_values=past_key_values,
            last_hidden_state=lm_output,
            attentions=None,
        )

    def forward_tts_lm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        lm_last_hidden_state: Optional[torch.FloatTensor] = None,
        tts_text_masks: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]:
        """
        Single pass of the TTS LM.

        - Overwrites tail embeddings with `lm_last_hidden_state`.
        - Adds type embedding via `tts_text_masks` (1=text, 0=speech).
        - Predicts EOS from last hidden state (binary classifier).
        - No loss / no full acoustic decoding here.

        Args:
            input_ids: (B, S) token ids.
            attention_mask: (B, S) mask.
            lm_last_hidden_state: (B, K, H) hidden states to splice into the tail.
            tts_text_masks: (B, 1) mask marking current position as text(1)/speech(0).
            past_key_values: cache from previous TTS steps.
            cache_position: positions for cached tokens.
            labels: unsupported (will raise).

        Returns:
            VibeVoiceCausalLMOutputWithPast with `logits` (EOS), `last_hidden_state`, `past_key_values`.
        """
        
        # Get embeddings
        if inputs_embeds is None:
            # Will be replaced with lm_last_hidden_state
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Replace the last part of inputs_embeds with lm_last_hidden_state
        start_idx = inputs_embeds.shape[1] - lm_last_hidden_state.shape[1]
        inputs_embeds[:, start_idx:, :] = lm_last_hidden_state
        
        # Adds type embedding via `tts_text_masks`.
        inputs_embeds = inputs_embeds + self.model.model.tts_input_types(tts_text_masks.long())

        state = self.get_coreml_state(past_key_values, self.tts_lm_mlmodel)

        MLMODEL_INPUT_LENGTH = 8
        input_dtype = inputs_embeds.dtype
        input_device = inputs_embeds.device
        input_length = inputs_embeds.shape[1]
        coreml_inputs_embeds = torch.transpose(inputs_embeds, 1, 2).unsqueeze(-2)
        coreml_inputs_embeds = torch.nn.functional.pad(
            coreml_inputs_embeds,
            (0, MLMODEL_INPUT_LENGTH - input_length)
        ).cpu().float().numpy()

        coreml_outputs = self.tts_lm_mlmodel.predict({
            "inputs_embeds": coreml_inputs_embeds,
            "position_id": cache_position[:1].cpu().numpy().astype(np.int32),
            # "position_id": cache_position[:1].cpu().numpy().astype(np.int32) if position_ids is not None else np.array([0], dtype=np.int32),
            # "position_id": np.array([0], dtype=np.int32),
        }, state)
        lm_output = torch.from_numpy(coreml_outputs["output"][:, :, :, :input_length].squeeze(-2)).transpose(1, 2).to(device=input_device, dtype=input_dtype)
        logits = self.model.tts_eos_classifier(lm_output[:, -1, :])
                

        return VibeVoiceCausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            last_hidden_state=lm_output,
            attentions=None,
        )

    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
        """
        CoreML-enhanced version of sample_speech_tokens.
        Uses CoreML for the prediction head and PyTorch for the scheduler.
        """
        self.model.noise_scheduler.set_timesteps(self.model.ddpm_inference_steps)
        
        # Use PyTorch for the condition concatenation and initial noise
        condition = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        speech = torch.randn(condition.shape[0], self.model.config.acoustic_vae_dim).to(condition)
        
        # Iterate through timesteps
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            
            # Use CoreML model for prediction head computation
            if self.diffusion_head_mlmodel is not None:
                # Prepare inputs for CoreML model
                noisy_images_np = combined.cpu().float().numpy()
                # timesteps = t.repeat(combined.shape[0])
                timesteps = self.model.prediction_head.t_embedder.timestep_embedding(
                    t.view(1).to(combined),
                    self.model.prediction_head.t_embedder.frequency_embedding_size,
                ).cpu().float().numpy()
                condition_np = condition.cpu().float().numpy()
                
                # CoreML inference
                eps_coreml = self.diffusion_head_mlmodel.predict({
                    "noisy_images": noisy_images_np[..., None, None],
                    "timesteps": timesteps.reshape(-1, 1, 1),
                    "condition": condition_np[..., None, None],
                })
                eps = torch.from_numpy(eps_coreml["predicted_noise"].squeeze()).to(combined.device, dtype=combined.dtype)
            else:
                # Fallback to PyTorch if CoreML model not available
                eps = self.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            
            # Continue with PyTorch scheduler (kept in fp32 for precision)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            
            # Use PyTorch scheduler (single step)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        
        return speech[: len(speech) // 2]
    
    def generate_numpy(self, *args, **kwargs) -> NumPyGenerationOutput:
        """
        Numpy-only generation method.
        
        This method provides a pure numpy inference pipeline that eliminates
        PyTorch dependency while leveraging CoreML models for the neural network
        computations.
        
        Args:
            Same as the regular generate() method but returns NumPyGenerationOutput.
            
        Returns:
            NumPyGenerationOutput with numpy arrays instead of torch tensors.
        """
        # Create numpy generator
        numpy_gen = NumPyStreamingGenerator(
            config=self.model.config,
            lm_mlmodel_path=self.lm_mlmodel._mlpackage_path if hasattr(self.lm_mlmodel, '_mlpackage_path') else None,
            tts_lm_mlmodel_path=self.tts_lm_mlmodel._mlpackage_path if hasattr(self.tts_lm_mlmodel, '_mlpackage_path') else None,
            diffusion_head_mlmodel_path=self.diffusion_head_mlmodel._mlpackage_path if self.diffusion_head_mlmodel and hasattr(self.diffusion_head_mlmodel, '_mlpackage_path') else None,
            speech_scaling_factor=float(self.model.speech_scaling_factor.item()) if hasattr(self.model.speech_scaling_factor, 'item') else float(self.model.speech_scaling_factor),
            speech_bias_factor=float(self.model.speech_bias_factor.item()) if hasattr(self.model.speech_bias_factor, 'item') else float(self.model.speech_bias_factor),
            acoustic_vae_dim=self.model.config.acoustic_vae_dim,
            hidden_size=self.model.config.decoder_config.hidden_size,
            ddpm_num_inference_steps=self.model.ddpm_inference_steps,
        )
        
        # Set embeddings from the model's embedding table
        embed_tokens = self.model.get_input_embeddings().weight.detach().cpu().numpy()
        numpy_gen.set_embed_tokens(embed_tokens)
        
        # Set EOS classifier weights (2-layer MLP)
        # BinaryClassifier: fc1: hidden -> hidden, fc2: hidden -> 1
        eos_classifier = self.model.tts_eos_classifier
        numpy_gen.set_tts_eos_classifier({
            'fc1_weight': eos_classifier.fc1.weight.detach().cpu().numpy(),
            'fc1_bias': eos_classifier.fc1.bias.detach().cpu().numpy() if eos_classifier.fc1.bias is not None else None,
            'fc2_weight': eos_classifier.fc2.weight.detach().cpu().numpy(),
            'fc2_bias': eos_classifier.fc2.bias.detach().cpu().numpy() if eos_classifier.fc2.bias is not None else None,
        })
        
        # Set speech connector weights
        # SpeechConnector: fc1: 64 -> hidden, norm, fc2: hidden -> hidden
        connector = self.model.model.acoustic_connector
        numpy_gen.set_speech_connector({
            'fc1_weight': connector.fc1.weight.detach().cpu().numpy(),
            'fc1_bias': connector.fc1.bias.detach().cpu().numpy() if connector.fc1.bias is not None else None,
            'fc2_weight': connector.fc2.weight.detach().cpu().numpy(),
            'fc2_bias': connector.fc2.bias.detach().cpu().numpy() if connector.fc2.bias is not None else None,
            'norm_weight': connector.norm.weight.detach().cpu().numpy() if hasattr(connector.norm, 'weight') else None,
            'norm_bias': connector.norm.bias.detach().cpu().numpy() if hasattr(connector.norm, 'bias') else None,
        })
        
        # Call the numpy generate method
        return numpy_gen.generate(*args, **kwargs)


__all__ = [
    "NumPyDPMSolverMultistepScheduler",
    "NumPyStreamingGenerator",
    "NumPyGenerationOutput",
    "NumPyAcousticDetokenizerCoreMLModel",
    "torch_to_numpy",
    "numpy_to_torch",
    "WrappedInferenceModel",
    "try_loading_or_compile_mlmodel",
]
