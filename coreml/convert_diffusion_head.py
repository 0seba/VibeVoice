from httpx import patch
import torch
import numpy as np
import coremltools as ct

from vibevoice.modular.modular_vibevoice_diffusion_head import FinalLayer, HeadLayer, VibeVoiceDiffusionHead, modulate, RMSNorm
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from coreml.layers import *

class HeadLayerANEWrapper(nn.Module):
    target_classes = [HeadLayer]

    def __init__(self, layer: HeadLayer):
        super().__init__()
        self.layer = layer

    def forward(self, x, c):
        shift_ffn, scale_ffn, gate_ffn = self.layer.adaLN_modulation(c).chunk(3, dim=1)
        x = x + gate_ffn * self.layer.ffn(modulate(self.layer.norm(x), shift_ffn, scale_ffn))
        return x
    

class FinalLayerANEWrapper(nn.Module):
    target_classes = [FinalLayer]

    def __init__(self, layer: FinalLayer):
        super().__init__()
        self.layer = layer

    def forward(self, x, c):
        shift, scale = self.layer.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.layer.norm_final(x), shift, scale)
        x = self.layer.linear(x)
        return x
    
class VibeVoiceDiffusionHeadANEWrapper(nn.Module):
    target_classes = [VibeVoiceDiffusionHead]

    def __init__(self, diffusion_head: VibeVoiceDiffusionHead):
        super().__init__()
        self.layer = diffusion_head

    def forward(
        self,
        noisy_images,
        timesteps,
        condition,
    ):
        """
        Forward pass of the prediction head.
        
        Args:
            noisy_images (`torch.Tensor`): Noisy images/latents to denoise
            timesteps (`torch.Tensor`): Timesteps for diffusion
            condition (`torch.Tensor`): Conditioning information
            
        Returns:
            `torch.Tensor`: The predicted noise/velocity
        """
        x = self.layer.noisy_images_proj(noisy_images)
        t = self.layer.t_embedder.mlp(timesteps)
        # t = self.t_embedder(timesteps)
        condition = self.layer.cond_proj(condition)
        c = condition + t
        
        for layer in self.layer.layers:
            x = layer(x, c)
            
        x = self.layer.final_layer(x, c)
        return x

def convert_diffusion_head(target_model, output_path):
    """Convert the diffusion head to CoreML"""
    with torch.inference_mode():
        target_model.eval().float()
        
        patch_for_ane(target_model, ANELinearWrapper)
        patch_for_ane(target_model, WrappedRMSNorm, [RMSNorm])
        patch_for_ane(target_model, HeadLayerANEWrapper)
        patch_for_ane(target_model, FinalLayerANEWrapper)

        # Wrap the diffusion head for CoreML compatibility
        wrapped_diffusion_head = VibeVoiceDiffusionHeadANEWrapper(target_model)
        wrapped_diffusion_head.eval().float()

    example_inputs = (
        torch.randn(2, target_model.config.latent_size, 1, 1).float(),  # noisy_images
        torch.ones(256, 1, 1).float(),  # timesteps
        torch.randn(2, target_model.config.hidden_size, 1, 1).float(),  # condition
    )
    
    with torch.inference_mode():
        traced_model = torch.jit.trace(wrapped_diffusion_head, example_inputs=example_inputs)

    base_mlmodel = ct.convert(
        traced_model,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(shape=example_inputs[0].shape, name="noisy_images"),
            ct.TensorType(shape=example_inputs[1].shape, name="timesteps"),
            ct.TensorType(shape=example_inputs[2].shape, name="condition"),
        ],
        outputs=[
            ct.TensorType(name="predicted_noise"),
        ],
    )
    base_mlmodel.save(output_path)
    return base_mlmodel


def test_diffusion_head(torch_model, coreml_model):
    """Test diffusion head conversion"""
    with torch.inference_mode():
        torch_model = torch_model.eval().float()
        
        # Test with random inputs
        noisy_images = torch.randn(1, 64).float()
        timesteps = torch.tensor([500]).long()
        condition = torch.randn(1, 896).float()  # Assuming hidden_size = 896
        
        torch_output = torch_model(noisy_images, timesteps, condition)
        
        # CoreML inference
        coreml_output = coreml_model.predict({
            "noisy_images": noisy_images.numpy(),
            "timesteps": timesteps.numpy(),
            "condition": condition.numpy(),
        })
        
        # Compare outputs
        torch_np = torch_output.detach().numpy()
        coreml_np = coreml_output["predicted_noise"]
        
        print("Torch output shape:", torch_np.shape)
        print("CoreML output shape:", coreml_np.shape)
        print("Max difference:", np.max(np.abs(torch_np - coreml_np)))
        print("Mean difference:", np.mean(np.abs(torch_np - coreml_np)))
        
        return torch_np, coreml_np


if __name__ == "__main__":
    # Load the model
    model_path = "microsoft/VibeVoice-Realtime-0.5B"
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print(model.model.prediction_head)
    
    # Convert diffusion head
    print("Converting diffusion head...")
    diffusion_head_model = convert_diffusion_head(
        model.model.prediction_head, 
        "diffusion_head_model.mlpackage"
    )
    
    # # Test the conversion
    # print("\nTesting diffusion head...")
    # test_diffusion_head(model.model.prediction_head, diffusion_head_model)
    
    # print("\nConversion completed successfully!")