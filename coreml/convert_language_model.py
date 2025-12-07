import torch
import numpy as np
import coremltools as ct

from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from coreml.wrapped_inference import load_past_key_values_into_state
from coreml.layers import *

def main():
    model_path = "microsoft/VibeVoice-Realtime-0.5B"
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        # attn_implementation=attn_impl_primary,
    )

    target_model = model.model.language_model
    
    with torch.inference_mode():
        target_model.eval().float()
        patch_for_ane(target_model, AttentionANEWraperChannelsFirstWithCache)
        patch_for_ane(target_model, ANELinearWrapper)
        patch_for_ane(target_model, WrappedRMSNorm, )
        wrapped_base_llm = LMModelANEWrapperWithCache(target_model, device="cpu", channels_first=True)
        wrapped_base_llm.eval().float()

        
    print(target_model)

    example_inputs = (
        torch.randn((1, 896, 1, 1)).float(),
        torch.zeros((1,), dtype=torch.int32),
    )
    with torch.inference_mode():
        traced_model = torch.jit.trace(wrapped_base_llm, example_inputs=example_inputs)

    bsz = 32
    base_mlmodel = ct.convert(
        traced_model,
        # convert_to="milinternal",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        # skip_model_load=skip_model_load,
        inputs=[
            ct.TensorType(shape=(1, 896, 1, bsz), name="inputs_embeds"),
            ct.TensorType(shape=example_inputs[1].shape, name="position_id", dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="output"),
        ],
        states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=wrapped_base_llm.key_cache.shape,
                    ),
                    name="key_cache",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=wrapped_base_llm.value_cache.shape,
                    ),
                    name="value_cache",
                ),
            ],
    )
    base_mlmodel.save(f"vibe_voice_lm_model_seqlen_{bsz}.mlpackage")

def test(torch_model, coreml_model, coreml_model_length):
    with torch.inference_mode():
        torch_model = torch_model.eval().float()

        seqlen = 31
        inputs_embeds = torch.randn(1, seqlen, 896).to(torch.bfloat16)

        initial_output = torch_model(
            inputs_embeds=inputs_embeds,
            use_cache=True
        )

        new_inputs_embeds = torch.randn(1, seqlen, 896)
        outputs_with_cache = torch_model(
            inputs_embeds=new_inputs_embeds,
            cache_position=torch.arange(seqlen, seqlen * 2), # .view(1, seqlen),
            past_key_values=initial_output.past_key_values,
            use_cache=True,
        )

    coreml_input_embeds = torch.transpose(inputs_embeds, 1, 2).unsqueeze(-2)
    coreml_input_embeds = torch.nn.functional.pad(coreml_input_embeds, (0, coreml_model_length - seqlen)).float().numpy()

    state = coreml_model.make_state()
    # coreml_output = coreml_model.predict({
    #     "inputs_embeds": coreml_input_embeds,
    #     "position_id": np.array([0], dtype=np.int32),
    # }, state)

    load_past_key_values_into_state(initial_output.past_key_values, state)

    coreml_new_inputs_embeds = torch.transpose(new_inputs_embeds, 1, 2).unsqueeze(-2)
    coreml_new_inputs_embeds = torch.nn.functional.pad(coreml_new_inputs_embeds, (0, coreml_model_length - seqlen)).float().numpy()
    coreml_output_with_cache = coreml_model.predict({
        "inputs_embeds": coreml_new_inputs_embeds,
        "position_id": np.array([seqlen], dtype=np.int32),
    }, state)

    target_hidden_states = initial_output.last_hidden_state.transpose(1, 2).unsqueeze(-2)

    # ANE Float16 gives results that are more than a bit off so an allclose
    # would need a high tolerance, but a quick manual inspection we hope that
    # they are close enough and test-in-prod.


    # print("CoreML")
    # first_coreml_output = coreml_output["output"][:, :, :, :seqlen].flatten()
    # print(first_coreml_output)
    # print("Torch")
    # first_target_output = target_hidden_states[:, :, :, :seqlen].float().numpy().flatten()
    # print(first_target_output)
    # print("Difference")
    # difference = first_coreml_output - first_target_output

    # abs_diff = np.abs(difference)
    # rel_diff = np.abs(difference) / (np.abs(first_target_output) + 1e-7)

    # print("-" * 20)
    # abs_diff_indices = np.argsort(-abs_diff)[:18]
    # rel_diff_indices = np.argsort(-rel_diff)[:18]
    # print("Top 20 absolute differences:")
    # print(abs_diff[abs_diff_indices])
    # print("Corresponding CoreML outputs:")
    # print(first_coreml_output[abs_diff_indices])
    # print("Corresponding Torch outputs:")
    # print(first_target_output[abs_diff_indices])
    # print("Corresponding relative differences:")
    # print(rel_diff[abs_diff_indices])

    # print("-" * 20)
    # print("Top 20 relative differences:")
    # print(rel_diff[rel_diff_indices])
    # print("Corresponding CoreML outputs:")
    # print(first_coreml_output[rel_diff_indices])
    # print("Corresponding Torch outputs:")
    # print(first_target_output[rel_diff_indices])
    # print("Corresponding absolute differences:")
    # print(abs_diff[rel_diff_indices])

    # print("-" * 20)
    # are_close = np.bitwise_or(
    #     abs_diff < 5e-2,
    #     rel_diff < 5.5e-2,
    # )
    # are_not_close = ~are_close
    # print("Corresponding CoreML outputs for not close:")
    # print(first_coreml_output[are_not_close])
    # print("Corresponding Torch outputs for not close:")
    # print(first_target_output[are_not_close])
    # print("Corresponding absolute differences for not close:")
    # print(abs_diff[are_not_close])
    # print("Corresponding relative differences for not close:")
    # print(rel_diff[are_not_close])

    # assert np.all(are_close), "Outputs do not match for input without cache"

    second_target_output = outputs_with_cache.last_hidden_state.transpose(1, 2).unsqueeze(-2)

    print("-" * 40)
    print("CoreML")
    second_coreml_output = coreml_output_with_cache["output"][:, :, :, :seqlen].flatten()
    print(second_coreml_output)
    print("Torch")
    second_target_output = second_target_output[:, :, :, :seqlen].float().numpy().flatten()
    print(second_target_output)
    print("Difference")
    difference = second_coreml_output - second_target_output

    abs_diff = np.abs(difference)
    rel_diff = np.abs(difference) / (np.abs(second_target_output) + 1e-7)

    abs_diff_indices = np.argsort(-abs_diff)[:18]
    rel_diff_indices = np.argsort(-rel_diff)[:18]
    print("Top 20 absolute differences:")
    print(abs_diff[abs_diff_indices])
    print("Corresponding CoreML outputs:")
    print(second_coreml_output[abs_diff_indices])
    print("Corresponding Torch outputs:")
    print(second_target_output[abs_diff_indices])
    print("Corresponding relative differences:")
    print(rel_diff[abs_diff_indices])

    print("Top 20 relative differences:")
    print(rel_diff[rel_diff_indices])
    print("Corresponding CoreML outputs:")
    print(second_coreml_output[rel_diff_indices])
    print("Corresponding Torch outputs:")
    print(second_target_output[rel_diff_indices])
    print("Corresponding absolute differences:")
    print(abs_diff[rel_diff_indices])

    assert np.all(
        np.bitwise_or(
            abs_diff < 5e-2,
            rel_diff < 5.5e-2,
        )
    ), "Outputs do not match for input with cache"



if __name__ == "__main__":
    # main()

    torch_model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        "microsoft/VibeVoice-Realtime-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ).model.language_model
    coreml_model = ct.models.MLModel("vibe_voice_lm_model_seqlen_32.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
    test(torch_model, coreml_model, coreml_model_length=32)