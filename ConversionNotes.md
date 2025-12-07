We'll begin studying the spech generation workflow in the `demo/realtime_model_inference_from_file.py` file.

Inside of the `main` function the first class instantiated is the `VoiceMapper` generates a list of preset voices based on the `.pt` files in the `voices` folder and then returns the path to a voice file when looking based on the voice name, no need to modify that yet.

Then an instance of `VibeVoiceStreamingProcessor` is created. This class loads the text tokenizer, that loads, resamples,  normalizes and may apply other preprocessing transformations to the audio. Won't modify this class yet.

When printing the outputs `all_prefilled_outputs` is the result of `torch.load` a voice preset we get:
- lm BaseModelOutputWithPast with last hidden states and kv cache
- tts_lm with the same
- neg_lm with the same
- neg_tts_lm with the same

when printing `inputs` which is the result of 
```python
inputs = processor.process_input_with_cached_prompt(
    text=full_script,
    cached_prompt=all_prefilled_outputs,
    padding=True,
    return_tensors="pt",
    return_attention_mask=True,
)
```
we get a dictionary with the following keys:
- input_ids full of 151655
- tts_lm_input_ids full of 151655
- tts_text_ids with ids
- attention_mask with 1s
- tts_lm_attention_mask with 1s
- tts_text_attention_mask with 1s
- speech_input_mask all false
- speech_tensors is none
- speech_masks is none

Before these prints a instance of the class `VibeVoiceStreamingForConditionalGenerationInference` is created.
After that model.generate is called. Let's dive into that now.

There is a lot of data preparation which I'm going to skip for now and jump to the generation loop. The generation loop has two levels. First and an outer level in which the text tokens (without any speech conditioning?) are passed through the `lm` model in chunks of size `TTS_TEXT_WINDOW_SIZE` (default value 5), the last hidden state for each text token is obtained and then passed to the `tts_lm` to prefill it's kv cache.
The last hidden state of the `tts_lm` is then used to generate a speech latent which is used to autoregressively generate audio in the inner loop.

We will first convert the `lm` model to CoreML and in the code replace the uses of that model with the CoreML wrapping and unwrapping to numpy a torch.
First impression is that the `lm` model receives size 5 inputs, but it only receives text without any speech (prompt or feedback), so for performance reasons we feed the `lm` longer spans of tokens and chunk them to the size used by the `tts_lm` this will help reduce latency.

We need to modify the `lm` model to make the graph suitable to be converted to CoreML and run on the NE. The changes are:
- Switch from linear layers to convolutions, shape `(B, C, 1, L)` for the activations
- Use the negative concat trick of anemll for the RMSNorm to avoid overflows
- Precompute RoPE embeddings for all positions and use indexing to select the ones needed for the current position
- Implement KV-caching with buffers for CoreML state compatibility
These changes are implemented in the `coreml/layers.py` file and the `lm` model is patched in the `coreml/convert_language_model.py` file.

What we'll do now is replace all the calls to `lm` inside of the `.generate` with calls to the CoreML model. Since in the current `generate` method the inputs are of length 5 we'll just pad them to fill the size 32 of our model.

First thing we have to do is load the past key values from torch into CoreML, in the `devving.ipynb` notebook we print the shapes to see what is happening and in the `coreml/wrapped_inference.py` there is short function to do this.