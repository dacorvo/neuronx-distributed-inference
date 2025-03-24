"""
This is a temporary file to get the testing running for new package.

Some of the utitlies functions need to be redo or removed.
"""
# flake8: noqa

import warnings
from contextlib import nullcontext
from typing import Union

import torch
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from neuronx_distributed_inference.utils.constants import *
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    warnings.warn(
        "Intel extension for pytorch not found. For faster CPU references install `intel-extension-for-pytorch`.",
        category=UserWarning,
    )
    ipex = None

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


def get_generate_outputs_from_token_ids(
    model,
    token_ids,
    tokenizer,
    attention_mask=None,
    is_hf=False,
    draft_model=None,
    **generate_kwargs,
):
    if not is_hf:
        # Update generation kwargs to run Neuron model.
        if draft_model is not None:
            draft_generation_model = HuggingFaceGenerationAdapter(draft_model)
            draft_generation_model.generation_config.update(
                num_assistant_tokens=model.neuron_config.speculation_length
            )

            generate_kwargs.update(
                {
                    "assistant_model": draft_generation_model,
                    "do_sample": False,
                }
            )

    # If an attention mask is provided, the inputs are also expected to be padded to the correct shape.
    if attention_mask is None:
        print("attention mask not provided, padding inputs and generating a mask")

        tokenizer.pad_token_id = tokenizer.eos_token_id

        padding_side = "left" if is_hf else "right"
        inputs = tokenizer.pad(
            {"input_ids": token_ids},
            padding_side=padding_side,
            return_attention_mask=True,
            return_tensors="pt",
        )

        token_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        attention_mask[token_ids == tokenizer.pad_token_id] = 0

    generation_model = model if is_hf else HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(token_ids, attention_mask=attention_mask, **generate_kwargs)

    if not is_hf:
        model.reset()
        if draft_model is not None:
            draft_model.reset()

    if isinstance(outputs, SampleOutput.__args__):
        # Get token ids from output when return_dict_in_generate=True
        output_ids = outputs.sequences
    else:
        output_ids = outputs
    output_tokens = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return outputs, output_tokens


def get_generate_outputs(
    model, prompts, tokenizer, is_hf=False, draft_model=None, device="neuron", **generate_kwargs
):
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_hf:
        tokenizer.padding_side = "left"
    else:
        # FIXME: add cpu generation
        if device == "cpu":
            assert "get_generate_outputs from CPU yet avaialble"
        tokenizer.padding_side = "right"

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")

    is_bfloat16 = (
        model.dtype == torch.bfloat16
        if is_hf
        else model.neuron_config.torch_dtype == torch.bfloat16
    )
    use_ipex = ipex and is_bfloat16
    if use_ipex:
        model = ipex.optimize(model, dtype=model.config.torch_dtype)
        model = torch.compile(model, backend="ipex")

    with torch.cpu.amp.autocast() if use_ipex else nullcontext():
        return get_generate_outputs_from_token_ids(
            model,
            inputs.input_ids,
            tokenizer,
            attention_mask=inputs.attention_mask,
            is_hf=is_hf,
            draft_model=draft_model,
            **generate_kwargs,
        )
