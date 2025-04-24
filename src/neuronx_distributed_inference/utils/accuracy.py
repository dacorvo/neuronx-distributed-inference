"""
This is a temporary file to get the testing running for new package.

Some of the utitlies functions need to be redo or removed.
"""
# flake8: noqa

from typing import Union

from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput


SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


def get_generate_outputs(
    model, prompts, tokenizer, draft_model=None, **generate_kwargs
):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")

    token_ids = inputs.input_ids
    attention_mask=inputs.attention_mask

    # Update generation kwargs to run Neuron model.
    if draft_model is not None:
        draft_model.generation_config.update(
            num_assistant_tokens=model.neuron_config.speculation_length
        )

        generate_kwargs.update(
            {
                "assistant_model": draft_model,
                "do_sample": False,
            }
        )

    outputs = model.generate(token_ids, attention_mask=attention_mask, **generate_kwargs)

    model.reset()
    if draft_model is not None:
        draft_model.reset()

    if isinstance(outputs, SampleOutput.__args__):
        # Get token ids from output when return_dict_in_generate=True
        output_ids = outputs.sequences
    else:
        output_ids = outputs
    generated_texts = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return outputs, generated_texts
