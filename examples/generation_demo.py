import argparse
import torch

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

model_path = "/home/ubuntu/neuronx-distributed-inference/model_hf/Llama-3.2-1B/"
traced_model_path = "/home/ubuntu/neuronx-distributed-inference/traced_model/Llama-3.2-1B/"

torch.manual_seed(0)


def run_llama_generate(sequence_parallel_enabled: bool  = False):
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "do_sample": True,
        "top_k": 1,
        "pad_token_id": generation_config.eos_token_id,
    }
    generation_config.update(**generation_config_kwargs)

    neuron_config = NeuronConfig(
        tp_degree=2,
        batch_size=1,
        seq_len=4096,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        flash_decoding_enabled=False,
        fused_qkv=True,
        torch_dtype=torch.bfloat16,
        sequence_parallel_enabled=sequence_parallel_enabled,
    )
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronLlamaForCausalLM(model_path, config)
    model.compile(traced_model_path)
    tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronLlamaForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompts = ["I believe the meaning of life is", "The color of the sky is"]
    sampling_params = prepare_sampling_params(batch_size=neuron_config.batch_size, top_k=[10, 5], top_p=[0.5, 0.9],  temperature=[0.9, 0.5])
    print(f"Prompts: {prompts}")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=512,
        sampling_params=sampling_params,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_parallel_enabled", action="store_true", help="Enable sequence parallelism.")
    args = parser.parse_args()
    run_llama_generate(args.sequence_parallel_enabled)
