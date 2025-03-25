import argparse
import copy
import os
import time
from typing import Type

import torch
from neuronx_distributed.quantization.quantization_config import QuantizationType
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import (
    OnDeviceSamplingConfig,
    to_torch_dtype,
)
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.models.mixtral.modeling_mixtral import NeuronMixtralForCausalLM
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
from neuronx_distributed_inference.utils.distributed import get_init_rank, get_init_world_size
from neuronx_distributed_inference.utils.random import set_random_seed

set_random_seed(0)


MODEL_TYPES = {
    "llama": {"causal-lm": NeuronLlamaForCausalLM},
    "mixtral": {"causal-lm": NeuronMixtralForCausalLM},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=MODEL_TYPES.keys(), required=True)
    parser.add_argument("--task-type", type=str, required=True)
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser("run")
    setup_run_parser(run_parser)

    return parser.parse_args()


def setup_run_parser(run_parser: argparse.ArgumentParser):
    run_parser.add_argument("--model-path", type=str, required=True)
    run_parser.add_argument("--compiled-model-path", type=str, required=True)

    # Generation
    run_parser.add_argument("--prompt", dest="prompts", type=str, action="append", required=True)
    run_parser.add_argument("--top-k", type=int, default=1)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--temperature", type=float, default=1.0)
    run_parser.add_argument("--global-topk", type=int)
    run_parser.add_argument("--do-sample", action="store_true", default=False)
    run_parser.add_argument("--dynamic", action="store_true", default=False)
    run_parser.add_argument("--pad-token-id", type=int, default=0)

    # Basic config
    run_parser.add_argument("--torch-dtype", type=to_torch_dtype)
    run_parser.add_argument("--batch-size", type=int)
    run_parser.add_argument("--padding-side", type=str)
    run_parser.add_argument("--seq-len", type=int)
    run_parser.add_argument("--n-active-tokens", type=int)
    run_parser.add_argument("--n-positions", type=int)
    run_parser.add_argument("--max-context-length", type=int)
    run_parser.add_argument("--max-new-tokens", type=int)
    run_parser.add_argument("--max-length", type=int)
    run_parser.add_argument("--rpl-reduce-dtype", type=to_torch_dtype)
    run_parser.add_argument("--output-logits", action="store_true")
    run_parser.add_argument("--vocab-parallel", action="store_true")

    # Attention
    run_parser.add_argument("--fused-qkv", action="store_true")
    run_parser.add_argument("--sequence-parallel-enabled", action="store_true")
    run_parser.add_argument("--flash-decoding-enabled", action="store_true")

    # Continuous batching
    run_parser.add_argument("--ctx-batch-size", type=int)
    run_parser.add_argument("--tkg-batch-size", type=int)
    run_parser.add_argument("--max-batch-size", type=int)
    run_parser.add_argument("--is-continuous-batching", action="store_true")

    # On device sampling
    run_parser.add_argument("--on-device-sampling", action="store_true")

    # Bucketing
    run_parser.add_argument("--enable-bucketing", action="store_true")
    run_parser.add_argument("--bucket-n-active-tokens", action="store_true")
    run_parser.add_argument("--context-encoding-buckets", nargs="+", type=int)
    run_parser.add_argument("--token-generation-buckets", nargs="+", type=int)

    # Quantization
    run_parser.add_argument("--quantized", action="store_true")
    run_parser.add_argument("--quantized-checkpoints-path", type=str)
    run_parser.add_argument(
        "--quantization-type", type=str, choices=[t.value for t in QuantizationType]
    )
    run_parser.add_argument("--kv-cache-quant", action="store_true")
    run_parser.add_argument("--quantization-dtype", type=str)

    # MoE
    run_parser.add_argument("--capacity-factor", type=float)

    # Speculative decoding
    run_parser.add_argument("--draft-model-path", type=str)
    run_parser.add_argument("--draft-model-tp-degree", type=int, default=None)
    run_parser.add_argument("--compiled-draft-model-path", type=str)
    run_parser.add_argument(
        "--no-trace-tokengen-model", dest="trace_tokengen_model", action="store_false"
    )
    run_parser.add_argument("--speculation-length", type=int)
    run_parser.add_argument("--spec-batch-size", type=int)

    # Parallelism
    run_parser.add_argument("--tp-degree", type=int)
    run_parser.add_argument("--pp-degree", type=int)
    run_parser.add_argument("--ep-degree", type=int)
    run_parser.add_argument("--world-size", type=int)
    run_parser.add_argument("--start_rank_id", type=int)
    run_parser.add_argument("--local_ranks_size", type=int)
    run_parser.add_argument(
        "--enable-torch-dist",
        action="store_true",
        help="Use torch.distributed (gloo) backend when running multi-node examples. "
        "This is useful for ensuring processes on different nodes are in sync",
    )
    run_parser.add_argument(
        "--skip-save-sharded-checkpoint", dest="save_sharded_checkpoint", action="store_false"
    )

    # async
    run_parser.add_argument("--async", action="store_true")

    # Kernels
    run_parser.add_argument("--qkv-kernel-enabled", action="store_true")
    run_parser.add_argument("--attn-kernel-enabled", action="store_true")
    run_parser.add_argument("--mlp-kernel-enabled", action="store_true")
    run_parser.add_argument("--quantized-mlp-kernel-enabled", action="store_true")
    run_parser.add_argument("--rmsnorm-quantize-kernel-enabled", action="store_true")
    run_parser.add_argument("--quantized-kernel-lower-bound", type=float, default=1200.0)
    run_parser.add_argument("--mlp-kernel-fuse-residual-add", action="store_true")

    # Compiler Args
    run_parser.add_argument("--logical-neuron-cores", type=int, default=1)
    run_parser.add_argument("--cc-pipeline-tiling-factor", type=int, default=2)

    # optional demo arguments
    run_parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="skip model compilation. If this option is set, then compiled model must be "
        "present at path specified by --compiled-model-path argument",
    )
    run_parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only perform model compilation.",
    )
    run_parser.add_argument(
        "--hlo-debug",
        action="store_true",
        help="Adds metadata into the generated HLO. This metadata maps the HLO "
        "operators to the corresponding lines in the PyTorch code",
    )


def run_inference(model_cls: Type[NeuronApplicationBase], args):
    # Initialize configs.
    print("Loading configs...")

    if args.skip_compile:
        # Reload configuration
        config = AutoConfig.from_pretrained(args.compiled_model_path)
        neuron_config = model_cls.get_neuron_config_cls().load(args.compiled_model_path)
    else:
        # Skip values not specified in the args to avoid setting values to None in the config.
        config_kwargs = copy.deepcopy(vars(args))
        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
        if args.on_device_sampling:
            config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**config_kwargs)

        if (args.quantized and args.quantization_dtype == "f8e4m3") or args.kv_cache_quant:
            os.environ["XLA_HANDLE_SPECIAL_SCALAR"] = "1"

        neuron_config = model_cls.get_neuron_config_cls()(**config_kwargs)

        config = AutoConfig.from_pretrained(args.model_path)

    # Initialize draft model.
    draft_model = None
    if neuron_config.speculation_length > 0 and args.draft_model_path is not None:
        # Reset speculation options to defaults for the draft model.
        draft_neuron_config = copy.deepcopy(neuron_config)
        draft_neuron_config.speculation_length = 0
        draft_neuron_config.trace_tokengen_model = True

        if args.draft_model_tp_degree is not None:
            draft_neuron_config.tp_degree = args.draft_model_tp_degree

        draft_config = AutoConfig.from_pretrained(args.draft_model_path)
        draft_model = model_cls(draft_config, draft_neuron_config)

    model = model_cls(config, neuron_config)

    # Quantize model.
    if neuron_config.quantized:
        model_cls.save_quantized_state_dict(args.model_path, config, neuron_config)

    # Compile and save model.
    if not args.skip_compile:
        print("\nCompiling and saving model...")
        compiling_start_time = time.monotonic()
        model.compile(args.compiled_model_path, debug=args.hlo_debug)
        if draft_model is not None:
            print("\nCompiling and saving draft model...")
            draft_model.compile(args.compiled_draft_model_path)
        compiling_end_time = time.monotonic()
        total_compiling_time = compiling_end_time - compiling_start_time
        print(f"Compiling and tracing time: {total_compiling_time} seconds")
        if args.save_sharded_checkpoint:
            print("\nSharding and saving weights ...")
            sharding_start_time = time.monotonic()
            model.shard_checkpoint(args.model_path, args.compiled_model_path)
            if draft_model is not None:
                draft_model.shard_checkpoint(args.draft_model_path, args.compiled_draft_model_path)
            sharding_end_time = time.monotonic()
            print(f"Sharding and saving time: {sharding_end_time - sharding_start_time} seconds")
        if args.compile_only:
            return

    if args.enable_torch_dist:
        torch.distributed.barrier()

    # Load compiled model to Neuron.
    print("\nLoading model to Neuron...")
    loading_start_time = time.monotonic()
    weight_path = args.compiled_model_path if neuron_config.save_sharded_checkpoint else args.model_path
    model.load(args.compiled_model_path, weight_path)
    loading_end_time = time.monotonic()
    model_loading_time = loading_end_time - loading_start_time
    print(f"Total model loading time: {model_loading_time} seconds")

    if draft_model is not None:
        print("\nLoading draft model to Neuron...")
        draft_weight_path = args.compiled_draft_model_path if draft_neuron_config.save_sharded_checkpoint else args.draft_model_path
        draft_model.load(args.compiled_draft_model_path, draft_weight_path)

    if args.enable_torch_dist:
        torch.distributed.barrier()

    # Load tokenizer.
    tokenizer = load_tokenizer(args.model_path, args.compiled_model_path, neuron_config)

    # Configure generation config.
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config_args = [
        "do_sample",
        "top_k",
        "pad_token_id",
        "dynamic",
        "top_p",
        "temperature",
    ]
    generation_config_kwargs = {
        k: getattr(args, k) for k in generation_config_args if getattr(args, k) is not None
    }
    generation_config.update(**generation_config_kwargs)

    # Generate outputs.
    run_generation(
        model,
        tokenizer,
        args.prompts,
        generation_config,
        draft_model=draft_model,
        max_new_tokens=args.max_new_tokens
    )


def load_tokenizer(model_path, compiled_model_path, neuron_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(compiled_model_path)
    return tokenizer


def run_generation(
    model,
    tokenizer,
    prompts,
    generation_config,
    draft_model=None,
    max_new_tokens=None,
):
    print("\nGenerating outputs...")
    print(f"Prompts: {prompts}")

    _, generated_texts = get_generate_outputs(
        model,
        prompts,
        tokenizer,
        draft_model=draft_model,
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
    )

    print("Generated outputs:")
    for i, text in enumerate(generated_texts):
        print(f"Output {i}: {text}")


def main():
    args = parse_args()
    assert (
        args.task_type in MODEL_TYPES[args.model_type]
    ), f"Unsupported task: {args.model_type}/{args.task_type}"

    if args.enable_torch_dist:
        torch.distributed.init_process_group(
            backend="gloo",
            world_size=get_init_world_size(),
            rank=get_init_rank(),
        )
        node_rank = torch.distributed.get_rank()
        args.start_rank_id = node_rank * args.local_ranks_size
        torch.distributed.barrier()

    model_cls = MODEL_TYPES[args.model_type][args.task_type]
    run_inference(model_cls, args)


if __name__ == "__main__":
    main()
