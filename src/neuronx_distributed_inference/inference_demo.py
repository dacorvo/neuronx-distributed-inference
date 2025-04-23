import argparse
import copy
import os
import time
from typing import Type

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.pretrained_model import NxDPreTrainedModel
from neuronx_distributed_inference.models.config import to_torch_dtype
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


def setup_export_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--compiled-model-path", type=str, required=True)

    parser.add_argument("--pad-token-id", type=int, default=0)

    # Basic config
    parser.add_argument("--torch-dtype", type=to_torch_dtype)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--padding-side", type=str, default="right")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--n-active-tokens", type=int)
    parser.add_argument("--n-positions", type=int)
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--rpl-reduce-dtype", type=to_torch_dtype)
    parser.add_argument("--output-logits", action="store_true")
    parser.add_argument("--vocab-parallel", action="store_true")

    # Attention
    parser.add_argument("--fused-qkv", action="store_true")
    parser.add_argument("--sequence-parallel-enabled", action="store_true")
    parser.add_argument("--flash-decoding-enabled", action="store_true")

    # Continuous batching
    parser.add_argument("--ctx-batch-size", type=int)
    parser.add_argument("--tkg-batch-size", type=int)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--is-continuous-batching", action="store_true")

    # On device sampling
    parser.add_argument("--on-device-sampling", action="store_true")

    # Bucketing
    parser.add_argument("--enable-bucketing", action="store_true")

    # MoE
    parser.add_argument("--capacity-factor", type=float)

    # Speculative decoding
    parser.add_argument("--speculation-length", type=int, default=0)

    # Parallelism
    parser.add_argument("--tp-degree", type=int, default=1)
    parser.add_argument("--pp-degree", type=int, default=1)
    parser.add_argument("--ep-degree", type=int, default=1)
    parser.add_argument("--start_rank_id", type=int, default=0)
    parser.add_argument("--local_ranks_size", type=int)
    parser.add_argument(
        "--enable-torch-dist",
        action="store_true",
        help="Use torch.distributed (gloo) backend when running multi-node examples. "
        "This is useful for ensuring processes on different nodes are in sync",
    )

    # async
    parser.add_argument("--async", action="store_true")

    # Kernels
    parser.add_argument("--qkv-kernel-enabled", action="store_true")
    parser.add_argument("--attn-kernel-enabled", action="store_true")
    parser.add_argument("--mlp-kernel-enabled", action="store_true")
    parser.add_argument("--mlp-kernel-fuse-residual-add", action="store_true")

    # Compiler Args
    parser.add_argument("--logical-neuron-cores", type=int, default=1)
    parser.add_argument("--cc-pipeline-tiling-factor", type=int, default=2)


def setup_run_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--draft-model-path", type=str)

    # Generation
    parser.add_argument("--prompt", dest="prompts", type=str, action="append", required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-topk", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int)


def export_model(model_cls: Type[NxDPreTrainedModel], args):
    # Initialize configs.
    print("Loading configs...")
    config = AutoConfig.from_pretrained(args.model_path)
    neuron_config = model_cls.get_neuron_config_cls()(
        batch_size=args.batch_size,
        ctx_batch_size=args.ctx_batch_size,
        tkg_batch_size=args.tkg_batch_size,
        max_batch_size=args.max_batch_size,
        is_continuous_batching=args.is_continuous_batching,
        speculation_length=args.speculation_length,
        seq_len=args.seq_len,
        tp_degree=args.tp_degree,
        ep_degree=args.ep_degree,
        pp_degree=args.pp_degree,
        torch_dtype=args.torch_dtype,
        rpl_reduce_dtype=args.rpl_reduce_dtype,
        n_active_tokens=args.n_active_tokens,
        max_context_length=args.max_context_length,
        output_logits=args.output_logits,
        padding_side=args.padding_side,
        fused_qkv=args.fused_qkv,
        vocab_parallel=args.vocab_parallel,
        sequence_parallel_enabled=args.sequence_parallel_enabled,
        flash_decoding_enabled=args.flash_decoding_enabled,
        async_mode=getattr(args, "async"),
        attn_kernel_enabled=args.attn_kernel_enabled,
        qkv_kernel_enabled=args.qkv_kernel_enabled,
        mlp_kernel_enabled=args.mlp_kernel_enabled,
        mlp_kernel_fuse_residual_add=args.mlp_kernel_fuse_residual_add,
        enable_bucketing=args.enable_bucketing,
        logical_nc_config=args.logical_neuron_cores,
        cc_pipeline_tiling_factor=args.cc_pipeline_tiling_factor,
        on_device_sampling=args.on_device_sampling,
    )

    # Compile and save model.
    print("\nCompiling and saving model...")
    compiling_start_time = time.monotonic()
    model = model_cls.export(args.model_path, config, neuron_config)
    compiling_end_time = time.monotonic()
    print(f"Compiling time: {compiling_end_time - compiling_start_time} seconds")
    model.save_pretrained(args.compiled_model_path)
    saving_end_time = time.monotonic()
    print(f"Saving time: {saving_end_time - compiling_end_time} seconds")
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side=neuron_config.padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(args.compiled_model_path)
    print("\nTokenizer saved.")
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config.save_pretrained(args.compiled_model_path)
    print("\nGeneration config saved.")


def run_inference(model_cls: Type[NxDPreTrainedModel], args):
    # Initialize configs.
    print("Loading configs...")

    # Reload configuration
    neuron_config = model_cls.get_neuron_config_cls().load(args.model_path)

    if args.enable_torch_dist:
        assert neuron_config.start_rank_id == args.start_rank_id
        assert neuron_config.local_ranks_size == args.local_ranks_size

    do_speculate = neuron_config.speculation_length > 0

    if args.enable_torch_dist:
        torch.distributed.barrier()

    # Load compiled model to Neuron.
    print("\nLoading model to Neuron...")
    loading_start_time = time.monotonic()
    model = model_cls.from_pretrained(args.model_path)
    loading_end_time = time.monotonic()
    model_loading_time = loading_end_time - loading_start_time
    print(f"Total model loading time: {model_loading_time} seconds")

    if do_speculate:
        print("\nLoading draft model to Neuron...")
        draft_model = model_cls.from_pretrained(args.draft_model_path)

    if args.enable_torch_dist:
        torch.distributed.barrier()

    # Load tokenizer.
    tokenizer = load_tokenizer(args.model_path, neuron_config)

    # Configure generation config.
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config_args = [
        "do_sample",
        "top_k",
        "pad_token_id",
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
        draft_model=draft_model if do_speculate else None,
        max_new_tokens=args.max_new_tokens
    )


def load_tokenizer(model_path, neuron_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
    tokenizer.pad_token = tokenizer.eos_token
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=MODEL_TYPES.keys(), required=True)
    parser.add_argument("--task-type", type=str, required=True)
    parser.add_argument("--enable-torch-dist", action="store_true")
    subparsers = parser.add_subparsers(dest="action", required=True)

    export_parser = subparsers.add_parser("export")
    setup_export_parser(export_parser)

    run_parser = subparsers.add_parser("run")
    setup_run_parser(run_parser)

    args = parser.parse_args()
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
    if args.action == "export":
        export_model(model_cls, args)
    elif args.action == "run":
        run_inference(model_cls, args)


if __name__ == "__main__":
    main()
