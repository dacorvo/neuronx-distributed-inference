import json
import logging
import os
from typing import Dict, List, Type, Optional, Union

import torch

NEURON_CONFIG_FILE = "neuron_config.json"


def to_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    assert dtype_str in dtype_mapping, f"Unsupported dtype: {dtype_str}"
    return dtype_mapping[dtype_str]


def to_dict(obj):
    if type(obj) is dict:
        return {k: to_dict(v) for k, v in obj.items()}
    elif type(obj) is list:
        return [to_dict(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    elif type(obj) is torch.dtype:
        return str(obj).split(".")[1]
    else:
        return obj


class IncompatibleConfigError(ValueError):
    pass


class NeuronConfig:
    """
    Base config class for inference in NxD.

    This class contains attributes that are needed for various inference
    optimization/features in NxD.
    """

    def __init__(self,
                 batch_size: Optional[int] = 1,
                 ctx_batch_size: Optional[int] = None,
                 tkg_batch_size: Optional[int] = None,
                 max_batch_size: Optional[int] = None,
                 is_continuous_batching: Optional[bool] = False,
                 seq_len: Optional[int] = 128,
                 tp_degree: Optional[int] = 1,
                 ep_degree: Optional[int] = 1,
                 pp_degree: Optional[int] = 1,
                 torch_dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
                 rpl_reduce_dtype: Optional[Union[str, torch.dtype]] = None,
                 n_active_tokens: Optional[int] = None,
                 max_context_length: Optional[int] = None,
                 output_logits: Optional[bool] = False,
                 padding_side: Optional[str] = "right",
                 fused_qkv: Optional[bool] = False,
                 vocab_parallel: Optional[bool] = False,
                 sequence_parallel_enabled: Optional[bool] = False,
                 is_chunked_prefill: Optional[bool] = False,
                 flash_decoding_enabled: Optional[bool] = False,
                 async_mode: Optional[bool] = False,
                 qk_layernorm: Optional[bool] = False,
                 attn_kernel_enabled: Optional[bool] = False,
                 qkv_kernel_enabled: Optional[bool] = False,
                 mlp_kernel_enabled: Optional[bool] = False,
                 mlp_kernel_fuse_residual_add: Optional[bool] = False,
                 enable_bucketing: Optional[bool] = False,
                 **kwargs) -> None:
        # Basic config for inference in NxD
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tp_degree = tp_degree
        self.torch_dtype = torch_dtype
        if isinstance(self.torch_dtype, str):
            self.torch_dtype = to_torch_dtype(self.torch_dtype)
        self.n_active_tokens = self.seq_len if n_active_tokens is None else n_active_tokens
        self.output_logits = output_logits

        self.padding_side = padding_side

        self.rpl_reduce_dtype = torch_dtype if rpl_reduce_dtype is None else rpl_reduce_dtype
        if isinstance(self.rpl_reduce_dtype, str):
            self.rpl_reduce_dtype = to_torch_dtype(self.rpl_reduce_dtype)

        # fallback to sequence_length is for compatibility with vllm
        self.max_context_length = max_context_length
        if self.max_context_length is None:
            self.max_context_length = seq_len

        # Graph transforms
        self.fused_qkv = fused_qkv

        # Functional parallelism
        self.vocab_parallel = vocab_parallel
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.is_chunked_prefill = is_chunked_prefill

        # Continuous batching
        # TODO: Check if we really need different batch size for CTE and TKG, given
        # that we anyway provide two different config instance for them.
        self.ctx_batch_size = batch_size if ctx_batch_size is None else ctx_batch_size
        self.tkg_batch_size = batch_size if tkg_batch_size is None else tkg_batch_size
        self.max_batch_size = batch_size if max_batch_size is None else max_batch_size
        self.is_continuous_batching = is_continuous_batching

        # On-device sampling
        self.on_device_sampling_config = kwargs.pop("on_device_sampling_config", None)
        if type(self.on_device_sampling_config) is dict:
            self.on_device_sampling_config = OnDeviceSamplingConfig(
                **self.on_device_sampling_config
            )

        # async
        self.async_mode = async_mode

        # Bucketing
        self.enable_bucketing = enable_bucketing

        # Speculative decoding
        self.trace_tokengen_model = kwargs.pop("trace_tokengen_model", True)
        self.speculation_length = kwargs.pop("speculation_length", 0)
        self.spec_batch_size = kwargs.pop("spec_batch_size", self.batch_size)

        if self.speculation_length > 0 and self.async_mode:
            raise IncompatibleConfigError("Speculative Decoding is not yet supported with async.")


        # Distributed config
        self.pp_degree = pp_degree
        self.ep_degree = ep_degree
        self.save_sharded_checkpoint = kwargs.pop("save_sharded_checkpoint", True)

        # QK layer normalization
        self.qk_layernorm = qk_layernorm

        self.world_size = kwargs.pop("world_size", None)
        if self.world_size is None:
            self.world_size = self.tp_degree * self.pp_degree * self.ep_degree

        self.start_rank_id = kwargs.pop("start_rank_id", 0)
        self.local_ranks_size = kwargs.pop("local_ranks_size", None)

        if self.local_ranks_size is None:
            self.local_ranks_size = self.world_size

        # Flash decoding
        self.flash_decoding_enabled = flash_decoding_enabled
        self.num_cores_per_group = 1

        # Kernels
        self.attn_kernel_enabled = attn_kernel_enabled
        self.qkv_kernel_enabled = qkv_kernel_enabled
        self.mlp_kernel_enabled = mlp_kernel_enabled
        self.mlp_kernel_fuse_residual_add = mlp_kernel_fuse_residual_add

        # compiler flags
        self.logical_nc_config = kwargs.pop("logical_nc_config", 1)
        self.cc_pipeline_tiling_factor = kwargs.pop("cc_pipeline_tiling_factor", 2)
        self.target = kwargs.pop("target", None)

        # weights_to_skip_layout_optimization
        self.weights_to_skip_layout_optimization = []

        if kwargs:
            logging.warning(f"NeuronConfig init: Unexpected keyword arguments: {kwargs}")


    def save(self, model_path: Union[str, os.PathLike]):
        """
        Saves the config to a JSON file in the given model directory.
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        config_file = os.path.join(model_path, NEURON_CONFIG_FILE)
        self.to_json_file(config_file)

    def to_json_file(self, json_file: Union[str, os.PathLike]):
        with open(json_file, "w", encoding="utf-8") as writer:
            config_json = self.to_json_string()
            logging.debug(f"Saving config: {config_json}")
            writer.write(config_json + "\n")

    def to_json_string(self) -> str:
        config_dict = to_dict(self)
        return json.dumps(config_dict, indent=2, sort_keys=True)

    @classmethod
    def load(cls, model_path: Union[str, os.PathLike], **kwargs) -> "NeuronConfig":
        """
        Loads the config from the given model directory.

        The given kwargs override any properties of the same name from the JSON file.
        """
        config_file = os.path.join(model_path, NEURON_CONFIG_FILE)
        return cls.from_json_file(config_file, **kwargs)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike], **kwargs) -> "NeuronConfig":
        with open(json_file, "r", encoding="utf-8") as reader:
            config = cls.from_json_string(reader.read(), **kwargs)
            logging.info(f"Loaded Neuron config: {config.to_json_string()}")
            return config

    @classmethod
    def from_json_string(cls, json_string: str, **kwargs) -> "NeuronConfig":
        merged_kwargs = json.loads(json_string)
        merged_kwargs.update(kwargs)

        # Initialize NeuronConfig from dict.
        if "neuron_config" in merged_kwargs and isinstance(merged_kwargs["neuron_config"], dict):
            merged_kwargs["neuron_config"] = cls.get_neuron_config_cls()(
                **merged_kwargs["neuron_config"]
            )
        return cls(**merged_kwargs)


class MoENeuronConfig(NeuronConfig):
    """
    Base class for mixture of experts (MoE) config on Neuron.
    """

    def __init__(
        self,
        capacity_factor: float = None,
        glu_mlp: bool = True,
        **kwargs,
    ) -> None:
        self.capacity_factor = float(capacity_factor) if capacity_factor is not None else None
        self.glu_mlp = glu_mlp
        super().__init__(**kwargs)


class OnDeviceSamplingConfig:
    def __init__(self, **kwargs):
        self.do_sample = kwargs.pop("do_sample", True)
        self.top_k = kwargs.pop("top_k", 1)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.dynamic = kwargs.pop("dynamic", False)
        self.deterministic = kwargs.pop("deterministic", False)
        self.global_topk = kwargs.pop("global_topk", 256)
        self.on_device_sampling_config = kwargs.pop("on_device_sampling_config", True)
