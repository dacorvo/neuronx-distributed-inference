import copy
import logging
import os
import warnings
from functools import partial
from typing import List

import neuronx_distributed.trace.hlo_utils as hlo_utils
import torch
from neuronx_distributed.quantization.quantization_config import QuantizationType, QuantizedDtype
from neuronx_distributed.quantization.quantization_utils import (
    convert_qint8_to_int8_state_dict,
    quantize_pytorch_model_per_channel_symmetric,
    quantize_pytorch_model_per_tensor_symmetric,
)
from neuronx_distributed.trace.model_builder import ModelBuilder
from safetensors.torch import load_file
from transformers import PretrainedConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import NxDModelWrapper
from neuronx_distributed_inference.modules.checkpoint import (
    load_state_dict,
    prune_state_dict,
    save_state_dict_safetensors,
)
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group

COMPILED_MODEL_FILE_NAME = "model.pt"
logger = logging.getLogger("Neuron")


def normalize_path(path):
    """Normalize path separators and ensure path ends with a trailing slash"""
    normalized = os.path.normpath(path)
    return os.path.join(normalized, "")


def get_shards_path(dest_path):
    return os.path.join(dest_path, "weights")


class NeuronApplicationBase(torch.nn.Module):
    _STATE_DICT_MODEL_PREFIX = "model."
    _NEW_STATE_DICT_MODEL_PREFIX = ""
    _FUSED_PREFIX = ""

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NeuronConfig,
    ):
        super().__init__()

        self.config = copy.deepcopy(config)
        self.neuron_config = copy.deepcopy(neuron_config)
        # Override torch_dtype in config as it is used by the neuronx_distributed code to cast weights to the correct type
        self.config.torch_dtype = self.neuron_config.torch_dtype
        if neuron_config.flash_decoding_enabled:
            # FIXME: this should not be part of neuron_config but is used in downstream classes
            # Could it be deduced from tensor shapes ?
            self.neuron_config.num_cores_per_group = calculate_num_cores_per_group(
                config.num_attention_heads, config.num_key_value_heads, neuron_config.tp_degree
            )
        self.on_device_sampling = self.neuron_config.on_device_sampling_config is not None
        self.models: List[NxDModelWrapper] = []
        self.traced_model = None
        self.is_loaded_to_neuron = False

    def get_builder(self, debug=False, checkpoint_loader=None):
        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        builder = ModelBuilder(
            router=None,
            tp_degree=self.neuron_config.tp_degree,
            pp_degree=self.neuron_config.pp_degree,
            ep_degree=self.neuron_config.ep_degree,
            world_size=self.neuron_config.world_size,
            start_rank_id=self.neuron_config.start_rank_id,
            local_ranks_size=self.neuron_config.local_ranks_size,
            checkpoint_loader=checkpoint_loader,
            compiler_workdir=base_compile_work_dir,
            debug=debug,
            num_cores_per_group=self.neuron_config.num_cores_per_group,
            logical_neuron_cores=self.neuron_config.logical_neuron_cores,
            weights_to_skip_layout_optimization=self.neuron_config.weights_to_skip_layout_optimization,
        )
        for model in self.models:
            builder.add(
                key=model.tag,
                model_instance=model.get_model_instance(),
                example_inputs=model.input_generator(),
                compiler_args=self.get_compiler_args(),
                bucket_config=model.get_bucket_config(),
                priority_model_idx=model.priority_model_idx,
            )
        return builder

    def forward(self, **kwargs):
        """Forward pass for this model."""
        raise NotImplementedError("forward is not implemented")

    @classmethod
    def get_config_cls(cls) -> PretrainedConfig:
        """Gets the config class for this model."""
        raise NotImplementedError("get_config_cls is not implemented")

    @classmethod
    def get_neuron_config_cls(cls) -> NeuronConfig:
        raise NotImplementedError("get_neuron_config_cls is not implemented")

    def get_compiler_args(self) -> str:
        """Gets the Neuron compiler arguments to use when compiling this model."""
        return None

    def compile(self, compiled_model_path, debug=False):
        compiled_model_path = normalize_path(compiled_model_path)

        """Compiles this model and saves it to the given path."""
        self.config.save_pretrained(compiled_model_path)
        self.neuron_config.save(compiled_model_path)

        builder = self.get_builder(debug)

        traced_model = builder.trace(initialize_model_weights=False)
        torch.jit.save(traced_model, compiled_model_path + COMPILED_MODEL_FILE_NAME)
        del traced_model

    def shard_checkpoint(self, src_path, dest_path, debug=False):
        shards_path = get_shards_path(dest_path)
        checkpoint_loader = partial(self.checkpoint_loader_fn, src_path, self.config, self.neuron_config)
        builder = self.get_builder(debug, checkpoint_loader=checkpoint_loader)
        builder.shard_checkpoint(serialize_path=shards_path)

        if hlo_utils.NXD_LAYOUT_TRANSFORMATION_OPTIONS in os.environ:
            builder.transform_weight_layout_with_overriden_option(
                sharded_checkpoint_dir=shards_path
            )

    def load(self, compiled_model_path, weight_path, start_rank_id=None, local_ranks_size=None):
        compiled_model_path = normalize_path(compiled_model_path)
        weight_path = normalize_path(weight_path)

        """Loads the compiled model checkpoint to the Neuron device."""
        self.traced_model = torch.jit.load(compiled_model_path + COMPILED_MODEL_FILE_NAME)

        self.load_weights(
            weight_path, start_rank_id=start_rank_id, local_ranks_size=local_ranks_size
        )

        if self.neuron_config.torch_dtype != torch.float32:
            self.to(self.neuron_config.torch_dtype)

        for model_wrapper in self.models:
            model_wrapper.model = self.traced_model
        self.is_loaded_to_neuron = True

    def load_weights(self, weights_path, start_rank_id=None, local_ranks_size=None):
        weights_path = normalize_path(weights_path)

        """Loads the model weights to the Neuron device."""
        if self.traced_model is None:
            raise ValueError("Model is not loaded")

        if start_rank_id is None:
            start_rank_id = self.neuron_config.start_rank_id
        if local_ranks_size is None:
            local_ranks_size = self.neuron_config.local_ranks_size

        logging.info(
            f"loading models for ranks {start_rank_id}...{start_rank_id + local_ranks_size - 1}"
        )
        weights = []
        if self.neuron_config.save_sharded_checkpoint:
            shards_path = get_shards_path(weights_path)
            for rank in range(start_rank_id, start_rank_id + local_ranks_size):
                ckpt = load_file(
                    os.path.join(
                        shards_path, f"tp{rank}_sharded_checkpoint.safetensors"
                    )
                )
                weights.append(ckpt)
        else:
            print("There are no saved sharded checkpoints.")
            checkpoint_loader = partial(self.checkpoint_loader_fn, weights_path, self.config, self.neuron_config)
            builder = self.get_builder(checkpoint_loader=checkpoint_loader)
            source_model_key = list(builder.model_collection.keys())[0]
            for rank in range(start_rank_id, start_rank_id + local_ranks_size):
                print(f"Sharding and loading rank {rank}")
                ckpt = builder.shard_weights(
                    rank, builder.model_collection[source_model_key]
                )
                weights.append(ckpt)
        start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self.traced_model.nxd_model.initialize(weights, start_rank_tensor)

    def checkpoint_loader_fn(self, checkpoint_path, config, neuron_config):
        """This function loads the model's state dictionary and weights from the hf model"""

        if neuron_config.quantized:
            return self.get_quantized_state_dict(config, neuron_config)
        else:
            model_sd = self.get_state_dict(checkpoint_path, config, neuron_config)
            if neuron_config.torch_dtype != torch.float32:
                for name, param in model_sd.items():
                    if torch.is_floating_point(param) and param.dtype not in [torch.float8_e4m3fn]:
                        # only cast floating types
                        if name.endswith("scale"):
                            warnings.warn(
                                f"Found float32 weights in quantized checkpoint: {name}. Will skip converting to bfloat16 as its scale"
                            )
                        else:
                            warnings.warn(
                                f"Found float32 weights in quantized checkpoint: {name}. Will convert to bfloat16"
                            )
                            model_sd[name] = param.to(neuron_config.torch_dtype)
        return model_sd

    @classmethod
    def get_state_dict(cls, model_path: str, config: PretrainedConfig, neuron_config: NeuronConfig) -> dict:
        """Gets the state dict for this model."""
        model_sd = load_state_dict(model_path)
        param_name_list = list(model_sd.keys())
        for param_name in param_name_list:
            if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                updated_param_name = param_name.replace(
                    cls._STATE_DICT_MODEL_PREFIX, cls._NEW_STATE_DICT_MODEL_PREFIX, 1
                )
                model_sd[updated_param_name] = model_sd[param_name]
                del model_sd[param_name]
        model_sd = cls.convert_hf_to_neuron_state_dict(model_sd, config, neuron_config)
        if getattr(config, "tie_word_embeddings", False):
            cls.update_state_dict_for_tied_weights(model_sd)

        param_name_list = list(model_sd.keys())
        if cls._FUSED_PREFIX != "":
            for param_name in param_name_list:
                model_sd[f"{cls._FUSED_PREFIX}.{param_name}"] = model_sd[param_name]
                del model_sd[param_name]
        return model_sd

    @classmethod
    def get_quantized_state_dict(cls, config: PretrainedConfig, neuron_config: NeuronConfig, mmap: bool = False) -> dict:
        """
        This function loads the checkpointed float model state dictionary and weights from the quantized hf model
        This will be removed once we move to safe tensors in NxD
        """
        existing_checkpoint_path = neuron_config.quantized_checkpoints_path
        if not os.path.exists(existing_checkpoint_path):
            raise FileNotFoundError(
                f"Quantized checkpoint file not found: {existing_checkpoint_path}"
            )

        print(f"Using existing checkpoint: {existing_checkpoint_path}")
        if os.path.isdir(existing_checkpoint_path):
            model_quant_sd = load_state_dict(existing_checkpoint_path)
        else:
            model_quant_sd = torch.load(existing_checkpoint_path)

        # For the case when huggingface models come with existing prefixes. We do not allow
        # Any prefix like "model."
        param_name_list = list(model_quant_sd.keys())
        for param_name in param_name_list:
            if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                updated_param_name = param_name.replace(
                    cls._STATE_DICT_MODEL_PREFIX, cls._NEW_STATE_DICT_MODEL_PREFIX, 1
                )
                model_quant_sd[updated_param_name] = model_quant_sd[param_name]
                del model_quant_sd[param_name]

        model_quant_sd = cls.convert_hf_to_neuron_state_dict(model_quant_sd, config)

        # Make sure that the non quantized weights are in bfloat16 and not float32
        if neuron_config.torch_dtype == torch.bfloat16:
            for name, param in model_quant_sd.items():
                # TODO: Reduce and clean-up these warnings
                if param is not None and param.dtype == torch.float32:
                    if name.endswith(".scale"):
                        warnings.warn(
                            f"Found float32 weights in quantized checkpoint: {name}. Will skip converting to bfloat16 as its scale"
                        )
                    else:
                        warnings.warn(
                            f"Found float32 weights in quantized checkpoint: {name}. Will convert to bfloat16"
                        )
                        model_quant_sd[name] = param.bfloat16()

        return model_quant_sd

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: PretrainedConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        return state_dict

    @classmethod
    def save_quantized_state_dict(cls, model_path: str, config: PretrainedConfig, neuron_config: NeuronConfig):
        """
        Quantizes the model and saves the quantized checkpoint to `neuron_config.quantized_checkpoints_path`.
        """
        model_path = normalize_path(model_path)
        quantized_state_dict = cls.generate_quantized_state_dict(model_path, neuron_config)

        # Prune None values in the quantized_state_dict. torch.save crashes if None values exist.
        quantized_state_dict = prune_state_dict(quantized_state_dict)
        if os.path.isdir(neuron_config.quantized_checkpoints_path):
            logging.info(
                "Saving quantized state dict as safetensors to: %s",
                neuron_config.quantized_checkpoints_path,
            )
            save_state_dict_safetensors(
                state_dict=quantized_state_dict,
                state_dict_dir=neuron_config.quantized_checkpoints_path,
            )
        else:
            logging.info(
                "Saving quantized state dict as torch pt file to: %s",
                neuron_config.quantized_checkpoints_path,
            )
            torch.save(quantized_state_dict, neuron_config.quantized_checkpoints_path)

    @classmethod
    def generate_quantized_state_dict(cls, model_path: str, neuron_config: NeuronConfig) -> dict:
        """Generates the quantized state dict for this model."""
        hf_model = cls.load_hf_model(model_path)
        quantization_type = QuantizationType(neuron_config.quantization_type)
        quantized_dtype = QuantizedDtype.get_dtype(neuron_config.quantization_dtype)
        if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            hf_model_quant = quantize_pytorch_model_per_tensor_symmetric(
                float_model=hf_model, inplace=True, dtype=quantized_dtype
            )
        elif quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            hf_model_quant = quantize_pytorch_model_per_channel_symmetric(
                float_model=hf_model, inplace=True, dtype=quantized_dtype
            )
        else:
            raise RuntimeError(f"{neuron_config.quantization_type} not supported")

        return cls.prepare_quantized_state_dict(hf_model_quant)

    @classmethod
    def prepare_quantized_state_dict(cls, hf_model_quant) -> dict:
        """Can be overriden to customize the quantized state dict in generate_quantized_state_dict."""
        model_quant_sd = hf_model_quant.model.state_dict()
        convert_qint8_to_int8_state_dict(model_quant_sd)
        return model_quant_sd

    @staticmethod
    def load_hf_model(model_path):
        """Loads the HuggingFace model from the given checkpoint path."""
        raise NotImplementedError("load_hf_model is not implemented")

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Implement state_dict update for each model class with tied weights"""
        raise NotImplementedError("State-dict update not implemented")

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # We dont want HF to move parameters to device
        return torch.device("cpu")

    def reset(self):
        """Resets the model state. Can be implemented by subclasses."""
        pass
