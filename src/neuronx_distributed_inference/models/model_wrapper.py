import torch
from abc import abstractmethod
from typing import List
from torch_neuronx import BucketModelConfig
from neuronx_distributed.trace.model_builder import BaseModelInstance


class NxDModelWrapper(torch.nn.Module):
    def __init__(self, tag: str, priority_model_idx: int):
        super().__init__()
        self.tag = tag
        self.priority_model_idx = priority_model_idx

    @abstractmethod
    def input_generator(self) -> List[torch.Tensor]:
        """Return the list of the model input tensors

        Used at compilation time only when tracing the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_instance(self) -> BaseModelInstance:
        """Return the underlying ModelInstance

        Used at compilation time only when tracing the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_bucket_config(self) -> BucketModelConfig:
        """Return the bucket configuration

        Used at compilation time only when tracing the model.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args) -> List[torch.Tensor]:
        raise NotImplementedError
