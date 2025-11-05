# type: ignore
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence
import torch
import numpy as np


@dataclass
class SamplerInput:
    xs: torch.Tensor
    kwargs: Dict


@dataclass
class LSpaceParameterisation:
    loop_momenta: np.ndarray
    success: np.ndarray
    timing: float


@dataclass
class XSpaceParameterisation:
    xs: np.ndarray
    success: np.ndarray
    timing: float


@dataclass
class GraphProperties:
    edge_src_dst_vertices: list[(int, int)]
    edge_masses: list[int]
    edge_ismassive: list[bool]
    edge_momentum_shifts: list[list[float]]
    graph_external_vertices: list[int]
    graph_signature: list[list[int]]
    momtrop_edge_weights: list[int]


class Sampler(ABC):
    def __init__(self, settings_file: str, **kwargs):
        pass

    @abstractmethod
    def parameterise(self, input: SamplerInput) -> LSpaceParameterisation:
        """
        Args:
            SamplerInput: Contains discrete and continuous points combined into a single tensor,
                          optional kwargs
        Returns:
            LSpaceParameterisation: loop_momenta, success flags array and timing in seconds
        Raises:
            ValueError
        """
        pass

    @abstractmethod
    def get_discrete_dims(self) -> List[int]:
        """
        Returns:
            List of shape of the discrete dimensions of the sampler
        """
        pass

    @abstractmethod
    def get_continuous_dim(self) -> int:
        """
        Returns:
            The continuous dimension of the sampler
        """
        pass

    @abstractmethod
    def discrete_prior_prob_function(self, input: SamplerInput, dim: int) -> torch.Tensor:
        """
        Args:
            SamplerInput: Contains discrete and continuous points combined into a single tensor,
                          optional kwargs
        Returns:

        """

    def set_params(self, **kwargs):
        """Update the parameters of the integrand."""
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")


class Integrand(ABC):
    pass


class DiscreteLayer(ABC):
    pass
