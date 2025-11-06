# type: ignore
import torch
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

from madnis.integrator import Integrand as MadnisIntegrand
from src.parser import SettingsParser


"""
Example chain for Kaapo vacuum diagrams:
MadNIS -> Momtrop -> inverse spherical param -> Kaapos modified spherical param ->  Integrand
-----------------------------------------------------------------------------------
So then, what should be passed along?
-----------------------------------------------------------------------------------
MadNIS or HAVANA ##################################################################
args: kwargs (settings_files)
consumes: Nothing
outputs x_all
passes along: kwargs
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
MOMTROP or FORWARD PARAM ##########################################################
args: x_all, kwargs
consumes x_all[:, continuous_dim:] and xall[:, :len(discrete_dims)].long()
outputs: jacobians, loop_momenta
passes along: remaining x_all, kwargs
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
INVERSE SPHERICAL PARAM or BACKWARD PARAM #########################################
args: jacobians, loop_momenta, x_all, kwargs
consumes: Nothing
outputs: jacobians, x_points
passes along: remaining x_all, kwargs
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
KAAPOS MODIFIED SPHERICAL PARAM or FORWARD PARAM ##################################
args: jacobians, x_points, x_all, kwargs
consumes: Nothing
outputs: jacobians, loop_momenta
passes along: remaining x_all, kwargs
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
INTEGRAND #########################################################################
args: jacobians, loop_momenta, x_all, kwargs
consumes: Remaining x_all
outputs: func_val
passes along: Nothing

HOW ABOUT:
Each parameterisation could add all of its data to the x_all tensor
"""


@dataclass
class SamplerInput:
    x_all: torch.Tensor
    kwargs: Dict


@dataclass(frozen=True)
class ParameterisationResult:
    points: np.ndarray
    jacobians: np.ndarray
    success: np.ndarray
    timing: Dict[str, float]


@dataclass
class LSpaceParameterisationResult(ParameterisationResult):
    loop_momenta: np.ndarray
    jacobians: np.ndarray
    success: np.ndarray
    timing: Dict[str, float]


@dataclass
class XSpaceParameterisationResult(ParameterisationResult):
    x_points: np.ndarray
    jacobians: np.ndarray
    success: np.ndarray
    timing: Dict[str, float]


@dataclass
class GraphProperties:
    edge_src_dst_vertices: list[(int, int)]
    edge_masses: list[int]
    edge_ismassive: list[bool]
    edge_momentum_shifts: list[list[float]]
    graph_external_vertices: list[int]
    graph_signature: list[list[int]]
    momtrop_edge_weights: list[int]

    def __post_init__(self):
        self.n_loops = len(self.graph_signature[0])
        self.n_edges = len(self.edge_masses)


class Parameterisation(ABC):
    n_spatial_dims = 3
    def __init__(self, graph_properties: GraphProperties,
                 next_param: Parameterisation,
                 **kwargs,):
        self.identifier = "ABC_parameterisation"
        self.next_param = next_param
        self.graph_propertes = graph_properties
        self.continuous_dim_out = 0
        self.continuous_dim_in = 0
        self.discrete_dims = self._layer_discrete_dims()
        self.set_params(**kwargs)

    @abstractmethod
    def parameterise(self, input: SamplerInput,
                     ) -> Tuple[LSpaceParameterisation, SamplerInput]:
        """
        Args:
            SamplerInput: Contains discrete and continuous points combined into a single tensor,
                          optional kwargs
            prev_param: In case of chained parameterisations, the result of a BackwardParameterisation
        Returns:
            LSpaceParameterisation: loop_momenta, success flags array and timing in seconds
            SamplerInput: Data to be passed along the chain
        Raises:
            ValueError
        """
        pass

    @abstractmethod
    def _layer_discrete_dims(self) -> List[int]:
        """
        Returns:
            List of shape of the discrete dimensions of this layer in the parameterisation chain.
            Intended to be used to initialize the value for self.discrete_dims
        """
        pass

    @abstractmethod
    def _layer_continuous_dim_in(self) -> int:
        """
        Returns:
            The continuous dimension of the input of this layer in the parameterisation chain.
            Intended to be used to initialize the value for self.continuous_dim_in
        """
        return self.n_spatial_dims*self.graph_propertes.n_loops

    @abstractmethod
    def _layer_continuous_dim_out(self) -> int:
        """
        Returns:
            The continuous dimension of the output of this layer in the parameterisation chain.
            Intended to be used to initialize the value for self.continuous_dim_in
        """
        return self.n_spatial_dims*self.graph_propertes.n_loops

    def get_chain_discrete_dims(self) -> List[int]:
        """
        Returns:
            List of shape of the discrete dimensions of this and following layers in the 
            parameterisation chain.
            Intended to be used to set the discrete dimensions for an Integrator.
        """
        if self.next_param is None:
            return self.discrete_dims
        return self.discrete_dims + self.next_param.get_discrete_dims()

    def get_chain_continuous_dim(self) -> int:
        """
        Returns:
            The continuous dimension of this and following layers in the parameterisation chain.
            Intended to be used to set the number of continuous dimensions for an Integrator.
        """
        continuous_dim = self.continuous_dim_in - self.continuous_dim_out
        if self.next_param is None:
            return self.continuous_dim_in
        return max(self.continuous_dim_in, continuous_dim + self.next_param.get_discrete_dims())

    def discrete_prior_prob_function(self, indices: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Args:
            indices: indices of the discrete channel
            dim: current layer of generated indices
        Returns:
            torch tensor of shape (indices.shape[0],): probability of the prior distribution for given indices
        """
        return torch.ones(indices.shape[0])

    def set_params(self, **kwargs):
        """Update the parameters of the integrand."""
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")


class Integrand(ABC):
    def __init__(self, settings_file: str, **kwargs):
        Parser = SettingsParser(settings_file)
        self.settings = Parser.get_settings()
        self.graph_properties = Parser.get_graph_properties()

    @abstractmethod
    def worker(self, **kwargs) -> None:
        pass


class DiscreteLayer(ABC):
    pass
