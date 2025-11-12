# type: ignore
import time
import torch
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import momtrop
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
#########################################
INVERSE SPHERICAL PARAM or BACKWARD PARAM
args: jacobians, loop_momenta, x_all, kwargs
consumes: Nothing
outputs: jacobians, x_points
passes along: remaining x_all, kwargs
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
##################################
KAAPOS MODIFIED SPHERICAL PARAM or FORWARD PARAM
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
class LayerOutput:
    x_all: torch.Tensor

    def __post_init__(self):
        self.timing = {"init": time.time()}

    def overwrite_x_all(self, new_x_all: torch.Tensor, identifier: str) -> None:
        self.x_all = new_x_all
        self.timing.update({identifier: time.time()})


@dataclass
class GraphProperties:
    edge_src_dst_vertices: list[(int, int)]
    edge_masses: list[int]
    edge_momentum_shifts: list[list[float]]
    graph_external_vertices: list[int]
    graph_signature: list[list[int]]
    momtrop_edge_weights: list[int]

    def __post_init__(self):
        TOLERANCE = 1E-10
        self.n_loops: int = len(self.graph_signature[0])
        self.n_edges: int = len(self.edge_masses)
        self.edge_ismassive: list[bool] = [
            mass > TOLERANCE for mass in self.edge_masses]


class Parameterisation(ABC):
    n_spatial_dims = 3

    def __init__(self,
                 identifier: str,
                 graph_properties: GraphProperties,
                 next_param: Parameterisation | None = None,
                 is_first_layer: bool = False
                 ):
        self.identifier = identifier
        self.graph_properties = graph_properties
        self.next_param: Parameterisation = next_param
        self.is_first_layer = is_first_layer

        self.layer_continuous_dim_in = self._layer_continuous_dim_in()
        self.layer_continuous_dim_out = self._layer_continuous_dim_out()
        self.layer_continuous_dim_consumed = self.layer_continuous_dim_in - \
            self.layer_continuous_dim_out
        self.layer_discrete_dims = self._layer_discrete_dims()
        self.layer_num_discrete_dims = len(self.layer_discrete_dims)

        self.chain_continuous_dim_in = self.get_chain_continuous_dim()
        self.chain_continuous_dim_out = self.chain_continuous_dim_in - \
            self.layer_continuous_dim_consumed
        self.chain_discrete_dims = self.get_chain_discrete_dims()

    @abstractmethod
    def _layer_parameterise(self, jacobians: torch.Tensor,
                            continuous: torch.Tensor,
                            discrete: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _layer_prior_prob_function(self, indices: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.ones(len(indices))

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
        pass

    @abstractmethod
    def _layer_continuous_dim_out(self) -> int:
        """
        Returns:
            The continuous dimension of the output of this layer in the parameterisation chain.
            Intended to be used to initialize the value for self.continuous_dim_in
        """
        pass

    def parameterise(self, layer_input: LayerOutput) -> LayerOutput:
        """
        Args:
            LayerOutput: Output of the previous Parameterisation layer (or sampler)
        Returns:
            LayerOutput: Parameterisation and data to be passed along the chain
        Raises:
            ValueError
        """
        param_input = self._to_layer_input(layer_input)
        param_output = self._layer_parameterise(*param_input)
        layer_output = self._to_layer_output(layer_input, param_output)

        if self.next_param is None:
            return layer_output

        return self.next_param.parameterise(layer_output)

    def get_chain_discrete_dims(self) -> List[int]:
        """
        Returns:
            List of shape of the discrete dimensions of this and following layers in the 
            parameterisation chain.
            Intended to be used to set the discrete dimensions for an Integrator.
        """
        if self.next_param is None:
            return self.layer_discrete_dims
        return self.layer_discrete_dims + self.next_param.get_chain_discrete_dims()

    def get_chain_continuous_dim(self) -> int:
        """
        Returns:
            The continuous dimension of this and following layers in the parameterisation chain.
            Intended to be used to set the number of continuous dimensions for an Integrator.
        """
        if self.next_param is None:
            return self.layer_continuous_dim_in

        return self.layer_continuous_dim_consumed + self.next_param.get_chain_continuous_dim()

    def discrete_prior_prob_function(self, indices: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Args:
            indices: indices of the discrete channel
            dim: current layer of generated indices
        Returns:
            torch tensor of shape (indices.shape[0],): probability of the prior distribution for given indices
        """
        layer_result = self._layer_prior_prob_function(indices, dim)
        if self.next_param is None:
            return layer_result

        return layer_result * self.next_param.discrete_prior_prob_function(indices, dim)

    def _to_layer_input(self, input: LayerOutput
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the part of the input data that is relevant to the current layer, respecting
        the structure of the chain.

        Args:
            input: LayerOutput of the previous layer
        Returns:
            Jacobians: 1-dim torch.Tensor
            Continuous Samples: torch.Tensor of shape(n_samples, self.continuous_dim)
            Discrete Samples: torch.Tensor of shape(n_samples, <=len(self.discrete_dims))
        Raises:
            ValueError
        """
        if self.is_first_layer:
            jacobians = torch.ones(len(input.x_all))
            x_all = input.x_all
        else:
            jacobians, x_all = input.x_all.tensor_split([1], axis=1)

        n_dim = x_all.shape[1]
        if n_dim < self.chain_continuous_dim_in:
            raise ValueError(f"Layer {self.identifier} has received {n_dim}-dimensional input, "
                             + f"expected at least {self.chain_continuous_dim_in}.")

        continuous, discrete = x_all.tensor_split(
            [self.chain_continuous_dim_in], axis=1)
        discrete_dim = discrete.shape[1]

        if discrete_dim > self.layer_num_discrete_dims:
            discrete = discrete[:, :self.layer_num_discrete_dims]

        return jacobians, continuous, discrete

    def _to_layer_output(self, layer_input: LayerOutput, param_output: torch.Tensor) -> LayerOutput:
        """
        Returns the output of the current layer in a form that respects the structure of the chain,
        in order to be passed down the chain.

        Args:
            layer_input: LayerOutput of the previous layer
            param_output: LayerOutput generated by the parameterisation step of the current layer
        Returns:
            LayerOutput
        """
        discrete_in = layer_input.x_all[:, self.chain_continuous_dim_in:]
        continuous, discrete_param = param_output.tensor_split(
            [self.chain_continuous_dim_out], axis=1)
        if discrete_in.shape[1] > self.layer_num_discrete_dims:
            discrete = torch.hstack(
                [discrete_in[:, self.layer_num_discrete_dims:], discrete_param])
        else:
            discrete = discrete_param

        x_all = torch.hstack([continuous, discrete])

        return layer_input.overwrite_x_all(x_all, self.identifier)

    def set_params(self, **kwargs):
        """Update the parameters of the parameterisation."""
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")


class MomtropParameterisation(Parameterisation):
    def __init__(self,
                 identifier: str,
                 graph_properties: GraphProperties,
                 next_param: Parameterisation | None = None,
                 ):
        super().__init__(identifier, graph_properties, next_param)
        mt_edges = [
            momtrop.Edge(src_dst, ismassive, weight) for src_dst, ismassive, weight
            in zip(self.graph_properties.edge_src_dst_vertices,
                   self.graph_properties.edge_ismassive,
                   self.graph_properties.momtrop_edge_weights)
        ]
        assym_graph = momtrop.Graph(
            mt_edges, self.graph_properties.graph_external_vertices)
        momentum_shifts = [momtrop.Vector(*shift) for shift
                           in self.graph_properties.edge_momentum_shifts]
        self.momtrop_edge_data = momtrop.EdgeData(
            self.graph_properties.edge_masses, momentum_shifts)
        self.momtrop_sampler = momtrop.Sampler(
            assym_graph, self.graph_properties.graph_signature)
        self.momtrop_sampler_settings = momtrop.Settings(False, False)

    def _layer_parameterise(self, jacobians: torch.Tensor,
                            continuous: torch.Tensor,
                            discrete: torch.Tensor) -> LayerOutput:

        if discrete.shape[1] == 0:
            full_graph = torch.arange(self.graph_properties.n_edges)
            discrete = torch.tile(full_graph, (len(jacobians), 1))
        else:
            discrete = [
                self._get_graph_from_edges_removed(edges_removed) for edges_removed in discrete.tolist()
            ]
        samples = self.sampler.sample_batch(
            continuous.tolist(), self.edge_data, self.sampler_settings, discrete)

        loop_momenta = torch.Tensor(
            samples.loop_momenta).reshape(len(jacobians), -1)
        jacobians = jacobians * torch.Tensor(samples.jacobians)

        return torch.hstack([jacobians, loop_momenta])

    def _get_graph_from_edges_removed(self, edges_removed: List[int] | None = None) -> List[int]:
        """
        Args:
            edges_removed: List of the edge indices that have already been forced
        Returns:
            List of shape (n_edges,) that appends the as-yet unforced edges to edges_removed
        """
        full_graph = list(range(self.graph_properties.n_edges))
        if edges_removed is None:
            return full_graph

        for edge in edges_removed:
            full_graph[edge] = None

        return edges_removed + [edge for edge in full_graph if edge is not None]

    def _layer_prior_prob_function(self, indices: torch.Tensor, _: int = 0) -> torch.Tensor:
        return torch.tensor(self.sampler.predict_discrete_probs(indices.tolist()))

    def _layer_continuous_dim_in(self) -> int:
        return self.momtrop_sampler.get_dimension()

    def _layer_continuous_dim_out(self) -> int:
        return self.n_spatial_dims * self.graph_properties.n_loops

    def _layer_discrete_dims(self) -> List[int]:
        n_edges = self.graph_properties.n_edges
        return n_edges * [n_edges]
