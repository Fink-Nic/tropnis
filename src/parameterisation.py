# type: ignore
import torch
import numpy as np
from time import perf_counter
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Sequence

import momtrop
from madnis.integrator import Integrand as MadnisIntegrand


@dataclass
class GraphProperties:
    edge_src_dst_vertices: list[(int, int)]
    edge_masses: list[int]
    edge_momentum_shifts: list[list[float]]
    graph_external_vertices: list[int]
    graph_signature: list[list[int]]
    momtrop_edge_weights: list[int] = field(default_factory=list)

    def __post_init__(self):
        TOLERANCE = 1E-10
        self.n_loops: int = len(self.graph_signature[0])
        self.n_edges: int = len(self.edge_masses)
        self.edge_ismassive: list[bool] = [
            mass > TOLERANCE for mass in self.edge_masses]


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
        self.timing: Dict[str, float] = {"init": perf_counter()}
        self.num_processing_steps: int = 0

    def overwrite_x_all(self, new_x_all: torch.Tensor, identifier: str) -> None:
        """
        Should be used to update the dataclass instance with the result of the 
        current layer. Automatically sets the processing time for the layer.
        """
        self.x_all = new_x_all
        self.timing.update({identifier: perf_counter()})
        self.num_processing_steps += 1

    def get_processing_times(self, mu_s_per_point: bool = False) -> Dict[str, float]:
        """
        Args:
            mu_s_per_point: If True, will return processing times in microseconds per 
            evaluated point.
        Returns:
            Processing time spent in each layer, in seconds.
        """
        self.timing: Dict[str, float]
        last_timestamp = self.timing['init']
        processing_times: Dict = {}
        total_time = 0.0
        for identifier in self.timing.keys():
            if identifier in ['init', 'total']:
                continue
            curr_timestamp = self.timing[identifier]
            processing_time = curr_timestamp - last_timestamp
            if mu_s_per_point:
                processing_time /= len(self.x_all) * 1E6
            processing_times[identifier] = processing_time
            total_time += processing_time
            last_timestamp = curr_timestamp
        processing_times['total'] = total_time

        return processing_times

    @staticmethod
    def join(instances: Sequence['LayerOutput'], n_cores: int = 1):
        if len(instances) < 1:
            return []

        joined_x_all = torch.vstack([i.x_all for i in instances])
        joined_timings = [i.get_processing_times() for i in instances]
        joined_outputs = LayerOutput(joined_x_all)
        last_timestamp = joined_outputs.timing['init']
        for identifier in joined_timings[0].keys():
            curr_p_time = sum(
                [t[identifier] for t in joined_timings]) / n_cores
            joined_outputs.timing[identifier] = last_timestamp + curr_p_time
            last_timestamp += curr_p_time

        return joined_outputs


class Parameterisation(ABC):
    N_SPATIAL_DIMS = 3
    DIM_FROM_JACOBIAN = 1
    identifier = "ABCParameterisation"

    def __init__(self,
                 graph_properties: GraphProperties,
                 identifier: Optional[str] = None,
                 next_param: Optional['Parameterisation'] = None,
                 is_first_layer: bool = False,
                 ):
        if identifier is not None:
            self.identifier = identifier
        self.graph_properties = graph_properties
        self.next_param = next_param
        self.is_first_layer = is_first_layer

        self.layer_continuous_dim_in = self._layer_continuous_dim_in()
        self.layer_continuous_dim_out = self._layer_continuous_dim_out()
        self.layer_discrete_dims = self._layer_discrete_dims()
        self.layer_num_discrete_dims = len(self.layer_discrete_dims)

        self.chain_continuous_dim_in = self.get_chain_continuous_dim()
        self.chain_discrete_dims = self.get_chain_discrete_dims()

    @abstractmethod
    def _layer_parameterise(self, continuous: torch.Tensor, discrete: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            continuous: continuous parameters
            discrete: discrete parameters
        Returns:
            jacobians, parameterised continuous output 
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
        layer_output = self._to_layer_output(layer_input, *param_output)

        if self.next_param is None:
            return layer_output

        return self.next_param.parameterise(layer_output)

    def discrete_prior_prob_function(self, indices: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Args:
            indices: indices of the discrete channel
            dim: current index on dim=1 of generated indices
        Returns:
            torch tensor of shape (indices.shape[0], self.layer_num_discrete_dims): 
            probability of the prior distribution for given indices.
            Default is flat probability distribution, 
            zero if indices.shape[0] = self.layer_num_discrete_dims
        """
        if dim < self.layer_num_discrete_dims or self.next_param is None:
            return self._layer_prior_prob_function(indices.long())

        indices = indices[:, self.layer_num_discrete_dims:]
        dim -= self.layer_num_discrete_dims

        return self.next_param.discrete_prior_prob_function(indices, dim)

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

        layer_dim = self.layer_continuous_dim_in - self.layer_continuous_dim_out

        return layer_dim + self.next_param.get_chain_continuous_dim()

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
            x_all = input.x_all
        else:
            x_all = input.x_all[:, self.DIM_FROM_JACOBIAN:]

        n_dim = x_all.shape[1]
        if n_dim < self.chain_continuous_dim_in:
            raise ValueError(f"Layer {self.identifier} has received {n_dim}-dimensional input, "
                             + f"expected at least {self.chain_continuous_dim_in}.")

        continuous, discrete = x_all.tensor_split(
            [self.chain_continuous_dim_in], dim=1)
        continuous = continuous[:, :self.layer_continuous_dim_in]

        if discrete.shape[1] > self.layer_num_discrete_dims:
            discrete = discrete[:, :self.layer_num_discrete_dims]

        return continuous, discrete.long()

    def _to_layer_output(self, layer_input: LayerOutput,
                         jacobians_param: torch.Tensor,
                         x_all_param: torch.Tensor) -> LayerOutput:
        """
        Returns the output of the current layer in a form that respects the structure of the chain,
        in order to be passed down the chain.

        Args:
            layer_input: LayerOutput of the previous layer
            *param_output: Output generated by the parameterisation step of the current layer
        Returns:
            LayerOutput
        """
        # Data from parameterisation
        jacobians = jacobians_param
        continuous_param, discrete_param = x_all_param.tensor_split(
            [self.layer_continuous_dim_out], dim=1)

        # Data from previous layer
        x_all = layer_input.x_all
        if not self.is_first_layer:
            jacobians *= x_all[:, 0].reshape(-1, 1)
            x_all = x_all[:, self.DIM_FROM_JACOBIAN:]

        # Pass along additional continuous dimensions that are required down the chain
        _, pass_continuous_in, discrete_in = x_all.tensor_split(
            [self.layer_continuous_dim_in, self.chain_continuous_dim_in], dim=1)

        # Combine the data from the previous layer and parameterisation
        continuous = torch.hstack([continuous_param, pass_continuous_in])
        if discrete_in.shape[1] > self.layer_num_discrete_dims:
            discrete = torch.hstack(
                [discrete_in[:, self.layer_num_discrete_dims:], discrete_param])
        else:
            discrete = discrete_param

        x_all = torch.hstack([jacobians, continuous, discrete])
        layer_input.overwrite_x_all(x_all, self.identifier)

        return layer_input

    def _layer_prior_prob_function(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.shape[1] == self.layer_num_discrete_dims:
            return torch.zeros_like(indices)

        norm_factor = self.layer_num_discrete_dims - indices.shape[1]
        prior = torch.ones((len(indices), self.layer_num_discrete_dims))
        if indices.shape[1] == 0:
            return prior / norm_factor

        rows = torch.arange(len(indices)).unsqueeze(1)
        prior[rows, indices] = 0

        return prior / norm_factor

    def _layer_discrete_dims(self) -> List[int]:
        """
        Returns:
            List of shape of the discrete dimensions of this layer in the parameterisation chain.
            Intended to be used to initialize the value for self.discrete_dims
        """
        return []

    def _layer_continuous_dim_in(self) -> int:
        """
        Intended to be used to initialize the value for self.layer_continuous_dim_in.

        Returns:
            The continuous dimension of the input of this layer in the parameterisation chain.
        """
        return self.N_SPATIAL_DIMS*self.graph_properties.n_loops

    def _layer_continuous_dim_out(self) -> int:
        """
        Intended to be used to initialize the value for self.layer_continuous_dim_out.

        Returns:
            The continuous dimension of the output of this layer in the parameterisation chain.
        """
        return self.N_SPATIAL_DIMS*self.graph_properties.n_loops

    def set_params(self, **kwargs: Dict):
        """Update the parameters of the parameterisation."""
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")


class LayeredParameterisation:
    identifier = "layered parameterisation"

    def __init__(self, graph_properties: GraphProperties,
                 param_settings: Dict[str, Dict],):
        param_layers: List[Parameterisation] = []
        num_layers = len(param_settings)
        for i_layer, (_, kwargs) in enumerate(param_settings.items()):
            if not "param_type" in kwargs.keys():
                raise KeyError(
                    "Each parameterisation layer must specify its parameterisation type.")

            param_type: str = kwargs.pop("param_type")
            if not "identifier" in kwargs.keys():
                kwargs.update({"identifier": param_type})
            is_first_layer = (i_layer + 1) == num_layers
            next_param = None if i_layer == 0 else param_layers[-1]
            kwargs.update({"is_first_layer": is_first_layer,
                           "next_param": next_param,
                           "graph_properties": graph_properties, })

            match param_type.lower():
                case "momtrop":
                    p = MomtropParameterisation(**kwargs)
                case "spherical":
                    p = SphericalParameterisation(**kwargs)
                case "inverse spherical":
                    p = InverseSphericalParameterisation(**kwargs)
                case "kaapo":
                    p = KaapoParameterisation(**kwargs)
                case _:
                    raise NotImplementedError(
                        f"Parameterisation {param_type} has not been implemented.")
            param_layers.append(p)

        self.param = param_layers[-1]
        self.continuous_dim = self.param.chain_continuous_dim_in
        self.discrete_dims = self.param.chain_discrete_dims

    def parameterise(self, layer_input: LayerOutput) -> LayerOutput:
        return self.param.parameterise(layer_input)

    def discrete_prior_prob_function(self, indices: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return self.param.discrete_prior_prob_function(indices, dim)


class MomtropParameterisation(Parameterisation):
    identifier = "momtrop"

    def __init__(self, **kwargs: Dict[str, any]):
        self.graph_properties: GraphProperties = kwargs["graph_properties"]
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
        super().__init__(**kwargs)

    def _layer_parameterise(self, continuous: torch.Tensor, discrete: torch.Tensor,
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        # For naive sampler implementation
        if discrete.shape[1] > 0:
            if discrete[0, 0] < 0:
                samples = self.momtrop_sampler.sample_batch(
                    continuous.tolist(), self.momtrop_edge_data, self.momtrop_sampler_settings)
                jacobians = torch.Tensor(samples.jacobians).reshape(-1, 1)
                loop_momenta = torch.Tensor(
                    samples.loop_momenta).reshape(len(continuous), -1)

                return jacobians, loop_momenta

        discrete = self._get_graph_from_edges_removed(discrete)
        samples = self.momtrop_sampler.sample_batch(
            continuous.tolist(), self.momtrop_edge_data, self.momtrop_sampler_settings, discrete)

        jacobians = torch.Tensor(samples.jacobians).reshape(-1, 1)
        loop_momenta = torch.Tensor(
            samples.loop_momenta).reshape(len(continuous), -1)

        return jacobians, loop_momenta

    def _get_graph_from_edges_removed(self, edges_removed: Optional[torch.Tensor] = None
                                      ) -> List[List[int]]:
        """
        Args:
            edges_removed: List of the edge indices that have already been forced
        Returns:
            List of shape (n_edges,) that appends the as-yet unforced edges to edges_removed
        """
        full_graph = torch.arange(self.layer_num_discrete_dims)
        if edges_removed is None:
            return [full_graph.tolist()]

        full_graph = torch.tile(full_graph, (len(edges_removed), 1))
        if edges_removed.shape[1] == 0:
            return full_graph.tolist()

        mask = torch.nn.functional.one_hot(
            edges_removed, num_classes=self.layer_num_discrete_dims)
        # Append the edges that are not in discrete, meaning where onehot is zero
        remaining_edges = full_graph[mask == 0].reshape(len(edges_removed), -1)

        return torch.hstack([edges_removed, remaining_edges]).tolist()

    """ def _layer_prior_prob_function(self, indices: torch.Tensor) -> torch.Tensor:
        rust_result = self.momtrop_sampler.predict_discrete_probs(
            indices.tolist())")

        return torch.tensor(rust_result) """

    def _layer_continuous_dim_in(self) -> int:
        return self.momtrop_sampler.get_dimension()

    def _layer_discrete_dims(self) -> List[int]:
        n_edges = self.graph_properties.n_edges
        return n_edges * [n_edges]


class SphericalParameterisation(Parameterisation):
    identifier = "spherical"

    def __init__(self,
                 conformal_scale: float,
                 origins: torch.Tensor | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.conformal_scale = conformal_scale
        self.n_loops = self.graph_properties.n_loops
        if origins is not None:
            self.origins = origins.expand((self.n_loops, self.N_SPATIAL_DIMS))
        else:
            self.origins = self.n_loops * [None]

    def _layer_parameterise(self, continuous: torch.Tensor, _: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        loop_momenta = torch.zeros_like(continuous)

        # Constant part of the jacobian
        jacobians = torch.ones((len(continuous), 1))
        jacobians *= (4*torch.pi * self.conformal_scale**3)**self.n_loops

        for i_loop in range(self.n_loops):
            origin = self.origins[i_loop]
            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)

            xs = continuous[:, _start: _end]
            x, y, z = xs.tensor_split([1, 2], dim=1)

            r = x/(1-x)
            cos_az = (2*y-1)
            sin_az = (torch.ones_like(x) - cos_az**2).sqrt()
            pol = 2*torch.pi*z

            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)
            ks = self.conformal_scale*r * \
                torch.hstack(
                    [sin_az * pol.cos(), sin_az * pol.sin(), cos_az])
            if origin is not None:
                ks -= origin
            loop_momenta[:, _start: _end] = ks
            # Calculate the jacobian determinant
            jacobians *= x**2 / (1 - x)**4

        return jacobians, loop_momenta


class InverseSphericalParameterisation(Parameterisation):
    identifier = "inverse spherical"

    def __init__(self,
                 conformal_scale: float,
                 origins: Optional[torch.Tensor] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.conformal_scale = conformal_scale
        self.n_loops = self.graph_properties.n_loops
        if origins is not None:
            self.origins = origins.expand((self.n_loops, self.N_SPATIAL_DIMS))
        else:
            self.origins = self.n_loops * [None]

    def _layer_parameterise(self, continuous: torch.Tensor, _: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = torch.zeros_like(continuous)

        # Constant part of the jacobian
        jacobians = torch.ones((len(continuous), 1))
        jacobians /= (4*torch.pi * self.conformal_scale**3)**self.n_loops

        for i_loop in range(self.n_loops):
            origin = self.origins[i_loop]
            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)
            ks = continuous[:, _start: _end]
            if origin is not None:
                ks += origin

            k0, k1, k2 = ks.tensor_split([1, 2], dim=1)

            r = ks.norm(dim=1, p=2).reshape(-1, 1)
            cos_az = k2 / r
            tan_pol = (k1 / k0).reshape(-1, 1)
            pol: torch.Tensor = tan_pol.arctan()
            # Accounting for missing quadrants of arctan
            pol += torch.pi*(1 - k0.sgn())/2 * k1.sgn()
            pol += torch.pi*(1 - pol.sgn())

            r /= self.conformal_scale
            x = r / (1 + r)
            y = (cos_az + 1) / 2
            z = pol / 2 / torch.pi
            xs[:, _start: _end] = torch.hstack([x, y, z])

            # Calculate the jacobian determinant
            jacobians /= x**2 / (1 - x)**4

        return jacobians, xs


class KaapoParameterisation(Parameterisation):
    identifier = "kaapo"

    def __init__(self, mu: float = torch.pi, a: float = 0.2, b: float = 1.0, **kwargs):
        self.mu = mu
        self.a = a
        self.b = b
        super().__init__(**kwargs)

    def _layer_parameterise(self, continuous: torch.Tensor, _: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_loops = self.graph_properties.n_loops

        # Massless propagators (for now)
        p_F = self.mu

        loop_momenta = torch.zeros_like(continuous)

        # The constant part of the jacobian
        jacobians = torch.ones((len(continuous), 1))
        jacobians *= (4 * torch.pi / self.a / self.b**self.a)**n_loops

        for i_loop in range(n_loops):
            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)

            xs = continuous[:, _start: _end]
            x, y, z = xs.tensor_split([1, 2], dim=1)

            cos_az = (2*y-1)
            sin_az = (torch.ones_like(x) - cos_az.pow(2)).sqrt()
            pol = 2*torch.pi*z

            peak_F: torch.Tensor = self.b**self.a * x / (1 - x) - p_F**self.a
            h = p_F + peak_F.sgn() * peak_F.abs().pow(1 / self.a)
            ks = h * torch.hstack(
                [sin_az * pol.cos(), sin_az * pol.sin(), cos_az])
            loop_momenta[:, _start: _end] = ks

            jacobians *= h.pow(2) * (h - p_F).abs().pow(1 - self.a) \
                * (peak_F.sgn() * (h - p_F).abs().pow(self.a) + p_F**self.a + self.b**self.a).pow(2)

        return jacobians, loop_momenta
