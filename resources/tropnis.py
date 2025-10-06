# type: ignore
from dataclasses import dataclass
import math
import momtrop
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydot
import numpy as np

from symbolica import E, S, Expression
from madnis.nn import Flow


class MaskedMLP(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        output_dims: list[int],
        layers: int = 3,
        nodes_per_feature: int = 8,
        activation=nn.LeakyReLU,
    ):
        super().__init__()

        in_degrees = []
        for i, in_dim in enumerate(input_dims):
            in_degrees.extend([i] * in_dim)
        hidden_degrees = torch.repeat_interleave(
            torch.arange(len(input_dims)), nodes_per_feature)
        out_degrees = []
        for i, out_dim in enumerate(output_dims):
            out_degrees.extend([i] * out_dim)
        hidden_layers = layers - 1
        layer_degrees = [
            torch.tensor(in_degrees), *([hidden_degrees]
                                        * hidden_layers), torch.tensor(out_degrees)
        ]

        self.in_slices = [[slice(0)] * layers]
        self.out_slices = [[slice(0)] * layers]
        hidden_dims = [nodes_per_feature] * hidden_layers
        for in_dim, out_dim in zip(input_dims, output_dims):
            self.in_slices.append([
                slice(0, prev_slice_in.stop + deg_in)
                for deg_in, prev_slice_in in zip([in_dim, *hidden_dims], self.in_slices[-1])
            ])
            self.out_slices.append([
                slice(prev_slice_out.stop, prev_slice_out.stop + deg_out)
                for deg_out, prev_slice_out in zip([*hidden_dims, out_dim], self.out_slices[-1])
            ])
        self.in_slices.pop(0)
        self.out_slices.pop(0)

        self.masks = nn.ParameterList()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for deg_in, deg_out in zip(layer_degrees[:-1], layer_degrees[1:]):
            self.masks.append(
                nn.Parameter(
                    (deg_out[:, None] >= deg_in[None, :]).float(), requires_grad=False)
            )
            self.weights.append(nn.Parameter(
                torch.empty((len(deg_out), len(deg_in)))))
            self.biases.append(nn.Parameter(torch.empty((len(deg_out),))))

        self.activation = activation()
        self.reset_parameters()

    def reset_parameters(self):
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(bias, -bound, bound)
        nn.init.zeros_(self.weights[-1])
        nn.init.zeros_(self.biases[-1])

    def forward(self, x: torch.Tensor):
        for weight, bias, mask in zip(self.weights[:-1], self.biases[:-1], self.masks[:-1]):
            x = self.activation(F.linear(x, mask * weight, bias))
        return F.linear(x, self.masks[-1] * self.weights[-1], self.biases[-1])

    def forward_cached(
        self, x: torch.Tensor, feature: int, cache: list[torch.Tensor] | None = None
    ):
        if cache is None:
            cache = [None] * len(self.weights)
        new_cache = []
        in_slices = self.in_slices[feature]
        out_slices = self.out_slices[feature]
        first = True
        for weight, bias, in_slice, out_slice, x_cached in zip(
            self.weights, self.biases, in_slices, out_slices, cache
        ):
            if first:
                first = False
            else:
                x = self.activation(x)
            if x_cached is not None:
                x = torch.cat((x_cached, x), dim=1)
            new_cache.append(x)
            x = F.linear(x, weight[out_slice, in_slice], bias[out_slice])
        return x, new_cache


Cache = tuple[torch.Tensor, list[torch.Tensor] | None]


class TropicalFlow(nn.Module):
    def __init__(
        self,
        continuous_dim: int,
        discrete_dims: list[int],
        conditional_dim: int,
        continuous_kwargs: dict,
        discrete_kwargs: dict,
    ):
        super().__init__()
        self.flow = Flow(
            dims_in=continuous_dim,
            dims_c=sum(discrete_dims) + conditional_dim,
            **continuous_kwargs
        )
        self.masked_net = MaskedMLP(
            input_dims=[conditional_dim, *discrete_dims[:-1]],
            output_dims=discrete_dims,
            **discrete_kwargs
        )
        self.discrete_dims = discrete_dims
        self.max_dim = max(discrete_dims)
        discrete_indices = []
        one_hot_mask = []
        for i, dim in enumerate(discrete_dims):
            discrete_indices.extend([i] * dim)
            one_hot_mask.extend([True] * dim + [False] * (self.max_dim - dim))
        self.register_buffer("discrete_indices",
                             torch.tensor(discrete_indices))
        self.register_buffer("one_hot_mask", torch.tensor(one_hot_mask))

    def log_prob(
        self,
        indices: torch.Tensor,
        x: torch.Tensor,
        discrete_probs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_discrete = F.one_hot(
            indices, self.max_dim
        ).to(x.dtype).flatten(start_dim=1)[:, self.one_hot_mask]
        if condition is None:
            input_disc = x_discrete
        else:
            input_disc = torch.cat((condition, x_discrete), dim=1)
        unnorm_prob_disc = self.masked_net(
            input_disc[:, :-self.discrete_dims[-1]]
        ).exp() * discrete_probs
        prob_norms = torch.zeros_like(indices, dtype=x.dtype).scatter_add_(
            1, self.discrete_indices[None, :].expand(
                x.shape[0], -1), unnorm_prob_disc
        )
        prob_sums = torch.zeros_like(prob_norms).scatter_add_(
            1, self.discrete_indices[None, :].expand(
                x.shape[0], -1), unnorm_prob_disc * x_discrete
        )
        prob_disc = torch.prod(prob_sums / prob_norms, dim=1)
        log_prob_cont = self.flow.log_prob(x, c=input_disc)
        return prob_disc.log() + log_prob_cont

    def init_cache(self, n: int, condition: torch.Tensor | None = None) -> Cache:
        return (torch.zeros((n, 0)) if condition is None else condition, torch.ones((n, )), None)

    def sample_discrete(
        self, dim: int, pred_probs: torch.Tensor, cache: Cache
    ) -> tuple[torch.Tensor, torch.Tensor, Cache]:
        x, prob, net_cache = cache
        y, net_cache = self.masked_net.forward_cached(x, dim, net_cache)
        unnorm_probs = y.exp() * pred_probs
        cdf = unnorm_probs.cumsum(dim=1)
        norm = cdf[:, -1]
        cdf = cdf / norm[:, None]
        r = torch.rand((y.shape[0], 1))
        samples = torch.searchsorted(cdf, r)[:, 0]
        prob = torch.gather(unnorm_probs, 1, samples[:, None])[
            :, 0] / norm * prob
        x_one_hot = F.one_hot(samples, self.discrete_dims[dim]).to(y.dtype)
        return samples, (x_one_hot, prob, net_cache)

    def sample_continuous(self, cache: Cache) -> tuple[torch.Tensor, torch.Tensor]:
        x, prob, net_cache = cache
        condition = torch.cat((net_cache[0], x), dim=1)
        flow_samples, flow_log_prob = self.flow.sample(
            c=condition, return_log_prob=True)
        return flow_samples, prob.log() + flow_log_prob


@dataclass
class SampleBatch:
    x: torch.Tensor
    indices: torch.Tensor
    prob: torch.Tensor
    discrete_probs: torch.Tensor
    func_val: torch.Tensor


@dataclass
class MomtropSamplerProperties:
    edge_weights: list[int]
    edge_masses: list[int]
    edges: list[momtrop.Edge]
    external_vertices: list[int]
    signature: list[list[int]]
    momentum_shifts: list[momtrop.Vector]


class TriangleIntegrand:
    continuous_dim = 7
    discrete_dims = [3, 3]
    mt_weight = 0.7

    def __init__(self, integrand, m_psi: float = 0.02, p: list[float] = [0.005, 0., 0., 0.005], q: list[float] = [0.005, 0., 0., -0.005],
                 weight: float = 0.5):
        isMassive = m_psi > 1e-10
        self.m_psi = m_psi
        self.p = p
        self.q = q
        self.weight = weight
        self.integrand = integrand
        edge_1 = momtrop.Edge((0, 1), isMassive, self.mt_weight)
        edge_2 = momtrop.Edge((1, 2), isMassive, self.mt_weight)
        edge_3 = momtrop.Edge((2, 0), isMassive, self.mt_weight)

        assym_graph = momtrop.Graph([edge_1, edge_2, edge_3], [0, 1, 2])
        signature = [[1], [1], [1]]

        self.sampler = momtrop.Sampler(assym_graph, signature)
        self.edge_data = momtrop.EdgeData(
            [m_psi, m_psi, m_psi],
            [momtrop.Vector(0., 0., 0.),
             momtrop.Vector(-q[1], -q[2], -q[3]),
             momtrop.Vector(p[1], p[2], p[3])
             ])
        self.settings = momtrop.Settings(False, False)
        self.continuous_dim = self.sampler.get_dimension()

    def predict_discrete_probs(self, dim: int, indices: torch.Tensor) -> torch.Tensor:
        rust_result = self.sampler.predict_discrete_probs(indices.tolist())

        return torch.tensor(rust_result)

    def get_subgraph_from_edges_removed(self, edges_removed):
        result = []

        for edge in [0, 1, 2]:
            if edge not in edges_removed:
                result.append(edge)

        return result

    def momtrop_parameterize(self, xs: list[float], force_sector: list[float] | None = None) -> tuple[list[float], float]:
        sample = self.sampler.sample_point(
            xs, self.edge_data, self.settings, force_sector)
        k = sample.loop_momenta[0].to_list()

        return k, sample.jacobian

    def integrate_point(self, xs: list[float], indices: list[int] | None = None) -> float:
        force_sector = indices + self.get_subgraph_from_edges_removed(indices)
        k, jac = self.momtrop_parameterize(xs, force_sector)
        jac *= self.sampler.get_sector_prob(force_sector)

        return self.integrand(self.m_psi, k, self.q, self.p, self.weight-self.mt_weight)*jac

    def __call__(self, indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        result = list(map(self.integrate_point, x.tolist(), indices.tolist()))

        return torch.tensor(result)


def test_numerator(loop_momenta: np.ndarray) -> np.ndarray:
    return np.ones_like(loop_momenta, shape=loop_momenta.shape[0])


class MomtropIntegrand:
    def __init__(self, dot_file: str, numerator: callable):
        self.sampler, self.sampler_properties = self._generate_momtrop_sampler_from_dot_file(
            dot_file)
        self.edge_data = momtrop.EdgeData(
            self.sampler_properties.edge_masses,
            self.sampler_properties.momentum_shifts)
        self.settings = momtrop.Settings(False, False)
        self.continuous_dim = self.sampler.get_dimension()
        self.discrete_dims = self.get_discrete_dims()
        self.numerator = numerator

    def _generate_momtrop_sampler_from_dot_file(self, dot_file: str):
        # Scalar particle properties
        scalars = {
            "scalar_0": {"mass": 0.},
            "scalar_1": {"mass": 0.02},
            "scalar_2": {"mass": 2.}
        }
        # External momenta
        ext_momenta = [
            [0.005, 0.0, 0.0, 0.005],
            [0.005, 0.0, 0.0, -0.005],
            [0.09903067841547669,
             0.03734458315384239,
             0.06393671561700484,
             -0.0657613394982612]
        ]
        # Import the dot graph
        graph = pydot.graph_from_dot_file(dot_file)[0]
        edges = graph.get_edges()
        vertices = graph.get_nodes()

        VERTICES = []
        LMB_EDGES = []
        EXT_VERTICES = []
        INT_EDGES = []

        # Filter out the external vertices
        for vert in vertices:
            if vert.get("num") is not None:
                VERTICES.append(vert)

        # Add vertex ID for momtrop
        for v_id, vert in enumerate(VERTICES):
            vert.set("v_id", v_id)

        # Filter edges and add additional attributes for momtrop
        for edge in edges:
            edge.set("src", edge.get_source().split(':')[0])
            edge.set("dst", edge.get_destination().split(':')[0])

            if edge.get("lmb_id") is not None:
                LMB_EDGES.append(edge)

            if edge.get("source") is None:
                EXT_VERTICES.append(graph.get_node(edge.get("dst"))[0])
            elif edge.get("sink") is None:
                EXT_VERTICES.append(graph.get_node(edge.get("src"))[0])
            else:
                INT_EDGES.append(edge)
                particle = edge.get("particle")[1:-1]
                edge.set("mass", scalars.get(particle).get("mass"))
                src_vert = graph.get_node(edge.get("src"))[0]
                dst_vert = graph.get_node(edge.get("dst"))[0]
                edge.set("src_id", src_vert.get("v_id"))
                edge.set("dst_id", dst_vert.get("v_id"))

        # Symbolica setup for LMB representation parsing
        # P: External momenta
        # K: Internal momenta
        # x_, a_: wildcards
        P, K = S('P', 'K')
        x_, a_ = S('x_', 'a_')

        # Set up momtrop sampler
        TOLERANCE = 1E-10
        n_loops = len(LMB_EDGES)
        n_int = len(INT_EDGES)
        n_ext = len(edges) - n_int
        mt_weight = (3*n_loops + 3/2)/n_int/2

        mt_edges = []
        mt_masses = []
        mt_signature = []
        mt_offsets = []
        mt_externals = sorted([v.get("v_id") for v in EXT_VERTICES])

        for edge in INT_EDGES:
            # Generate the momtrop edge
            src_id = edge.get("src_id")
            dst_id = edge.get("dst_id")
            mass = edge.get("mass")
            mt_edges.append(
                momtrop.Edge((src_id, dst_id), mass > TOLERANCE, mt_weight)
            )
            mt_masses.append(mass)

            # LMB representation parsing
            e: Expression = E(edge.get("lmb_rep")[1:-1])
            e = e.replace(P(x_, a_), P(x_-1))
            e = e.replace(K(x_, a_), K(x_))
            lmb_sig = [int(e.coefficient(K(lmb_id)).to_sympy())
                       for lmb_id in range(n_loops)]
            mt_signature.append(lmb_sig)

            offset_sig = [float(e.coefficient(P(ext_id)).to_sympy())
                          for ext_id in range(n_ext - 1)]
            offset = [0. for _ in range(3)]
            for coeff, ext_mom in zip(offset_sig, ext_momenta):
                for i in range(3):
                    offset[i] += coeff*ext_mom[i+1]

            mt_offsets.append(momtrop.Vector(*offset))

        assym_graph = momtrop.Graph(mt_edges, mt_externals)
        sampler = momtrop.Sampler(assym_graph, mt_signature)

        sampler_properties = MomtropSamplerProperties(
            edge_weights=mt_weight,
            edge_masses=mt_masses,
            edges=mt_edges,
            external_vertices=mt_externals,
            signature=mt_signature,
            momentum_shifts=mt_offsets,
        )

        return sampler, sampler_properties

    def get_discrete_dims(self) -> list[int]:
        # TODO: Implement for arbitrary diagrams
        num_edges = len(self.sampler_properties.edges)
        return num_edges*[num_edges]

    def predict_discrete_probs(self, dim: int, indices: torch.Tensor) -> torch.Tensor:
        rust_result = self.sampler.predict_discrete_probs(indices.tolist())

        return torch.tensor(rust_result)

    def get_subgraph_from_edges_removed(self, edges_removed: list[int]) -> list[int]:
        # TODO: implement this for arbitrary diagrams
        edges = list(range(len(self.sampler_properties.edges)))
        result = []

        for edge in edges:
            if edge not in edges_removed:
                result.append(edge)

        return result

    def momtrop_parameterize_batch(self, xs: list[list[float]], force_sector: list[list[int]] | None = None
                                   ) -> tuple[np.ndarray, np.ndarray]:
        samples = self.sampler.sample_batch(
            xs, self.edge_data, self.settings, force_sector)

        loop_momenta = np.array(samples.loop_momenta).reshape(-1, 3)
        jacobians = np.array(samples.jacobians)

        return loop_momenta, jacobians

    def integrate_batch(self, xs: list[list[float]], indices: list[list[int]] | None = None
                        ) -> np.ndarray:
        force_sector = [
            ind + self.get_subgraph_from_edges_removed(ind) for ind in indices]
        loop_momenta, jacs = self.momtrop_parameterize_batch(xs, force_sector)

        return self.numerator(loop_momenta)*jacs

    def __call__(self, indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        result = self.integrate_batch(x.tolist(), indices.tolist())

        return torch.tensor(result)


class TropicalIntegrator:
    def __init__(self, integrand, lr=3e-4, batch_size=1024, continuous_kwargs={}, discrete_kwargs={}):
        self.integrand = integrand
        self.flow = TropicalFlow(
            continuous_dim=integrand.continuous_dim,
            discrete_dims=integrand.discrete_dims,
            conditional_dim=0,
            continuous_kwargs=continuous_kwargs,
            discrete_kwargs=discrete_kwargs,
        )
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr)
        self.batch_size = batch_size

    def sample(self, n: int):
        with torch.no_grad():
            discrete_count = len(self.integrand.discrete_dims)
            indices = torch.zeros((n, discrete_count), dtype=torch.int64)
            cache = self.flow.init_cache(n)
            discrete_probs = []
            for i in range(discrete_count):
                pred_probs = self.integrand.predict_discrete_probs(
                    i, indices[:, :i])
                discrete_probs.append(pred_probs)
                indices[:, i], cache = self.flow.sample_discrete(
                    i, pred_probs, cache)

            x, log_prob = self.flow.sample_continuous(cache)
            func_val = self.integrand(indices, x)
            return SampleBatch(
                x, indices, log_prob.exp(), torch.cat(discrete_probs, dim=1), func_val
            )

    def optimization_step(self, samples: SampleBatch) -> float:
        self.optimizer.zero_grad()
        log_prob = self.flow.log_prob(
            samples.indices, samples.x, samples.discrete_probs)
        loss = -torch.mean(samples.func_val.abs() / samples.prob * log_prob)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, iterations: int, n_log=100, **opts):
        loss = 0.
        for i in range(iterations):
            samples = self.sample(self.batch_size)
            loss += self.optimization_step(samples)
            if (i + 1) % n_log == 0:
                print(f"Batch {i+1}: loss={loss / n_log:.6f}")
                loss = 0.

    def integrate(self, n: int) -> tuple[float, float]:
        samples = self.sample(n)
        weights = samples.func_val / samples.prob
        integral = weights.mean().item()
        error = weights.std().item() / math.sqrt(n)
        return integral, error


def integrate_flat(integrand, n):
    indices = torch.stack(
        [torch.randint(0, dim, (n, )) for dim in integrand.discrete_dims], dim=1
    )
    prob = 1 / torch.tensor(integrand.discrete_dims).prod()
    x = torch.rand((n, integrand.continuous_dim))
    weights = integrand(indices, x) / prob
    integral = weights.mean().item()
    error = weights.std().item() / math.sqrt(n)
    return integral, error
