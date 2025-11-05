# type: ignore
from dataclasses import dataclass
from typing import Callable, Sequence, List, Tuple
import time
import math
import momtrop
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multiprocessing import Process, Queue

from madnis.nn import Flow
from madnis.integrator import Integrand as MadnisIntegrand
from resources.parser import RunCardParser


def test_integrand(loop_momenta: np.ndarray) -> np.ndarray:
    return np.ones_like(loop_momenta, shape=loop_momenta.shape[0])


def chunks(ary: Sequence, n_chunks: int) -> List[Sequence]:
    l = len(ary)
    if n_chunks > l or n_chunks < 1:
        raise ValueError("the number of chunks should be at least 1, and at most len(ary)")
    n_long = l % n_chunks
    len_long = l // n_chunks + 1
    total_long = n_long*len_long
    len_short = l // n_chunks

    long_chunks = [ ary[start:start+len_long] 
                   for start in range(0, total_long, len_long) ]
    short_chunks = [ ary[start:start+len_short] 
                    for start in range(total_long, l, len_short) ]
    
    return long_chunks + short_chunks
    

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


Cache = tuple[torch.Tensor, List[torch.Tensor] | None]


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


class MomtropIntegrand:
    def __init__(self, 
    runcard_file: str, 
    integrand: Callable[[np.ndarray], np.ndarray] | None = None, 
    n_cores: int | None = None,
    parser_verbosity: bool = False,
    ):
        self.runcard = runcard_file
        self.Parser = RunCardParser(runcard_file, parser_verbosity)
        self.sampler, self.sampler_properties = self.Parser.generate_momtrop_sampler_from_dot_file()
        self.edge_data = momtrop.EdgeData(
            self.sampler_properties.edge_masses,
            self.sampler_properties.momentum_shifts)
        self.settings = momtrop.Settings(False, False)
        self.continuous_dim = self.sampler.get_dimension()
        self.discrete_dims = self.get_discrete_dims()
        if integrand is None:
            self.integrand = self.Parser.get_gl_integrand()
        else:
            self.integrand = integrand
        if n_cores is None:
            try:
                self.n_cores = self.Parser.settings['n_cores']
            except KeyError:
                self.n_cores = 1
        else:
            self.n_cores = n_cores

    def get_discrete_dims(self) -> List[int]:
        # TODO: Implement for arbitrary diagrams
        num_edges = len(self.sampler_properties.edges)
        return num_edges*[num_edges]

    def predict_discrete_probs(self, dim: int, indices: torch.Tensor) -> torch.Tensor:
        rust_result = self.sampler.predict_discrete_probs(indices.tolist())

        return torch.tensor(rust_result)
    
    def madnis_predict_discrete_probs(self, indices: torch.Tensor, dim: int) -> torch.Tensor:

        return self.predict_discrete_probs(dim, indices)

    def get_subgraph_from_edges_removed(self, edges_removed: List[int]) -> List[int]:
        edges = list(range(self.discrete_dims[0]))

        for edge in edges_removed:
            edges[edge] = None

        return [ edge for edge in edges if edge is not None ]

    def momtrop_parameterize_batch(self, xs: List[List[float]], force_sector: List[List[int]] | None = None
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        samples = self.sampler.sample_batch(
            xs, self.edge_data, self.settings, force_sector)

        loop_momenta = np.array(samples.loop_momenta).reshape(-1, 3)
        jacobians = np.array(samples.jacobians)

        return loop_momenta, jacobians

    def integrate_batch(self, xs: torch.Tensor, 
                        indices: torch.Tensor | None = None,
                        gammaloop_state = None,
                        ) -> torch.Tensor:
        force_sector = [
            ind + self.get_subgraph_from_edges_removed(ind) for ind in indices.tolist()]
        loop_momenta, jacs = self.momtrop_parameterize_batch(xs.tolist(), force_sector)
        
        return torch.tensor(self.integrand(loop_momenta, gammaloop_state)*jacs)

    def eval_integrand(self, x_all: torch.Tensor) -> torch.Tensor:
        x = x_all[:, -self.continuous_dim:]
        indices = x_all[:, :-self.continuous_dim].long()

        return self.__call__(indices, x)
        

    def madnis_integrand(self) -> MadnisIntegrand:
        return MadnisIntegrand(
            function=self.eval_integrand,
            input_dim=self.continuous_dim + len(self.discrete_dims),
            discrete_dims=self.discrete_dims,
            discrete_prior_prob_function=self.madnis_predict_discrete_probs,
        )

    def __call__(self, indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert (ln:=len(indices)) == len(x), "indices and x must be of same length"

        n_cores = min(ln, self.n_cores)
        
        # Symbolica worker approach
        def worker(q_in: Queue, q_out: Queue, worker_id) -> None:
            gammaloop_state = self.gammaloop_states[worker_id]
            while True:
                data = q_in.get()
                if data is None:
                    break
                args_chunk, chunk_id = data
                xs, indices = args_chunk
                res = self.integrate_batch(xs, indices, gammaloop_state)
                q_out.put((res, chunk_id))
        
        MAX_CHUNK_SIZE = 1024
        chunks_per_worker = math.ceil(ln / n_cores / MAX_CHUNK_SIZE)
        n_chunks = n_cores * chunks_per_worker
        x_chunks = [ xc for xc in chunks(x, n_chunks) ]
        indices_chunks = [ ic for ic in chunks(indices, n_chunks) ]
        args_chunked = zip(x_chunks, indices_chunks)

        in_queues: List[Queue] = []
        out_queues: List[Queue] = []
        processes: List[Process] = []

        for worker_id in range(n_cores):
            q_in, q_out = Queue(), Queue()
            p = Process(target=worker, args=(q_in, q_out, worker_id))
            p.start()
            in_queues.append(q_in)
            out_queues.append(q_out)
            processes.append(p)
        
        curr_chunk_id = 0
        for q_in in in_queues:
            for _ in range(chunks_per_worker):
                q_in.put((next(args_chunked), curr_chunk_id))
                curr_chunk_id += 1

        result_sorted = [None]*n_chunks
        for q_out in out_queues:
            for _ in range(chunks_per_worker):
                res, chunk_id = q_out.get()
                result_sorted[chunk_id] = res
        
        for q_in in in_queues:
            q_in.put(None)

        for p in processes:
            p.join()

        for q in in_queues + out_queues:
            q.close()
            q.join_thread()

        return torch.hstack(result_sorted)
        
        # for q in in_queues + out_queues:
        #     q.close()
        #     q.join_thread()
        # for p in processes:
        #     p.join()
            
        # except KeyboardInterrupt:
        #     print("\n[!] Caught KeyboardInterrupt. Cleaning up...")
        #     # Kill all workers
        #     for p in processes:
        #         if p.is_alive():
        #             p.terminate()
        #     for p in processes:
        #         p.join()

        # finally:
        #     for q in in_queues + out_queues:
        #         q.close()
        #         q.join_thread()

        return torch.hstack(result_sorted)

        # cloning data in case I don't understand how racing conditions work with sequences
        """
        x_chunks = [ xc.detach().clone() for xc in chunks(x, n_cores) ]
        indices_chunks = [ ic.detach().clone() for ic in chunks(indices, n_cores) ]
        args_chunked = zip(x_chunks, indices_chunks) """
        # pass only slices of the data
        # args_chunked = zip(chunks(x, n_cores), chunks(indices, n_cores))

        # Not working chunked Process/Queue approach
        """ result_queue = Queue()

        def worker(args_chunk, chunk_index):
            result_queue.put( (chunk_index, self.integrate_batch(*args_chunk)) )

        processes: List(Process) = []
        
        for chunk_index, args_chunk in enumerate(args_chunks):
            process = Process(target=worker, args=(args_chunk, chunk_index))
            processes.append(process)
            process.start()
        
        result = [result_queue.get() for p in processes]
        result_sorted = [None] * n_cores

        for index, res in result:
            result_sorted[index] = res

        for p in processes:
            p.join()
        
        return torch.hstack(result_sorted) """

        # (Sometimes) working manager approach
        """
        with Manager() as manager:
            result_sorted = manager.list([None]*n_cores)

            def worker(args_chunk, chunk_id):
                result_sorted[chunk_id] = self.integrate_batch(*args_chunk)
                print(f"Worker {chunk_id} has finished.")
        
            processes: List(Process) = []

            for chunk_id, args_chunk in enumerate(args_chunked):
                process = Process(target=worker, args=(args_chunk, chunk_id))
                processes.append(process)
                process.start()
            
            for process in processes: List(Process):
                process.join()

            return torch.hstack(list(result_sorted))
        """


@dataclass
class SamplingTimes:
    subgraph: float
    momtrop: float
    integrand: float


@dataclass
class SampleMetrics:
    result: float
    error: float
    n_eval: int
    sampling_times: SamplingTimes


class GammaLoopWorker():
    def __init__(self, 
                 runcard: str,
                 daemon: bool = False):
        self.integrand = MomtropIntegrand(runcard)
        self.gammaloop_state = self.integrand.Parser.get_gl_state()
        self.daemon = daemon
        self.p: Process = None
    
    def worker(self, q_in: Queue, q_out: Queue):
        for data in iter(q_in.get, "STOP"):
            args_chunk, chunk_id = data
            xs, indices = args_chunk
            res = self.integrand.integrate_batch(xs, indices, self.gammaloop_state)
            q_out.put((res, chunk_id))

    def run(self, q_in: Queue, q_out: Queue):
        self.p = Process(target=self.worker, args=(q_in, q_out), daemon=self.daemon).start()


class GammaLoopIntegrand:
    def __init__(self, integrand: MomtropIntegrand, n_cores: int = 1):
        self.MAX_CHUNK_SIZE = 2048
        self.integrand = integrand
        self.n_cores = n_cores
        self.workers = [GammaLoopWorker(self.integrand.runcard) for _ in range(self.n_cores)]

    def __call__(self, indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert (ln:=len(indices)) == len(x), "indices and x must be of same length"
        n_cores = min(ln, self.n_cores)

        chunks_per_worker = math.ceil(ln / n_cores / self.MAX_CHUNK_SIZE)
        n_chunks = n_cores * chunks_per_worker
        x_chunks = [ xc for xc in chunks(x, n_chunks) ]
        indices_chunks = [ ic for ic in chunks(indices, n_chunks) ]
        args_chunked = zip(x_chunks, indices_chunks)

        q_in, q_out = Queue(), Queue()
        for worker in self.workers[:n_cores]:
            worker.run(q_in, q_out)

        for curr_chunk_id, args in enumerate(args_chunked):
            q_in.put((args, curr_chunk_id))

        result_sorted = [None]*n_chunks
        for _ in range(n_chunks):
            data = q_out.get()
            res, chunk_id = data
            result_sorted[chunk_id] = res

        for _ in range(n_cores):
            q_in.put("STOP")
        
        # for worker in self.workers[:n_cores]:
        #     worker.join()
        
        for q in (q_in, q_out):
            q.close()
            q.join_thread()
        
        return torch.hstack(result_sorted)
    
    # def __del__(self):
    #     for w in self.workers:
    #         w.join()
        
    #     for q in self.in_queues + self.out_queues:
    #         q.close()
    #         q.join_thread()


class TropicalIntegrator:
    def __init__(self, integrand: MomtropIntegrand, lr=1e-3, batch_size=1024, continuous_kwargs={}, discrete_kwargs={}):
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
        self.step = 0

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
            self.step += 1
            if (self.step) % n_log == 0:
                print(f"Batch {self.step}: loss={loss / n_log:.6g}")
                loss = 0.
        

    def integrate(self, n: int) -> tuple[float, float]:
        samples = self.sample(n)
        weights = samples.func_val / samples.prob
        integral = weights.mean().item()
        error = weights.std().item() / math.sqrt(n)
        return integral, error
