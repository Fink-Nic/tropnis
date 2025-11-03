# type: ignore
import math
import torch
import numpy as np
from time import time
from dataclasses import dataclass
from typing import Callable, List, Tuple
from multiprocessing import Process, Queue, Event

import momtrop
from resources.parser import SettingsParser, MomtropSamplerProperties
from resources.helpers import chunks
from madnis.integrator import Integrand as MadnisIntegrand


@dataclass
class SamplingTimings:
    subgraph: float
    momtrop: float
    integrand: float


@dataclass
class IntegrandSample:
    samples: torch.Tensor
    n_eval: int
    timings: SamplingTimings


def merge(sample_batches: List[IntegrandSample]) -> IntegrandSample:
    samples = []
    n_eval = 0
    subgraph_time = 0
    momtrop_time = 0
    integrand_time = 0
    for sample in sample_batches:
        samples.append(sample.samples)
        n_eval += sample.n_eval
        subgraph_time += sample.timings.subgraph
        momtrop_time += sample.timings.momtrop
        integrand_time += sample.timings.integrand

    return IntegrandSample(torch.hstack(samples), n_eval,
                           SamplingTimings(subgraph_time,
                                           momtrop_time,
                                           integrand_time))


class MomtropIntegrand:
    def __init__(self,
                 settings_file: str,
                 integrand: Callable[[np.ndarray], np.ndarray] | None = None,
                 verbose: bool = False,
                 sampler_properties: MomtropSamplerProperties | None = None,
                 ):
        self.settings_file = settings_file
        self.Parser = SettingsParser(settings_file, verbose)
        self.settings = self.Parser.settings
        self.verbose = verbose
        if sampler_properties is None:
            self.sampler, self.sampler_properties = self.Parser.generate_momtrop_sampler_from_dot_file()
        else:
            self.sampler_properties = sampler_properties
            edge_src_dst = sampler_properties.edge_src_dst
            edge_ismassive = sampler_properties.edge_ismassive
            edge_weights = sampler_properties.edge_weights
            mt_edges = [momtrop.Edge(src_dst, ismassive, weight) for src_dst, ismassive, weight
                        in zip(edge_src_dst, edge_ismassive, edge_weights)]
            assym_graph = momtrop.Graph(mt_edges, sampler_properties.graph_externals)
            self.sampler = momtrop.Sampler(assym_graph, sampler_properties.graph_signature)
            
        momentum_shifts = [momtrop.Vector(*offset) for offset 
                           in self.sampler_properties.momentum_shifts]
        self.edge_data = momtrop.EdgeData(
            self.sampler_properties.edge_masses, momentum_shifts)
        self.sampler_settings = momtrop.Settings(False, False)
        self.continuous_dim = self.sampler.get_dimension()
        self.discrete_dims = self.get_discrete_dims()
        if integrand is None:
            self.integrand = self.Parser.get_gl_integrand()
        else:
            self.integrand = integrand

    def get_discrete_dims(self) -> List[int]:
        # TODO: Implement for arbitrary diagrams
        num_edges = len(self.sampler_properties.edge_weights)
        return num_edges*[num_edges]

    def predict_discrete_probs(self, indices: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.tensor(self.sampler.predict_discrete_probs(indices.tolist()))

    def get_subgraph_from_edges_removed(self, edges_removed: List[int]) -> List[int]:
        edges = list(range(self.discrete_dims[0]))

        for edge in edges_removed:
            edges[edge] = None

        return [edge for edge in edges if edge is not None]

    def momtrop_parameterize_batch(self, xs: List[List[float]], force_sector: List[List[int]] | None = None
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        samples = self.sampler.sample_batch(
            xs, self.edge_data, self.sampler_settings, force_sector)

        loop_momenta = np.array(samples.loop_momenta).reshape(len(xs), -1)
        jacobians = np.array(samples.jacobians)

        return loop_momenta, jacobians

    def integrate_batch(self, xs: torch.Tensor,
                        indices: torch.Tensor | None = None,
                        gammaloop_state=None,
                        ) -> IntegrandSample:
        t_last = time()
        force_sector = [ind + self.get_subgraph_from_edges_removed(ind)
                        for ind in indices.tolist()]
        subgraph_time = - t_last + (t_last := time())
        loop_momenta, jacs = self.momtrop_parameterize_batch(
            xs.tolist(), force_sector)
        momtrop_time = - t_last + (t_last := time())
        samples = torch.tensor(self.integrand(
            loop_momenta, gammaloop_state)*jacs)
        integrand_time = - t_last + (t_last := time())

        return IntegrandSample(samples, samples.shape[0],
                               SamplingTimings(
                                   subgraph_time,
                                   momtrop_time,
                                   integrand_time))

    def eval_integrand(self, x_all: torch.Tensor) -> torch.Tensor:
        x = x_all[:, -self.continuous_dim:]
        indices = x_all[:, :-self.continuous_dim].long()

        return self.__call__(x, indices).samples

    def madnis_integrand(self) -> MadnisIntegrand:
        return MadnisIntegrand(
            function=self.eval_integrand,
            input_dim=self.continuous_dim + len(self.discrete_dims),
            discrete_dims=self.discrete_dims,
            discrete_prior_prob_function=self.predict_discrete_probs,
        )

    def __call__(self, x: torch.Tensor, indices: torch.Tensor) -> IntegrandSample:
        assert len(indices) == len(x), "indices and x must be of same length"

        return self.integrate_batch(x, indices)


class GammaLoopWorker:
    daemon: bool = False

    def __init__(self,
                 settings_file: str,
                 q_in: Queue,
                 q_out: Queue,
                 ):
        Process(target=self.worker, args=(
            settings_file, q_in, q_out), daemon=self.daemon).start()

    def worker(self, settings_file, q_in: Queue, q_out: Queue):
        integrand = MomtropIntegrand(settings_file)
        gammaloop_state = integrand.Parser.get_gl_state()
        q_out.put("STARTED")
        for data in iter(q_in.get, "STOP"):
            args_chunk, chunk_id = data
            xs, indices = args_chunk
            res = integrand.integrate_batch(
                xs, indices, gammaloop_state)
            q_out.put((res, chunk_id))

    # def run(self, q_in: Queue, q_out: Queue):
    #     Process(target=self.worker, args=(runcard,
    #         q_in, q_out), daemon=self.daemon).start()

def gammaloop_worker(settings_file: str, 
                     q_in: Queue, q_out: Queue, stop_event,
                     sampler_properties: MomtropSamplerProperties | None = None,) -> None:
    integrand = MomtropIntegrand(settings_file, sampler_properties=sampler_properties)
    gammaloop_state = integrand.Parser.get_gl_state()
    q_out.put("STARTED")
    while not stop_event.is_set():
        try:
            data = q_in.get(timeout=0.5)
        except:
            continue
        if data == "STOP":
            break
        args_chunk, chunk_id = data
        xs, indices = args_chunk
        res = integrand.integrate_batch(
            xs, indices, gammaloop_state)
        q_out.put((res, chunk_id))

class GammaLoopIntegrand:
    MAX_CHUNK_SIZE = 10_000
    q_in, q_out = Queue(), Queue()
    stop_event = Event()

    def __init__(self, settings_file: str, n_cores: int | None = None, verbose: bool = False):
        self.momtrop_integrand = MomtropIntegrand(settings_file, verbose)
        self.verbose = verbose
        self.settings = self.momtrop_integrand.settings
        self.sampler_properties = self.momtrop_integrand.sampler_properties
        if n_cores is None:
            self.n_cores  = self.settings['tropnis']['n_cores']
        else:
            self.n_cores = n_cores
        for _ in range(self.n_cores):
            Process(target=gammaloop_worker, 
                    args=(settings_file, self.q_in, self.q_out, self.stop_event, self.sampler_properties), 
                    daemon=False).start()
        for core in range(self.n_cores):
            output = self.q_out.get()
            if output =="STARTED":
                if self.verbose:
                    print(f"Core {core} has been initialized.")
            else:
                raise ValueError("Unexpected initialization value in queue: {output}")

    def eval_integrand(self, x_all: torch.Tensor) -> torch.Tensor:
        continuous_dim = self.momtrop_integrand.continuous_dim
        x = x_all[:, -continuous_dim:]
        indices = x_all[:, :-continuous_dim].long()

        return self.__call__(x, indices).samples

    def madnis_integrand(self) -> MadnisIntegrand:
        continuous_dim = self.momtrop_integrand.continuous_dim
        discrete_dims = self.momtrop_integrand.discrete_dims
        return MadnisIntegrand(
            function=self.eval_integrand,
            input_dim=continuous_dim + len(discrete_dims),
            discrete_dims=discrete_dims,
            discrete_prior_prob_function=self.momtrop_integrand.predict_discrete_probs,
        )
    
    def end(self) -> None:
        self.stop_event.set()
        try:
            for _ in range(self.n_cores):
                self.q_in.put("STOP")
        except:
            print(f"Queues have already been closed")
        
    def __call__(self, xs: torch.Tensor, indices: torch.Tensor) -> IntegrandSample:
        assert (ln := len(indices)) == len(
            xs), "indices and x must be of same length"
        n_cores = min(ln, self.n_cores)

        chunks_per_worker = math.ceil(ln / n_cores / self.MAX_CHUNK_SIZE)
        n_chunks = n_cores * chunks_per_worker
        x_chunks = [xc for xc in chunks(xs, n_chunks)]
        indices_chunks = [ic for ic in chunks(indices, n_chunks)]
        args_chunked = zip(x_chunks, indices_chunks)

        for curr_chunk_id, args in enumerate(args_chunked):
            self.q_in.put((args, curr_chunk_id))

        result_sorted = [None]*n_chunks
        for _ in range(n_chunks):
            data = self.q_out.get()
            res, chunk_id = data
            result_sorted[chunk_id] = res

        return merge(result_sorted)
    
    def __del__(self):
        self.end()
