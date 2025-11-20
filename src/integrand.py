# type: ignore
import math
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Callable, Optional
from multiprocessing import Process, Queue, Event

try:
    from gammaloop import GammaLoopAPI
except:
    print("No gammaloop python API found.")
try:
    import kaapos.samplers as ksamplers
    import kaapos.integrand as kintegrand
except:
    print("No thermal integrand python module found.")
from madnis.integrator import Integrand as MadnisIntegrand
from .parameterisation import LayerOutput, LayeredParameterisation, GraphProperties
from .helpers import chunks


class Integrand(ABC):
    identifier = "ABCIntegrand"

    def __init__(self,
                 continuous_dim: int = 0,
                 discrete_dims: List[int] = [],
                 identifier: Optional[str] = None,
                 eval_real_part: bool = True,):
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims
        if identifier is not None:
            self.identifier = identifier
        self.eval_real_part = eval_real_part

    @abstractmethod
    def _evaluate_batch(self, continuous: torch.Tensor, discrete: torch.Tensor) -> torch.Tensor:
        pass

    def evaluate_batch(self, layer_input: LayerOutput) -> LayerOutput:
        jacobians, continuous, discrete = layer_input.x_all.tensor_split(
            [1, layer_input.x_all.shape[1] - len(self.discrete_dims)], dim=1)

        res = jacobians * \
            self._evaluate_batch(continuous, discrete).reshape(-1, 1)
        layer_input.overwrite_x_all(res, self.identifier)

        return layer_input

    def discrete_prior_prob_function(self, indices: torch.Tensor, _: int = 0) -> torch.Tensor:
        if indices.shape[1] == len(self.discrete_dims):
            return torch.zeros_like(indices)

        norm_factor = len(self.discrete_dims) - indices.shape[1]
        prior = torch.ones((len(indices), len(self.discrete_dims)))
        if indices.shape[1] == 0:
            return prior / norm_factor

        rows = torch.arange(len(indices)).unsqueeze(1)
        prior[rows, indices] = 0

        return prior / norm_factor

    def __call__(self, layer_input: LayerOutput) -> LayerOutput:
        return self.evaluate_batch(layer_input)

    @staticmethod
    def get_integrand_instance(integrand_kwargs: Dict[str, any]) -> 'Integrand':
        integrand_type = integrand_kwargs.pop('integrand_type')
        match integrand_type:
            case 'test':
                return TestIntegrand(**integrand_kwargs)
            case 'gammaloop':
                return GammaLoopIntegrand(**integrand_kwargs)
            case 'kaapo':
                return KaapoIntegrand(**integrand_kwargs)
            case _:
                raise NotImplementedError(
                    f"Integrand {integrand_type} has not been implemented.")


class TestIntegrand(Integrand):
    """
    Implements a normalized multivariate gaussian that integrates to unity.
    """
    identifier = "test integrand"

    def __init__(self, offset: Optional[torch.Tensor] = None,
                 sigma: float = 100,
                 const_f: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.offset = offset
        self.sigma = sigma
        self.const_f = const_f

    def _evaluate_batch(self, continuous: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        if self.const_f:
            return torch.ones((continuous.shape[0], 1))

        if self.offset is not None:
            continuous -= self.offset.reshape(1, -1)
        norm_factor = (2*torch.pi * self.sigma**2)**(continuous.shape[1]/2)

        return (-(continuous.norm(p=2, dim=1) / self.sigma).pow(2)/2).exp() / norm_factor


class GammaLoopIntegrand(Integrand):
    """
    Implements a callable gammaloop state integrand.
    """
    identifier = "gammaloop integrand"

    def __init__(self,
                 gammaloop_state_path: str,
                 process_id: int = 0,
                 integrand_name: str = "default",
                 use_f128: bool = False,
                 momentum_space: bool = True,
                 **kwargs):
        self.gammaloop_state = GammaLoopAPI(gammaloop_state_path)
        self.process_id = process_id
        self.integrand_name = integrand_name
        self.use_f128 = use_f128
        self.momentum_space = momentum_space
        super().__init__(**kwargs)

    def _evaluate_batch(self, continuous: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        discrete_dims = np.zeros(
            (continuous.shape[0], 1), dtype=np.uint64)
        res, _ = self.gammaloop_state.batched_inspect(
            points=continuous.detach().cpu().numpy(), momentum_space=self.momentum_space,
            process_id=self.process_id,
            integrand_name=self.integrand_name,
            use_f128=self.use_f128,  discrete_dims=discrete_dims
        )
        res = res.real if self.real else res.imag
        return torch.from_numpy(np.float64(res))


class KaapoIntegrand(Integrand):
    """
    Implements a callable thermal integrand.
    """
    identifier = "kaapo integrand"

    def __init__(self,
                 path_to_example: str,
                 params: List[float] = [math.pi, 1.0, 1/math.pi],
                 symbolica_integrand_kwargs: Dict[str, any] = {
                     'force_rebuild': False,
                     'sum_orientations': True,
                     'runtime_summation': False,
                     'stability_tolerance': 1e-14,
                     'stability_abs_threshold': 1e-15,
                     'stability_abs_tolerance': 1e-15,
                     'escalate_large_weight_multiplier': 0.9,
                     'n_shots': 3,
                     'rotation_seed': 1337, },
                 symbolica_integrand_prec_kwargs: Dict[str, any] = {
                     'sum_orientations': True,
                     'runtime_summation': False,
                     'prec': 200,
                     'n_shots': 1,
                     'escalate_large_weight_multiplier': -1.0
                 },
                 use_prec: bool = True,
                 **kwargs):
        self.path_to_example = path_to_example
        self.params = np.array(params)
        self.symbolica_integrand_kwargs = symbolica_integrand_kwargs
        self.symbolica_integrand_prec_kwargs = symbolica_integrand_prec_kwargs
        self.use_prec = use_prec
        self.integrand_fast = kintegrand.SymbolicaIntegrand(
            path_to_example=self.path_to_example,
            params=self.params,
            **self.symbolica_integrand_kwargs,
        )
        self.integrand_prec = kintegrand.SymbolicaIntegrandPrec(
            path_to_example=self.path_to_example,
            params=self.params,
            **self.symbolica_integrand_prec_kwargs,
        )
        self.stack = kintegrand.StableStack([
            kintegrand.PrecisionLevel(
                integrand=self.integrand_fast, level_id=0),
            kintegrand.PrecisionLevel(
                integrand=self.integrand_prec, level_id=1),
        ])
        super().__init__(**kwargs)

    def _evaluate_batch(self, continuous: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        if not self.use_prec:
            raise NotImplementedError(
                "Non-precision symbolica integrand not implemented for KaaposIntegrand.")

        n_points = continuous.shape[0]
        loop_momenta = continuous.reshape(
            n_points, self.integrand_fast.n_loops, self.integrand_fast.dim).detach().cpu().numpy()
        input = ksamplers.SamplerResult(
            weight_array=np.ones((n_points, 1)),
            jacobian_array=np.ones(n_points),
            loop_momentum_array=loop_momenta
        )
        output = self.stack.evaluate(input)

        return output.values


class ParameterisedIntegrand:
    def __init__(self,
                 graph_properties: GraphProperties,
                 param_kwargs: Dict[str, any],
                 integrand_kwargs: Dict[str, any],
                 condition_integrand_first: bool = False,
                 verbose: bool = False,):
        self.graph_properties = graph_properties
        self.param_kwargs = param_kwargs
        self.integrand_kwargs = integrand_kwargs
        self.condition_integrand_first = condition_integrand_first
        self.verbose = verbose

        self.param = LayeredParameterisation(
            self.graph_properties, self.param_kwargs)
        self.integrand = Integrand.get_integrand_instance(
            self.integrand_kwargs)
        self.continuous_dim = self._get_continuous_dims()
        self.discrete_dims = self._get_discrete_dims()

    def eval_integrand(self, layer_input: LayerOutput) -> LayerOutput:
        parameterised = self.param.parameterise(layer_input)

        return self.integrand.evaluate_batch(parameterised)

    def discrete_prior_prob_function(self, indices: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.condition_integrand_first:
            n_dim = len(self.integrand.discrete_dims)
            prior1: Callable = self.integrand.discrete_prior_prob_function
            prior2: Callable = self.param.discrete_prior_prob_function
        else:
            n_dim = len(self.param.discrete_dims)
            prior1: Callable = self.param.discrete_prior_prob_function
            prior2: Callable = self.integrand.discrete_prior_prob_function

        if dim < n_dim:
            return prior1(indices, dim)

        indices = indices[:, n_dim:]
        dim -= n_dim

        return prior2(indices, dim)

    def get_madnis_integrand(self) -> MadnisIntegrand:
        return MadnisIntegrand(
            function=self.eval_integrand,
            input_dim=self.continuous_dim + len(self.discrete_dims),
            discrete_dims=self.discrete_dims,
            discrete_prior_prob_function=self.discrete_prior_prob_function,
        )

    def _get_discrete_dims(self) -> List[int]:
        if self.condition_integrand_first:
            return self.integrand.discrete_dims + self.param.discrete_dims

        return self.param.discrete_dims + self.integrand.discrete_dims

    def _get_continuous_dims(self) -> int:
        return self.integrand.continuous_dim + self.param.continuous_dim


class MPIntegrand:
    MAX_CHUNK_SIZE = 10_000
    MIN_CHUNK_SIZE = 10
    identifier = "multiprocessing integrand"

    def __init__(self, graph_properties: GraphProperties,
                 param_kwargs: Dict[str, any],
                 integrand_kwargs: Dict[str, any],
                 n_cores: int = 1,
                 condition_integrand_first: bool = False,
                 verbose: bool = False,
                 return_layeroutput: bool = False,):

        self.graph_properties = graph_properties
        self.param_kwargs = param_kwargs
        self.integrand_kwargs = integrand_kwargs
        self.n_cores = n_cores
        self.condition_integrand_first = condition_integrand_first
        self.verbose = verbose
        self.return_layeroutput = return_layeroutput
        self.q_in, self.q_out, self.q_discr = [Queue() for _ in range(3)]
        self.stop_event = Event()

        worker_args = (
            (self.q_in, self.q_out, self.q_discr),
            self.stop_event,
            self.graph_properties,
            self.param_kwargs,
            self.integrand_kwargs,
            self.condition_integrand_first,
        )
        for _ in range(self.n_cores):
            Process(target=self.gammaloop_worker,
                    args=worker_args,
                    daemon=False).start()
        for core in range(self.n_cores):
            output = self.q_out.get()
            if output == "STARTED":
                if self.verbose:
                    print(f"Core {core} has been initialized.")
            else:
                raise ValueError(
                    "Unexpected initialization value in queue: {output}")
        # Get properties without creating additional instances
        self.q_in.put(('init', None, None))
        self.continuous_dim, self.discrete_dims = self.q_out.get()
        if self.verbose:
            print(f"Sucessfully initalized integrand properties:")
            print(f"continuous_dim: {self.continuous_dim}")
            print(f"discrete_dims: {self.discrete_dims}")

    @staticmethod
    def gammaloop_worker(queues: Sequence[Queue],
                         stop_event,
                         graph_properties: GraphProperties,
                         param_kwargs: Dict[str, any],
                         integrand_kwargs: Dict[str, any],
                         condition_integrand_first: bool,) -> None:
        integrand = ParameterisedIntegrand(graph_properties,
                                           param_kwargs,
                                           integrand_kwargs,
                                           condition_integrand_first,)
        q_in, q_out, q_discr = queues
        q_out.put("STARTED")
        while not stop_event.is_set():
            try:
                data = q_in.get(timeout=0.5)
            except:
                continue
            job_type, chunk_id, args = data
            job_type: str
            match job_type.lower():
                case 'eval':
                    input = LayerOutput(args)
                    res = integrand.eval_integrand(input)
                    q_out.put((chunk_id, res))
                case 'prior':
                    indices, dim = args
                    res = integrand.discrete_prior_prob_function(indices, dim)
                    q_discr.put((chunk_id, res))
                case 'init':
                    q_out.put(
                        (integrand.continuous_dim, integrand.discrete_dims)
                    )
                case _:
                    print("CRITICAL WARNING:")
                    print(
                        f"gammaloop worker has received invalid job type \"{job_type.upper()}\"")
                    print("Consider terminating the program.")

    def eval_integrand(self, input: LayerOutput) -> LayerOutput | torch.Tensor:
        job_type = "eval"
        x_all = input.x_all
        n_samples = len(x_all)
        n_cores = min(math.ceil(n_samples / self.MIN_CHUNK_SIZE), self.n_cores)

        chunks_per_worker = math.ceil(
            n_samples / n_cores / self.MAX_CHUNK_SIZE)
        n_chunks = n_cores * chunks_per_worker
        x_chunks = chunks(x_all, n_chunks)

        for chunk_id, args in enumerate(x_chunks):
            self.q_in.put((job_type, chunk_id, args))

        result_sorted = [None]*n_chunks
        for _ in range(n_chunks):
            if self.stop_event.is_set():
                return
            data = self.q_out.get()
            chunk_id, res = data
            result_sorted[chunk_id] = res

        output = LayerOutput.join(result_sorted, n_cores=self.n_cores)

        if self.verbose:
            times = output.get_processing_times()
            print("PROCESSING TIMES:")
            for identifier, time in times.items():
                print(f"{identifier.upper()}: {time:.4f}")

        if self.return_layeroutput:
            return output

        return output.x_all

    def discrete_prior_prob_function(self, indices: torch.Tensor, dim: int = 0) -> torch.Tensor:
        job_type = "prior"
        n_samples = len(indices)
        n_cores = min(math.ceil(n_samples / self.MIN_CHUNK_SIZE), self.n_cores)

        chunks_per_worker = math.ceil(
            n_samples / n_cores / self.MAX_CHUNK_SIZE)
        n_chunks = n_cores * chunks_per_worker
        ind_chunks = chunks(indices, n_chunks)

        for chunk_id, ind in enumerate(ind_chunks):
            args = (ind, dim)
            self.q_in.put((job_type, chunk_id, args))

        result_sorted = [None]*n_chunks
        for _ in range(n_chunks):
            if self.stop_event.is_set():
                return
            data = self.q_discr.get()
            chunk_id, res = data
            result_sorted[chunk_id] = res

        return torch.vstack(result_sorted)

    def get_madnis_integrand(self) -> MadnisIntegrand:
        return MadnisIntegrand(
            function=self.eval_integrand,
            input_dim=self.continuous_dim + len(self.discrete_dims),
            discrete_dims=self.discrete_dims,
            discrete_prior_prob_function=self.discrete_prior_prob_function,
        )

    def end(self) -> None:
        self.stop_event.set()
        """ 
        try:
            for _ in range(self.n_cores):
                self.q_in.put("STOP")
        except:
            print(f"Queues have already been closed") 
        """

        print(f"{self.identifier.upper()} has successfully terminated.")

    def __del__(self) -> None:
        self.end()
