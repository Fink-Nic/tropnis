# type: ignore
import torch
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Callable
from multiprocessing import Process, Queue, Event

import momtrop
try:
    from gammaloop import GammaLoopAPI
except:
    pass
from madnis.integrator import Integrand as MadnisIntegrand
try:
    from src.parser import SettingsParser
except:
    pass
from src.parameterisation import Parameterisation, LayerOutput, GraphProperties


@dataclass
class IntegrandResult:
    samples: np.ndarray


class Integrand(ABC):
    identifier = "ABCIntegrand"

    def __init__(self,
                 continuous_dim: int,
                 discrete_dims: List[int] = []):
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims

    @abstractmethod
    def _evaluate_batch(self, continuous: torch.Tensor, discrete: torch.Tensor) -> torch.Tensor:
        pass

    def evaluate_batch(self, layer_input: LayerOutput) -> LayerOutput:
        jacobians, continuous, discrete = layer_input.x_all.tensor_split(
            [1, 1 + self.continuous_dim], dim=1)

        res = jacobians * \
            self._evaluate_batch(continuous, discrete).reshape(-1, 1)
        layer_input.overwrite_x_all(res, self.identifier)

        return layer_input

    def __call__(self, layer_input: LayerOutput) -> LayerOutput:
        return self.evaluate_batch(layer_input)


class TestIntegrand(Integrand):
    """
    Implements a normalized multivariate gaussian that integrates to unity.
    """
    identifier = "Test Integrand"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate_batch(self, continuous: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        norm_factor = (2*torch.pi)**(continuous.shape[1]/2)

        return (-continuous.norm(p=2, dim=1).pow(2)/2).exp() / norm_factor


class GammaLoopIntegrand(Integrand):
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
        return res


class ParameterisedIntegrand:
    MAX_CHUNK_SIZE = 10_000
    q_in, q_out = Queue(), Queue()
    stop_event = Event()

    def __init__(self,
                 settings_file,
                 condition_integrand_first: bool = False,
                 ):
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims
        self.param_init = param_init
        self.param_kwargs = param_kwargs
        self.condition_integrand_first = condition_integrand_first
        self.param = self._get_param_instance()

    @abstractmethod
    def evaluate_batch(self, continuous: torch.Tensor, discrete: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def parameterise_batch(self, continous: torch.Tensor, discrete: torch.Tensor) -> torch.Tensor:
        pass

    def get_chain_discrete_dims(self) -> List[int]:
        if self.condition_integrand_first:
            return self.discrete_dims + self.param.get_chain_discrete_dims()

        return self.param.get_chain_discrete_dims + self.discrete_dims

    def get_chain_continuous_dims(self) -> int:
        return self.continuous_dim + self.param.get_chain_continuous_dim()

    def _get_param_instance(self) -> Parameterisation:
        return self.param_init(**self.param_kwargs)

    def discrete_prior_prob_function(self, indices: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return self.param.discrete_prior_prob_function(indices, dim)

    def get_madnis_integrand(self) -> MadnisIntegrand:
        continuous_dim = self.get_chain_continuous_dims()
        discrete_dims = self.get_chain_discrete_dims()
        return MadnisIntegrand(
            function=self.eval_integrand,
            input_dim=continuous_dim + len(discrete_dims),
            discrete_dims=discrete_dims,
            discrete_prior_prob_function=self.discrete_prior_prob_function,
        )

    def eval_integrand(self, layer_input: torch.Tensor) -> torch.Tensor:
        pass

    def end(self) -> None:
        self.stop_event.set()
        try:
            for _ in range(self.n_cores):
                self.q_in.put("STOP")
        except:
            print(f"Queues have already been closed")

    def __del__(self):
        self.end()
