# type: ignore
import torch
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Callable
from multiprocessing import Process, Queue, Event

import momtrop
from madnis.integrator import Integrand as MadnisIntegrand
from src.parser import SettingsParser
from src.parameterisation import Parameterisation, LayerOutput


@dataclass
class IntegrandResult:
    samples: torch.Tensor


class Integrand(ABC):
    MAX_CHUNK_SIZE = 10_000
    q_in, q_out = Queue(), Queue()
    stop_event = Event()

    def __init__(self,
                 continuous_dim: int = 0,
                 discrete_dims: List[int] = [],
                 param_init: Callable[any, Parameterisation],
                 param_kwargs: Dict[str, any] = {},
                 condition_integrand_first: bool = False,
                 ):
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims
        self.param_init = param_init
        self.param_kwargs = param_kwargs
        self.condition_integrand_first = condition_integrand_first
        self.param = self._get_param_instance()

    @abstractmethod
    def integrate_batch(self, continuous: torch.Tensor, discrete: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def parameterise_batch(self, continous: torch.Tensor, discrete: torch.Tensor) -> torch.Tensor:
        pass

    def get_chain_discrete_dims(self) -> List[int]:
        if self.condition_integrand_first:
            return self.discrete_dims + self.param.get_chain_discrete_dims()

        return self.param.get_chain_discrete_dims + self.discrete_dims

    def get_chain_continuous_dims(self) -> int:
        return self.continuous_dim + self.param.get_chain_continuous_dim

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

    def eval_integrand(self, x_all: torch.Tensor) -> torch.Tensor:
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
