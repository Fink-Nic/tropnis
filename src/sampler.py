# type: ignore
import torch
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Callable, Optional
from multiprocessing import Process, Queue, Event

try:
    from gammaloop import GammaLoopAPI
except:
    pass
from madnis.integrator import Integrand as MadnisIntegrand
except:
    pass
from .parameterisation import Parameterisation, LayerOutput, LayeredParameterisation, GraphProperties
from kaapos.integrands import symbolica_integrand


class Sampler(ABC):
    identifier = "ABCSampler"

    def __init__(self,
                 continuous_dim: int,
                 discrete_dims: List[int] = [],
                 identifier: Optional[str] = None,):
        if identifier is not None:
            self.identifier = identifier
