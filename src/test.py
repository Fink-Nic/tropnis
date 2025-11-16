import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Dict, List

GRAPH_PROPERTIES = [10, 20.2, 123, "str"]

test_dict: Dict[str, Dict] = {"layer_0": {"a": 1, "b": 2, },
                              "layer_1": {"a": 1, "b": 2, "GRAPH_PROPERTIES": "IF YOU CAN SEE THIS, YOU SUCK."},
                              "layer_2": {"a": 1, "b": 2, "GRAPH_PROPERTIES": 1}, }

for dic in test_dict.values():
    if "GRAPH_PROPERTIES" in dic.keys():
        dic["GRAPH_PROPERTIES"] = GRAPH_PROPERTIES
        print(dic)

print(test_dict)
