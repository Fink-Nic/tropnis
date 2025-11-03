# type: ignore
import numpy as np
import json
import os
import math
from typing import List, Sequence, Dict
from collections.abc import Mapping


with open(os.path.join(os.path.dirname(__file__), "PATHS.json")) as f:
    PATHS = json.load(f)


def test_integrand(loop_momenta: np.ndarray) -> np.ndarray:
    return np.ones_like(loop_momenta, shape=loop_momenta.shape[0])


def chunks(ary: Sequence, n_chunks: int) -> List[Sequence]:
    l = len(ary)
    if n_chunks > l or n_chunks < 1:
        raise ValueError(
            "the number of chunks should be at least 1, and at most len(ary)")
    n_long = l % n_chunks
    len_long = l // n_chunks + 1
    total_long = n_long*len_long
    len_short = l // n_chunks

    long_chunks = [ary[start:start+len_long]
                   for start in range(0, total_long, len_long)]
    short_chunks = [ary[start:start+len_short]
                    for start in range(total_long, l, len_short)]

    return long_chunks + short_chunks


def deep_update(orig_dict: Dict, new_dict: Dict):
    for key, val in new_dict.items():
        if isinstance(val, Mapping):
            tmp = deep_update(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def error_fmter(value, error, prec: int | None = None):
    log10v, log10e = math.log10(abs(value)), math.log10(error)

    if prec is None:
        error_prec = 2
        prec = error_prec + math.floor(log10v) - math.floor(log10e)
    else:
        error_prec = prec - math.floor(log10v) + math.floor(log10e)

    # Case of small error, user should increase precision or use default
    if error_prec <= 0:
        print(
            f'Automatically adjusted precision for the error formatter to {prec - error_prec + 1}.')
        prec = prec - error_prec + 1
        error_prec = 1

    # General string formatter returns non-scientific representation for -4<log10(value)<5
    # 'Hashtag' option for g formatter forces trailing zeros
    if log10v < prec-1 and log10v >= -2:
        value_str = f'{value:#.{prec}g}'
        if log10e >= 0:
            error_str = f'{error:#.{error_prec}g}'
            return f'{value_str}({error_str})'
        else:
            error_str = f'{error:.{error_prec}e}'
            return f'{value_str}({error_str[0]}{error_str[2:error_prec+1]})'

    # In case of scientific representation, reduce prec by one to match actual shown digits with prec
    prec -= 1
    error_prec -= 1
    value_str = f'{value:.{prec}e}'
    error_str = f'{error:.{error_prec}e}'

    return f'{value_str[:prec+2]}({error_str[0]}{error_str[2:error_prec+2]}){value_str[prec+3:]}'