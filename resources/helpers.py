# type: ignore
import numpy as np
import json
import os
import math
from typing import List, Sequence, Dict
from collections.abc import Mapping


# Load the file system structure specified in PATHS.json
with open(os.path.join(os.path.dirname(__file__), "PATHS.json")) as f:
    PATHS = json.load(f)


def test_integrand(loop_momenta: np.ndarray) -> np.ndarray:
    return np.ones_like(loop_momenta, shape=loop_momenta.shape[0])


def chunks(ary: Sequence, n_chunks: int) -> List[Sequence]:
    """
    Like numpy.array_split, but works for all sequences
    """
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
    """
    Used to Overwrite the default settings with the specified settings file
    """
    for key, val in new_dict.items():
        if isinstance(val, Mapping):
            tmp = deep_update(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def error_fmter(value: float, error: float, prec_error: int = 2) -> str:
    """
    Format a value and its error in scientific notation with a given number of significant digits for the error.

    Examples:
        value = 1234.5678, error = 111.11, prec_error = 2 -> "1.23(11)e+04"
        value = 12.345, error = 111.111, prec_error = 1 -> "1.2(11.1)e+01"
        value = 0.0123, error = 0.001234, prec_error = 3 -> "1.230(123)e-02"
    """
    if error <= 0:
        raise ValueError("Error must be positive.")
    if prec_error < 1:
        raise ValueError(
            "Number of significant digits should be at least 1, or there is no reason to use this function.")

    if value == 0:
        log10val = 0
    else:
        log10val = math.floor(math.log10(abs(value)))
    exp10val = 10**log10val

    # Normalize both value and error to the same order of magnitude
    val_norm = value / exp10val
    err_norm = error / exp10val

    # Set prec: the significant number of digits such that prec_error number
    # of significant digits are shown for the error
    log10err_norm = math.floor(math.log10(err_norm))

    if log10err_norm >= 0:
        prec = prec_error
    else:
        prec = prec_error - log10err_norm

    # Get digits without scientific notation
    val_str = f"{val_norm:.{prec}f}"
    if log10err_norm >= 0:
        err_str = f"{err_norm:.{prec}f}"
    else:
        err_str = f"{err_norm:.{prec - 1}f}".replace(".", "")[-prec_error:]
    # I don't think this can happen since error>0, but if the error is somehow rounded
    # down to zero, err_str will be empty and we default to
    if not err_str:
        err_str = '0' * prec_error

    return f"{val_str}({err_str})e{log10val:+03d}"
