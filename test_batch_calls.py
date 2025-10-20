# type: ignore
import torch
import math
import numpy as np
from time import time
from madnis.integrator import Integrator
from resources.tropnis import MomtropIntegrand, TropicalIntegrator, TriangleIntegrand
from triangle import ltd_triangle

# Triangle kinematics
m_psi = 0.02
p = [0.005, 0., 0., 0.005]
q = [0.005, 0., 0., -0.005]

runcard_file = "/shared/vhirshi/nic/tropnis/gl_files/runcards/generate_pentagon_euclidean.toml"
verbose = True

# Training parameter
n_train = 0
n_steps = 20
batch_size = 2**15
n_log = 10
n_samples = 30_000
n_cores = 8

# Gammaloop result for the scalar box
# target_int = 0.0010449992991079754
# target_err = 5.673200100564505E-7
# target_rsd = target_err / target_int * math.sqrt(1e6)

# # Gammaloop result for the photons eu
# # RE
# target_int = 2.601715451759867e-11
# target_err = 1.140976195795837e-10
# target_rsd = target_err / target_int * math.sqrt(1e6)
# # IM
# target_int = 3.548019462435879e-10
# target_err = 2.0509353198538228e-10
# target_rsd = target_err / target_int * math.sqrt(1e6)
# "result":{"re":2.601715451759867e-11,"im":3.548019462435879e-10},"error":{"re":1.140976195795837e-10,"im":2.0509353198538228e-10}

# # Gammaloop result for the pentagon_euclidean
# # RE
# target_int = 0.04186664630521326
# target_err = 0.007694827471977602
# target_rsd = target_err / target_int * math.sqrt(1e6)
# # IM
target_int = 0.005171160614788395
target_err = 0.0009551847364931251
target_rsd = target_err / target_int * math.sqrt(1e6)
# ,"result":{"re":0.04186664630521326,"im":0.005171160614788395},"error":{"re":0.007694827471977602,"im":0.0009551847364931251}

print(f"Gammaloop Result:    {
    target_int:.8g} +- {target_err:.8g}, RSD = {target_rsd:.2f}")

def triangle_integrand(loop_momenta: np.ndarray) -> np.ndarray:
    result = [ltd_triangle(m_psi, k.tolist(), q, p, 0.5) for k in loop_momenta]

    return np.array(result)

torch.set_default_dtype(torch.float64)

time_last = time()
integrand = MomtropIntegrand(
    runcard_file,
    n_cores = n_cores,
    parser_verbosity = verbose,)
integrator = Integrator(
    integrand.madnis_integrand(),
    discrete_model="transformer",
    discrete_flow_kwargs=dict(
        embedding_dim=64,
        feedforward_dim=128,
        heads=8,
        mlp_units=128,
        transformer_layers=2,
    ),
    batch_size = batch_size,
)

print(f"Initializing the Integrand and Integrator took \
      { - time_last + (time_last:=time()):.2f}s")

"""
indices = torch.arange(integrand.discrete_dims[0]).repeat(n_samples, 1)
xs = torch.rand(size=(n_samples, integrand.continuous_dim))
time_last = time()
samples = integrand(indices, xs)
int_momtrop = samples.mean().item()
err_momtrop = samples.std().item() 
"""
metrics = integrator.integration_metrics(n_samples)
print(f"Evaluating {n_samples} samples using {integrand.n_cores} \
      cores took { - time_last + (time_last:=time()):.2f}s")

print(f"Momtrop Result (before training):     {
    metrics.integral:.8g} +- {metrics.error:.8g}, RSD = {metrics.rel_stddev:.2f}")

def callback(status) -> None:
    if (status.step + 1) % n_log == 0:
        print(f"Step {status.step + 1}: loss={status.loss:.5f}")

for i in range(n_train):
    integrator.train(n_steps, callback)
    metrics = integrator.integration_metrics(n_samples)
    print(f"Trained Result after {integrator.step} steps of {batch_size}:     {
        metrics.integral:.8g} +- {metrics.error:.8g}, RSD = {metrics.rel_stddev:.2f}")

print(f"Gammaloop Result:    {
    target_int:.8g} +- {target_err:.8g}, RSD = {target_rsd:.2f}")