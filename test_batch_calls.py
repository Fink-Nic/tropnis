# type: ignore
import torch
import math
import numpy as np
from resources.tropnis import test_numerator, MomtropIntegrand, TropicalIntegrator, TriangleIntegrand
from triangle import ltd_triangle

m_psi = 0.02
p = [0.005, 0., 0., 0.005]
q = [0.005, 0., 0., -0.005]


def triangle_numerator(loop_momenta: np.ndarray) -> np.ndarray:
    result = [ltd_triangle(m_psi, k.tolist(), q, p, 0.5) for k in loop_momenta]

    return np.array(result)


torch.set_default_dtype(torch.float64)
integrand = MomtropIntegrand(
    "gl_files/runcards/test_triangle_runcard.toml", triangle_numerator)
integrator = TropicalIntegrator(integrand, batch_size=1024)

n_train = 50
n_log = 5
n_samples = 10000
integrator.train(n_train, n_log)
int_flow, err_flow = integrator.integrate(n_samples)
rsd_flow = err_flow / int_flow * math.sqrt(n_samples)
print(f"Trained flow:     {
    int_flow:.8f} +- {err_flow:.8f}, RSD = {rsd_flow:.2f}")
