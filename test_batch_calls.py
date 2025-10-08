# type: ignore
import torch
import math
import numpy as np
from resources.tropnis import test_integrand, MomtropIntegrand, TropicalIntegrator, TriangleIntegrand
from triangle import ltd_triangle

# Triangle kinematics
m_psi = 0.02
p = [0.005, 0., 0., 0.005]
q = [0.005, 0., 0., -0.005]

# Training parameter
n_train = 0
n_log = 5
n_samples = 100000

# Gammaloop result for the scalar box
target_int = 0.0010449992991079754
target_err = 5.673200100564505E-7

def triangle_integrand(loop_momenta: np.ndarray) -> np.ndarray:
    result = [ltd_triangle(m_psi, k.tolist(), q, p, 0.5) for k in loop_momenta]

    return np.array(result)

torch.set_default_dtype(torch.float64)
integrand = MomtropIntegrand(
    "/shared/vhirshi/nic/tropnis/gl_files/runcards/generate_scalar_box.toml")
integrator = TropicalIntegrator(integrand, batch_size=1024)

int_momtrop, err_momtrop = integrator.integrate(n_samples)
rsd_momtrop = err_momtrop / int_momtrop * math.sqrt(n_samples)
print(f"Gammaloop Result:    {
    target_int:.8f} +- {target_err:.8f}")
print(f"Momtrop Result (before training):     {
    int_momtrop:.8f} +- {err_momtrop:.8f}, RSD = {rsd_momtrop:.2f}")

if n_train == 0:
    quit()
    
integrator.train(n_train, n_log)
int_flow, err_flow = integrator.integrate(n_samples)
rsd_flow = err_flow / int_flow * math.sqrt(n_samples)
print(f"Trained Result:     {
    int_flow:.8f} +- {err_flow:.8f}, RSD = {rsd_flow:.2f}")
print(f"Gammaloop Result:    {
target_int:.8f} +- {target_err:.8f}")
