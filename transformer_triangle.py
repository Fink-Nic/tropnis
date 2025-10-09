# type:ignore
from madnis.integrator import Integrator
import scipy.stats
import torch
import math
import numpy as np
from resources.tropnis import test_numerator, MomtropIntegrand, TropicalIntegrator, TriangleIntegrand, MomtropTransformerIntegrand
from triangle import ltd_triangle
import matplotlib.pyplot as plt

m_psi = 0.02
p = [0.005, 0., 0., 0.005]
q = [0.005, 0., 0., -0.005]

n_train = 500
n_log = 20
n_plt = 20
n_samples = 10_000

arr_rsd_flow = []
arr_rsd_made = []
arr_rsd_trns = []
train_arr = np.arange(n_plt, n_train+1, n_plt)


def triangle_integrand(loop_momenta: np.ndarray) -> np.ndarray:
    result = [ltd_triangle(m_psi, k.tolist(), q, p, 0.5) for k in loop_momenta]

    return np.array(result)


torch.set_default_dtype(torch.float64)
integrand = MomtropIntegrand(
    "gl_files/runcards/test_triangle_runcard.toml", triangle_integrand)
integrator = TropicalIntegrator(integrand, batch_size=1024)

res_momtrop, err_momtrop = integrator.integrate(n_samples)
rsd_momtrop = err_momtrop / res_momtrop * math.sqrt(n_samples)
print(
    f"Momtrop integration result:   {res_momtrop:.5f} +- {err_momtrop:.5f}, RSD={rsd_momtrop:.2f}")

for _ in train_arr:
    integrator.train(n_plt, n_log)
    res_flow, err_flow = integrator.integrate(n_samples)
    rsd_flow = err_flow / res_flow * math.sqrt(n_samples)
    arr_rsd_flow.append(rsd_flow)


print(f"Flow integration result:     {
    res_flow:.5f} +- {err_flow:.5f}, RSD = {rsd_flow:.2f}")


integrand = MomtropTransformerIntegrand(
    "gl_files/runcards/test_triangle_runcard.toml", triangle_integrand)


def callback_made(status):
    if (status.step + 1) % n_log == 0:
        print(f"Batch {status.step + 1}: loss={status.loss:.5f}")
    if (status.step + 1) % n_plt == 0:
        res_made, err_made = integrator.integrate(n_samples)
        rsd_made = err_made / res_made * math.sqrt(n_samples)
        arr_rsd_made.append(rsd_made)


integrator = Integrator(
    integrand.madnis_integrand(),
    discrete_model="made",
    discrete_flow_kwargs=dict(
    )
)

integrator.train(n_train, callback_made)
res_made, err_made = integrator.integrate(n_samples)
rsd_made = err_made / res_made * math.sqrt(n_samples)
print(
    f"Made integration result:   {res_made:.5f} +- {err_made:.5f}, RSD={rsd_made:.2f}")


def callback_trns(status):
    if (status.step + 1) % n_log == 0:
        print(f"Batch {status.step + 1}: loss={status.loss:.5f}")
    if (status.step + 1) % n_plt == 0:
        res_trns, err_trns = integrator.integrate(n_samples)
        rsd_trns = err_trns / res_trns * math.sqrt(n_samples)
        arr_rsd_trns.append(rsd_trns)


integrator = Integrator(
    integrand.madnis_integrand(),
    discrete_model="transformer",
    discrete_flow_kwargs=dict(
        embedding_dim=64,
        feedforward_dim=64,
        heads=4,
        mlp_units=64,
        transformer_layers=1,
    )
)

integrator.train(n_train, callback_trns)
res_trns, err_trns = integrator.integrate(n_samples)
rsd_trns = err_trns / res_trns * math.sqrt(n_samples)
print(
    f"Transformer integration result:   {res_trns:.5f} +- {err_trns:.5f}, RSD={rsd_trns:.2f}")

fig, ax = plt.subplots()
ax.scatter([0], [rsd_momtrop], c="green", label="momtrop")
ax.scatter(train_arr, arr_rsd_flow, c="black", label="flow")
ax.scatter(train_arr, arr_rsd_made, c="blue", label="made")
ax.scatter(train_arr, arr_rsd_trns, c="red", label="transformer")
ax.set_xlabel("Training iteration")
ax.set_ylabel("RSD")
ax.set_ylim(bottom=0)
ax.legend()
fig.suptitle("Performance Comparison for triangle integrand of MadNiS samplers")
plt.show()
