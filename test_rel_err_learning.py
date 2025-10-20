# type: ignore

from madnis.integrator import Integrator
import torch
import math
import numpy as np
from resources.tropnis import MomtropIntegrand

n_cores = 16
n_err_probe = 2**15
n_total_training_samples = 400_000

output_file = "/shared/vhirshi/nic/tropnis/outputs/rel_err_learning_data_pentagon.npy"
runcard_file = "/shared/vhirshi/nic/tropnis/gl_files/runcards/generate_pentagon_euclidean.toml"
rel_errors = [0.6, 0.4, 0.3] # 0.10, 0.05, 0.035, 
min_batch_size = 2_000
n_update_batch_size = 5
n_log = 2

torch.set_default_dtype(torch.float64)

def update_batch_size(target_rel_error: float) -> None:
    global spls
    global batchsizes
    global rsds
    metrics = integrator.integration_metrics(n_err_probe)
    curr_rel_error = metrics.error/(metrics.error + abs(metrics.integral))
    updated_batch_size = int((curr_rel_error / target_rel_error)**2 * n_err_probe) + 1

    integrator.batch_size = max(updated_batch_size, min_batch_size)
    spls.append(completed_training_samples)
    batchsizes.append(integrator.batch_size)
    rsds.append(curr_rel_error * math.sqrt(n_err_probe))

def generate_callback_trns(target_rel_error: float):
    def callback_trns(status) -> None:
        global completed_training_samples
        global losses
        step = status.step
        loss = status.loss

        if (step + 1) % n_update_batch_size == 0 or step < n_update_batch_size:
            update_batch_size(target_rel_error)
            losses.append(loss)
        
        batch_size = integrator.batch_size
        if (step + 1) % n_log == 0 or step == 0:
            print(f"Step {step + 1}: loss={loss:.5f}, batch_size={batch_size}")
        
        completed_training_samples += batch_size

    return callback_trns

integrand = MomtropIntegrand(
    runcard_file,
    n_cores = n_cores)

arr_rsds = []
arr_spls = []
arr_batchsizes = []
arr_losses = []

for target_rel_error in rel_errors:
    completed_training_samples = 0
    rsds = []
    spls = []
    batchsizes = []
    losses = []
    
    integrator = Integrator(
        integrand.madnis_integrand(),
        discrete_model="transformer",
        discrete_flow_kwargs=dict(
            embedding_dim=64,
            feedforward_dim=64,
            heads=4,
            mlp_units=64,
            transformer_layers=1,
        ),
        batch_size = n_err_probe,
    )
    res_init, err_init = integrator.integrate(n_err_probe)
    rel_err_init = abs(err_init / res_init)
    rsd_init = rel_err_init * math.sqrt(n_err_probe)
    print(
        f"Initial error probe using {n_err_probe} samples:   \
            {res_init:.5f} +- {err_init:.5f}, RSD={rsd_init:.2f}")

    callback_trns = generate_callback_trns(target_rel_error)
    while completed_training_samples < n_total_training_samples:
        integrator.train(1, callback_trns)

    res, err = integrator.integrate(n_err_probe)
    rel_err = abs(err / res)
    rsd = rel_err * math.sqrt(n_err_probe)
    print(
        f"After {completed_training_samples} training samples aiming for a relative error of {target_rel_error:.3f}: \n \
        A probe of {n_err_probe} samples:   \
        {res:.5f} +- {err:.5f}, {rel_err=:.5f}, RSD={rsd:.2f}")
    
    batchsizes.append(integrator.batch_size)
    rsds.append(rsd)
    spls.append(completed_training_samples)
    arr_batchsizes.append(np.array(batchsizes))
    arr_rsds.append(np.array(rsds))
    arr_spls.append(np.array(spls))
    arr_losses.append(np.array(losses))

# Save the data in the same order it will (has to) be read out
with open(output_file, 'wb') as f:
    np.save(f, np.array([min_batch_size]))
    np.save(f, np.array(rel_errors))
    for rsds, spls, batchsizes, losses in zip(arr_rsds, arr_spls, arr_batchsizes, arr_losses):
        np.save(f, rsds)
        np.save(f, spls)
        np.save(f, batchsizes)
        np.save(f, losses)
