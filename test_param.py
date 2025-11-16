# type: ignore
import torch
import time
import toml

import src.parameterisation as par
from src.integrand import TestIntegrand
from src.helpers import error_fmter

use_settings = False
n_points = 10_000
settings_path = "dev_settings/1L_phys_layer_test.toml"

torch.set_default_dtype(torch.float64)

graph_properties = par.GraphProperties(
    edge_src_dst_vertices=[(3, 4), (4, 5), (5, 0), (0, 1), (1, 2), (2, 3)],
    edge_masses=[1500.0, 1500.0,
                 1500.0, 1500.0, 1500.0, 1500.0],
    edge_momentum_shifts=[
        [-231.8312618377385,
         -659.5630405248305,
         450.2378756570221],
        [-231.8312618377385,
         -359.5630405248305,
         50.23787565702207],
        [-209.73057155004852,
         -399.64339371651585,
         126.0433066139587],
        [-105.8809596665922,
         -97.7096383269757,
         49.54838522679282],
        [0.0, 0.0, 0.0],
        [-231.8312618377385,
         -359.5630405248305,
         50.23787565702207]
    ],
    graph_external_vertices=[
        0, 1, 2, 3, 4, 5],
    graph_signature=[
        [1], [1], [1], [1], [1], [1]],
    momtrop_edge_weights=[0.555555555555, 0.555555555555, 0.555555555555,
                          0.555555555555, 0.555555555555, 0.555555555555]
)
if use_settings:
    from src.parser import SettingsParser
    Settings = SettingsParser(settings_path)
    graph_properties = Settings.get_graph_properties()

integrand = TestIntegrand(continuous_dim=3*graph_properties.n_loops)

time_last = time.time()

spherical_kwargs = {"conformal_scale": 1.0,
                    "identifier": "spherical",
                    "graph_properties": graph_properties,
                    "is_first_layer": True, }
sph_param = par.SphericalParameterisation(**spherical_kwargs)
inverse_spherical_kwargs = {"conformal_scale": 1.0,
                            "identifier": "inverse spherical",
                            "graph_properties": graph_properties, }
inv_sph_param = par.InverseSphericalParameterisation(
    **inverse_spherical_kwargs)
neutral_kwargs = {"conformal_scale": 1.0,
                  "identifier": "neutral",
                  "graph_properties": graph_properties,
                  "next_param": inv_sph_param,
                  "is_first_layer": True, }
neutral_param = par.SphericalParameterisation(**neutral_kwargs)
momtrop_kwargs = {"graph_properties": graph_properties,
                  "identifier": "momtrop",
                  "is_first_layer": True, }
momtrop_param = par.MomtropParameterisation(**momtrop_kwargs)
kaapo_kwargs = {"graph_properties": graph_properties,
                "identifier": "kaapo",
                "is_first_layer": True,
                "mu": 0.0001}
kaapo_param = par.KaapoParameterisation(**kaapo_kwargs)
print(f"Number of Momtrop contdim: {momtrop_param.layer_continuous_dim_in}")
print(f"Initializing the Parameterisations took {
    - time_last + (time_last := time.time()):.2f}s")

hline = "----------------------------------------------------------------------------"

for param in [sph_param, neutral_param, momtrop_param, kaapo_param]:
    param: par.Parameterisation
    print(hline)
    print(f"Testing {param.identifier.upper()}:")
    print(hline)
    n_continuous = param.chain_continuous_dim_in
    xs = torch.rand(size=(n_points, n_continuous))
    input = par.LayerOutput(xs)
    output = integrand.evaluate_batch(param.parameterise(input))
    res, err = output.x_all.mean().item(), output.x_all.std().item() / n_points**0.5
    print(f"Integration result: {res:.4f} +- {err:.4f}")
    # print(f"Integration result: {error_fmter(res, err)}")
    print(f"Timing: {output.get_processing_times()}")

print(hline)
print("Testing LayerParameterisation")
print(hline)

with open(settings_path, 'r') as f:
    settings = toml.load(f)

param_settings = settings["parameterisation"]

layered_param = par.LayeredParameterisation(graph_properties, param_settings)
n_continuous = layered_param.continuous_dim
xs = torch.rand(size=(n_points, n_continuous))
input = par.LayerOutput(xs)
output = integrand.evaluate_batch(layered_param.parameterise(input))
res, err = output.x_all.mean().item(), output.x_all.std().item() / n_points**0.5
# print(f"Integration result: {error_fmter(res, err)}")
print(f"Timing: {output.get_processing_times()}")
