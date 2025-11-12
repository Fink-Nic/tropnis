# type: ignore
import torch
import time

import src.parameterisation as par
from src.parser import SettingsParser, GraphProperties

use_settings = False
n_points = 3
settings_path = "settings/physical_1L_6photons.toml"
graph_properties = GraphProperties(edge_src_dst_vertices=[(3, 4), (4, 5), (5, 0), (0, 1), (1, 2), (2, 3)], 
                                   edge_masses=[1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0], 
                                   edge_momentum_shifts=[
                                       [-231.8312618377385, -659.5630405248305, 450.2378756570221], 
                                       [-231.8312618377385, -359.5630405248305, 50.23787565702207], 
                                       [-209.73057155004852, -399.64339371651585, 126.0433066139587], 
                                       [-105.8809596665922, -97.7096383269757, 49.54838522679282], 
                                       [0.0, 0.0, 0.0], [-231.8312618377385, -359.5630405248305, 50.23787565702207]
                                       ], 
                                       graph_external_vertices=[0, 1, 2, 3, 4, 5], 
                                       graph_signature=[[1], [1], [1], [1], [1], [1]], 
                                       momtrop_edge_weights=[0.555555555555, 0.555555555555, 0.555555555555, 
                                                             0.555555555555, 0.555555555555, 0.555555555555]
                                                             )
if use_settings:
    Settings = SettingsParser(settings_path)
    graph_properties = Settings.get_graph_properties()

time_last = time.time()

spherical_kwargs = {"conformal_scale": 10.0,
                    "n_loops": graph_properties.n_loops,
                    "identifier": "spherical",}
sph_param = par.SphericalParameterisation(**spherical_kwargs)
inverse_spherical_kwargs = {"conformal_scale": 10.0,
                            "n_loops": graph_properties.n_loops,
                            "identifier": "inverse spherical",
                            "next_param": sph_param,}
inv_sph_param = par.InverseSphericalParameterisation(**inverse_spherical_kwargs)
momtrop_kwargs = {"graph_properties": graph_properties,
                  "identifier": "momtrop",
                  "next_param": inv_sph_param,
                  "is_first_layer": True,}
param = par.MomtropParameterisation(**momtrop_kwargs)
print(f"Number of Momtrop contdim: {param._layer_continuous_dim_in()}")
print(f"Initializing the Parameterisation took {
    - time_last + (time_last := time.time()):.2f}s")

print("Trying to get the dimensions of the chain.")
n_continuous = param.get_chain_continuous_dim()
n_discrete = len(param.get_chain_discrete_dims())

hline = "----------------------------------------------------------------------------"
print(hline)
print("Testing with no discrete input.")
print(hline)
xs = torch.rand(size=(n_points,n_continuous))
input = par.LayerOutput(xs)
print(f"{input=}")
output = param.parameterise(input)
print(f"{output=}")
print(f"Timing: {output.get_processing_times()}")

print(hline)
print("Testing with matching discrete input.")
print(hline)
full_graph = torch.arange(graph_properties.n_edges)
discrete = torch.tile(full_graph, (n_points, 1))
input = par.LayerOutput(torch.hstack([xs, discrete]))
print(f"{input=}")
output = param.parameterise(input)
print(f"{output=}")
print(f"Timing: {output.get_processing_times()}")

print(hline)
print("Testing with too large discrete input.")
print(hline)
xs = torch.rand(size=(n_points,n_continuous))
full_graph = torch.arange(graph_properties.n_edges + 2)
discrete = torch.tile(full_graph, (n_points, 1))
input = par.LayerOutput(torch.hstack([xs, discrete]))
print(f"{input=}")
output = param.parameterise(input)
print(f"{output=}")
print(f"Timing: {output.get_processing_times()}")
