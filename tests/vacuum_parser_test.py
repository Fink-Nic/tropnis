# type: ignore
from resources.parser import RunCardParser
import momtrop
import torch
import numpy as np

diagram = "sunrise_kaapo"
prefix = "/shared/vhirshi"
nic_files = prefix+"/nic/tropnis/gl_files"
gl_files = prefix+"/gammaloop_hedge_numerator"
runcard_file = nic_files+"/runcards/generate_"+diagram+".toml"
gl_integration_result = gl_files+"/gl_states/integration_"+diagram+"/integration_results.txt"

Parser = RunCardParser(runcard_file, True)
sampler, sampler_properties = Parser.generate_momtrop_sampler_from_dot_file()
edge_data = momtrop.EdgeData(
        sampler_properties.edge_masses,
        sampler_properties.momentum_shifts)
settings = momtrop.Settings(False, False)
continuous_dim = sampler.get_dimension()
discrete_dim = len(sampler_properties.edges)

n_samples = 10
indices = torch.arange(discrete_dim).repeat(n_samples, 1)
xs = torch.rand(size=(n_samples, continuous_dim))
samples = sampler.sample_batch(xs.tolist(), edge_data, settings, indices.tolist())

loop_momenta = np.array(samples.loop_momenta).reshape(-1, 3)
jacobians = np.array(samples.jacobians)

print(loop_momenta)
print(jacobians)