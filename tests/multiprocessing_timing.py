# type: ignore
import os
import torch
import numpy as np
import signal
import argparse
from time import time, sleep
from resources.tropnis import MomtropIntegrand, GammaLoopIntegrand
from resources.helpers import PATHS



def main() -> None:
    # -------------------------------------------------------------------------------------
    #                        Easier access to the default values
    # -------------------------------------------------------------------------------------
    n_cores = [4, 4] # [1, 4, 8, 16, 32, 64]
    n_samples = [2**10, 2**12] # [2**10, 2**13, 2**16, 2**18]
    # -------------------------------------------------------------------------------------
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        parser = argparse.ArgumentParser(prog="multiprocessing_timing")
        parser.add_argument('--settings', '-s', type=str,
                            help="The settings .toml file.")
        parser.add_argument('--n_cores', '-nc', type=int, nargs='+', default=n_cores,
                            help="Number of cores, provided as a list. Default = %(default)s")
        parser.add_argument('--n_samples', '-ns', type=int, nargs='+', default=n_samples,
                            help="Number of samples, provided as a list. Default = %(default)s")
        args = parser.parse_args()
        settings_file = args.settings
        n_cores = args.n_cores
        n_samples = args.n_samples

        mt_integrand = MomtropIntegrand(settings_file)
        continuous_dim = mt_integrand.continuous_dim
        discrete_dim = mt_integrand.discrete_dims[0]
        gammaloop_state = mt_integrand.settings['gammaloop_state']['state_name']
        subfolder_path = os.path.join(PATHS['tropnis'], "outputs", "multiprocessing_timing")
        out_file = os.path.join(subfolder_path, f"{gammaloop_state}.txt")

        print(f"Working on runcard {settings_file}")
        print(f"Output will be at {subfolder_path}")

        state = mt_integrand.Parser.get_gl_state()
        process_id = mt_integrand.settings['gammaloop_state']['process_id']
        integrand_name = mt_integrand.settings['gammaloop_state']['integrand_name']
        n_loops = len(mt_integrand.sampler_properties.edge_masses)
        dummy_momenta = np.random.rand(3*n_loops*n_samples[0])*100
        time_before_gl_batched_inspect = time()
        res, jac = state.batched_inspect(
            points=dummy_momenta.reshape(-1,3), momentum_space=True,
            process_id=process_id,
            integrand_name=integrand_name,
            use_f128=False,  discrete_dims=np.zeros((n_samples[0], 1), dtype=np.uint64)
        )
        time_gl_batched_inspect = time() - time_before_gl_batched_inspect

        total_times = []
        subgraph_times = []
        momtrop_times = []
        integrand_times = []
        unaccounted_times = []
        initialisation_times = []
        for cores in n_cores:
            time_before_initialisation = time()
            gl_integrand = GammaLoopIntegrand(settings_file, cores)
            initialisation_time = time() - time_before_initialisation
            initialisation_times.append(initialisation_time)
            print(f"Initalising {cores} cores took {initialisation_time:.2f}s")
            times_total = []
            times_subgraph = []
            times_momtrop = []
            times_integrand = []
            times_unaccounted = []
            for samples in n_samples:
                if ((cores == 1 and samples > 2**12) or 
                    (cores == 4 and samples > 2**14) or
                    (cores < 32 and samples > 2**17)):
                    times_total.append(0.)
                    times_subgraph.append(0.)
                    times_momtrop.append(0.)
                    times_integrand.append(0.)
                    times_unaccounted.append(0.)
                    continue

                indices = torch.arange(discrete_dim).repeat(samples, 1)
                xs = torch.rand(size=(samples, continuous_dim))
                time_before_sampling = time()
                print("before sampling")
                sample = gl_integrand(xs, indices)
                print("after sampling")
                total_time = time() - time_before_sampling
                subgraph_time = sample.timings.subgraph / cores
                momtrop_time = sample.timings.momtrop / cores
                integrand_time = sample.timings.integrand / cores
                unaccounted_time = total_time - subgraph_time - momtrop_time - integrand_time
                times_total.append(total_time)
                times_subgraph.append(subgraph_time)
                times_momtrop.append(momtrop_time)
                times_integrand.append(integrand_time)
                times_unaccounted.append(unaccounted_time)

                print(f"Finished {cores=}, {samples=} in {total_time:.2f}s")
            
            gl_integrand.__del__()
            print(f"ended {cores=}")

            total_times.append(times_total)
            subgraph_times.append(times_subgraph)
            momtrop_times.append(times_momtrop)
            integrand_times.append(times_integrand)
            unaccounted_times.append(times_unaccounted)


        total_times = np.array(total_times).T
        subgraph_times = np.array(subgraph_times).T
        momtrop_times = np.array(momtrop_times).T
        integrand_times = np.array(integrand_times).T
        unaccounted_times = np.array(unaccounted_times).T
        all_times = [
            total_times,
            integrand_times,
            momtrop_times,
            subgraph_times,
            unaccounted_times
            ]
        headers = [
            "Total time",
            "Gammaloop integrand time",
            "Momtrop parameterisation time",
            "Subgraph completion time",
            "Unaccounted time",
            ]

        # Write all times to file
        padding = " "
        width_samples = 10
        width_times = 8
        width_total = width_samples + width_times*len(n_cores)
        dashed_line = width_total*"-"+"\n"
        empty_line = width_total*" "+"\n"

        with open(out_file, 'w') as f:
            print("got there")
            f.write("Multiprocessing efficiency for number of cores vs ")
            f.write(f"sample sizes of state {gammaloop_state} \n\n")

            f.write(f"Gammaloop batched_inspect for {n_samples[0]} samples took {\
                time_gl_batched_inspect:.2f}s \n \n")

            for header, times in zip(headers, all_times):
                f.write(dashed_line)
                f.write(f"{header:{'#'}^{width_total}}"+"\n")
                f.write(dashed_line+"\n")
                f.write(width_samples*padding)
                for cores in n_cores:
                    f.write(f"{str(cores):{padding}>{width_times}}")
                f.write("\n"+empty_line)

                for i in range(len(n_samples)):
                    f.write(f"{str(n_samples[i]):{padding}<{width_samples}}")
                    for t in times[i]:
                        f.write(f"{t:{width_times}.2f}")
                    f.write("\n")
                f.write("\n")

            f.write(dashed_line)
            f.write(f"{'Initialisation time':{'#'}^{width_total}}")
            f.write("\n")
            f.write(dashed_line+"\n")
            f.write(width_samples*padding)
            for cores in n_cores:
                f.write(f"{str(cores):{padding}>{width_times}}")
            f.write("\n"+empty_line)
            f.write(width_samples*padding)
            for t in initialisation_times:
                f.write(f"{t:{width_times}.2f}")
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        gl_integrand.end()
    finally:
        gl_integrand.end()

if __name__ == "__main__":
    main()
