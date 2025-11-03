# type: ignore
import torch
import math
import argparse
import signal
from time import time
from madnis.integrator import Integrator
from resources.tropnis import GammaLoopIntegrand


def main() -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        parser = argparse.ArgumentParser(prog="gammaloop_state_test")
        parser.add_argument('--settings', '-s', type=str,
                            help="The settings .toml file.")
        args = parser.parse_args()
        settings_file = args.settings

        # Initialize the gammaloop integrand and madnis integrator
        torch.set_default_dtype(torch.float64)

        time_last = time()
        integrand = GammaLoopIntegrand(
            settings_file)
        discrete_model = integrand.settings['tropnis']['discrete_model']
        batch_size = integrand.settings['tropnis']['batch_size']
        integrator = Integrator(
            integrand.madnis_integrand(),
            discrete_model=discrete_model,
            discrete_flow_kwargs=integrand.momtrop_integrand.Parser.settings[discrete_model],
            batch_size=batch_size,
        )
        print(f"Initializing the Integrand and Integrator took {
            - time_last + (time_last := time()):.2f}s")
        
        # Training parameters
        n_training_steps = 1

        # Parse GammaLoop results
        gl_res = integrand.momtrop_integrand.Parser.get_gammaloop_integration_result()
        if gl_res is not None:
            RE_OR_IM = 're' if integrand.settings['tropnis']['evaluate_real_part'] else 'im'
            gl_int = gl_res['result'][RE_OR_IM]
            gl_err = gl_res['error'][RE_OR_IM]
            gl_rsd = abs(gl_err / gl_int) * math.sqrt(gl_res['neval'])

            print(f"Gammaloop Result:    {
                gl_int:.8g} +- {gl_err:.8g}, RSD = {gl_rsd:.2f}")

        time_last = time()
        metrics = integrator.integration_metrics(batch_size)
        print(f"Evaluating {batch_size} samples using {integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s")

        print(f"Momtrop Result (before training) using {batch_size} samples:     {
            metrics.integral:.8g} +- {metrics.error:.8g}, RSD = {metrics.rel_stddev:.2f}")

        def callback(status) -> None:
            print(f"Step {status.step + 1}: loss={status.loss:.5f}")

        integrator.train(n_training_steps, callback)

        print(f"Test successfully completed!")
        print(f"The gammaloop state {integrand.settings['gammaloop_state']['state_name']} should be good to go.")
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        integrand.end()
    finally:
        integrand.end()

if __name__ == "__main__":
    main()
