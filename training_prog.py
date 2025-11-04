# type: ignore
import torch
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import signal
from datetime import datetime
from time import time
from madnis.integrator import Integrator
from resources.tropnis import GammaLoopIntegrand
from resources.helpers import PATHS, error_fmter


def main() -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        parser = argparse.ArgumentParser(prog="training_prog")
        parser.add_argument('--settings', '-s', type=str,
                            help="The settings .toml file.")
        parser.add_argument('--comment', type=str, default='No comment.',
                            help="Add a comment to the output summary file.")
        parser.add_argument('--no_output', action='store_true', default=False,
                            help="Enable this flag to not output plot/summary file.")
        args = parser.parse_args()
        settings_file = args.settings

        print(f"Working on settings {settings_file}")
        if not args.no_output:
            subfolder_path = os.path.join(
                PATHS['tropnis'], "outputs", "training_prog")
            print(f"Output will be at {subfolder_path}")

        # Initialize the gammaloop integrand and madnis integrator
        torch.set_default_dtype(torch.float64)
        time_last = time()
        integrand = GammaLoopIntegrand(
            settings_file)
        discrete_model = integrand.settings['tropnis']['discrete_model']
        discrete_model_params = integrand.momtrop_integrand.Parser.settings[discrete_model]
        batch_size = integrand.settings['tropnis']['batch_size']
        gammaloop_state = integrand.settings['gammaloop_state']['state_name']
        integrator = Integrator(
            integrand.madnis_integrand(),
            discrete_model=discrete_model,
            discrete_flow_kwargs=discrete_model_params,
            batch_size=batch_size,
        )
        print(f"Initializing the Integrand and Integrator took {
            - time_last + (time_last := time()):.2f}s")

        # Training parameters
        params = integrand.settings['plotting_params']['training_prog']
        n_training_steps = params['n_training_steps']
        n_log = params['n_log']
        n_plot_rsd = params['n_plot_rsd']
        n_plot_loss = params['n_plot_loss']
        n_samples = params['n_samples']
        n_samples_after_training = params['n_samples_after_training']

        # Callback for the madnis integrator
        def callback(status) -> None:
            step = status.step + 1
            if step % n_log == 0:
                print(f"Step {status.step + 1}: loss={status.loss:.5f}")
            if step % n_plot_loss == 0:
                losses.append(status.loss)
                steps_losses.append(step)
            if step % n_plot_rsd == 0:
                metrics = integrator.integration_metrics(n_samples)
                rsd = metrics.rel_stddev
                print(f"Trained Result after {step} steps of {batch_size}: {
                    metrics.integral:.8g} +- {metrics.error:.8g}, RSD = {rsd:.3f}")
                rsds.append(rsd)
                steps_rsds.append(step)

        # Parse GammaLoop results
        gl_res = integrand.momtrop_integrand.Parser.get_gammaloop_integration_result()
        if gl_res is not None:
            RE_OR_IM = 're' if integrand.settings['tropnis']['evaluate_real_part'] else 'im'
            gl_int = gl_res['result'][RE_OR_IM]
            gl_err = gl_res['error'][RE_OR_IM]
            gl_neval = gl_res['neval']
            gl_rsd = abs(gl_err / gl_int) * math.sqrt(gl_neval)

            print(
                f"Gammaloop Result: {gl_int:.8g} +- {gl_err:.8g}, RSD = {gl_rsd:.3f}")

        time_last = time()
        metrics = integrator.integration_metrics(n_samples)
        print(f"Evaluating {n_samples} samples using {integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s")

        momtrop_int = metrics.integral
        momtrop_err = metrics.error
        momtrop_rsd = metrics.rel_stddev
        print(
            f"Momtrop Result (before training) using {n_samples} samples: {
                momtrop_int:.8g} +- {momtrop_err:.8g}, RSD = {momtrop_rsd:.3f}")

        # Plotting setup
        losses = []
        rsds = [metrics.rel_stddev]
        steps_losses = []
        steps_rsds = [0]

        integrator.train(n_training_steps, callback)

        if gl_res is not None:
            print(
                f"Gammaloop Result: {gl_int:.8g} +- {gl_err:.8g}, RSD = {gl_rsd:.3f}")

        # Take the final snapshot
        metrics = integrator.integration_metrics(n_samples_after_training)
        trained_int = metrics.integral
        trained_err = metrics.error
        trained_rsd = metrics.rel_stddev
        print(f"Trained Result after {integrator.step} steps of {batch_size}, using a sample size of {n_samples_after_training}: {
            trained_int:.8g} +- {trained_err:.8g}, RSD = {trained_rsd:.3f}")

        # IMPORTANT: close the worker functions, or your script will hang
        integrand.end()

        if args.no_output:
            quit()

        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        losses, steps_losses = np.array(losses), np.array(steps_losses)
        rsds, steps_rsds = np.array(rsds), np.array(steps_rsds)

        fig, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
        axs[0].plot(steps_losses, losses)
        axs[0].set_ylabel("loss")
        axs[1].scatter(steps_rsds, rsds)
        axs[1].set_ylabel("RSD")
        axs[1].set_xlabel("Training steps")
        fig.suptitle(f"Training progression for {gammaloop_state}")
        filename = gammaloop_state+datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        plt.savefig(os.path.join(subfolder_path, filename+".png"),
                    dpi=300, bbox_inches='tight')

        with open(os.path.join(subfolder_path, filename+".txt"), 'w') as f:
            sep = '-'
            width = 60
            line = width*sep+"\n"
            f.write(f"Comment: {args.comment} \n")
            f.write(line)
            f.write(f"{' Training Parameters ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"{gammaloop_state=}\n")
            f.write(f"{batch_size=}\n")
            f.write(f"{n_training_steps=}\n")
            f.write(f"{discrete_model=}\n")
            f.write(f"Integrated phase: {RE_OR_IM}\n")
            try:
                for key, value in discrete_model_params.items():
                    f.write(f"{key}={value}\n")
            except:
                pass
            if gl_res is not None:
                f.write(f"\n{line}")
                f.write(f"{' Gammaloop Results ':{'#'}^{width}}\n")
                f.write(line)
                f.write(f"Integral: {error_fmter(gl_int, gl_err)}\n")
                f.write(f"RSD: {gl_rsd:.3f}\n")
                f.write(f"Number of samples: {gl_neval}\n")
            f.write(f"\n{line}")
            f.write(f"{' Momtrop Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"Integral: {error_fmter(momtrop_int, momtrop_err)}\n")
            f.write(f"RSD: {momtrop_rsd:.3f}\n")
            f.write(f"Number of samples: {n_samples}\n")
            f.write(f"\n{line}")
            f.write(f"{' Trained Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"Integral: {error_fmter(trained_int, trained_err)}\n")
            f.write(f"RSD: {trained_rsd:.3f}\n")
            f.write(f"Number of samples: {n_samples_after_training}\n")
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        integrand.end()
    finally:
        integrand.end()


if __name__ == "__main__":
    main()
