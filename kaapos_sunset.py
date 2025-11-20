# type: ignore

def main() -> None:
    import torch
    import time

    import src.parameterisation as par
    from src.integrand import ParameterisedIntegrand, MPIntegrand
    from src.helpers import error_fmter
    from src.parser import SettingsParser
    import signal

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        n_points = 200_000
        MAX_CHUNK_SIZE = 100_000
        n_cores = 8
        settings_path = "dev_settings/kaapos_sunset.toml"
        path_to_example = "/shared/vhirshi/nic/kaapos_dev/kaapos/examples/sunset"
        
        T = 1.0
        m_UV = 2*torch.pi * T
        mu = torch.pi * T * 0.0
        beta = 1.0 / T
        a = 0.95
        b = T
        params = [m_UV, mu, beta]

        torch.set_default_dtype(torch.float64)
        torch.manual_seed(1)
        Parser = SettingsParser(settings_path)
        Parser.dot_path = "/shared/vhirshi/nic/tropnis/sunset.dot"
        Parser.model_path = "/shared/vhirshi/nic/tropnis/legacy/gl_files/models/scalars.json"
        settings = Parser.settings
        graph_properties = Parser.get_graph_properties()

        param_settings = {
            'layer_0': {
                'param_type': 'kaapo',
                'mu': mu,
                'a': a,
                'b': b,
            }
        }
        """ 
        param_settings = {
            'layer_0': {
                'param_type': 'spherical',
                'conformal_scale': mu,
            }
        }
        """
        integrand_settings = {
            'integrand_type': 'kaapo',
            'params': params,
            'path_to_example': path_to_example,
        }

        hline = "----------------------------------------------------------------------------"
        print(hline)
        print("Testing Kaapo parameterisation and integrand")
        print(hline)

        integrand = MPIntegrand(
            graph_properties,
            param_settings,
            integrand_settings,
            verbose = True,
            n_cores = n_cores,
            return_layeroutput = True,
        )
        integrand.MAX_CHUNK_SIZE = MAX_CHUNK_SIZE
        n_continuous = integrand.continuous_dim
        xs = torch.rand(size=(n_points, n_continuous))
        input = par.LayerOutput(xs)
        output = integrand.eval_integrand(input)
        res, err = output.x_all.mean().item(), output.x_all.std().item() / n_points**0.5
        norm_factor = (2*torch.pi)**(-3*graph_properties.n_loops)
        print(f"Integration result: {error_fmter(res, err)}")
        print(f"Integration result: {res:.4g} +- {err:.4g}")
        print(f"Scaled integration result: {error_fmter(res * norm_factor, err * norm_factor)}")
        print(f"Scaled integration result: {res * norm_factor:.4g} +- {err * norm_factor:.4g}")
        timings = [f"{key}: {value:.4f}s" for key, value 
                in output.get_processing_times().items()]
        print(f"Timing: \n{timings}")

    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        integrand.end()
    finally:
        integrand.end()


if __name__ == "__main__":
    main()
