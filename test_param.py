# type: ignore

def main() -> None:
    import torch
    import time
    import toml

    import src.parameterisation as par
    from src.integrand import TestIntegrand, MPIntegrand, ParameterisedIntegrand, KaapoIntegrand
    from src.helpers import error_fmter

    dont_test_mp = False
    use_settings = True
    n_points = 10_000
    n_cores = 8
    n_points_mp = 50_000
    MAX_CHUNK_SIZE = 10_000
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

    kspace_integrand = TestIntegrand()
    xspace_integrand = TestIntegrand(const_f=True)

    time_last = time.perf_counter()

    spherical_kwargs = {"conformal_scale": 1.0,
                        "graph_properties": graph_properties,
                        "is_first_layer": True, }
    sph_param = par.SphericalParameterisation(**spherical_kwargs)
    inverse_spherical_kwargs = {"conformal_scale": 1.0,
                                "graph_properties": graph_properties, }
    inv_sph_param = par.InverseSphericalParameterisation(
        **inverse_spherical_kwargs)
    neutral_kwargs = {"conformal_scale": 1.0,
                    "graph_properties": graph_properties,
                    "next_param": inv_sph_param,
                    "is_first_layer": True, }
    neutral_param = par.SphericalParameterisation(**neutral_kwargs)
    momtrop_kwargs = {"graph_properties": graph_properties,
                    "is_first_layer": True, }
    momtrop_param = par.MomtropParameterisation(**momtrop_kwargs)
    kaapo_kwargs = {"graph_properties": graph_properties,
                    "is_first_layer": True,
                    "mu": 0.0001}
    kaapo_param = par.KaapoParameterisation(**kaapo_kwargs)
    print(f"Number of Momtrop contdim: {momtrop_param.layer_continuous_dim_in}")
    print(f"Initializing the Parameterisations took {
        - time_last + (time_last := time.perf_counter()):.2f}s")

    with open(settings_path, 'r') as f:
        settings = toml.load(f)
    param_settings = settings['parameterisation']
    layered_param = par.LayeredParameterisation(graph_properties, param_settings)
    discrete = -1*torch.ones(n_points).reshape(-1, 1)

    hline = "----------------------------------------------------------------------------"
    params = [sph_param, neutral_param, momtrop_param, kaapo_param, layered_param]
    integrands = [kspace_integrand, xspace_integrand,
                kspace_integrand, kspace_integrand, kspace_integrand]
    for param, grand in zip(params, integrands):
        param: par.Parameterisation
        print(hline)
        print(f"Testing {param.identifier.upper()}:")
        print(hline)
        try:
            n_continuous = param.chain_continuous_dim_in
        except:
            n_continuous = param.continuous_dim
        xs = torch.rand(size=(n_points, n_continuous))
        if param.identifier in ['momtrop', 'layered parameterisation']:
            xs = torch.hstack([xs, discrete])
        input = par.LayerOutput(xs)
        time_last = time.perf_counter()
        output = grand.evaluate_batch(param.parameterise(input))
        print(f"Evaluating {param.identifier.upper()} took {
            - time_last + (time_last := time.perf_counter()):.4f}s")
        res, err = output.x_all.mean().item(), output.x_all.std().item() / n_points**0.5
        print(f"Integration result: {res:.4f} +- {err:.4f}")
        # print(f"Integration result: {error_fmter(res, err)}")
        timings = [f"{key}: {value:.4f}s" for key,
                value in output.get_processing_times().items()]
        print(f"Timing: \n{timings}")

    print(hline)
    print("Testing LayerParameterisation")
    print(hline)

    print(hline)
    print("Testing discrete prior probability function.")
    print(hline)

    print("Matching discrete dims:")
    n_discrete = len(layered_param.discrete_dims)
    indices = torch.arange(n_discrete).reshape(1, -1)
    indices = torch.tile(indices, (3, 1))
    print(f"{indices=}")
    print(f"{layered_param.discrete_prior_prob_function(indices, n_discrete - 1)}")

    print("Only one:")
    print(
        f"{layered_param.discrete_prior_prob_function(indices[:, :1] + torch.arange(3).reshape(-1, 1), 0)}")

    print("Empty indices:")
    indices = torch.Tensor([]).reshape(1, -1)
    print(f"{layered_param.discrete_prior_prob_function(indices, -1)}")

    print(hline)
    print("Testing Kaapo parameterisation and integrand")
    print(hline)

    graph_properties.n_loops = 2
    path_to_example = "/shared/vhirshi/nic/kaapos_dev/kaapos/examples/sunset"
    m_UV = 2*torch.pi
    mu = torch.pi
    beta = 1.0
    kaapo_kwargs = {"graph_properties": graph_properties,
                    "is_first_layer": True,
                    "mu": mu,
                    "a": 0.4,
                    "b": 1.0,}
    kaapo_param = par.KaapoParameterisation(**kaapo_kwargs)
    n_continuous = kaapo_param.chain_continuous_dim_in
    kaapo_integrand = KaapoIntegrand(path_to_example,
                                     params=[m_UV, mu, beta],
                                     use_prec=True,)
    xs = torch.rand(size=(n_points, n_continuous))
    input = par.LayerOutput(xs)
    parameterised = kaapo_param.parameterise(input)
    output = kaapo_integrand.evaluate_batch(parameterised)
    res, err = output.x_all.mean().item(), output.x_all.std().item() / n_points**0.5
    print(f"Integration result: {res:.4f} +- {err:.4f}")
    timings = [f"{key}: {value:.4f}s" for key, value 
               in output.get_processing_times().items()]
    print(f"Timing: \n{timings}")
    graph_properties.n_loops = 1

    if dont_test_mp:
        quit()  

    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        print(hline)
        print("Testing multiprocessing (parameterised) integrand.")
        print(hline)
        with open('/shared/vhirshi/nic/tropnis/dev_settings/physical_1L_6photons.toml', 'r') as f:
            settings = toml.load(f)

        param_settings = settings['parameterisation']
        integrand_kwargs = {
            'integrand_type': 'gammaloop',
            'gammaloop_state_path': '/shared/vhirshi/gammaloop_hedge_numerator/gl_states/physical_1L_6photons_output',
            'eval_real_part': True,
        }
        time_last = time.perf_counter()
        integrand = MPIntegrand(
            graph_properties,
            param_settings,
            integrand_kwargs,
            n_cores=n_cores,
            verbose=False,
            return_layeroutput=True,
        )
        print(f"Initializing the MPIntegrand took {
            - time_last + (time_last := time.perf_counter()):.2f}s")
        integrand.MAX_CHUNK_SIZE = MAX_CHUNK_SIZE
        n_continuous = integrand.continuous_dim
        continuous = torch.rand(size=(n_points_mp, n_continuous))
        discrete = -1*torch.ones(n_points_mp).reshape(-1, 1)
        xs = torch.hstack([continuous, discrete])
        input = par.LayerOutput(xs)
        for _ in range(3):
            time_last = time.perf_counter()
            output = integrand.eval_integrand(input)
            output: par.LayerOutput
            timings = [f"{key}: {value:.4f}s" for key,
                       value in output.get_processing_times().items()]
            print(f"Timing: \n{timings}")
            print(f"Evaluating the integrand took {
                - time_last + (time_last := time.perf_counter()):.4f}s")
        x_all = output.x_all
        res, err = x_all.mean().item(), x_all.std().item() / n_points_mp**0.5
        print(f"Integration result: {error_fmter(res, err)}")

        print(hline)
        print("Testing discrete prior probability function.")
        print(hline)

        print("Matching discrete dims:")
        n_discrete = len(integrand.discrete_dims)
        indices = torch.arange(n_discrete).reshape(1, -1)
        indices = torch.tile(indices, (3, 1))
        print(f"{indices=}")
        print(f"{integrand.discrete_prior_prob_function(indices, n_discrete - 1)}")

        print("Only one:")
        print(
            f"{integrand.discrete_prior_prob_function(indices[:, :1] + torch.arange(3).reshape(-1, 1), 0)}")

        print("Empty indices:")
        indices = torch.Tensor([]).reshape(1, -1)
        print(f"{integrand.discrete_prior_prob_function(indices, -1)}")

    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        integrand.end()
    finally:
        integrand.end()


if __name__ == "__main__":
    main()
