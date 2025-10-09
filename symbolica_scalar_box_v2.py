# type: ignore
from symbolica import NumericalIntegrator, Sample
import gammaloop
import numpy as np
import math

state = gammaloop.GammaLoopState(
    "/Users/vjhirsch/Documents/Work/gammaloop_hedge_numerator/NicFink/scalar_box_output")

# Conformal mapping
scale = 0.3


def conf(c: float):
    return scale*math.log(c/(1-c))

# Derivate of the conformal mapping


def d_conf(c: float):
    return scale/(c-c**2)


def cartesian_parameterize(xs: list[float]) -> tuple[list[float], float]:
    x, y, z = xs

    cartesian = [conf(x), conf(y), conf(z)]
    jacobian = d_conf(x)*d_conf(y)*d_conf(z)
    return cartesian, jacobian


def spherical_parameterize(xs: list[float]) -> tuple[list[float], float]:
    x, y, z = xs

    r = x/(1-x)
    cos_az = 2*y-1
    sin_az = math.sqrt(abs(1 - cos_az**2))
    pol = 2*math.pi*z
    spherical = [
        scale*r*sin_az*math.cos(pol),
        scale*r*sin_az*math.sin(pol),
        scale*r*cos_az
    ]

    jacobian = 4*math.pi*scale**3*x**2/(1-x)**4

    return spherical, jacobian


def integrand_momentum_space(samples: list[Sample], parameterize):
    loop_momenta = [parameterize(sample.c)[0] for sample in samples]
    param_jac = np.array([parameterize(sample.c)[1] for sample in samples])
    loop_momenta = np.array(loop_momenta)
    discrete_dims = np.array(len(samples)*[[0]], dtype=np.uint64)

    # Try integrating the unit hypercube, integral should yield 1
    # res = np.array([complex(0.0, 0.0) if any(lmi > 1. or lmi < 0. for lmi in lm)
    #                else complex(1.0, 0.0) for lm in loop_momenta], dtype=np.complex128)
    res, gl_jac = state.batched_inspect(
        points=loop_momenta, momentum_space=True,
        process_id=0,
        integrand_name='default',
        use_f128=False,  discrete_dims=discrete_dims
    )

    # return (res.real*gl_jac*param_jac).tolist()
    return (res.real*param_jac).tolist()


def integrand_momentum_space_non_batched(samples: list[Sample], parameterize):
    loop_momenta = [parameterize(sample.c)[0] for sample in samples]
    param_jac = np.array([parameterize(sample.c)[1] for sample in samples])
    loop_momenta = np.array(loop_momenta)
    discrete_dims = np.array(len(samples)*[[0]], dtype=np.uint64)

    # Try integrating the unit hypercube, integral should yield 1
    # res = np.array([complex(0.0, 0.0) if any(lmi > 1. or lmi < 0. for lmi in lm)
    #                else complex(1.0, 0.0) for lm in loop_momenta], dtype=np.complex128)
    res = []
    for lm, dm in zip(loop_momenta, discrete_dims):
        r = state.inspect(
            point=lm, momentum_space=True,
            process_id=0,
            integrand_name='default',
            use_f128=False,  discrete_dim=dm,
            force_radius=False
        )
        res.append(r[0])
    res = np.array(res, dtype=np.complex128)
    # return (res.real*gl_jac*param_jac).tolist()
    return (res.real*param_jac).tolist()


def integrand_momentum_space_cartesian(samples: list[Sample]):
    return integrand_momentum_space(samples, cartesian_parameterize)


def integrand_momentum_space_spherical(samples: list[Sample]):
    return integrand_momentum_space(samples, spherical_parameterize)


def integrand_x_space(samples: list[Sample]):
    samples = [sample.c for sample in samples]
    xs = np.array(samples)
    discrete_dims = np.array(len(samples)*[[0]], dtype=np.uint64)
    res, _ = state.batched_inspect(
        points=xs, momentum_space=False,
        process_id=0,
        integrand_name='default',
        use_f128=False,  discrete_dims=discrete_dims
    )

    return res.real.tolist()


print("---------- BEGINNING X SPACE -----------")
NumericalIntegrator.continuous(3).integrate(integrand_x_space, show_stats=True,
                                            max_n_iter=3, n_samples_per_iter=1000, min_error=1e-4)

print("---------- BEGINNING CARTESIAN -----------")
NumericalIntegrator.continuous(3).integrate(integrand_momentum_space_cartesian, show_stats=True,
                                            max_n_iter=3, n_samples_per_iter=1000, min_error=1e-4)

print("---------- BEGINNING SPHERICAL -----------")
NumericalIntegrator.continuous(3).integrate(integrand_momentum_space_spherical, show_stats=True,
                                            max_n_iter=3, n_samples_per_iter=1000, min_error=1e-4)
