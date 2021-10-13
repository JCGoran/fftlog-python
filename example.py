import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import fftlog

BENCHMARK_DIR = 'benchmark/'

def plot_benchmark(
    input_file : str = 'power_test.dat',
    size : int = 2048,
    savepath : str = '',
):
    """Plots the result of the FFTlog with a known benchmark for various inputs."""

    # load a simple power spectrum (generated with CLASS)
    data = np.loadtxt(input_file)

    k = data[:, 0]
    pk = data[:, 1]

    # map between the benchmark integral filenames and the pairs (l, n)
    integrals = {
        0 : (0, 0),
        1 : (2, 0),
        2 : (4, 0),
        3 : (1, 1),
        4 : (3, 1),
        5 : (0, 2),
        6 : (2, 2),
        7 : (1, 3),
    }

    # input is k [array], pk [array]
    result = fftlog.FFTlog(k, pk)

    scaling = 8
    fig, axes = plt.subplots(
        nrows=len(integrals),
        ncols=2,
        figsize=(scaling, scaling * len(integrals) / 3)
    )

    # set sensible limits
    for index, axis in enumerate(axes):
        l, n = integrals[index]

        # this does the actual transform with the input being:
        # l [int], n [double or int], r_0 [double]
        # (in this case, since input k is in [h/Mpc], and input P(k) is in [Mpc/h]^3,
        # this needs to be in Mpc/h, i.e. here we set r_0 = 1 Mpc/h)
        # NOTE: results are stored in class members `x` (the separations), and `y` (the FFTlog)

        result.transform(param_bessel=l, param_power=n, size=size, x0=1)

        # we don't care about results above 1000 Mpc/h
        criterion = result.x <= 1000

        axis_plot, axis_error = axis

        axis_plot.set_xlim(
            min(result.x[criterion]),
            max(result.x[criterion]),
        )

        # plot r, FFTlog(r) * r^2
        axis_plot.plot(
            result.x,
            result.y * (result.x)**2,
            label='FFTlog Python',
        )

        # benchmark to compare it with
        benchmark = np.loadtxt(os.path.join(BENCHMARK_DIR, f'integral{index}.dat'))

        # the integral needs rescaling as benchmark uses dimensionless units
        factor = 2997.9

        # benchmark is such that it outputs output_x^(n - l) * output_y if n > l,
        # so we rescale it back;
        # the epsilon is needed so we don't divide by zero
        epsilon = 1e-20
        if n > l:
            benchmark[:, 1] = benchmark[:, 1] / (benchmark[:, 0]**(n - l) + epsilon)

        # plot the benchmark
        axis_plot.plot(
            benchmark[:, 0] * factor,
            benchmark[:, 1] * (factor * benchmark[:, 0])**2,
            ls='--',
            label='benchmark',
        )
        axis_plot.set_title(f'l = {l}, n = {n}')
        axis_plot.set_xlabel('r [Mpc/h]')
        axis_plot.set_ylabel('ξ(r) × r² [Mpc²/h²]')
        axis_plot.grid()
        axis_plot.legend()

        fftlog_interp = interp1d(
            result.x[criterion],
            result.y[criterion],
            kind='cubic',
            fill_value='extrapolate'
        )
        benchmark_interp = interp1d(
            factor * benchmark[:, 0],
            benchmark[:, 1],
            kind='cubic',
            fill_value='extrapolate'
        )

        axis_error.set_xlim(
            min(result.x[criterion]),
            max(result.x[criterion]),
        )

        # plot the error w.r.t. the benchmark
        axis_error.plot(
            result.x[criterion],
            100 * np.abs(
                1 - fftlog_interp(result.x[criterion]) / benchmark_interp(result.x[criterion])
            ),
        )

        axis_error.set_yscale('log')
        axis_error.set_xlabel('r [Mpc/h]')
        axis_error.set_ylabel('rel. err. (in %)')
        axis_error.grid()

    fig.tight_layout()

    # if we're saving it, we don't use `plt.show`
    if savepath:
        fig.savefig(savepath, dpi=300)
    # otherwise, display it
    else:
        plt.show()

if __name__ == '__main__':
    plot_benchmark()
