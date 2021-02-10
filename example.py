import numpy as np
import fftlog
import matplotlib.pyplot as plt

# load a simple power spectrum (generated with CLASS)
data = np.loadtxt('power_test.dat')

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

benchmark = 0
l, n = integrals[benchmark]

# input is k [array], pk [array], l [double or int], n [double or int], sampling points [int]
result = fftlog.FFTlog(k, pk, l, n, 2048)

# this does the actual transform with the input being r_0
# (in this case, since input k is in [h/Mpc], and input P(k) is in [Mpc/h]^3,
# this needs to be in Mpc/h, i.e. here we set r_0 = 1 Mpc/h)
# NOTE: results are stored in class members `x_fft` (the separations), and `y_fft` (the FFTlog)
result.transform(1)

# we don't care about results above 1000 Mpc/h
criterion = result.x_fft <= 1000

# set sensible limits
plt.xlim(
    min(result.x_fft[criterion]),
    max(result.x_fft[criterion]),
)

# plot r, FFTlog(r) * r^2
plt.plot(
    result.x_fft,
    result.y_fft * (result.x_fft)**2,
    label='FFTlog Python',
)

# benchmark to compare it with
benchmark = np.loadtxt(f'integral{benchmark}.dat')

# the integral needs rescaling as benchmark uses dimensionless units
factor = 2997.9

# benchmark is such that it outputs output_x^(n - l) * output_y if n > l, so we rescale it back;
# the epsilon is needed so we don't divide by zero
epsilon = 1e-20
if n > l:
    benchmark[:, 1] = benchmark[:, 1] / (benchmark[:, 0]**(n - l) + epsilon)

# plot the benchmark
plt.plot(
    benchmark[:, 0] * factor,
    benchmark[:, 1] * (factor * benchmark[:, 0])**2,
    ls='--',
    label='benchmark',
)

plt.grid()
plt.legend()

# display it
plt.show()
