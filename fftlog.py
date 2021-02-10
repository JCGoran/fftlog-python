import numpy as np
from scipy.interpolate import interp1d
from scipy.special import loggamma
from numpy.fft import rfft, irfft

def _select_bias(
    l: float,
    nu: float,
):
    """Computes the bias parameter q(nu); eq. (20) of https://arxiv.org/abs/1709.02401"""
    # these numbers were taken from https://github.com/hsgg/twoFAST.jl
    # they do not appear directly in https://arxiv.org/abs/1709.02401
    n1 = 0.9
    n2 = 0.9999
    qmin = max(n2 - 1.0 - nu, -l)
    qmax = min(n1 + 3.0 - nu, 2.0)
    qbest = (2. + n1 + n2 ) / 2. - nu
    q = qbest
    if not (qmin < q and q < qmax):
        q = (qmin + 2. * qmax) / 3.
    return q

def _window(
    value: float,
    xmin: float,
    xmax: float,
    xleft: float,
    xright: float,
):
    """Computes the window function"""
    result = 0
    if (xmin <= xleft and xleft <= xright and xright <= xmax):
        if (value > xleft and value < xright and value > xmin and value < xmax):
            result = 1.0
        elif (value <= xmin or value >= xmax):
            result = 0.0
        else:
            if (value < xleft and value > xmin):
                result = (value - xmin) / (xleft - xmin)
            elif (value > xright and value < xmax):
                result = (xmax - value) / (xmax - xright)
            result = result - np.sin(2 * np.pi * result) / 2. / np.pi
        return result

def _coefficients(
    t: float,
    q: float,
    l: float,
    alpha: float,
):
    """Computes the coefficients M^{q(nu)}_{\ell}; eq. (16) of https://arxiv.org/abs/1709.02401"""
    n = q - 1 - t * 1j
    return \
        pow(alpha, t * 1j - q) \
      * pow(2, n - 1) \
      * np.sqrt(np.pi) \
      * np.exp(
            loggamma((1 + l + n) / 2) - loggamma((2 + l - n) / 2)
        )


class FFTlog:

    def __init__(
        self,
        x,
        y,
        param_bessel: float, # formally the \ell parameter
        param_power: float, # formally the n parameter
        size: int, # number of sampling points for the FFTlog
        kind='cubic', # interpolation type; same options as `kind` parameter of `scipy.interpolate.interp1d`
    ):

        self.xmin = min(x)
        self.xmax = max(x)
        self.size = size
        self.param_bessel = param_bessel
        self.param_power = param_power
        self.x_fft = None
        self.y_fft = None

        # setting up the interpolation
        self._interpolation = interp1d(x, y, kind=kind)

    def _fft_input(
        self,
        q: float,
    ):
        halfsize = self.size // 2 + 1
        L = 2 * np.pi * self.size / np.log(self.xmax / self.xmin)

        input_x_mod = np.zeros(self.size)

        for i in range(self.size):
            input_x_mod[i] = self.xmin * pow(self.xmax / self.xmin, i / self.size)

        input_y_mod = np.zeros(self.size)

        for i in range(self.size):
            input_y_mod[i] = \
                pow(self.xmax / self.xmin, (3. - q) * i / self.size) \
               *self._interpolation(self.xmin * pow(self.xmax / self.xmin, i / self.size)) \
               *_window(
                    self.xmin*pow(self.xmax / self.xmin, i / self.size),
                    self.xmin,
                    self.xmin*pow(self.xmax / self.xmin, (self.size - 1) / self.size),
                    # these numbers were taken from https://github.com/hsgg/twoFAST.jl
                    # they do not appear directly in https://arxiv.org/abs/1709.02401
                    np.exp(0.46) * self.xmin,
                    np.exp(-0.46) * self.xmin * pow(self.xmax/self.xmin, (self.size - 1) / self.size)
                )

        input_y_fft = rfft(input_y_mod)

        output_y = np.zeros(halfsize, dtype = "complex_")

        for i in range(halfsize):
            output_y[i] = \
                _window(
                    input_x_mod[halfsize - 2 + i],
                    self.xmin,
                    self.xmin * pow(self.xmax / self.xmin, (self.size - 1) / self.size),
                    np.exp(0.46) * self.xmin,
                    np.exp(-0.46) * self.xmin * pow(self.xmax / self.xmin, (self.size - 1) / self.size)
                ) \
              * np.conj(input_y_fft[i]) \
              / L

        return output_y

    def transform(
        self,
        x0: float, # smallest value of the output; should be roughly 1 / max(x)
    ):
        halfsize = self.size // 2 + 1
        bias = _select_bias(self.param_bessel, self.param_power)
        G = np.log(self.xmax / self.xmin)

        input_y_fft = self._fft_input(
            bias + self.param_power,
        )

        output_x = np.array([
            x0 \
          * pow(self.xmax / self.xmin, i / self.size) \
            for i in range(self.size)
        ])

        prefactors = np.array([
            self.xmin**3 \
          * pow(self.xmax / self.xmin, -(bias + self.param_power) * i / self.size) \
          / np.pi \
          / pow(x0 * self.xmin, self.param_power) \
          / G \
            for i in range(self.size)
        ])

        temp_input = np.array(
            [
                input_y_fft[i] \
              * _coefficients(2 * np.pi * i / G, bias, self.param_bessel, self.xmin * x0) \
                for i in range(halfsize)
            ],
            dtype="complex_"
        )

        temp_output_y = irfft(temp_input)

        for i in range(self.size):
            temp_output_y[i] *= prefactors[i]

        self.x_fft = output_x
        self.y_fft = self.size * temp_output_y

        # in case users want to immediately assign the return values
        return self.x_fft, self.y_fft

