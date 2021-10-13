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
    r"""Computes the coefficients M^{q(nu)}_{\ell}; eq. (16) of https://arxiv.org/abs/1709.02401"""
    n = q - 1 - t * 1j
    return \
        pow(alpha, t * 1j - q) \
      * pow(2, n - 1) \
      * np.sqrt(np.pi) \
      * np.exp(
            loggamma((1 + l + n) / 2) - loggamma((2 + l - n) / 2)
        )


class Parameter:
    """Class containing the parameters for the FFTlog."""
    def __init__(self, param_bessel, param_power, size, x0):
        """Constructor."""
        if size < 0:
            raise ValueError(
                'The argument `size` cannot be negative.'
            )

        if int(param_bessel) != param_bessel or param_bessel < 0:
            raise TypeError(
                'The argument `param_bessel` must be a non-negative integer'
            )

        self._param_bessel = param_bessel
        self._param_power = param_power
        self._size = size
        self._x0 = x0

    def __str__(self):
        return str({
            'param_bessel' : self._param_bessel,
            'param_power' : self._param_power,
            'size' : self._size,
            'x0' : self._x0
        })

    def __eq__(self, a):
        if not isinstance(a, Parameter):
            raise TypeError

        return self._size == a.size \
        and self._param_bessel == a.param_bessel \
        and self._param_power == a.param_power \
        and self._x0 == a.x0

    @property
    def size(self):
        """Returns the size."""
        return self._size

    @property
    def param_bessel(self):
        """Returns the order of the spherical Bessel function."""
        return self._param_bessel

    @property
    def param_power(self):
        """Returns the inverse power with which the spherical Bessel function is multiplied."""
        return self._param_power

    @property
    def x0(self):
        """Returns the smallest value in the output x-array."""
        return self._x0


class FFTlog:
    """Main class for computing the FFTlog."""

    class _Wrapper:
        """Wrapper for Parameter class which has the output x-array and y-array."""
        def __init__(self, parameter, x, y):
            """Constructor."""
            if not isinstance(parameter, Parameter):
                raise TypeError
            self._parameter = parameter
            self._x = x
            self._y = y

        def __str__(self):
            return str(self._parameter)

        @property
        def parameter(self):
            """Returns an instance of the Parameter class."""
            return self._parameter

        @property
        def x(self):
            """Returns the output x-array."""
            return self._x

        @property
        def y(self):
            """Returns the output y-array."""
            return self._y

    def __init__(
        self,
        x,
        y,
        kind='cubic',
    ):
        """Constructor for the FFTlog class.
Parameters:
x : array
    An array of input values (the independent variable).
y : array
    An array of input values (the dependent variable).
kind : str, optional
    The type of interpolation to use;
    same options as `kind` parameter of `scipy.interpolate.interp1d`.
    Default: 'cubic'"""

        if not np.allclose(np.sort(x), x):
            raise ValueError(
                'The input x-array must be in ascending order.'
            )

        self._xmin = np.min(x)
        self._xmax = np.max(x)

        self._x_fft = None
        self._y_fft = None

        # setting up the interpolation
        if len(x) != len(y):
            raise ValueError(
                'The input x-array and input y-arrays must have the same size.'
            )
        self._interpolation = interp1d(x, y, kind=kind)

        # for keeping track of what's already been computed
        self._cache = []

    @property
    def interpolation(self):
        """Returns the interpolation object of the input."""
        return self._interpolation

    @property
    def x(self):
        """Returns the most recently computed output x-array."""
        return self._x_fft

    @property
    def y(self):
        """Returns the most recently computed output y-array."""
        return self._y_fft

    @property
    def cache(self):
        """Returns the list of already computed values."""
        return self._cache

    def cache_clear(self):
        """Clears the cache."""
        self._cache = []

    def _fft_input(
        self,
        q: float,
        size : int,
    ):
        halfsize = size // 2 + 1
        L = 2 * np.pi * size / np.log(self._xmax / self._xmin)

        input_x_mod = np.zeros(size)

        for i in range(size):
            input_x_mod[i] = self._xmin * pow(self._xmax / self._xmin, i / size)

        input_y_mod = np.zeros(size)

        for i in range(size):
            input_y_mod[i] = \
                pow(self._xmax / self._xmin, (3. - q) * i / size) \
               *self._interpolation(self._xmin * pow(self._xmax / self._xmin, i / size)) \
               *_window(
                    self._xmin*pow(self._xmax / self._xmin, i / size),
                    self._xmin,
                    self._xmin*pow(self._xmax / self._xmin, (size - 1) / size),
                    # these numbers were taken from https://github.com/hsgg/twoFAST.jl
                    # they do not appear directly in https://arxiv.org/abs/1709.02401
                    np.exp(0.46) * self._xmin,
                    np.exp(-0.46) * self._xmin * pow(self._xmax/self._xmin, (size - 1) / size)
                )

        input_y_fft = rfft(input_y_mod)

        output_y = np.zeros(halfsize, dtype = "complex_")

        for i in range(halfsize):
            output_y[i] = \
                _window(
                    input_x_mod[halfsize - 2 + i],
                    self._xmin,
                    self._xmin * pow(self._xmax / self._xmin, (size - 1) / size),
                    np.exp(0.46) * self._xmin,
                    np.exp(-0.46) * self._xmin * pow(self._xmax / self._xmin, (size - 1) / size)
                ) \
              * np.conj(input_y_fft[i]) \
              / L

        return output_y

    def transform(
        self,
        param_bessel : int,
        param_power : float,
        size : int,
        x0: float = None,
        cache : bool = True,
    ):
        """Performs the FFTlog transform.
Parameters:
param_bessel : int
    The order of the spherical Bessel function.
param_power : float
    The order of the power parameter.
size : int
    The number of sampling points for the FFTlog. Usually 1024 or above is sufficient.
x0 : float, optional
    The smallest value of the output x-array; should be roughly equal to 1 / max(input x-array).
    Default: 1 / max(input x-array)
cache : bool, optional
    Controls whether or not the cache should be used.
    Default: True"""

        x0 = x0 if x0 else 1. / self._xmax

        parameter = Parameter(
            param_bessel=param_bessel,
            param_power=param_power,
            size=size,
            x0=x0
        )

        if cache:
            # check if we haven't already computed this one; if we did, we
            # return the cached element
            for element in self._cache:
                if element.parameter == parameter:
                    self._x_fft = element.x
                    self._y_fft = element.y
                    return self.x, self.y

        halfsize = size // 2 + 1
        bias = _select_bias(param_bessel, param_power)
        G = np.log(self._xmax / self._xmin)

        input_y_fft = self._fft_input(
            bias + param_power,
            size
        )

        output_x = np.array([
            x0 \
          * pow(self._xmax / self._xmin, i / size) \
            for i in range(size)
        ])

        prefactors = np.array([
            self._xmin**3 \
          * pow(self._xmax / self._xmin, -(bias + param_power) * i / size) \
          / np.pi \
          / pow(x0 * self._xmin, param_power) \
          / G \
            for i in range(size)
        ])

        temp_input = np.array(
            [
                input_y_fft[i] \
              * _coefficients(2 * np.pi * i / G, bias, param_bessel, self._xmin * x0) \
                for i in range(halfsize)
            ],
            dtype="complex_"
        )

        temp_output_y = irfft(temp_input)

        for i in range(size):
            temp_output_y[i] *= prefactors[i]

        self._x_fft = output_x
        self._y_fft = size * temp_output_y

        if cache:
            self._cache.append(
                self._Wrapper(
                    parameter,
                    self._x_fft,
                    self._y_fft,
                )
            )

        # in case users want to immediately assign the return values
        return np.array((self._x_fft, self._y_fft))
