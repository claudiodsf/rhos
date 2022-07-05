# -*- coding: utf8 -*-
"""
Recursive high-order statistics for Python.

:copyright:
    2022 Claudio Satriano <satriano@ipgp.fr>
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from ._version import get_versions
import numpy as np
from scipy.signal import lfilter
from typing import TypeVar
ArrayLike = TypeVar('ArrayLike')
__version__ = get_versions()['version']
del get_versions


def rec_mean_py(signal: ArrayLike, C: float) -> np.ndarray:
    """
    Recursive mean of a signal.

        μ[i] = C·signal[i] + (1-C)·μ[i-1]


    Parameters
    ----------
    signal : ArrayLike
        signal to compute recursive mean for
    C : float
        decay constant, in the [0, 1] interval

    Returns
    -------
    numpy.ndarray
        the recursive mean, with the same length than signal

    Raises
    ------
    ValueError
        if C is not in the [0, 1] interval

    Warning
    -------
    This is a pure python reference implementation.
    Use :func:`recursive_mean` for a faster implementation.

    """
    signal = np.asarray(np.atleast_1d(signal), dtype=float)
    C = float(C)
    if not 0 <= C <= 1:
        msg = 'C must be in the [0, 1] interval'
        raise ValueError(msg)
    mean = np.zeros_like(signal)
    for i in range(len(signal)):
        mean[i] = C * signal[i] + (1 - C) * mean[i-1]
    return mean


def rec_mean(signal: ArrayLike, C: float) -> np.ndarray:
    """
    Recursive mean of a signal.

        μ[i] = C·signal[i] + (1-C)·μ[i-1]


    Parameters
    ----------
    signal : ArrayLike
        signal to compute recursive mean for
    C : float
        decay constant, in the [0, 1] interval

    Returns
    -------
    numpy.ndarray
        the recursive mean, with the same length than signal

    Raises
    ------
    ValueError
        if C is not in the [0, 1] interval

    Note
    ----
    Fast implementation, using :func:`scipy.signal.lfilter()`.

    """
    signal = np.asarray(np.atleast_1d(signal), dtype=float)
    C = float(C)
    if not 0 <= C <= 1:
        msg = 'C must be in the [0, 1] interval'
        raise ValueError(msg)
    a = (1, -(1-C))
    b = (C, )
    mean = lfilter(b, a, signal)
    return mean


def rec_variance_py(
        signal: ArrayLike,
        C: float,
        definition: int = 0) -> np.ndarray:
    """
    Recursive variance of a signal.

    Defined as in :cite:t:`Poiata2016` (definition 0):

        σ²[i] = C·(signal[i]-μ[i-1])² + (1-C)·σ²[i-1]

    Or, defined as in :cite:t:`Langet2014` (definition 1):

        σ²[i] = C·(signal[i]-μ[i])² + (1-C)·σ²[i-1]

    For both definitions:

        μ[i] = C·signal[i] + (1-C)·μ[i-1]


    Parameters
    ----------
    signal : ArrayLike
        signal to compute recursive variance for
    C : float
        decay constant, in the [0, 1] interval
    definition : int
        which formula to use

    Returns
    -------
    numpy.ndarray
        the recursive variance, with the same length than signal

    Raises
    ------
    ValueError
        if C is not in the [0, 1] interval
    ValueError
        if definition is not 0 or 1

    Warning
    -------
    This is a pure python reference implementation.
    Use :func:`recursive_variance` for a faster implementation.

    """
    signal = np.asarray(np.atleast_1d(signal), dtype=float)
    C = float(C)
    definition = int(definition)
    if not 0 <= C <= 1:
        msg = 'C must be in the [0, 1] interval'
        raise ValueError(msg)
    if definition not in [0, 1]:
        msg = 'definition must be either 0 or 1'
        raise ValueError(msg)
    mean = np.zeros_like(signal)
    var = np.zeros_like(signal)
    if definition == 0:
        for i in range(len(signal)):
            mean[i] = C * signal[i] + (1 - C) * mean[i-1]
            var[i] = C * (signal[i] - mean[i-1])**2 + (1 - C) * var[i-1]
    elif definition == 1:
        for i in range(len(signal)):
            mean[i] = C * signal[i] + (1 - C) * mean[i-1]
            var[i] = C * (signal[i] - mean[i])**2 + (1 - C) * var[i-1]
    return var


def rec_variance(
        signal: ArrayLike,
        C: float,
        definition: int = 0) -> np.ndarray:
    """
    Recursive variance of a signal.

    Defined as in :cite:t:`Poiata2016` (definition 0):

        σ²[i] = C·(signal[i]-μ[i-1])² + (1-C)·σ²[i-1]

    Or, defined as in :cite:t:`Langet2014` (definition 1):

        σ²[i] = C·(signal[i]-μ[i])² + (1-C)·σ²[i-1]

    For both definitions:

        μ[i] = C·signal[i] + (1-C)·μ[i-1]


    Parameters
    ----------
    signal : ArrayLike
        signal to compute recursive variance for
    C : float
        decay constant, in the [0, 1] interval
    definition : int
        which formula to use

    Returns
    -------
    numpy.ndarray
        the recursive variance, with the same length than signal

    Raises
    ------
    ValueError
        if C is not in the [0, 1] interval
    ValueError
        if definition is not 0 or 1

    Note
    ----
    Fast implementation, using :func:`scipy.signal.lfilter()`.

    """
    signal = np.asarray(np.atleast_1d(signal), dtype=float)
    C = float(C)
    definition = int(definition)
    if not 0 <= C <= 1:
        msg = 'C must be in the [0, 1] interval'
        raise ValueError(msg)
    if definition not in [0, 1]:
        msg = 'definition must be either 0 or 1'
        raise ValueError(msg)
    a = (1, -(1-C))
    b = (C, )
    mean = lfilter(b, a, signal)
    if definition == 0:
        dev = signal[:]**2
        dev[1:] = (signal[1:] - mean[:-1])**2
    elif definition == 1:
        dev = (signal - mean)**2
    var = lfilter(b, a, dev)
    return var


def rec_hos_py(
        signal: ArrayLike,
        C: float,
        order: int = 4,
        var_min: float = -1,
        definition: int = 0) -> np.ndarray:
    """
    Recursive high order statistics (hos) of a signal.

    Defined as in `BackTrackBB <https://backtrackbb.github.io>`_
    (definition 0):

        hos[i] = C·(signal[i]-μ[i-1])ⁿ / (σ²[i])ⁿᐟ² + (1-C)·hos[i-1]

    with

        σ²[i] = C·(signal[i]-μ[i-1])² + (1-C)·σ²[i-1]

    Note
    ----
    This is the actual implementation in the BackTrackBB source code, which
    does not correspond to equation 7 of :cite:t:`Poiata2016` or
    equation 1 of :cite:t:`Poiata2018`.


    Or, defined as in :cite:t:`Langet2014` (definition 1):

        hos[i] = C·(signal[i]-μ[i])ⁿ / (σ²[i])ⁿᐟ² + (1-C)·hos[i-1]

    with

        σ²[i] = C·(signal[i]-μ[i])² + (1-C)·σ²[i-1]

    For both definitions:

        μ[i] = C·signal[i] + (1-C)·μ[i-1]


    Parameters
    ----------
    signal : ArrayLike
        signal to compute recursive hos for
    C : float
        decay constant, in the [0, 1] interval
    order : int
        hos order
    var_min : float
        values of variance σ² (hos denominator) smaller than
        `var_min` will be replaced by `var_min`
    definition : int
        which formula to use

    Returns
    -------
    numpy.ndarray
        the recursive hos, with the same length than signal

    Raises
    ------
    ValueError
        if C is not in the [0, 1] interval
    ValueError
        if definition is not 0 or 1

    Warning
    -------
    This is a pure python reference implementation.
    Use :func:`recursive_hos` for a faster implementation.

    """
    signal = np.asarray(np.atleast_1d(signal), dtype=float)
    C = float(C)
    order = int(order)
    var_min = float(var_min)
    definition = int(definition)
    if not 0 <= C <= 1:
        msg = 'C must be in the [0, 1] interval'
        raise ValueError(msg)
    if definition not in [0, 1]:
        msg = 'definition must be either 0 or 1'
        raise ValueError(msg)
    mean = np.zeros_like(signal)
    var = np.ones_like(signal)
    hos = np.zeros_like(signal)
    n_win = int(1./C)
    # initialize:
    for i in range(n_win):
        mean[-1] = C * signal[i] + (1 - C) * mean[-1]
        var[-1] = C * (signal[i] - mean[-1])**2 + (1 - C) * var[-1]
    if definition == 0:
        for i in range(len(signal)):
            mean[i] = C * signal[i] + (1 - C) * mean[i-1]
            var[i] = C * (signal[i] - mean[i-1])**2 + (1 - C) * var[i-1]
            if var[i] > var_min:
                _var = var[i] or 1e-9
            else:
                _var = var_min
            hos[i] = C * ((signal[i] - mean[i-1])**order / _var**(order/2))
            hos[i] += (1 - C) * hos[i-1]
    elif definition == 1:
        for i in range(len(signal)):
            mean[i] = C * signal[i] + (1 - C) * mean[i-1]
            var[i] = C * (signal[i] - mean[i])**2 + (1 - C) * var[i-1]
            if var[i] > var_min:
                _var = var[i] or 1e-9
            else:
                _var = var_min
            hos[i] = C * ((signal[i] - mean[i])**order / _var**(order/2))
            hos[i] += (1 - C) * hos[i-1]
    return hos


def rec_hos(
        signal: ArrayLike,
        C: float,
        order: int = 4,
        var_min: float = -1,
        definition: int = 0) -> np.ndarray:
    """
    Recursive high order statistics (hos) of a signal.

    Defined as in `BackTrackBB <https://backtrackbb.github.io>`_
    (definition 0):

        hos[i] = C·(signal[i]-μ[i-1])ⁿ / (σ²[i])ⁿᐟ² + (1-C)·hos[i-1]

    with

        σ²[i] = C·(signal[i]-μ[i-1])² + (1-C)·σ²[i-1]

    Note
    ----
    This is the actual implementation in the BackTrackBB source code, which
    does not correspond to equation 7 of :cite:t:`Poiata2016` or
    equation 1 of :cite:t:`Poiata2018`.


    Or, defined as in :cite:t:`Langet2014` (definition 1):

        hos[i] = C·(signal[i]-μ[i])ⁿ / (σ²[i])ⁿᐟ² + (1-C)·hos[i-1]

    with

        σ²[i] = C·(signal[i]-μ[i])² + (1-C)·σ²[i-1]

    For both definitions:

        μ[i] = C·signal[i] + (1-C)·μ[i-1]


    Parameters
    ----------
    signal : ArrayLike
        signal to compute recursive hos for
    C : float
        decay constant, in the [0, 1] interval
    order : int
        hos order
    var_min : float
        values of variance σ² (hos denominator) smaller than
        `var_min` will be replaced by `var_min`
    definition : int
        which formula to use

    Returns
    -------
    numpy.ndarray
        the recursive hos, with the same length than signal

    Raises
    ------
    ValueError
        if C is not in the [0, 1] interval
    ValueError
        if definition is not 0 or 1

    Note
    ----
    Fast implementation, using :func:`scipy.signal.lfilter()`.

    """
    signal = np.asarray(np.atleast_1d(signal), dtype=float)
    C = float(C)
    order = int(order)
    var_min = float(var_min)
    definition = int(definition)
    if not 0 <= C <= 1:
        msg = 'C must be in the [0, 1] interval'
        raise ValueError(msg)
    # initialize:
    n_win = int(1./C)
    mean0 = 0
    var0 = 1
    for i in range(n_win):
        mean0 = C * signal[i] + (1 - C) * mean0
        var0 = C * (signal[i]-mean0)**2 + (1 - C) * var0
    a = (1, -(1-C))
    b = (C, )
    mean, _ = lfilter(b, a, signal, zi=[(1-C)*mean0])
    dev = signal**2
    dev[1:] = (signal[1:] - mean[:-1])**2
    var, _ = lfilter(b, a, dev, zi=[(1-C)*var0])
    dev = signal**order
    dev[1:] = (signal[1:] - mean[:-1])**order
    var[var == 0] = 1e-9
    var[var < var_min] = var_min
    dev /= var**(order/2)
    hos = lfilter(b, a, dev)
    return hos
