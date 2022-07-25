Welcome to rhos documentation!
====================================

rhos (recursive high order statistics) is a python module to compute recursive
mean, variance and
`high-order statistics <https://en.wikipedia.org/wiki/Higher-order_statistics>`_
on 1D signals.

Each function is implemented in pure python (functions ending in ``_py``,
useful for algorithm reference), as well as using a faster approach based
on :func:`scipy.signal.lfilter()`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   functions
   bibliography


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
