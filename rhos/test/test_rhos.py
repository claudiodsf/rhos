# -*- coding: utf8 -*-
"""
Unit tests for rhos.

:copyright:
    2022 Claudio Satriano <satriano@ipgp.fr>
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import sys
from pathlib import Path
if __name__ == '__main__' and __package__ is None:
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[1]
    sys.path.insert(0, str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:  # Already removed
        pass
    __package__ = 'rhos'
from rhos import (
    rec_mean, rec_mean_py,
    rec_variance, rec_variance_py,
    rec_hos, rec_hos_py)
import numpy as np
import unittest
from numpy.testing import assert_allclose


class TestRHOS(unittest.TestCase):
    def setUp(self):
        self.signal = np.zeros(20)
        self.signal[2] = 1
        self.out_expected_rec_mean = np.array([
            0.00000000e+00, 0.00000000e+00, 5.00000000e-01, 2.50000000e-01,
            1.25000000e-01, 6.25000000e-02, 3.12500000e-02, 1.56250000e-02,
            7.81250000e-03, 3.90625000e-03, 1.95312500e-03, 9.76562500e-04,
            4.88281250e-04, 2.44140625e-04, 1.22070312e-04, 6.10351562e-05,
            3.05175781e-05, 1.52587891e-05, 7.62939453e-06, 3.81469727e-06
        ])
        self.out_expected_rec_var_def0 = np.array([
            0.00000000e+00, 0.00000000e+00, 5.00000000e-01, 3.75000000e-01,
            2.18750000e-01, 1.17187500e-01, 6.05468750e-02, 3.07617188e-02,
            1.55029297e-02, 7.78198242e-03, 3.89862061e-03, 1.95121765e-03,
            9.76085663e-04, 4.88162041e-04, 2.44110823e-04, 1.22062862e-04,
            6.10332936e-05, 3.05171125e-05, 1.52586726e-05, 7.62936543e-06
        ])
        self.out_expected_rec_var_def1 = np.array([
            0.00000000e+00, 0.00000000e+00, 1.25000000e-01, 9.37500000e-02,
            5.46875000e-02, 2.92968750e-02, 1.51367188e-02, 7.69042969e-03,
            3.87573242e-03, 1.94549561e-03, 9.74655151e-04, 4.87804413e-04,
            2.44021416e-04, 1.22040510e-04, 6.10277057e-05, 3.05157155e-05,
            1.52583234e-05, 7.62927812e-06, 3.81466816e-06, 1.90734136e-06
        ])
        self.out_expected_rec_hos_def0_order4 = np.array([
            0.00000000e+00, 0.00000000e+00, 1.77162630e+00, 1.09061315e+00,
            5.83356515e-01, 3.00002919e-01, 1.51954585e-01, 7.64506651e-02,
            3.83418759e-02, 1.91998526e-02, 9.60712758e-03, 4.80536069e-03,
            2.40312915e-03, 1.20167672e-03, 6.00866390e-04, 3.00440202e-04,
            1.50221853e-04, 7.51113642e-05, 3.75557915e-05, 1.87779231e-05
        ])
        self.out_expected_rec_hos_def0_order8 = np.array([
            0.00000000e+00, 0.00000000e+00, 6.27731948e+00, 3.22254582e+00,
            1.61416851e+00, 8.07222853e-01, 4.03619056e-01, 2.01809976e-01,
            1.00905015e-01, 5.04525093e-02, 2.52262547e-02, 1.26131274e-02,
            6.30656369e-03, 3.15328184e-03, 1.57664092e-03, 7.88320461e-04,
            3.94160230e-04, 1.97080115e-04, 9.85400576e-05, 4.92700288e-05
        ])
        self.out_expected_rec_hos_def1_order4 = np.array([
            0.00000000e+00, 0.00000000e+00, 1.28000000e+00, 8.03265306e-01,
            4.32882653e-01, 2.23361742e-01, 1.13313524e-01, 5.70535086e-02,
            2.86245586e-02, 1.43365603e-02, 7.17432929e-03, 3.58867431e-03,
            1.79471424e-03, 8.97451352e-04, 4.48749228e-04, 2.24380502e-04,
            1.12191723e-04, 5.60962293e-05, 2.80482066e-05, 1.40241263e-05
        ])
        self.out_expected_rec_hos_def1_order8 = np.array([
            0.00000000e+00, 0.00000000e+00, 3.27680000e+00, 1.69171112e+00,
            8.47808685e-01, 4.24000127e-01, 2.12005395e-01, 1.06003012e-01,
            5.30015252e-02, 2.65007638e-02, 1.32503820e-02, 6.62519098e-03,
            3.31259549e-03, 1.65629775e-03, 8.28148873e-04, 4.14074436e-04,
            2.07037218e-04, 1.03518609e-04, 5.17593046e-05, 2.58796523e-05
        ])

    def test_rec_mean(self):
        out = rec_mean(self.signal, C=0.5)
        assert_allclose(out, self.out_expected_rec_mean)

    def test_rec_mean_py(self):
        out = rec_mean_py(self.signal, C=0.5)
        assert_allclose(out, self.out_expected_rec_mean)

    def test_rec_variance(self):
        # definition 0
        out = rec_variance(self.signal, C=0.5, definition=0)
        assert_allclose(out, self.out_expected_rec_var_def0)
        # definition 1
        out = rec_variance(self.signal, C=0.5, definition=1)
        assert_allclose(out, self.out_expected_rec_var_def1)

    def test_rec_variance_py(self):
        # definition 0
        out = rec_variance_py(self.signal, C=0.5, definition=0)
        assert_allclose(out, self.out_expected_rec_var_def0)
        # definition 1
        out = rec_variance_py(self.signal, C=0.5, definition=1)
        assert_allclose(out, self.out_expected_rec_var_def1)

    def test_rec_hos(self):
        # definition 0, order 4
        out = rec_hos(self.signal, C=0.5, order=4, definition=0)
        assert_allclose(out, self.out_expected_rec_hos_def0_order4)
        # definition 0, order 8
        out = rec_hos(self.signal, C=0.5, order=8, definition=0)
        assert_allclose(out, self.out_expected_rec_hos_def0_order8)
        # definition 1, order 4
        out = rec_hos(self.signal, C=0.5, order=4, definition=1)
        assert_allclose(out, self.out_expected_rec_hos_def1_order4)
        # definition 1, order 8
        out = rec_hos(self.signal, C=0.5, order=8, definition=1)
        assert_allclose(out, self.out_expected_rec_hos_def1_order8)

    def test_rec_hos_py(self):
        # definition 0, order 4
        out = rec_hos_py(self.signal, C=0.5, order=4, definition=0)
        assert_allclose(out, self.out_expected_rec_hos_def0_order4)
        # definition 0, order 8
        out = rec_hos_py(self.signal, C=0.5, order=8, definition=0)
        assert_allclose(out, self.out_expected_rec_hos_def0_order8)
        # definition 1, order 4
        out = rec_hos_py(self.signal, C=0.5, order=4, definition=1)
        assert_allclose(out, self.out_expected_rec_hos_def1_order4)
        # definition 1, order 8
        out = rec_hos_py(self.signal, C=0.5, order=8, definition=1)
        assert_allclose(out, self.out_expected_rec_hos_def1_order8)


if __name__ == '__main__':
    unittest.main()
