import unittest
from geometry.quaternions import Quaternions
import numpy as np
from geometry.quaternion import Quaternion


class TestQuaternions(unittest.TestCase):
    def setUp(self):
        self.qs = Quaternions(
            np.array([
                [0, 0, 0, 1],
                [0.4472136, 0, 0, 0.8944272]
            ])
        )

    def test_abs(self):
        self.assertCountEqual(
            abs(self.qs),
            np.vectorize(lambda *args: abs(Quaternion(*args)))(*self.qs.data.T)
        )

    def test_norm(self):
        self.assertCountEqual(
            self.qs.norm(),
            np.vectorize(lambda *args: Quaternion(*args).norm())(*self.qs.data.T)
        )

    def test_conjugate(self):
        self.assertCountEqual(
            self.qs.conjugate().x,
            np.vectorize(lambda *args: Quaternion(*args).conjugate().x)(*self.qs.data.T)
        )

        self.assertCountEqual(
            self.qs.conjugate().w,
            np.vectorize(lambda *args: Quaternion(*args).conjugate().w)(*self.qs.data.T)
        )
    
    def test_inverse(self):
        self.assertCountEqual(
            self.qs.inverse().x,
            np.vectorize(lambda *args: Quaternion(*args).inverse().x)(*self.qs.data.T)
        )

        self.assertCountEqual(
            self.qs.inverse().w,
            np.vectorize(lambda *args: Quaternion(*args).inverse().w)(*self.qs.data.T)
        )

    def test_mul(self):
        def sqr(*args):
            return Quaternion(*args) * Quaternion(*args)

        self.assertCountEqual(
            (self.qs * self.qs).w,
            np.vectorize(lambda *args: sqr(*args).w)(*self.qs.data.T)
        )