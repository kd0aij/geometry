import unittest

from geometry.points import Points

import numpy as np
import pandas as pd

from geometry import Point


class TestPoints(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]])
        self.pnts = Points(self.points)

    def test_xyz(self):
        self.assertCountEqual(
            self.pnts.x,
            np.vectorize(lambda *args: Point(*args).x)(*self.points.T)
        )

        self.assertCountEqual(
            self.pnts.y,
            np.vectorize(lambda *args: Point(*args).y)(*self.points.T)
        )
        self.assertCountEqual(
            self.pnts.z,
            np.vectorize(lambda *args: Point(*args).z)(*self.points.T)
        )

    def test_count(self):
        self.assertEqual(self.pnts.count, 4)

    def test_abs(self):
        self.assertCountEqual(
            abs(self.pnts),
            np.vectorize(lambda *args: abs(Point(*args)))(*self.points.T)
        )

    def test_add(self):

        def dbl(*args):
            return Point(*args) + Point(*args)

        self.assertCountEqual(
            (self.pnts + self.pnts).x,
            np.vectorize(lambda *args: dbl(*args).x)(*self.points.T)
        )
        self.assertCountEqual(
            (self.pnts + self.pnts).y,
            np.vectorize(lambda *args: dbl(*args).y)(*self.points.T)
        )

    def test_mul_points_points(self):
        def sqr(*args):
            return Point(*args) * Point(*args)

        self.assertCountEqual(
            (self.pnts * self.pnts).x,
            np.vectorize(lambda *args: sqr(*args).x)(*self.points.T)
        )
        self.assertCountEqual(
            (self.pnts * self.pnts).y,
            np.vectorize(lambda *args: sqr(*args).y)(*self.points.T)
        )

    def test_mul_points_scalar(self):
        def dbl(*args):
            return 2 * Point(*args)

        self.assertCountEqual(
            (2 * self.pnts).x,
            np.vectorize(lambda *args: dbl(*args).x)(*self.points.T)
        )

    def test_mul_points_array(self):
        mults = np.random.random(4)

        def mul(*args):
            return Point(*args[:3]) * args[3]

        self.assertCountEqual(
            (mults * self.pnts).x,
            np.vectorize(lambda *args: mul(*args).x)(
                *np.column_stack([self.points, mults]).T
            )
        )

        with self.assertRaises(NotImplementedError):
            self.pnts * np.random.random(5)

    def test_neg(self):
        self.assertCountEqual(
            (-self.pnts).y,
            np.vectorize(lambda *args: -Point(*args).y)(*self.points.T)
        )

    def test_truediv_scalar(self):
        self.assertCountEqual(
            (self.pnts / 2).x,
            np.vectorize(lambda *args: (Point(*args) / 2).x)(*self.points.T)
        )

    def test_truediv_array(self):
        pass

    def test_truediv_points_points(self):
        pass

    def test_scale_scalar(self):
        self.assertCountEqual(
            abs(self.pnts.scale(1)),
            np.vectorize(lambda *args: abs(Point(*args).scale(1)))(*self.points.T)
        )

    def test_trigs(self):
        self.assertCountEqual(
            self.pnts.sines().x,
            np.vectorize(lambda *args: Point(*args).sines.x)(*self.points.T)
        )
        self.assertCountEqual(
            self.pnts.cosines().x,
            np.vectorize(lambda *args: Point(*args).cosines.x)(*self.points.T)
        )
    
    