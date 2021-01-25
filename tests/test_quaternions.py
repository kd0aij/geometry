import unittest
from geometry.quaternions import Quaternions
import numpy as np
from geometry.quaternion import Quaternion
from geometry.points import Points
from geometry.point import Point


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
        np.testing.assert_array_equal(
            self.qs.norm().data,
            np.array(np.vectorize(
                lambda *args: tuple(Quaternion(*args).norm())
            )(*self.qs.data.T)).T
        )

    def test_conjugate(self):
        np.testing.assert_array_equal(
            self.qs.conjugate().data,
            np.array(np.vectorize(
                lambda *args: tuple(Quaternion(*args).conjugate())
            )(*self.qs.data.T)).T
        )

    def test_inverse(self):
        np.testing.assert_array_equal(
            self.qs.inverse().data,
            np.array(np.vectorize(
                lambda *args: tuple(Quaternion(*args).inverse())
            )(*self.qs.data.T)).T
        )

    def test_mul(self):
        def sqr(*args):
            return Quaternion(*args) * Quaternion(*args)

        self.assertCountEqual(
            (self.qs * self.qs).w,
            np.vectorize(lambda *args: sqr(*args).w)(*self.qs.data.T)
        )

    def test_from_euler(self):
        points = Points(np.array([
            [0, 0, 0],
            [0, 0, np.pi / 2],
            [1, 1,  0]
        ]))

        np.testing.assert_array_equal(
            Quaternions.from_euler(points).data,
            np.array(np.vectorize(
                lambda *args: tuple(Quaternion.from_euler(Point(*args)))
            )(*points.data.T)).T
        )

    def test_transform_point(self):
        np.testing.assert_array_equal(
            self.qs.transform_point(Point(1, 1, 1)).data,
            np.array(np.vectorize(
                lambda *args: tuple(Quaternion(*
                                               args).transform_point(Point(1, 1, 1)))
            )(*self.qs.data.T)).T
        )

    def test_from_axis_angle(self):
        points = Points(np.random.random((100,3)))

        np.testing.assert_array_equal(
            Quaternions.from_axis_angle(points).data,
            np.array(np.vectorize(
                lambda *args: tuple(Quaternion.from_axis_angle(Point(*args)))
            )(*points.data.T)).T
        )


    def test_to_axis_angle(self):
        qs = Quaternions(np.random.random((100,4))).norm()

        np.testing.assert_array_equal(
            qs.to_axis_angle().data,
            np.array(np.vectorize(
                lambda *args: tuple(Quaternion(*args).to_axis_angle())
            )(*qs.data.T)).T
        )
