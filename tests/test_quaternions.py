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
        q1 = Quaternions(np.random.random((100, 4))).norm()
        q2 = Quaternions(np.random.random((100, 4))).norm()

        #Quaternions * Quaternions
        np.testing.assert_array_almost_equal(
            (q1 * q2).data,
            np.array(np.vectorize(
                lambda *args: tuple(
                    Quaternion(*args[0:4]) * Quaternion(*args[4:8])
                )
            )(*np.column_stack([q1.data, q2.data]).T)).T,
            err_msg="failed to do Quaternions * Quaternions"
        )

        np.testing.assert_array_almost_equal(
            (q1 * Quaternion(1, 2, 3, 4).norm()).data,
            np.array(np.vectorize(
                lambda *args: tuple(
                    Quaternion(*args[0:4]) * Quaternion(1, 2, 3, 4).norm()
                )
            )(*q1.data.T)).T,
            err_msg="failed to do Quaternions * Quaternion"
        )

        np.testing.assert_array_almost_equal(
            (Quaternion(1, 2, 3, 4).norm() * q1).data,
            np.array(np.vectorize(
                lambda *args: tuple(
                    Quaternion(1, 2, 3, 4).norm() * Quaternion(*args[0:4])
                )
            )(*q1.data.T)).T,
            err_msg="failed to do Quaternion * Quaternions"
        )

    def test_from_euler(self):
        points = Points(np.array([
            [0, 0, 0],
            [0, 0, np.pi / 2],
            [1, 1, 0]
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
        points = Points(np.random.random((100, 3)))

        np.testing.assert_array_equal(
            Quaternions.from_axis_angle(points).data,
            np.array(np.vectorize(
                lambda *args: tuple(Quaternion.from_axis_angle(Point(*args)))
            )(*points.data.T)).T
        )

    def test_from_axis_angle_zero(self):
        pnts = np.concatenate(
            [np.zeros((1, 3)), np.random.random((5, 3))])

        np.testing.assert_array_equal(
            Quaternions.from_axis_angle(Points(pnts)).data,
            np.array(np.vectorize(
                lambda *args: tuple(Quaternion.from_axis_angle(Point(*args)))
            )(*pnts.T)).T
        )

    def test_to_axis_angle(self):
        qs = Quaternions(np.random.random((100, 4))).norm()

        np.testing.assert_array_equal(
            qs.to_axis_angle().data,
            np.array(np.vectorize(
                lambda *args: tuple(Quaternion(*args).to_axis_angle())
            )(*qs.data.T)).T
        )
