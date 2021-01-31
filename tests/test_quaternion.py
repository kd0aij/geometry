import unittest
from geometry import Quaternion, Point
from math import pi
import numpy as np


class TestQuaternion(unittest.TestCase):
    def test_from_to_euler(self):
        points = [
            Point(0, 0, 0),
            Point(0, 0, pi / 2),
            Point(1, 1, 0)
        ]
        for eul in points:
            quat = Quaternion.from_euler(eul)
            eul2 = quat.to_euler()

            self.assertAlmostEqual(
                eul.x, eul2.x, msg="input:\n" + str(eul) + "\nQuaternion:\n" + str(quat) + "\nResult:\n" + str(eul2))
            self.assertAlmostEqual(
                eul.y, eul2.y, msg="input:\n" + str(eul) + "\nQuaternion:\n" + str(quat) + "\nResult:\n" + str(eul2))
            self.assertAlmostEqual(
                eul.z, eul2.z, msg="input:\n" + str(eul) + "\nQuaternion:\n" + str(quat) + "\nResult:\n" + str(eul2))

    def test_transform_point(self):
        eul = Point(1, 1, 0)
        quat = Quaternion.from_euler(eul)

        point = Point(1, 2, 3)
        epoint = point.rotate(eul.to_rotation_matrix())

        qpoint = quat.transform_point(point)
        self.assertIsInstance(qpoint, Point)
        self.assertAlmostEqual(epoint.x, qpoint.x)
        self.assertAlmostEqual(epoint.y, qpoint.y)
        self.assertAlmostEqual(epoint.z, qpoint.z)

    def test_from_rotation_matrix(self):

        rmats = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            Point(1, 1, 0).to_rotation_matrix(),
            Point(0.7, -1.2, 1).to_rotation_matrix()
        ]

        for rmat in rmats:
            quat = Quaternion.from_rotation_matrix(np.array(rmat))

            rmat2 = quat.to_rotation_matrix()

            for i in range(0, 3):
                for j in range(0, 3):
                    self.assertAlmostEqual(rmat[i][j], rmat2[i][j])

    def test_iter(self):
        quat = Quaternion(1, 2, 3, 4)
        self.assertEqual(tuple(quat), (1, 2, 3, 4))
        self.assertEqual(list(quat), [1, 2, 3, 4])

    def test_axis_rates(self):
        q = Quaternion.from_euler(Point(0, 0, np.pi / 2))
        qdot = Quaternion.from_euler(Point(np.radians(5), 0, np.pi / 2))

        rates = Quaternion.axis_rates(q, qdot)
        self.assertAlmostEqual(np.degrees(rates.x), 0)
        self.assertAlmostEqual(np.degrees(rates.y), 5)
        self.assertAlmostEqual(np.degrees(rates.z), 0)
        # TODO I think we should get better precision than this...

    def test_body_axis_rates(self):
        q = Quaternion.from_euler(Point(0, 0, np.pi / 2))
        qdot = Quaternion.from_euler(Point(np.radians(5), 0, np.pi / 2))

        rates = Quaternion.body_axis_rates(q, qdot)
        self.assertAlmostEqual(np.degrees(rates.x), 5)
        self.assertAlmostEqual(np.degrees(rates.y), 0)
        self.assertAlmostEqual(np.degrees(rates.z), 0)

    def test_eaxisrates(self):
        r = np.array(Point(np.pi / 2, 0, 0).to_rotation_matrix())
        rdot = np.array(Point(np.pi / 2 + np.radians(5),
                              0, 0).to_rotation_matrix())

        a = np.dot(rdot, r.T)
        self.assertAlmostEqual(a[0, 1], 0)
        self.assertAlmostEqual(a[0, 2], 0)
        self.assertAlmostEqual(np.degrees(a[1, 2]), -5, 1)

    def test_rotate(self):
        q = Quaternion.from_euler(Point(0, 0, 0))
        qdot = q.rotate(Point(0, 0, np.radians(5)))
        self.assertAlmostEqual(qdot.transform_point(
            Point(1, 0, 0)).y, np.sin(np.radians(5)))

    def test_body_rotate(self):
        q = Quaternion.from_euler(Point(0, 0, np.pi / 2))
        qdot = q.body_rotate(Point(np.radians(5), 0, 0))

        self.assertAlmostEqual(qdot.transform_point(
            Point(0, 1, 0)).z, np.sin(np.radians(5)))

    def test_body_rotate_zero(self):
        qinit = Quaternion.from_euler(Point(0, 0, 0))
        qdot = qinit.body_rotate(Point(0, 0, 0))

        np.testing.assert_array_equal(list(qinit), list(qdot))

    def test_from_axis_angle_zero(self):
        qinit = Quaternion.from_axis_angle(Point(0, 0, 0), 1)
        np.testing.assert_array_equal(list(qinit), [1, 0, 0, 0])

        qb = Quaternion.from_axis_angle(Point(1, 0, 0))

        self.assertNotEqual(abs(qb), 0)

        pnts = np.concatenate(
            [np.zeros((1, 3)), np.random.random((5, 3))])

        #pnts = np.random.random((5, 3))

        def tfunc(*args):
            print(args)
            quat = Quaternion.from_axis_angle(Point(*args), 1)
            print(tuple(quat))
            return tuple(quat)

        quats = np.array(np.vectorize(
            tfunc
        )(*pnts.T))
        print(quats.T)
        self.assertNotEqual(abs(Quaternion(*quats.T[1])), 0)
