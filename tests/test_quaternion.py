import unittest
from geometry.quaternion import Quaternion
from geometry.point import Point
from math import pi


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

        self.assertAlmostEqual(epoint.x, qpoint.x)
        self.assertAlmostEqual(epoint.y, qpoint.y)
        self.assertAlmostEqual(epoint.z, qpoint.z)
