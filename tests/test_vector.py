import unittest
from geometry.vector import Vector, is_point_in_plane, vector_projection
from geometry.point import Point, is_parallel


class TestVector(unittest.TestCase):
    def setUp(self):
        self.vector1 = Vector(
            origin=Point(0, 0, 0),
            direction=Point(1, 0, 0)
        )
        self.vector2 = Vector(
            origin=Point(0, 0, 0),
            direction=Point(1, 1, 0)
        )

        self.plane0 = Vector(Point(0, 0, 0), Point(0, 0, 1))
        self.plane1 = Vector(Point(1, 1, 1), Point(1, 0, 0))
        self.plane3 = Vector(Point(1, 1, 1), Point(1, 0, 0))

    def test_is_point_in_plane(self):
        self.assertTrue(is_point_in_plane(
            Point(0, 2, 3),
            self.vector1
        ))

        self.assertTrue(is_point_in_plane(
            Point(1, -1, 5),
            self.vector2
        ))

        self.assertFalse(is_point_in_plane(
            Point(1, 1, 5),
            self.vector2
        ))

    def test_vector_projection(self):
        self.assertTrue(is_parallel(
            vector_projection(Point(100, 544, 4144), Point(1, 1, 1)),
            Point(1, 1, 1)
        ))

    def test_to_list(self):
        self.assertEqual(self.vector1.to_list(), [0, 0, 0, 1, 0, 0])
        self.assertEqual(self.vector2.to_list(), [0, 0, 0, 1, 1, 0])
        self.assertEqual(self.plane1.to_list(), [1, 1, 1, 1, 0, 0])
        self.assertEqual(self.plane3.to_list(), [1, 1, 1, 1, 0, 0])
