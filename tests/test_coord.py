import unittest
from geometry.coordinate_frame import Coord, Transformation
from geometry import Point
from math import sqrt
import numpy as np


class TestTransformation(unittest.TestCase):
    def test_translate(self):
        ca = Coord.from_nothing()
        cb = Coord.from_nothing().translate(Point(1, 0, 0))
        transform = Transformation(ca, cb)
        self.assertEqual(transform.translate(Point(0, 0, 0)), Point(1, 0, 0))

    def _test_rotate(self, c1, c2, p1, p2):
        transform = Transformation(c1, c2)
        p1b = transform.rotate(p1)
        self.assertEqual(p1b, p2, str(p1b) + ' != ' + str(p2))

    def test_rotate(self):
        self._test_rotate(
            Coord.from_nothing(),
            Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, -1, 0)),
            Point(1, 1, 0),
            Point(1, -1, 0)
        )

        self._test_rotate(
            Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, 1, 0)),
            Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, -1, 0)),
            Point(1, 1, 0),
            Point(-1, -1, 0)
        )

#        self._test_rotate(
 #           Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, 1, 0)),
  #          Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(1, -1, 0)),
   #         Point(1, 1, 0),
    #        Point(0, -sqrt(2), 0)
     #   )

    def test_pos_vec(self):
        ptupe = (np.ones(3), np.ones(3), np.zeros(3))
        transform = Transformation(Coord.from_nothing(), Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, -1, 0)))
        res = transform.pos_vec(*ptupe)
        self.assertEqual(res[0][0], 1)
        self.assertEqual(res[1][0], -1)
        self.assertEqual(res[2][0], 0)


    def test_transform(self):
        pass