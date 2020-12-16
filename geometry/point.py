"""
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
from math import pi, cos, sin, sqrt, acos, asin
import numpy as np
import unittest
from typing import List
from enum import Enum
from numbers import Number
from typing import List, Union


class Point(object):
    __slots__ = ["x", "y", "z"]

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    fit_id = 0

    def to_list(self):
        return [self.x, self.y, self.z]

    def to_tuple(self):
        return(self.x, self.y, self.z)

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }

    def __abs__(self):
        # TODO make this numpy array safe
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(
                x=self.x + other.x,
                y=self.y + other.y,
                z=self.z + other.z
            )
        elif isinstance(other, Number):
            return Point(
                x=self.x + other,
                y=self.y + other,
                z=self.z + other
            )
        else:
            return NotImplementedError

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(
                x=self.x - other.x,
                y=self.y - other.y,
                z=self.z - other.z
            )
        elif isinstance(other, Number):
            return Point(
                x=self.x - other,
                y=self.y - other,
                z=self.z - other
            )
        else:
            return NotImplementedError

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.to_list() == other.to_list()
        else:
            try:
                return (other == self.x) and (other == self.y) and (other == self.z)
            finally:
                return NotImplementedError

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(x=other.x * self.x, y=other.y * self.y, z=other.z * self.z)
        elif isinstance(other, Number):
            return Point(x=other * self.x, y=other * self.y, z=other * self.z)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, Point) or isinstance(other, Number):
            return self.__mul__(other)
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, Point):
            return Point(x=self.x / other.x, y=self.y / other.y, z=self.z / other.z)
        elif isinstance(other, Number):
            return Point(x=self.x / other, y=self.y / other, z=self.z / other)
        else:
            raise NotImplementedError

    def __lt__(self, other):
        if isinstance(other, Point):
            return abs(self) < abs(other)
        else:
            raise NotImplementedError

    def __gt__(self, other):
        if isinstance(other, Point):
            return abs(self) > abs(other)
        else:
            raise NotImplementedError

    def __neg__(self):
        return -1 * self

    def scale(self, value):
        return self.__mul__(value / self.__abs__())

    def is_equal(self, tolerance=0):
        if abs(self.x - self.y) > tolerance:
            return False
        if abs(self.x - self.z) > tolerance:
            return False
        return True

    def unit(self):
        return self.scale(1)

    @property
    def cosines(self):
        return Point(x=cos(self.x), y=cos(self.y), z=cos(self.z))

    @property
    def sines(self):
        return Point(x=sin(self.x), y=sin(self.y), z=sin(self.z))

    def rotate(self, rotation_matrix=List[List[float]]):
        return Point(
            self.x * rotation_matrix[0][0] + self.y *
            rotation_matrix[0][1] + self.z * rotation_matrix[0][2],
            self.x * rotation_matrix[1][0] + self.y *
            rotation_matrix[1][1] + self.z * rotation_matrix[1][2],
            self.x * rotation_matrix[2][0] + self.y *
            rotation_matrix[2][1] + self.z * rotation_matrix[2][2],
        )

    def unit_cost(self, other):
        return abs(self - other)

    def to_rotation_matrix(self):
        '''returns the rotation matrix based on apoint representing Euler angles'''
        s = self.sines
        c = self.cosines
        return [
            [c.z * c.y, c.z * s.y * s.x - c.x * s.z, c.x * c.z * s.y + s.x * s.z],
            [c.y * s.z, c.x * c.z + s.x * s.y *
                s.z, -1 * c.z * s.x + c.x * s.y * s.z],
            [-1 * s.y, c.y * s.x, c.x * c.y]
        ]


def np_arrays_from_point_list(points: List[Point]):
    data = np.empty((len(points), 3))
    for i, point in enumerate(points):
        data[i, :] = point.to_list()
    return data


def dot_product(p1: Point, p2: Point):
    return np.dot(p1.to_list(), p2.to_list())


def cos_angle_between(p1: Point, p2: Point):
    raisezero([p1, p2])
    return dot_product(p1.unit, p2.unit)


def cross_product(p1: Point, p2: Point) -> Point:
    return Point(
        x=p1.y * p2.z - p1.z * p2.y,
        y=p1.z * p2.x - p1.x * p2.z,
        z=p1.x * p2.y - p1.y * p2.x
    )


def scalar_projection(from_vec: Point, to_vec: Point):
    try:
        return cos_angle_between(from_vec, to_vec) * abs(from_vec)
    except ValueError:
        return 0


def vector_projection(from_vec: Point, to_vec: Point) -> Point:
    if abs(from_vec) == 0:
        return Point()
    return to_vec.scale(scalar_projection(from_vec, to_vec))


def is_parallel(p1: Point, p2: Point, tolerance=0.000001):
    raisezero([p1, p2])
    if p1 == p2:
        return True
    return abs(abs(cos_angle_between(p1, p2)) - 1) < tolerance


def is_anti_parallel(p1: Point, p2: Point, tolerance=0.000001):
    raisezero([p1, p2])
    if p1 == - p2:
        return True
    return abs(cos_angle_between(p1, p2) + 1) < tolerance


def is_perpendicular(p1: Point, p2: Point, tolerance=0.000001):
    raisezero([p1, p2])
    return abs(dot_product(p1, p2)) < tolerance


def min_angle_between(p1: Point, p2: Point):
    raisezero([p1, p2])
    angle = angle_between(p1, p2) % pi
    return min(angle, pi - angle)


def angle_between(p1: Point, p2: Point):
    raisezero([p1, p2])
    return acos(cos_angle_between(p1, p2))


def arbitrary_perpendicular(v: Point) -> Point:
    raisezero(v)
    if v.x == 0 and v.y == 0:
        return Point(0, 1, 0)
    return Point(-v.y, v.x, 0).unit


def raisezero(points: Union[Point, List[Point]]):
    if isinstance(points, Point):
        _raisezero(points)
    else:
        for point in points:
            _raisezero(point)


def _raisezero(point: Point, tolerance=0.000001):
    if abs(point) < tolerance:
        raise ValueError('magnitude less than tolerance')


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.p0 = Point(x=0, y=0, z=0)
        self.px = Point(x=1, y=0, z=0)
        self.pxy = Point(x=1, y=1, z=0)
        self.px_minus = Point(x=-3, y=0, z=0)
        self.py = Point(x=0, y=100, z=0)
        self.pz = Point(x=0, y=0, z=11)
        self.palmostequal = Point(x=1.1, y=1.05, z=1.15)
        self.pdiv1 = Point(x=1, y=2, z=3)
        self.pdiv2 = Point(x=1, y=4, z=9)

    def test_angle_betweenzero_length(self):
        with self.assertRaises(ValueError):
            angle_between(self.p0, self.p0)

    def test_angle_between(self):
        self.assertAlmostEqual(angle_between(self.px, self.pxy), pi / 4)

    def test_angle_between_90(self):
        self.assertAlmostEqual(angle_between(self.px, self.py), pi / 2)

    def test_angle_between_opposite(self):
        self.assertAlmostEqual(angle_between(self.px, self.px_minus), pi)

    def test_angle_between_more_than_90(self):
        self.assertAlmostEqual(angle_between(
            self.px_minus, self.pxy), 3 * pi / 4)

    def test_min_angle_between_more_than_90(self):
        self.assertAlmostEqual(min_angle_between(
            self.px_minus, self.pxy), pi / 4)

    def test_is_equal(self):
        self.assertTrue(self.p0.is_equal())
        self.assertTrue(self.palmostequal.is_equal(0.1))
        self.assertFalse(self.palmostequal.is_equal(0.025))

    def test_is_parallel(self):
        self.assertFalse(is_parallel(
            Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
            Point(x=0, y=-0.5150380749100639, z=0.8571673007021064)
        ))

        with self.assertRaises(TypeError):
            is_parallel(Point(), Point())

        self.assertTrue(
            is_parallel(
                Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
                Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
                0.001
            )
        )
        self.assertTrue(
            is_parallel(
                Point(x=0, y=0.5, z=0.5),
                Point(x=0, y=-1, z=-1)
            )
        )

    def test_is_anti_parallel(self):
        self.assertFalse(
            is_parallel(
                Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
                Point(x=0, y=-0.5150380749100639, z=0.8571673007021064)
            )
        )
        self.assertTrue(
            is_parallel(
                Point(x=1, y=1, z=1),
                Point(x=1, y=1, z=1)
            )
        )
        self.assertTrue(
            is_parallel(
                Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
                Point(x=0, y=-0.8571673007021064, z=-0.5150380749100639),
                0.001
            )
        )

    def test_eq(self):
        self.assertTrue(Point(0, 0, 0) == Point(0, 0, 0))
        self.assertTrue(Point(1, 2, 3) == Point(1, 2, 3))

    def test_abs(self):
        self.assertEqual(abs(Point(x=0, y=0, z=0)), 0)
        self.assertEqual(abs(Point(x=1, y=0, z=0)), 1)
        self.assertEqual(abs(Point(x=0, y=100, z=0)), 100)

    def test_div(self):
        self.assertAlmostEqual(self.pdiv2 / self.pdiv1, Point(x=1, y=2, z=3))
        self.assertAlmostEqual(self.pdiv2 / 1, self.pdiv2)

    def test_rotate(self):
        self.assertEqual(
            self.px.rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            self.px
        )
        self.assertEqual(
            self.px.rotate([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            Point(0, 1, 0)
        )

    def test_scalar_projection(self):
        self.assertEqual(scalar_projection(Point(1, 1, 0), Point(1, 0, 0)), 1)
        self.assertEqual(scalar_projection(Point(0, 0, 0), Point(1, 1, 0)), 0)
        self.assertEqual(scalar_projection(Point(1, 1, 0), Point(0, 0, 0)), 0)

    def test_is_perpendicular(self):
        self.assertTrue(is_perpendicular(Point(1, 0, 0), Point(0, 1, 0)))
        self.assertTrue(is_perpendicular(Point(1, 0, 0), Point(0, 0, 1)))
        self.assertTrue(is_perpendicular(Point(0, 1, 0), Point(1, 0, 0)))
        self.assertFalse(is_perpendicular(Point(1, 0, 0), Point(1, 1, 0)))


if __name__ == "__main__":
    unittest.main()
