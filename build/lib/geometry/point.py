from math import pi, cos, sin, sqrt, acos, asin
import numpy as np
import unittest
from typing import List
from enum import Enum
from numbers import Number
from typing import List, Union


class PointParams(Enum):
	X = 0
	Y = 1
	Z = 2


class Point(object):
	__slots__ = ["_x", "_y", "_z"]
	
	def __init__(self, x=None, y=None, z=None, *, array: List = None):
		if x is None:
			self._x = 0
		else:
			self._x = x
		if y is None:
			self._y = 0
		else:
			self._y = y
		if z is None:
			self._z = 0
		else:
			self._z = z
		if array:
			self._x = array[0]
			self._y = array[1]
			if len(array) > 2:
				self._z = array[2]
	
	fit_id = 0
	
	@property
	def x(self):
		return self._x
	
	@x.setter
	def x(self, value):
		self._x = value
	
	@property
	def y(self):
		return self._y
	
	@y.setter
	def y(self, value):
		self._y = value
	
	@property
	def z(self):
		return self._z
	
	@z.setter
	def z(self, value):
		self._z = value
	
	@property
	def array(self):
		return [self._x, self._y, self._z]
	
	@array.setter
	def array(self, value):
		self._x = value[0]
		self._y = value[1]
		self._z = value[2]
	
	def __abs__(self):
		return sqrt(self._x ** 2 + self._y ** 2 + self._z ** 2)
	
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
			return NotImplemented
	
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
			return NotImplemented
	
	def __eq__(self, other):
		if isinstance(other, Point):
			return self.array == other.array
		else:
			try:
				return (other == self._x) and (other == self._y) and (other == self._z)
			finally:
				return NotImplemented
	
	def __mul__(self, other):
		if isinstance(other, Point):
			return Point(x=other.x * self._x, y=other.y * self._y, z=other.z * self._z)
		elif isinstance(other, Number):
			return Point(x=other * self._x, y=other * self._y, z=other * self._z)
		else:
			raise NotImplemented
	
	def __rmul__(self, other):
		if isinstance(other, Point) or isinstance(other, Number):
			return self.__mul__(other)
		else:
			raise NotImplemented
	
	def __truediv__(self, other):
		if isinstance(other, Point):
			return Point(x=self._x / other.x, y=self._y / other.y, z=self._z / other.z)
		elif isinstance(other, Number):
			return Point(x=self._x / other, y=self._y / other, z=self._z / other)
		else:
			raise NotImplemented
	
	def __lt__(self, other):
		if isinstance(other, Point):
			return abs(self) < abs(other)
		else:
			raise NotImplemented
	
	def __gt__(self, other):
		if isinstance(other, Point):
			return abs(self) > abs(other)
		else:
			raise NotImplemented
	
	def __neg__(self):
		return Point(-self._x, -self._y, -self._z)
	
	def set_length(self, value):
		magnitude = self.__abs__()
		return Point(
			x=value * self._x / magnitude,
			y=value * self._y / magnitude,
			z=value * self._z / magnitude
		)
	
	def is_equal(self, tolerance=0):
		if abs(self.x - self.y) > tolerance:
			return False
		if abs(self.x - self.z) > tolerance:
			return False
		return True
	
	@property
	def unit(self):
		return self.set_length(1)
	
	@property
	def cosines(self):
		return Point(x=cos(self._x), y=cos(self._y), z=cos(self._z))
	
	@property
	def sines(self):
		return Point(x=sin(self._x), y=sin(self._y), z=sin(self._z))
	
	def rotate(self, rotation_matrix=List[List[float]]):
		return Point(
			x=self.x * rotation_matrix[0][0] + self.y * rotation_matrix[0][1] + self.z * rotation_matrix[0][2],
			y=self.x * rotation_matrix[1][0] + self.y * rotation_matrix[1][1] + self.z * rotation_matrix[1][2],
			z=self.x * rotation_matrix[2][0] + self.y * rotation_matrix[2][1] + self.z * rotation_matrix[2][2],
		)
	
	def copy(self):
		return Point(x=self._x, y=self._y, z=self._z)
	
	def unit_cost(self, other):
		return abs(self - other)
	
	def to_dict(self):
		return {
			"x": self._x,
			"y": self._y,
			"z": self._z
		}

	def to_rotation_matrix(self):
		s = self.sines
		c = self.cosines
		return [
			[c.y * c.x, c.z * s.x * s.y - c.x * s.z, c.x * c.z * s.y + s.x * s.z],
			[c.y * s.z, c.x * c.z + s.x * s.y * s.z, -c.z * s.x + c.x * s.y * s.z],
			[-s.y, c.y * s.x, c.x * c.y]
		]

def mean_point(points: List[Point]):
	data = np_arrays_from_point_list(points)
	point = data.mean(axis=0)
	return Point(array=point.tolist())


def vector_projection(from_vec: Point, to_vec: Point) -> Point:
	if abs(from_vec) == 0:
		return Point()
	return to_vec.set_length(scalar_projection(from_vec, to_vec))


def scalar_projection(from_vec: Point, to_vec: Point):
	try:
		return cos_angle_between(from_vec, to_vec) * abs(from_vec)
	except ValueError:
		return 0


def is_parallel(p1: Point, p2: Point, tolerance=0.000001):
	raise_zero([p1, p2])
	if p1 == p2:
		return True
	return abs(abs(cos_angle_between(p1, p2)) - 1) < tolerance


def is_anti_parallel(p1: Point, p2: Point, tolerance=0.000001):
	raise_zero([p1, p2])
	if p1 == - p2:
		return True
	return abs(cos_angle_between(p1, p2) + 1) < tolerance


def is_perpendicular(p1: Point, p2: Point, tolerance=0.000001):
	raise_zero([p1, p2])
	return abs(dot_product(p1, p2)) < tolerance


def min_angle_between(p1: Point, p2: Point):
	raise_zero([p1, p2])
	angle = angle_between(p1, p2) % pi
	return min(angle, pi - angle)


def angle_between(p1: Point, p2: Point):
	raise_zero([p1, p2])
	return acos(cos_angle_between(p1, p2))


def cos_angle_between(p1: Point, p2: Point):
	raise_zero([p1, p2])
	return dot_product(p1.unit, p2.unit)


def cross_product(p1: Point, p2: Point) -> Point:
	return Point(
		x=p1.y * p2.z - p1.z * p2.y,
		y=p1.z * p2.x - p1.x * p2.z,
		z=p1.x * p2.y - p1.y * p2.x
	)


def dot_product(p1: Point, p2: Point):
	return np.dot(p1.array, p2.array)


def arbitrary_perpendicular(v: Point) -> Point:
	raise_zero(v)
	if v.x == 0 and v.y == 0:
		return Point(array=[0, 1, 0])
	return Point(-v.y, v.x, 0).unit


def np_arrays_from_point_list(points: List[Point]):
	data = np.empty((len(points), 3))
	for i, point in enumerate(points):
		data[i, :] = point.array
	return data


def point_list_from_np_array(data: np.ndarray) -> List[Point]:
	points = []
	for row in data.tolist():
		points.append(Point(array=row))
	return points


def raise_zero(points: Union[Point, List[Point]]):
	if isinstance(points, Point):
		_raise_zero(points)
	else:
		for point in points:
			_raise_zero(point)


def _raise_zero(point: Point, tolerance=0.000001):
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
	
	def test_angle_between_zero_length(self):
		with self.assertRaises(ValueError):
			angle_between(self.p0, self.p0)
	
	def test_angle_between(self):
		self.assertAlmostEqual(angle_between(self.px, self.pxy), pi / 4)
	
	def test_angle_between_90(self):
		self.assertAlmostEqual(angle_between(self.px, self.py), pi / 2)
	
	def test_angle_between_opposite(self):
		self.assertAlmostEqual(angle_between(self.px, self.px_minus), pi)
	
	def test_angle_between_more_than_90(self):
		self.assertAlmostEqual(angle_between(self.px_minus, self.pxy), 3 * pi / 4)
	
	def test_min_angle_between_more_than_90(self):
		self.assertAlmostEqual(min_angle_between(self.px_minus, self.pxy), pi / 4)
	
	def test_is_equal(self):
		self.assertTrue(self.p0.is_equal())
		self.assertTrue(self.palmostequal.is_equal(0.1))
		self.assertFalse(self.palmostequal.is_equal(0.025))
	
	def test_is_parallel(self):
		self.assertFalse(is_parallel(
			Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
			Point(x=0, y=-0.5150380749100639, z=0.8571673007021064)
		))
		
		with self.assertRaises(ValueError):
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
		self.assertTrue(Point(array=[0, 0, 0]) == Point(array=[0, 0, 0]))
		self.assertTrue(Point(array=[1, 2, 3]) == Point(array=[1, 2, 3]))
	
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
	
	def test_mean(self):
		self.assertEqual(
			Point(0, 0, 0),
			mean_point([Point(0, 0, 0), Point(0, 0, 0)])
		)
	
	def test_scalar_projection(self):
		self.assertEqual(scalar_projection(Point(1, 1, 0), Point(1, 0, 0)), 1)
		self.assertEqual(scalar_projection(Point(0, 0, 0), Point(1, 1, 0)), 0)
		self.assertEqual(scalar_projection(Point(1, 1, 0), Point(0, 0, 0)), 0)
	
	def test_is_perpendicular(self):
		self.assertTrue(is_perpendicular(Point(x=1), Point(y=1)))
		self.assertTrue(is_perpendicular(Point(x=1), Point(z=1)))
		self.assertTrue(is_perpendicular(Point(y=1), Point(x=1)))
		self.assertFalse(is_perpendicular(Point(x=1), Point(x=1, y=1)))
		self.assertFalse(is_perpendicular(Point(x=1), Point(x=1)))


if __name__ == "__main__":
	unittest.main()