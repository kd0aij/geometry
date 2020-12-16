import numpy as np
from geometry.point import Point, scalar_projection, is_parallel, np_arrays_from_point_list, mean_point, \
	vector_projection, cross_product, dot_product, point_list_from_np_array
from typing import List, Union
from enum import Enum
import unittest


class VectorParams(Enum):
	oX = 0
	oY = 1
	oZ = 2
	vX = 3
	vY = 4
	vZ = 5


class Vector(object):
	__slots__ = ["_origin", "_direction"]
	
	def __init__(self, origin: Point = None, direction: Point = None):
		if not origin:
			self._origin = Point()
		else:
			self._origin = origin
		if not direction:
			self._direction = Point(x=1)
		else:
			self._direction = direction
	
	fit_id = 1
	
	@property
	def direction(self):
		return self._direction
	
	@direction.setter
	def direction(self, value: Point):
		if type(value) is Point:
			self._direction = value
		else:
			raise TypeError('direction must be a Point')
	
	@property
	def origin(self) -> Point:
		return self._origin
	
	@origin.setter
	def origin(self, value: Point):
		if type(value) is Point:
			self._origin = value
		else:
			raise TypeError('origin must be a Point')
	
	@property
	def array(self) -> List[float]:
		return self.origin.array + self.direction.array
	
	@array.setter
	def array(self, value: List[float]):
		self.origin.array = [value[VectorParams.oX.value], value[VectorParams.oY.value], value[VectorParams.oZ.value]]
		self.direction.array = [value[VectorParams.vX.value], value[VectorParams.vY.value],
								value[VectorParams.vZ.value]]
	
	def __eq__(self, other):
		if isinstance(other, Vector):
			return self.origin == other.origin and is_parallel(self.direction, other.direction)
		else:
			return NotImplemented
	
	def copy(self):
		return Vector(self.origin.copy(), self.direction.copy())
	
	def unit_cost(self, point: Point):
		return abs(distance_to_vector(point, self))


class Plane(Vector):
	def __init__(self, origin: Point = None, direction: Point = None):
		if not origin:
			self._origin = Point()
		else:
			self._origin = origin
		if not direction:
			self._direction = Point(z=1)
		else:
			self._direction = direction
		super().__init__(self._origin, self._direction)
	
	fit_id = 2
	
	def unit_cost(self, point: Point):
		return abs(distance_to_plane(point, self))
	
	def reverse(self):
		return Plane(origin=self.origin.copy(), direction=-self.direction)


def plane_plane_plane_intersect(plane1: Plane, plane2: Plane, plane3: Plane):
	if is_parallel(plane1.direction, plane2.direction) or is_parallel(plane1.direction,
																	  plane3.direction) or is_parallel(plane2.direction,
																									   plane3.direction):
		raise ValueError
	return vector_plane_intersect(plane_plane_intersect(plane1, plane2), plane3)


def vector_plane_intersect(vector: Vector, plane: Plane) -> Point:
	return vector.origin + distance_along_vector_to_plane(vector, plane) * vector.direction.unit


def distance_along_vector_to_plane(vector: Vector, plane: Plane) -> float:
	return dot_product(
		plane.origin - vector.origin,
		plane.direction.unit
	) / dot_product(
		vector.direction.unit,
		plane.direction.unit
	)


def is_coplanar(p1: Union[Plane, Vector, Point], p2: Union[Plane, Vector, Point], tolerance=0.001):
	if isinstance(p1, Point) and isinstance(p2, Point):
		return True
	elif isinstance(p1, Point) and isinstance(p2, (Plane, Vector)):
		return is_point_in_plane(p1, p2, tolerance)
	elif isinstance(p1, (Plane, Vector)) and isinstance(p2, Point):
		return is_point_in_plane(p2, p1, tolerance)
	elif isinstance(p1, (Plane, Vector)) and isinstance(p2, (Plane, Vector)):
		return is_point_in_plane(p1.origin, p2, tolerance) and is_parallel(p1.direction, p2.direction)
	else:
		raise NotImplemented


def is_point_in_plane(point: Point, plane: Union[Plane, Vector], tolerance=0.001):
	return abs(distance_to_plane(point, plane)) < tolerance


def distance_to_plane(point: Point, plane: Union[Plane, Vector]) -> float:
	return scalar_projection(point - plane.origin, plane.direction)


def distance_to_vector(point: Point, vector: Vector) -> Point:
	return project_point_to_vector(point, vector) - point


def project_point_to_vector(point: Point, vector: Vector) -> Point:
	return vector_projection(point - vector.origin, vector.direction) + vector.origin


def plane_plane_intersect(plane1: Plane, plane2: Plane) -> Vector:
	if is_parallel(plane1.direction, plane2.direction):
		raise ValueError
	axis_direction = cross_product(plane1.direction, plane2.direction)
	return Vector(
		origin=vector_plane_intersect(
			Vector(
				origin=plane1.origin,
				direction=cross_product(plane1.direction, axis_direction)
			),
			plane2
		),
		direction=axis_direction
	)



class TestVector(unittest.TestCase):
	def setUp(self):
		self.vector0 = Vector()
		self.vector1 = Vector(
			origin=Point(),
			direction=Point(x=1)
		)
		self.vector2 = Vector(
			origin=Point(),
			direction=Point(1, 1, 0)
		)
		
		self.plane0 = Plane()
		self.plane1 = Plane(origin=Point(1, 1, 1))
		self.plane2 = Plane(direction=Point(1, 0, 0))
		self.plane3 = Plane(Point(1, 1, 1), Point(1, 0, 0))
		
		point_array = np.zeros((5, 3))
		point_array[:, 0] = np.linspace(0, 10, 5)
		point_array[:, 1] = np.linspace(0, 10, 5)
		self.points1 = point_list_from_np_array(point_array)
		
		point_array2 = np.zeros((5, 3))
		point_array2[:, 0] = np.linspace(0, 10, 5)
		point_array2[:, 1] = np.linspace(0, -10, 5)
		self.points2 = self.points1 + point_list_from_np_array(point_array2)
	
	def test_init(self):
		self.assertEqual(Vector(direction=Point(x=1)), Vector(origin=Point(), direction=Point(x=1)))
		self.assertEqual(Vector(), Vector(direction=Point(x=1)))
	
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
	
	def test_array(self):
		self.assertEqual(self.vector1.array, [0, 0, 0, 1, 0, 0])
		self.assertEqual(self.vector2.array, [0, 0, 0, 1, 1, 0])
		self.assertEqual(Vector().array, [0, 0, 0, 1, 0, 0])
		self.assertEqual(Plane().array, [0, 0, 0, 0, 0, 1])
		self.assertEqual(self.plane1.array, [1, 1, 1, 0, 0, 1])
		self.assertEqual(self.plane2.array, [0, 0, 0, 1, 0, 0])
		self.assertEqual(self.plane3.array, [1, 1, 1, 1, 0, 0])

if __name__=="__main__":
    unittest.main()