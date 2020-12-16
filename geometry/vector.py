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
import numpy as np
from geometry.point import Point, scalar_projection, is_parallel, vector_projection, cross_product, dot_product
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
	__slots__ = ["origin", "direction"]
	fit_id = 1
	
	def __init__(self, origin: Point = None, direction: Point = None):
		if not origin:
			self.origin = Point()
		else:
			self.origin = origin
		if not direction:
			self.direction = Point(1, 0, 0)
		else:
			self.direction = direction
		
	def to_list(self) -> List[float]:
		return self.origin.to_list() + self.direction.to_list()
		
	def __eq__(self, other):
		if isinstance(other, Vector):
			return self.origin == other.origin and is_parallel(self.direction, other.direction)
		else:
			return NotImplementedError
	
	def copy(self):
		return Vector(self.origin.copy(), self.direction.copy())



def distance_along_vector_to_plane(vector: Vector, plane: Vector) -> float:
	return dot_product(
		plane.origin - vector.origin,
		plane.direction.unit
	) / dot_product(
		vector.direction.unit,
		plane.direction.unit
	)


def vector_plane_intersect(vector: Vector, plane: Vector) -> Point:
	return vector.origin + distance_along_vector_to_plane(vector, plane) * vector.direction.unit


def plane_plane_plane_intersect(plane1: Vector, plane2: Vector, plane3: Vector):
	if is_parallel(plane1.direction, plane2.direction) or is_parallel(plane1.direction,
																	  plane3.direction) or is_parallel(plane2.direction,
																									   plane3.direction):
		raise ValueError
	return vector_plane_intersect(plane_plane_intersect(plane1, plane2), plane3)


def is_coplanar(p1: Union[Vector, Point], p2: Union[Vector, Point], tolerance=0.001):
	if isinstance(p1, Point) and isinstance(p2, Point):
		return True
	elif isinstance(p1, Point) and isinstance(p2, Vector):
		return is_point_in_plane(p1, p2, tolerance)
	elif isinstance(p1, Vector) and isinstance(p2, Point):
		return is_point_in_plane(p2, p1, tolerance)
	elif isinstance(p1, Vector) and isinstance(p2, Vector):
		return is_point_in_plane(p1.origin, p2, tolerance) and is_parallel(p1.direction, p2.direction)
	else:
		raise NotImplementedError


def is_point_in_plane(point: Point, plane: Vector, tolerance=0.001):
	return abs(distance_to_plane(point, plane)) < tolerance


def distance_to_plane(point: Point, plane: Vector) -> float:
	return scalar_projection(point - plane.origin, plane.direction)


def distance_to_vector(point: Point, vector: Vector) -> Point:
	return project_point_to_vector(point, vector) - point


def project_point_to_vector(point: Point, vector: Vector) -> Point:
	return vector_projection(point - vector.origin, vector.direction) + vector.origin


def plane_plane_intersect(plane1: Vector, plane2: Vector) -> Vector:
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
		
		point_array = np.zeros((5, 3))
		point_array[:, 0] = np.linspace(0, 10, 5)
		point_array[:, 1] = np.linspace(0, 10, 5)
		
		point_array2 = np.zeros((5, 3))
		point_array2[:, 0] = np.linspace(0, 10, 5)
		point_array2[:, 1] = np.linspace(0, -10, 5)
	
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

if __name__=="__main__":
    unittest.main()