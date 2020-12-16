from geometry.point import Point, cross_product, arbitrary_perpendicular, is_perpendicular
from geometry.vector import Plane, Vector
from geometry.line import Line
from typing import List, Union
from numpy.linalg import inv
import unittest


class CoordinateFrame(object):
	__slots__ = ["_origin", "_x_axis", "_y_axis", "_z_axis", "_rotation_matrix", "_inverse_rotation_matrix"]
	
	def __init__(
			self,
			origin: Point = None,
			x_axis: Point = None,
			y_axis: Point = None,
			z_axis: Point = None):
		if not origin:
			self._origin = Point()
		else:
			self._origin = origin
		self._x_axis = x_axis
		self._y_axis = y_axis
		self._z_axis = z_axis
		self._rotation_matrix = None
		self._inverse_rotation_matrix = None
		self._make_unit()
		self._fix_perpendicularity()
	
	def _make_unit(self):
		if self._x_axis:
			self._x_axis = self._x_axis.unit
		if self._y_axis:
			self._y_axis = self._y_axis.unit
		if self._z_axis:
			self._z_axis = self._z_axis.unit
	
	def _reset(self):
		self._x_axis = Point(array=[1, 0, 0])
		self._y_axis = Point(array=[0, 1, 0])
		self._z_axis = Point(array=[0, 0, 1])
	
	def _fix_perpendicularity(self):
		if not self._check_perpendicularity():
			pass
	
	def _check_perpendicularity(self):
		self._make_unit()
		return self.z_axis == cross_product(self.x_axis, self.y_axis)
	
	@property
	def origin(self) -> Point:
		return self._origin
	
	@origin.setter
	def origin(self, value: Point):
		self._origin = value
	
	@property
	def x_axis(self) -> Point:
		if not self._x_axis:
			if not self._y_axis and not self._z_axis:
				self._reset()
			elif not self._z_axis:
				self._z_axis = arbitrary_perpendicular(self._y_axis)
				self._x_axis = cross_product(self._y_axis, self._z_axis)
			elif not self._y_axis:
				self._x_axis = arbitrary_perpendicular(self._z_axis)
			else:
				self._x_axis = cross_product(self._y_axis, self._z_axis)
		return self._x_axis
	
	@x_axis.setter
	def x_axis(self, value: Point):
		self._x_axis = value
	
	@property
	def y_axis(self) -> Point:
		if not self._y_axis:
			if not self._x_axis and not self._z_axis:
				self._reset()
			elif not self._z_axis:
				self._y_axis = arbitrary_perpendicular(self._x_axis)
			elif not self._x_axis:
				self._x_axis = arbitrary_perpendicular(self._z_axis)
				self._y_axis = cross_product(self._z_axis, self._x_axis)
			else:
				self._y_axis = cross_product(self._z_axis, self._x_axis)
		return self._y_axis
	
	@y_axis.setter
	def y_axis(self, value: Point):
		self._y_axis = value
	
	@property
	def z_axis(self) -> Point:
		if not self._z_axis:
			if not self._x_axis and not self._y_axis:
				self._reset()
			elif not self._x_axis:
				self._z_axis = arbitrary_perpendicular(self._y_axis)
			elif not self._y_axis:
				self._y_axis = arbitrary_perpendicular(self._x_axis)
				self._z_axis = cross_product(self._x_axis, self._y_axis)
			else:
				self._z_axis = cross_product(self._x_axis, self._y_axis)
		return self._z_axis
	
	@z_axis.setter
	def z_axis(self, value: Point):
		self._z_axis = value
	
	@property
	def xy(self) -> Plane:
		return Plane(self._origin, self.x_axis)
	
	@property
	def yz(self) -> Plane:
		return Plane(self._origin, self.y_axis)
	
	@property
	def zx(self) -> Plane:
		return Plane(self._origin, self.z_axis)
	
	@property
	def x_vector(self) -> Vector:
		return Vector(self._origin, self.x_axis)
	
	@property
	def y_vector(self) -> Vector:
		return Vector(self._origin, self.y_axis)
	
	@property
	def z_vector(self) -> Vector:
		return Vector(self._origin, self.z_axis)
	
	def _reset_axes(self):
		return CoordinateFrame(origin=self.origin.copy())
	
	def euler_angles(self, angles: Point = Point(array=[0, 0, 0])):
		return self._reset_axes().euler_rotation(angles)
	
	def euler_rotation(self, angles: Point = Point(array=[0, 0, 0])):
		return self.rotate(angles.to_rotation_matrix())
	
	@property
	def rotation_matrix(self):
		if not self._rotation_matrix:
			self._rotation_matrix = [self._x_axis.array, self._y_axis.array, self._z_axis.array]
		return self._rotation_matrix
	
	@property
	def inverse_rotation_matrix(self):
		if not self._inverse_rotation_matrix:
			self._inverse_rotation_matrix = [
				[self.x_axis.x, self.y_axis.x, self.z_axis.x],
				[self.x_axis.y, self.y_axis.y, self.z_axis.y],
				[self.x_axis.z, self.y_axis.z, self.z_axis.z]
			]
		return self._inverse_rotation_matrix
	
	def rotate(self, rotation_matrix=List[List[float]]):
		return CoordinateFrame(
			origin=self._origin.copy(),
			x_axis=self._x_axis.rotate(rotation_matrix),
			y_axis=self._y_axis.rotate(rotation_matrix),
			z_axis=self._z_axis.rotate(rotation_matrix)
		)
	
	def __eq__(self, other):
		return \
			self.origin == other.origin and \
			self.x_axis == other.x_axis and \
			self.y_axis == other.y_axis and \
			self.z_axis == other.z_axis


def transform_point(points: Union[Point, List[Point]], from_coord: CoordinateFrame = None,
					to_coord: CoordinateFrame = None):
	if not from_coord:
		from_coord = CoordinateFrame()
	if not to_coord:
		to_coord = CoordinateFrame()
	if isinstance(points, list):
		out_points = []
		for point in points:
			out_points.append(_transform_point(point, from_coord, to_coord))
		return out_points
	elif isinstance(points, Point):
		return _transform_point(points, from_coord, to_coord)
	else:
		raise TypeError


def _transform_point(point: Point, from_coord: CoordinateFrame, to_coord: CoordinateFrame):
	temp_point = point.rotate(from_coord.inverse_rotation_matrix) + from_coord.origin - to_coord.origin
	return temp_point.rotate(to_coord.rotation_matrix)


def transform_line(lines: Union[Line, List[Line]], from_coord: CoordinateFrame = None,
				   to_coord: CoordinateFrame = None):
	if not from_coord:
		from_coord = CoordinateFrame()
	if not to_coord:
		to_coord = CoordinateFrame()
	if isinstance(lines, list):
		out_lines = []
		for line in lines:
			out_lines.append(_transform_line(line, from_coord, to_coord))
		return out_lines
	elif isinstance(lines, Line):
		return _transform_line(lines, from_coord, to_coord)
	else:
		raise TypeError


def _transform_line(line: Line, from_coord: CoordinateFrame, to_coord: CoordinateFrame):
	return Line(
		start=_transform_point(line.start, from_coord, to_coord),
		end=_transform_point(line.end, from_coord, to_coord)
	)


def det_3x3(mat=List[List[float]]):
	return \
		mat[0][0] * mat[1][1] * mat[2][2] + \
		mat[0][1] * mat[1][2] * mat[2][0] + \
		mat[1][0] * mat[2][1] * mat[0][2] - \
		mat[0][2] * mat[1][1] * mat[2][0] - \
		mat[0][1] * mat[1][0] * mat[2][2] - \
		mat[1][2] * mat[2][1] * mat[0][0]




class TestCoordinateFrame(unittest.TestCase):
	def setUp(self):
		self.cid0 = CoordinateFrame()
		self.cid1 = CoordinateFrame(
			origin=Point(1, 1, 1),
			x_axis=Point(0, 0, 1),
			y_axis=Point(0, 1, 0)
		)
		self.cid2 = CoordinateFrame(
			origin=Point(1, 2, 3),
			x_axis=Point(0, 1, 0)
		)
		self.cid3 = CoordinateFrame(
			x_axis=Point(0, 1, 0)
		)
		
		self.cid_euler_0 = CoordinateFrame().euler_angles(Point(0, 0, 0))
		self.cid_euler_1 = CoordinateFrame().euler_angles(Point(0, 90, 0))
	
	def test_creation(self):
		self.assertEqual(self.cid0.x_axis, Point(1, 0, 0))
		self.assertEqual(self.cid0.y_axis, Point(0, 1, 0))
		self.assertEqual(self.cid0.z_axis, Point(0, 0, 1))
		
		self.assertEqual(self.cid1.origin, Point(1, 1, 1))
		self.assertEqual(self.cid1.z_axis, Point(-1, 0, 0))
		
		self.assertEqual(self.cid2.origin, Point(1, 2, 3))
		self.assertEqual(self.cid2.x_axis, Point(0, 1, 0))
		self.assertTrue(is_perpendicular(self.cid2.y_axis, self.cid2.x_axis))
		self.assertTrue(is_perpendicular(self.cid2.y_axis, self.cid2.z_axis))
	
	def test_rotation_matrix(self):
		self.assertEqual(self.cid0.rotation_matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
		self.assertEqual(self.cid1.rotation_matrix, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
		self.assertEqual(self.cid2.rotation_matrix, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
	
	def test_inverse_rotation_matrix(self):
		self.assertEqual(self.cid0.inverse_rotation_matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	
	def test_euler_angles(self):
		self.assertEqual(self.cid_euler_0.rotation_matrix, self.cid0.rotation_matrix)
	
	def test_rotate(self):
		self.assertEqual(
			self.cid0.rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
			self.cid0
		)
		
		self.assertEqual(
			self.cid0.rotate([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).rotation_matrix,
			CoordinateFrame(
				Point(0, 0, 0),
				Point(0, 1, 0),
				Point(-1, 0, 0),
				Point(0, 0, 1)
			).rotation_matrix
		)
	
	def test_transform_point(self):
		self.assertEqual(
			transform_point(Point(1, 0, 0), self.cid0, self.cid3),
			Point(0, -1, 0)
		)
		
		self.assertEqual(
			[Point(0, -1, 0), Point(0, 0, 1)],
			transform_point([Point(1, 0, 0), Point(0, 0, 1)], self.cid0, self.cid3)
		)
		
		self.assertEqual(
			transform_point(
				Point(1, 2, 0),
				CoordinateFrame(
					origin=Point(1, 8, 0),
					x_axis=Point(-1, 0, 0),
					y_axis=Point(0, -1, 0)
				),
				CoordinateFrame(
					origin=Point(1, 3, 0),
					x_axis=Point(0, 1, 0),
					y_axis=Point(-1, 0, 0)
				)
			),
			Point(3, 1, 0)
		)


if __name__=="__main__":
    unittest.main()