from geometry.point import Point, angle_between
from geometry.vector import Vector, distance_to_vector, plane_plane_intersect, Plane, \
	vector_plane_intersect
from geometry.line import Line
from geometry.coordinate_frame import CoordinateFrame, transform_point
from enum import Enum
from typing import Union, List
import unittest


class Circle2DParams(Enum):
	oX = 0
	oY = 1
	RADIUS = 2


class Circle2D(object):
	__slots__ = ["_centre", "_radius"]
	
	def __init__(self, centre: Point = None, radius: float = None):
		self._centre = centre
		self._radius = radius
	
	fit_id = 3
	
	@property
	def radius(self) -> float:
		return self._radius
	
	@radius.setter
	def radius(self, value: float):
		self._radius = value
	
	@property
	def centre(self) -> Point:
		return self._centre
	
	@centre.setter
	def centre(self, value):
		self._centre = value
	
	@property
	def array(self):
		return [self._centre.x, self._centre.y, self._radius]
	
	@array.setter
	def array(self, value):
		self._centre.x = value[Circle2DParams.oX.value]
		self._centre.y = value[Circle2DParams.oY.value]
		self._radius = value[Circle2DParams.RADIUS.value]
	
	def copy(self):
		return Circle2D(self.centre.copy(), self.radius)
	
	def unit_cost(self, point: Point):
		return abs(distance_to_circle_2d(point, self))


class Circle3D(object):
	__slots__ = ["_coordinate_frame", "_circle"]
	
	def __init__(self, axis: Vector = None, radius: float = None, start: Point = None, mid: Point = None,
				 end: Point = None):
		if not axis and not radius:
			if start and mid and end:
				axis_vector = plane_plane_intersect(
					Line(start, mid).parametric_plane(0.5),
					Line(mid, end).parametric_plane(0.5)
				)
				centre_point = vector_plane_intersect(
					axis_vector,
					Plane(
						start,
						axis_vector.direction
					))
				self._coordinate_frame = CoordinateFrame(centre_point, z_axis=axis_vector.direction.unit)
				self._circle = Circle2D(Point(), abs(centre_point - start))
			else:
				self._coordinate_frame = None
				self._circle = None
		else:
			self._coordinate_frame = CoordinateFrame(origin=axis.origin, z_axis=axis.direction.unit)
			self._circle = Circle2D(centre=Point(0, 0, 0), radius=radius)
	
	fit_id = 4
	
	@property
	def circle(self) -> Circle2D:
		return self._circle
	
	@circle.setter
	def circle(self, value: Circle2D):
		self._circle = value
	
	@property
	def axis(self) -> Vector:
		return Vector(
			transform_point(self._circle.centre, from_coord=self._coordinate_frame),
			self._coordinate_frame.z_axis
		)
	
	@property
	def coordinate_frame(self) -> CoordinateFrame:
		return self._coordinate_frame
	
	@coordinate_frame.setter
	def coordinate_frame(self, value: CoordinateFrame):
		self._coordinate_frame = value
	
	def refresh_centre(self):
		if not self._circle.centre.x == 0 or not self._circle.centre.y == 0:
			self._coordinate_frame.origin = transform_point(self._circle.centre, from_coord=self._coordinate_frame)
			self.circle.centre = Point(0, 0, 0)
	
	def copy(self):
		self.refresh_centre()
		return Circle3D(self.axis, self._circle.radius)


class Arc(object):
	__slots__ = ["_start", "_mid", "_end", "_circle", "_angle", "_centre"]
	
	def __init__(self, start: Point = Point(x=-1), mid: Point = Point(y=1), end: Point = Point(x=1)):
		self._start = start
		self._mid = mid
		self._end = end
		self._circle = None
		self._angle = None
		self._centre = None
	
	@property
	def start(self) -> Point:
		return self._start
	
	@start.setter
	def start(self, value: Point):
		self._clear_derived_data()
		self._start = value
	
	@property
	def mid(self) -> Point:
		return self._mid
	
	@mid.setter
	def mid(self, value: Point):
		self._clear_derived_data()
		self._mid = value
	
	@property
	def end(self) -> Point:
		return self._end
	
	@end.setter
	def end(self, value: Point):
		self._clear_derived_data()
		self._end = value
	
	@property
	def circle(self) -> Circle3D:
		if not self._circle:
			self._circle = Circle3D(start=self._start, mid=self._mid, end=self._end)
		return self._circle
	
	@property
	def angle(self):
		#  TODO not sure what to do for angles > 90
		if not self._angle:
			self._angle = angle_between(self._start - self.circle.axis.origin, self._end - self.circle.axis.origin)
		return self._angle
	
	def _clear_derived_data(self):
		self._circle = None
		self._angle = None


def distance_to_circle_2d(point: Point, circle: Circle2D) -> Point:
	point_to_centre = distance_to_vector(point, Vector(circle.centre, Point(0, 0, 1)))
	if abs(point_to_centre) == 0:
		centre_to_perimeter = Point(1, 0, 0).set_length(circle.radius)
	else:
		centre_to_perimeter = -point_to_centre.set_length(circle.radius)
	return point_to_centre + centre_to_perimeter


def sum_distance_to_circle_3d(points: List[Point], circle: Circle3D) -> float:
	distance = 0
	for point in points:
		distance = distance + abs(_distance_to_circle_3d(point, circle))
	return distance


def distance_to_circle_3d(points: Union[Point, List[Point]], circle: Circle3D) -> Union[Point, List[Point]]:
	if isinstance(points, Point):
		return _distance_to_circle_3d(points, circle)
	elif isinstance(points, list):
		distances = []
		for point in points:
			distances.append(_distance_to_circle_3d(point, circle))
	else:
		raise NotImplemented


def _distance_to_circle_3d(point: Point, circle: Circle3D) -> Point:
	return distance_to_circle_2d(
		transform_point(point, to_coord=circle.coordinate_frame),
		circle.circle
	)



class TestCircle(unittest.TestCase):
	def setUp(self):
		self.circle_1 = Circle3D(
			Vector(
				origin=Point(),
				direction=Point(z=1)
			),
			10
		)
		self.circle_2 = Circle3D(
			Vector(
				origin=Point(),
				direction=Point(z=1)
			),
			10
		)
	
	def test_circle_init(self):
		self.assertEqual(
			self.circle_1.axis,
			Vector(
				origin=Point(),
				direction=Point(z=1)
			))
	
	def test_refresh_axis(self):
		self.circle_2.circle.centre = Point(1, 1, 0)
		self.assertEqual(self.circle_2.axis.origin, Point(-1, 1, 0))


class TestArc(unittest.TestCase):
	def setUp(self):
		self.arc1 = Arc(
			Point(-1, 0, 0),
			Point(0, 1, 0),
			Point(1, 0, 0)
		)
		self.arc2 = Arc(
			Point(-1, 1, 0),
			Point(0, 2, 0),
			Point(1, 1, 0)
		)
	
	def test_init(self):
		self.assertEqual(self.arc1.circle.axis.origin, Point(0, 0, 0))
		self.assertEqual(self.arc2.circle.axis.origin, Point(0, 1, 0))


if __name__=="__main__":
    unittest.main()