from typing import List
import unittest
from geometry.point import Point, is_parallel
from geometry.vector import Vector, Plane
from math import sqrt

class Line(object):
	__slots__ = ["_start", "_end"]
	
	def __init__(self, start: Point = None, end: Point = None):
		if not start:
			self._start = Point()
		else:
			self._start = start
		if not end:
			self._end = Point(z=1)
		else:
			self._end = end
	
	@property
	def start(self) -> Point:
		return self._start
	
	@start.setter
	def start(self, value: Point):
		self._start = value
	
	@property
	def end(self) -> Point:
		return self._end
	
	@end.setter
	def end(self, value: Point):
		self._end = value
	
	@property
	def vector(self) -> Vector:
		return Vector(self.start, self.end - self.start)
	
	@vector.setter
	def vector(self, value: Vector):
		self._start = value.origin
		self._end = self._start + value.direction
	
	@property
	def length(self):
		return abs(self.start - self.end)
	
	@length.setter
	def length(self, value: float):
		self._end = self.start + value * self.vector.direction.unit()
	
	def parametric_point(self, proportion: float) -> Point:
		return (1 - proportion) * self._start + proportion * self._end
	
	def parametric_plane(self, proportion: float) -> Plane:
		return Plane(self.parametric_point(proportion), self.vector.direction.unit)


def lines_from_loads_of_points(points: List[Point], tolerance=0.5) -> List[Line]:
	line = Line(points[0], points[1])
	lines = [line]
	for point in points[2:]:
		is_new, line = check_extend_line(line, point, tolerance)
		if is_new:
			lines.append(line)
		else:
			lines[-1] = line
	return lines


def check_extend_line(line: Line, new_end_point: Point, tolerance: float) -> [bool, Line]:
	extended_line = Line(line.start, new_end_point)
	if extended_line.vector.unit_cost(line.end) > tolerance:
		return True, Line(line.end, new_end_point)
	else:
		return False, extended_line



class TestLine(unittest.TestCase):
	def setUp(self):
		self.line0 = Line(Point(), Point(x=1))
		self.points = [Point(), Point(1), Point(2), Point(3, 1)]
	
	def test_vector(self):
		self.assertTrue(is_parallel(self.line0.vector.direction, Point(x=1)))
	
	def test_length(self):
		self.assertEqual(self.line0.length, 1)
	
	def test_parametric_point(self):
		self.assertEqual(self.line0.parametric_point(0.5), Point(x=0.5))
		self.assertEqual(self.line0.parametric_point(0.25), Point(x=0.25))
	
	def test_parametric_plane(self):
		self.assertEqual(self.line0.parametric_plane(0.5), Plane(Point(x=0.5), Point(x=1)))
	
	def test_check_extend_line(self):
		line = Line(Point(), Point(x=1))
		is_new, new_line = check_extend_line(line, Point(x=2), 1)
		self.assertFalse(is_new)
		self.assertEqual(new_line.length, 2)
		is_new, new_line = check_extend_line(line, Point(x=2, y=1), 1)
		self.assertFalse(is_new)
		self.assertEqual(new_line.length, Line(Point(), Point(2, 1)).length)
		is_new, new_line = check_extend_line(line, Point(x=2, y=1), 0.25)
		self.assertTrue(is_new)
		self.assertEqual(new_line.length, sqrt(2))
	
	def test_lines_from_points(self):
		lines = lines_from_loads_of_points(self.points, 1)
		self.assertEqual(len(lines), 1)
		lines = lines_from_loads_of_points(self.points, 0.25)
		self.assertEqual(len(lines), 2)
		self.assertEqual(lines[1].length, sqrt(2))

if __name__=="__main__":
    unittest.main()