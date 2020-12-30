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
from typing import List
import unittest
from geometry.point import Point, is_parallel
from geometry.vector import Vector, distance_to_vector
from math import sqrt


class Line(object):
    __slots__ = ["_start", "_end"]

    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    @property
    def vector(self) -> Vector:
        return Vector(self.start, self.end - self.start)

    @property
    def length(self):
        return abs(self.start - self.end)

    def parametric_point(self, proportion: float) -> Point:
        return (1 - proportion) * self._start + proportion * self._end

    def parametric_plane(self, proportion: float) -> Vector:
        return Vector(self.parametric_point(proportion), self.vector.direction.unit)


def check_extend_line(line: Line, new_end_point: Point, tolerance: float) -> [bool, Line]:
    extended_line = Line(line.start, new_end_point)
    if abs(distance_to_vector(line.end, extended_line.vector)) > tolerance:
        return True, Line(line.end, new_end_point)
    else:
        return False, extended_line


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


class TestLine(unittest.TestCase):
    def setUp(self):
        self.line0 = Line(Point(0, 0, 0), Point(1, 0, 0))
        self.points = [Point(0, 0, 0), Point(1, 0, 0),
                       Point(2, 0, 0), Point(3, 1, 0)]

    def test_vector(self):
        self.assertTrue(is_parallel(
            self.line0.vector.direction, Point(1, 0, 0)))

    def test_length(self):
        self.assertEqual(self.line0.length, 1)

    def test_parametric_point(self):
        self.assertEqual(self.line0.parametric_point(0.5), Point(0.5, 0, 0))
        self.assertEqual(self.line0.parametric_point(0.25), Point(0.25, 0, 0))

    def test_parametric_plane(self):
        self.assertEqual(self.line0.parametric_plane(
            0.5), Vector(Point(0.5, 0, 0), Point(1, 0, 0)))

    def test_check_extend_line(self):
        line = Line(Point(0, 0, 0), Point(1, 0, 0))
        is_new, new_line = check_extend_line(line, Point(2, 0, 0), 1)
        self.assertFalse(is_new)
        self.assertEqual(new_line.length, 2)
        is_new, new_line = check_extend_line(line, Point(2, 1, 0), 1)
        self.assertFalse(is_new)
        self.assertEqual(new_line.length, Line(
            Point(0, 0, 0), Point(2, 1, 0)).length)
        is_new, new_line = check_extend_line(line, Point(2, 1, 0), 0.25)
        self.assertTrue(is_new)
        self.assertEqual(new_line.length, sqrt(2))

    def test_lines_from_points(self):
        lines = lines_from_loads_of_points(self.points, 1)
        self.assertEqual(len(lines), 1)
        lines = lines_from_loads_of_points(self.points, 0.25)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1].length, sqrt(2))


if __name__ == "__main__":
    unittest.main()
