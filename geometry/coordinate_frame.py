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
from geometry.point import Point, cross_product, arbitrary_perpendicular, is_perpendicular
from geometry.vector import Vector
from geometry.line import Line
from typing import List, Union


class Coord(object):
    __slots__ = ["origin", "x_axis", "y_axis", "z_axis",
                 "_rotation_matrix", "_inverse_rotation_matrix"]

    def __init__(self, origin: Point, x_axis: Point, y_axis: Point, z_axis: Point):
        self.origin = origin
        self.x_axis = x_axis.unit()
        self.y_axis = y_axis.unit()
        self.z_axis = z_axis.unit()

    @staticmethod
    def from_nothing():
        return Coord(Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0), Point(0, 0, 1))

    @staticmethod
    def from_xy(origin: Point, x_axis: Point, y_axis: Point):
        return Coord(origin, x_axis, y_axis, cross_product(x_axis, y_axis))

    @staticmethod
    def from_yz(origin: Point, y_axis: Point, z_axis: Point):
        return Coord(origin, cross_product(y_axis, z_axis), y_axis, z_axis)

    @staticmethod
    def from_zx(origin: Point, z_axis: Point, x_axis: Point):
        return Coord(origin, x_axis, cross_product(z_axis, x_axis), z_axis)

    def euler_rotation(self, angles: Point = Point(0, 0, 0)):
        return self.rotate(angles.to_rotation_matrix())

    @property
    def rotation_matrix(self):
        return [self.x_axis.to_list(), self.y_axis.to_list(), self.z_axis.to_list()]

    @property
    def inverse_rotation_matrix(self):
        return [
            [self.x_axis.x, self.y_axis.x, self.z_axis.x],
            [self.x_axis.y, self.y_axis.y, self.z_axis.y],
            [self.x_axis.z, self.y_axis.z, self.z_axis.z]
        ]

    def rotate(self, rotation_matrix=List[List[float]]):
        return Coord(
            origin=self.origin,
            x_axis=self.x_axis.rotate(rotation_matrix),
            y_axis=self.y_axis.rotate(rotation_matrix),
            z_axis=self.z_axis.rotate(rotation_matrix)
        )

    def __eq__(self, other):
        return \
            self.origin == other.origin and \
            self.x_axis == other.x_axis and \
            self.y_axis == other.y_axis and \
            self.z_axis == other.z_axis


def _transform_point(point: Point, from_coord: Coord, to_coord: Coord):
    temp_point = point.rotate(
        from_coord.inverse_rotation_matrix
    ) + from_coord.origin - to_coord.origin
    return temp_point.rotate(to_coord.rotation_matrix)


def transform_point(points: Union[Point, List[Point]], from_coord: Coord = None,
                    to_coord: Coord = None):
    if not from_coord:
        from_coord = Coord.from_nothing()
    if not to_coord:
        to_coord = Coord.from_nothing()
    if isinstance(points, list):
        out_points = []
        for point in points:
            out_points.append(_transform_point(point, from_coord, to_coord))
        return out_points
    elif isinstance(points, Point):
        return _transform_point(points, from_coord, to_coord)
    else:
        raise TypeError


def transform_line(lines: Union[Line, List[Line]], from_coord: Coord = None,
                   to_coord: Coord = None):
    if not from_coord:
        from_coord = Coord.from_nothing()
    if not to_coord:
        to_coord = Coord.from_nothing()
    if isinstance(lines, list):
        out_lines = []
        for line in lines:
            out_lines.append(_transform_line(line, from_coord, to_coord))
        return out_lines
    elif isinstance(lines, Line):
        return _transform_line(lines, from_coord, to_coord)
    else:
        raise TypeError


def _transform_line(line: Line, from_coord: Coord, to_coord: Coord):
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

