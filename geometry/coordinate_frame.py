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
from geometry.quaternion import Quaternion
from typing import List, Union
import numpy as np


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
        return np.array([self.x_axis.to_list(), self.y_axis.to_list(), self.z_axis.to_list()])

    @property
    def inverse_rotation_matrix(self):
        return np.array([
            [self.x_axis.x, self.y_axis.x, self.z_axis.x],
            [self.x_axis.y, self.y_axis.y, self.z_axis.y],
            [self.x_axis.z, self.y_axis.z, self.z_axis.z]
        ])

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

    def translate(self, point):
        return Coord(self.origin + point, self.x_axis, self.y_axis, self.z_axis)


class Transformation():
    def __init__(self, coord_a: Coord, coord_b: Coord):
        self.coord_a = coord_a
        self.coord_b = coord_b

        self.translation = self.coord_b.origin - self.coord_a.origin
        self.rotation = np.dot(
            self.coord_b.inverse_rotation_matrix, self.coord_a.rotation_matrix)

        self.pos_vec = np.vectorize(
            lambda x, y, z: self.point(Point(x, y, z)))

        self.eul_vec = np.vectorize(
            lambda r, p, y: self.quat(
                Quaternion.from_euler(Point(r, p, y))).to_tuple()
        )

    def rotate(self, point: Point):
        return point.rotate(self.rotation)

    def translate(self, point: Point):
        return point + self.translation

    def point(self, point: Point):
        return self.rotate(self.translate(point))

    def quat(self, quat: Quaternion):
        return Quaternion(quat.w, self.rotate(quat.axis))
