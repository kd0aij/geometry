from . import Point, Quaternion, Coord
import numpy as np


class Transformation():
    def __init__(self, coord_a: Coord, coord_b: Coord):
        self.coord_a = coord_a
        self.coord_b = coord_b

        self.translation = self.coord_b.origin - self.coord_a.origin
        self.rotation = np.dot(
            self.coord_b.inverse_rotation_matrix, self.coord_a.rotation_matrix)

        self.pos_vec = np.vectorize(
            lambda *args: self.point(Point(*args)).to_tuple())

        self.eul_vec = np.vectorize(
            lambda *args: self.quat(
                Quaternion.from_euler(Point(*args))).to_tuple()
        )

    def rotate(self, point: Point):
        return point.rotate(self.rotation)

    def translate(self, point: Point):
        return point + self.translation

    def point(self, point: Point):
        return self.rotate(self.translate(point))

    def quat(self, quat: Quaternion):
        return Quaternion(quat.w, self.rotate(quat.axis))
