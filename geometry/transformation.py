from . import Point, Quaternion, Coord
import numpy as np


class Transformation():
    def __init__(self, translation: Point, rotation: Quaternion):
        self.translation = translation
        self.rotation = rotation

        self.pos_vec = np.vectorize(
            lambda *args: tuple(self.point(Point(*args))))

        self.eul_vec = np.vectorize(
            lambda *args: tuple(self.quat(
                Quaternion.from_euler(Point(*args)))
        ))

    @staticmethod
    def from_coords(coord_a: Coord, coord_b: Coord):
        return Transformation(
            coord_b.origin - coord_a.origin,
            Quaternion.from_rotation_matrix(
                np.dot(
                    coord_b.inverse_rotation_matrix,
                    coord_a.rotation_matrix
                ))
        )

    def rotate(self, point: Point):
        return self.rotation.transform_point(point)

    def translate(self, point: Point):
        return point + self.translation

    def point(self, point: Point):
        return self.rotate(self.translate(point))

    def quat(self, quat: Quaternion):
        return self.rotation * quat
