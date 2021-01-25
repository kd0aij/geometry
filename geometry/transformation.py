from . import Point, Quaternion, Coord, Points, Quaternions

import numpy as np
from typing import Union

class Transformation():
    def __init__(self, translation: Point, rotation: Quaternion):
        self.translation = translation
        self.rotation = rotation

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

    def rotate(self, point: Union[Point, Points]):
        return self.rotation.transform_point(point)

    def translate(self, point: Union[Point, Points]):
        return point + self.translation

    def point(self, point: Union[Point, Points]):
        return self.rotate(self.translate(point))

    def quat(self, quat: Union[Quaternion, Quaternions]):
        return self.rotation * quat
