
from . import Point, dot_product, cross_product
from math import atan2, asin, copysign, pi, sqrt
from typing import List, Dict
import numpy as np


class Quaternion():
    def __init__(self, w, axis: Point):

        self.w = w
        self.axis = axis

    @property
    def x(self):
        return self.axis.x

    @property
    def y(self):
        return self.axis.y

    @property
    def z(self):
        return self.axis.z

    @staticmethod
    def from_tuple(w, x, y, z):
        return Quaternion(w, Point(x,y,z))

    def to_tuple(self):
        return(self.w, self.x, self.y, self.z)

    def to_list(self):
        return [self.w, self.x, self.y, self.z]

    def __abs__(self):
        return sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def norm(self):
        ab = abs(self)
        return Quaternion(self.w / ab, self.axis / ab)

    def conjugate(self):
        return Quaternion(self.w, - self.axis)

    def inverse(self):
        return self.conjugate().norm()

    def __mul__(self, other):
        return Quaternion(
            self.w * other.w - dot_product(self.axis, other.axis),
            self.w * other.axis + other.w * self.axis +
            cross_product(self.axis, other.axis)
        )

    def transform_point(self, point: Point):
        '''Transform a point by the rotation described by self'''
        return self * Quaternion(0, point) * self.inverse()

    @staticmethod
    def from_euler(eul: Point):
        half = eul * 0.5
        c = half.cosines
        s = half.sines
        return Quaternion(
            w=c.y * c.z * c.x - s.y * s.z * s.x,
            axis=Point(
                x=s.y * s.z * c.x + c.y * c.z * s.x,
                y=s.y * c.z * c.x + c.y * s.z * s.x,
                z=c.y * s.z * c.x - s.y * c.z * s.x
            )
        )

    def to_euler(self):
        roll = atan2(
            2 * (self.w * self.x + self.y * self.z),
            1 - 2 * (self.x * self.x + self.y * self.y)
        )

        _sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(_sinp) >= 1:
            pitch = copysign(pi / 2, _sinp)
        else:
            pitch = asin(_sinp)

        yaw = atan2(
            2 * (self.w * self.z + self.x * self.y),
            1 - 2 * (self.y * self.y + self.z * self.z)
        )

        return Point(roll, pitch, yaw)

    def to_rotation_matrix(self):
        """http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        https://github.com/mortlind/pymath3d/blob/master/math3d/quaternion.py
        """
        n = self.norm()
        s, x, y, z = n.w, n.x, n.y, n.z
        x2, y2, z2 = n.x**2, n.y**2, n.z**2
        return [
            [1 - 2 * (y2 + z2), 2 * x * y - 2 * s * z, 2 * s * y + 2 * x * z],
            [2 * x * y + 2 * s * z, 1 - 2 * (x2 + z2), -2 * s * x + 2 * y * z],
            [-2 * s * y + 2 * x * z, 2 * s * x + 2 * y * z, 1 - 2 * (x2 + y2)]
        ]

    @staticmethod
    def from_rotation_matrix(M: np.ndarray):
        """http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
        https://github.com/mortlind/pymath3d/blob/master/math3d/quaternion.py
        """
        tr = M.trace() + 1 
        if tr > 1e-10:
            w = 0.5 / np.sqrt(tr)
            return Quaternion(
                0.25 / w,
                Point(
                    w * (M[2, 1] - M[1, 2]),
                    w * (M[0, 2] - M[2, 0]),
                    w * (M[1, 0] - M[0, 1])
                )
            ).norm()
        else:
            diag = M.diagonal()
            u = diag.argmax()
            v = (u + 1) % 3
            w = (v + 1) % 3
            r = np.sqrt(1 + M[u, u] - M[v, v] - M[w, w])
            if abs(r) < 1e-10:
                return Quaternion(
                    1,  # utils.flt(1.0)?
                    Point(0, 0, 0)
                )
            else:
                tworinv = 1.0 / (2 * r)
                return Quaternion(
                    (M[w, v] - M[v, w]) * tworinv,
                    Point(
                        0.5 * r,
                        (M[u, v] + M[v, u]) * tworinv,
                        (M[w, u] + M[u, w]) * tworinv
                    )).norm()

    def __str__(self):
        return "W:{w:.2f}\nX:{x:.2f}\nY:{y:.2f}\nZ:{z:.2f}".format(w=self.w, x=self.x, y=self.y, z=self.z)

    def to_dict(self, prefix=''):
        return dict({prefix + 'w': self.w}, **self.axis.to_dict(prefix))

    def rotate(self, rotation_matrix=List[List[float]]):
        return Quaternion(self.w, self.axis.rotate(rotation_matrix))

    @staticmethod
    def from_dict(value: Dict):
        return Quaternion(
            value['w'],
            Point.from_dict(value)
        )
