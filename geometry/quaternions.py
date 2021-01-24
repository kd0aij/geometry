from geometry.points import Points

import numpy as np


class Quaternions():
    def __init__(self, data):
        """Args: data (np.array): npoint * 4 array of point locations"""
        self.data = data

    @property
    def w(self):
        return self.data[:, 0]

    @property
    def x(self):
        return self.data[:, 1]

    @property
    def y(self):
        return self.data[:, 2]

    @property
    def z(self):
        return self.data[:, 3]

    def __abs__(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    @property
    def axis(self):
        return Points(self.data[:, 1:])

    def norm(self):
        dab = 1 / abs(self)
        return self * dab

    def conjugate(self):

        return Quaternions(self.data * np.array([1, -1, -1, -1]))

    def inverse(self):
        return self.conjugate().norm()

    @property
    def count(self):
        return self.data.shape[0]

    def __mul__(self, other):
        if isinstance(other, Quaternions):
            pass
        #    if not self.count == other.count:
        #        raise NotImplementedError
        #    else:
        #        w = self.w * other.w - np.tensordot( self.axis, other.axis)
#
        #        xyz = self.w * other.axis + other.w * self.axis + \
        #            np.cross(self.axis, other.axis)
        #        return Quaternions(np.column_stack(w, xyz))

        elif isinstance(other, float):
            return Quaternions(self.data * other)
        else:
            raise NotImplementedError
