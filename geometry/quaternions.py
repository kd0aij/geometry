from typing import Union
from geometry.quaternion import Quaternion
from numbers import Number
from geometry import Point, Points

import numpy as np
import pandas as pd


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

    @staticmethod
    def from_pandas(df):
        return Quaternions(np.array(df))

    def to_pandas(self, prefix='', suffix='', columns=['w', 'x', 'y', 'z']):
        return pd.DataFrame(self.data, columns=[prefix + col + suffix for col in columns])

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
            if not self.count == other.count:
                return NotImplemented
            else:
                w = self.w * other.w - self.axis.dot(other.axis)
#
                xyz = self.w * other.axis + other.w * self.axis + \
                    self.axis.cross(other.axis)

                return Quaternions(np.column_stack([w, xyz.data]))

        elif isinstance(other, Number):
            return Quaternions(self.data * other)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == self.count:
                    return Quaternions(self.data * other[:, np.newaxis])
                else:
                    return NotImplemented
            else:
                return NotImplemented
        elif isinstance(other, Quaternion):
            return self * Quaternions.from_quaternion(other, self.count)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Quaternions):
            return NotImplemented('this should have gone to __mul__')
        elif isinstance(other, float):
            return Quaternions(self.data * other)
        elif isinstance(other, Quaternion):
            return Quaternions.from_quaternion(other, self.count) * self
        else:
            return NotImplemented

    @staticmethod
    def from_quaternion(quat: Quaternion, count: int):
        return Quaternions(np.tile(list(quat), (count, 1)))

    @staticmethod
    def from_euler(eul: Points):
        half = eul * 0.5

        c = half.cosines()
        s = half.sines()

        return Quaternions(
            np.array([
                c.y * c.z * c.x + s.y * s.z * s.x,
                c.y * c.z * s.x - s.y * s.z * c.x,
                s.y * c.z * c.x + c.y * s.z * s.x,
                c.y * s.z * c.x - s.y * c.z * s.x
            ]).T
        )

    def transform_point(self, point: Union[Point, Points]):
        '''Transform a point by the rotation described by self'''
        if isinstance(point, Point):
            qdata = np.tile([0] + list(point), (self.count, 1))

            return (self * Quaternions(qdata) * self.inverse()).axis
        elif isinstance(point, Points):
            if point.count == self.count:

                qdata = np.column_stack((np.zeros(self.count), point.data))

                return (self * Quaternions(qdata) * self.inverse()).axis
            return NotImplemented
        else:
            return NotImplemented

    @staticmethod
    def from_axis_angle(angles: Points, factor: float = 1):
        ab = abs(angles)
        fact = ab * factor
        s = np.sin(fact)
        c = np.cos(fact)

        qdat = np.array([
            ab * c, angles.x * s, angles.y * s, angles.z * s
        ]).T

        qdat[abs(Quaternions(qdat)) == 0] = np.array([[1, 0, 0, 0]])
        return Quaternions(qdat)

    def to_axis_angle(self):
        """to a point of axis angles. must be normalized first."""
        angle = np.arccos(self.w)
        s = np.sqrt(1 - self.w**2)
        return self.axis * angle / s

    @staticmethod
    def axis_rates(q, qdot):
        wdash = qdot * q.conjugate()
        return wdash.norm().to_axis_angle() * 2

    @staticmethod
    def body_axis_rates(q, qdot):
        wdash = q.conjugate() * qdot
        return wdash.norm().to_axis_angle() * 2

    def rotate(self, rate: Points):
        return (Quaternions.from_axis_angle(rate, 0.5) * self).norm()

    def body_rotate(self, rate: Points):
        return (self * Quaternions.from_axis_angle(rate, 0.5)).norm()

    def diff(self, dt: np.array) -> Points:
        newqs = Quaternions.axis_rates(
            self,
            Quaternions(np.vstack([self.data[1:, :], self.data[-1, :]]))
        ) / dt
        return newqs.remove_outliers(2) # Bodge to get rid of phase jump
        


    def body_diff(self, dt: np.array) -> Points:
        newqs = Quaternions.body_axis_rates(
            self,
            Quaternions(np.vstack([self.data[1:, :], self.data[-1, :]]))
        ) / dt
        return newqs.remove_outliers(2) # Bodge to get rid of phase jump

