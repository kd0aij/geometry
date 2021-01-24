import re
import numpy as np
from numbers import Number


class Points(object):
    __array_priority__ = 15.0
    def __init__(self, data: np.array):
        """
        Args:
            data (np.array): npoint * 3 array of point locations
        """
        self.data = data

    @property
    def x(self):
        return self.data[:,0]

    @property
    def y(self):
        return self.data[:,1]

    @property
    def z(self):
        return self.data[:,2]

    @property
    def count(self):
        return self.data.shape[0]  

    def __abs__(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __add__(self, other):
        return Points(self.data + other.data)
    
    def __mul__(self, other):
        if isinstance(other, Points):
            return Points(self.data * other.data)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == self.count:
                    return Points(self.data * other[:, np.newaxis])
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif isinstance(other, Number):
            return Points(self.data * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Points):
            return Points(self.data / other.data)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == self.count:
                    return Points(self.data / other[:, np.newaxis])
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif isinstance(other, Number):
            return Points(self.data / other)

    def __neg__(self):
        return -1 * self

    def scale(self, value):
        if isinstance(value, Number):
            return (value / abs(self)) * self
        else:
            raise NotImplementedError

    def unit(self):
        return self.scale(1)

    def sines(self):
        return Points(np.sin(self.data))
    
    def cosines(self):
        return Points(np.cos(self.data))

    def acosines(self):
        return Points(np.acos(self.data))

    def asines(self):
        return Points(np.asin(self.data))

    