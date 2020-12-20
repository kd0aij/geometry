from geometry.point import Point
from math import atan2, asin, copysign, pi


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
    def from_euler(eul: Point):
        half = eul * 0.5
        c = half.cosines
        s = half.sines
        return Quaternion(
            w=c.x * c.y * c.y + s.x * s.y * s.z,
            axis=Point(
                x=s.x * c.y * c.z - c.x * s.y * s.z,
                y=c.x * s.y * c.z + s.x * c.y * s.z,
                z=c.x * c.y * s.z - s.x * s.y * c.z
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
