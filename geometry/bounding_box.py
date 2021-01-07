from .point import Point
from .quaternion import Quaternion
import numpy as np
import open3d as o3d
from geometry import Coord


class BoundingBox():
    def __init__(self, centre: Point, length: Point):
        self.centre = centre
        self.length = length

    @staticmethod
    def from_corners(self, min: Point, max: Point):
        return BoundingBox((min + max) * 0.5, max - min)

    @staticmethod
    def from_field(x: np.ndarray, y: np.ndarray, z: np.ndarray):
        max = Point(x.max(), y.max(), z.max())
        min = Point(x.min(), y.min(), z.min())
        return BoundingBox.from_corners(min, max)

    @property
    def volume(self) -> float:
        return self.length.x * self.length.y * self.length.z

    @property
    def max(self):
        return self.centre + 0.5 * self.length

    @property
    def min(self):
        return self.centre - 0.5 * self.length


class RotatedBoundingBox():
    def __init__(self, orientation: Quaternion, centre: Point, length: Point):
        self.orientation = orientation
        self.centre = centre
        self.length = length

    @staticmethod
    def from_field(x: np.ndarray, y: np.ndarray, z: np.ndarray):
        pcloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.array([x, y, z]).transpose()))
        obb = pcloud.get_oriented_bounding_box()
        corners = np.asarray(obb.get_box_points())

        return RotatedBoundingBox(
            Quaternion.from_rotation_matrix(np.array(obb.R)),
            Point(*tuple(obb.get_centre())),
            Point(0, 0, 0)
        )

