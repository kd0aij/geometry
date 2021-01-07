import unittest
from geometry.bounding_box import BoundingBox, RotatedBoundingBox
from geometry import Coord, Point
import numpy as np
import open3d as o3d


class TestRotatedBoundingBox(unittest.TestCase):
    def test_from_point_cloud(self):
        bb = RotatedBoundingBox.from_field(
            np.array([0, 0.0, 0.1, 10]),
            np.array([0, 0.1, 0.0, 10]),
            np.array([0, 0.0, 0.0, 10])
        )

        self.assertAlmostEqual(abs(bb.length), 17.32, 1)
        
        