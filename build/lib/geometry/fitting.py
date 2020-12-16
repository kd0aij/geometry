from geometry.point import Point, PointParams, mean_point, point_list_from_np_array, is_parallel
from geometry.vector import Vector, Plane, distance_to_vector, distance_to_plane, VectorParams
from geometry.circle import Circle3D, Circle2D, Circle2DParams, distance_to_circle_2d
from geometry.coordinate_frame import CoordinateFrame, transform_point
from enum import Enum, auto
from typing import List
from scipy.optimize import least_squares

import unittest
from math import sqrt
import numpy as np


class FitType(Enum):
	POINT = Point.fit_id
	VECTOR = Vector.fit_id
	PLANE = Plane.fit_id
	CIRCLE2D = Circle2D.fit_id


parameter_enums = {
	FitType.POINT: PointParams,
	FitType.VECTOR: VectorParams,
	FitType.PLANE: VectorParams,
	FitType.CIRCLE2D: Circle2DParams
}


class DataFitting(object):
	__slots__ = ["_points", "_shape", "_params", "_variable_ids", "_optimize_result"]
	
	"""find the best fit shape for a list of points"""
	
	def __init__(self, points: List[Point], initial_shape):
		self._points = points
		self._shape = initial_shape
		self._params = initial_shape.array
		self._variable_ids = []
		self._init_variable_ids()
		self._optimize_result = None
	
	@property
	def shape(self):
		return self._shape
	
	@shape.setter
	def shape(self, value):
		self._shape = value
	
	@property
	def variable_ids(self) -> List[int]:
		return self._variable_ids
	
	@variable_ids.setter
	def variable_ids(self, value: List[int]):
		self._variable_ids = value
	
	def _init_variable_ids(self):
		self._variable_ids = []
		for variable in parameter_enums[FitType(self._shape.fit_id)]:
			self._variable_ids.append(variable.value)
	
	def run_fit(self):
		self._optimize_result = least_squares(self._total_cost, self._params_to_values())
		return self
	
	def _values_to_params(self, values):
		for i, variable in enumerate(self._variable_ids):
			self._params[variable] = values[i]
	
	def _params_to_values(self):
		values = []
		for i in self._variable_ids:
			values.append(self._params[i])
		return values
	
	def _total_cost(self, values):
		self._values_to_params(values)
		self._shape.array = self._params
		cost = []
		for point in self._points:
			cost.append(self._shape.unit_cost(point))
		return cost


def fit_circle_3d(points: List[Point]):
	# TODO this function wont be needed if the class can handle multi stage fitting.
	plane_fit = DataFitting(points, Plane(origin=mean_point(points)))
	plane_fit.variable_ids = [
		parameter_enums[FitType.PLANE].vX.value,
		parameter_enums[FitType.PLANE].vY.value,
		parameter_enums[FitType.PLANE].vZ.value
	]
	plane_fit.run_fit()
	circle = Circle3D(plane_fit.shape, 1)
	circle_2d_points = transform_point(points, to_coord=circle.coordinate_frame)
	circle_fit = DataFitting(circle_2d_points, Circle2D(centre=mean_point(circle_2d_points), radius=1))
	circle_fit.variable_ids = [
		parameter_enums[FitType.CIRCLE2D].oX.value,
		parameter_enums[FitType.CIRCLE2D].oY.value,
		parameter_enums[FitType.CIRCLE2D].RADIUS.value
	]
	circle_fit.run_fit()
	circle.circle = circle_fit.shape
	circle.refresh_centre()
	return circle



class TestFitting(unittest.TestCase):
	def setUp(self):
		point_array = np.zeros((5, 3))
		point_array[:, 0] = np.linspace(0, 10, 5)
		point_array[:, 1] = np.linspace(0, 10, 5)
		self.points1 = point_list_from_np_array(point_array)
		
		point_array2 = np.zeros((5, 3))
		point_array2[:, 0] = np.linspace(0, 10, 5)
		point_array2[:, 1] = np.linspace(0, -10, 5)
		self.points2 = point_list_from_np_array(point_array2)
		
		self.points_circle = [
			Point(1, 1, 0),
			Point(1, -1, 0),
			Point(-1, -1, 0),
			Point(-1, 1, 0)
		]
	
	def test_fit_vector(self):
		fit1 = DataFitting(self.points1, Vector(mean_point(self.points1))).run_fit()
		self.assertTrue(is_parallel(fit1.shape.direction, Point(1, 1, 0)))
		self.assertAlmostEqual(abs(distance_to_vector(Point(1, 1, 0), fit1.shape)), 0, 5)
		self.assertAlmostEqual(abs(distance_to_vector(Point(0, 0, 0), fit1.shape)), 0, 5)
	
	def test_fit_plane(self):
		fit1 = DataFitting(self.points1 + self.points2, Plane()).run_fit()
		self.assertTrue(is_parallel(fit1.shape.direction, Point(0, 0, 1)))
		self.assertAlmostEqual(abs(distance_to_plane(Point(0, 0, 0), fit1.shape)), 0, 5)
		self.assertAlmostEqual(abs(distance_to_plane(Point(1, 1, 0), fit1.shape)), 0, 5)
	
	def test_fit_circle2D(self):
		#  TODO the initial guess needs to be pretty close to the end result otherwise it can get a false peak.
		fit = DataFitting(self.points_circle, Circle2D(Point(0.5, 0.5, 0), 2)).run_fit()
		self.assertAlmostEqual(fit.shape.radius, sqrt(2))
		self.assertAlmostEqual(abs(fit.shape.centre - Point(0, 0, 0)), 0)
	
	def test_fit_circle_3d(self):
		circle_3d = fit_circle_3d(self.points_circle)
		self.assertTrue(is_parallel(circle_3d.axis.direction, Point(z=1)))
		self.assertAlmostEqual(circle_3d.axis.origin, Point())
		pass
	
	def test_params(self):
		plane_fit = DataFitting(self.points_circle, Plane())
		plane_fit.variable_ids = [
			parameter_enums[FitType.PLANE].vX.value,
			parameter_enums[FitType.PLANE].vY.value,
			parameter_enums[FitType.PLANE].vZ.value
		]
		plane_fit.run_fit()


if __name__=="__main__":
    unittest.main()