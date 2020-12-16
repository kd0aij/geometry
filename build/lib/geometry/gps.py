import math
from geometry.point import Point
import unittest


class GPSPosition(object):
	approx_earth_radius = 6378137
	
	def __init__(self, latitude, longitude):
		self._latitude = latitude
		self._longitude = longitude
		
	@property
	def latitude(self):
		return self._latitude
	
	@latitude.setter
	def latitude(self, value):
		self._latitude = value
	
	@property
	def longitude(self):
		return self._longitude
	
	@longitude.setter
	def longitude(self, value):
		self._longitude = value
	
	def to_xy(self):
		lat = self._latitude * math.pi / 180
		lon = self._longitude * math.pi / 180
		
		return [
			GPSPosition.approx_earth_radius * math.cos(lat) * math.cos(lon),
			GPSPosition.approx_earth_radius * math.cos(lat) * math.sin(lon)
		]
	
	def to_seu(self, home):
		xy_o = home.to_xy()
		xy_p = self.to_xy()
		
		return [
			xy_p[0] - xy_o[0],
			xy_p[1] - xy_o[1]
		]

	def __str__(self):
		return 'lat: ' + str(self._latitude) + ', long: ' + str(self._longitude)


if __name__ =="__main__":
	home = GPSPosition(51.459387, -2.791393)
	
	new = GPSPosition(51.458876, -2.789092)
	coord=new.to_seu(home)
	print(coord)
	
	print(math.sqrt(coord[0] ** 2 + coord[1] ** 2))
