import numpy as np
from shapely.geometry import Polygon, Point, LineString
import random
import csv

def random_point_within(poly):
    min_x, min_y, max_x, max_y = poly.bounds

    x = random.uniform(min_x, max_x)
    #x_line = LineString([(x, min_y), (x, max_y)])
    #x_line_intercept_min, x_line_intercept_max = x_line.intersection(poly).xy[1].tolist()
    y = random.uniform(min_y, max_y)

    return Point([x, y])

    
#poly = Polygon([(141.4378366,-25.95915986), (165.4279876,-29.43400298), (163.1382942,-47.65345814), (133.1675418,-42.99807751)])
poly =Polygon([(45.43484,-115.4248),(47.38203,-110.73865),(47.62409,-106.96061),(47.57653,-101.86523),(46.10371,-99.53613),(44.53842,-102.85412),(44.06667,-106.35545),(42.04122,-108.44858),(39.41135,-111.25999),(39.70414,-115.05032),(41.88252,-118.35791),(44.2452,-118.34473),(46.09587,-119.39489),(47.24941,-116.58691)])
points = [random_point_within(poly) for i in range(20000)]
#checks = [int(point.within(poly)) for point in points]
#print (checks[2])

