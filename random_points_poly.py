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

    
poly = Polygon([(141.4378366,-25.95915986), (165.4279876,-29.43400298), (163.1382942,-47.65345814), (133.1675418,-42.99807751)])
points = [random_point_within(poly) for i in range(500)]
checks = [int(point.within(poly)) for point in points]
print (checks[2])
tmp =[]
for point in points:
    tmp = tmp + [ [str(point).split('(')[1].split(')')[0]]]
#print row
#print [[str(float((i[0].split(' ')[0]))/165.4279876)+' '+ str(float((i[0].split(' ')[1]))/-47.65345814)] for i in tmp]
row = [[str(float((i[0].split(' ')[0]))/165.4279876)+' '+ str(float((i[0].split(' ')[1]))/-47.65345814)+' ' + str(checks[j]) ] for i,j in zip(tmp,range(len(checks)))]
#print float(row[:][0].split(' ')[0])/165.4279876
with open('traindata.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(row)
csvFile.close()
