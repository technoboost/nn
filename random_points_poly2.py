import random
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate
import csv

def random_points_in_polygon(polygon, k):
    "Return list of k points chosen uniformly at random inside the polygon."
    areas = []
    transforms = []
    for t in triangulate(polygon):
        areas.append(t.area)
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        transforms.append([x1 - x0, x2 - x0, y2 - y0, y1 - y0, x0, y0])
    points = []
    for transform in random.choices(transforms, weights=areas, k=k):
        x, y = [random.random() for _ in range(2)]
        if x + y > 1:
            p = Point(1 - x, 1 - y)
        else:
            p = Point(x, y)
        points.append(affine_transform(p, transform))
    return points
    
poly = Polygon([(141.4378366,-25.95915986), (165.4279876,-29.43400298), (163.1382942,-47.65345814), (133.1675418,-42.99807751)])
points = random_points_in_polygon(poly, 500)
tmp =[]
for point in points:
    tmp = tmp + [ [str(point).split('(')[1].split(')')[0]]]

row = [[str(float((i[0].split(' ')[0]))/165.4279876)+' '+ str(float((i[0].split(' ')[1]))/-47.65345814)+ ' 1'] for i in tmp ]
print(row)
with open('traindata.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(row)
csvFile.close()


