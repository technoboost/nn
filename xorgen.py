import numpy as np
import random
import csv

def random_point_within():

    x = random.randint(0,1)
    y = random.randint(0,1)

    return [x, y]

    

points = [random_point_within() for i in range(500)]
checks = [str(point[0]^point[1]) for point in points]
print (checks[2])

row = [[str(i[0])+' '+ str(i[1])+' ' + str(checks[j]) ] for i,j in zip(points,range(len(checks)))]
#print float(row[:][0].split(' ')[0])/165.4279876
with open('traindata1.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(row)
csvFile.close()
