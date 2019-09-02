import numpy as np
from shapely.geometry import Polygon, Point, LineString
import random
import csv
rows = [] 
def read_point_without():
    filename = "testdata.csv"
  
# initializing the titles and rows list 
    #rows=[]
  
# reading csv file 
    with open(filename, 'r') as csvfile: 
    # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
  
    # extracting each data row one by one 
        #for row in csvreader: 
            #rows.append(row) 
  
    # get total number of rows 
        print("Total no. of rows: %d"%(csvreader.line_num))
def read_point_within():
   # csv file name 
    filename = "testdata.csv"
  
# initializing the titles and rows list 
    #rows=[]
  
# reading csv file 
    with open(filename, 'r') as csvfile: 
    # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
  
    # extracting each data row one by one 
        for row in csvreader: 
            rows.append(row) 
  
    # get total number of rows 
        print("Total no. of rows: %d"%(csvreader.line_num)) 

    
poly = Polygon([(141.4378366/165.4279876,-25.95915986/-47.65345814), (165.4279876/165.4279876,-29.43400298/-47.65345814), (163.1382942/165.4279876,-47.65345814/-47.65345814), (133.1675418/165.4279876,-42.99807751/-47.65345814)])
for i in range(50000):
    read_point_without()
read_point_within()
#for row in rows:
#    print (row[0].split(' ')[0]) 
points = [Point([float(row[0].split(' ')[0]), float(row[0].split(' ')[1])]) for row in rows]
checks = [int(point.within(poly)) for point in points]
print (checks[2])

