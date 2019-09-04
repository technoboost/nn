import numpy as np
from shapely.geometry import Polygon, Point, LineString
import random
import csv
import time

def random_point_within(poly):
    min_x, min_y, max_x, max_y = poly.bounds
    x = random.uniform(min_x, max_x)
    #x_line = LineString([(x, min_y), (x, max_y)])
    #x_line_intercept_min, x_line_intercept_max = x_line.intersection(poly).xy[1].tolist()
    y = random.uniform(min_y, max_y)

    return Point([x, y])

    
#poly=Polygon([(45.53317469253527,-117.9449293125),(45.53317469253527,-101.7730543125),(35.83104279748903,-113.5503980625)])#-117.9449293125,45.53317469253527,3
#poly = Polygon([(141.4378366,-25.95915986), (165.4279876,-29.43400298), (163.1382942,-47.65345814), (133.1675418,-42.99807751)])#4
#poly=Polygon([(46.02355455715567,-112.7593824375),(47.40939262435097,-106.5191480625),(44.976341351102235,-98.6089918125),(40.92584177700299,-98.1695386875),(38.2164759661061,-105.2007886875),(38.2164759661061,-114.3414136875),(41.71790878546075,-116.0113355625)])#47.40939262435097,-116.0113355625,7
#poly =Polygon([(45.43484,-115.4248),(47.38203,-110.73865),(47.62409,-106.96061),(47.57653,-101.86523),(46.10371,-99.53613),(44.53842,-102.85412),(44.06667,-106.35545),(42.04122,-108.44858),(39.41135,-111.25999),(39.70414,-115.05032),(41.88252,-118.35791),(44.2452,-118.34473),(46.09587,-119.39489),(47.24941,-116.58691)])#47.62409,-119.39489,14
#poly=Polygon([(45.4748629535656,-116.05186230802514),(34.90316080157916,-112.36045605802514),(34.68663370228233,-106.91123730802514),(33.44897064819814,-105.50498730802514),(33.44897064819814,-102.60459668302514),(31.895394390121627,-100.93467480802514),(32.267738706953864,-98.12217480802514),(36.19031247289031,-99.35264355802514),(38.35813044186262,-97.85850293302514),(40.19492127884649,-99.70420605802514),(41.194463001983976,-96.89170605802514),(44.165751790264345,-97.59483105802514),(43.404345731730906,-94.69444043302514),(47.114342487816394,-99.96787793302514),(43.786261056697555,-101.19834668302514),(43.59560637932381,-105.24131543302514),(45.28966708437165,-106.47178418302514),(45.96575314813056,-103.83506543302514),(47.531386077217384,-106.47178418302514),(43.72277683954778,-110.51475293302514),(45.96575314813056,-110.33897168302514),(45.84343492146515,-112.36045605802514),(43.659225278146266,-113.23936230802514)])#47.531386077217384,-116.05186230802514,22
poly=Polygon([(42.30563244756129,-115.9234449375),(44.28835234002869,-114.9566480625),(44.60208568691987,-112.7593824375),(45.471573946165506,-111.0015699375),(44.976341351102235,-108.9800855625),(45.100553956466314,-106.7828199375),(44.60208568691987,-106.3433668125),(44.60208568691987,-105.5523511875),(45.962492910366194,-105.2007886875),(45.962492910366194,-104.4097730625),(45.100553956466314,-104.4976636875),(44.91413393161562,-102.9156324375),(43.78287494202761,-104.6734449375),(43.655831975887644,-103.8824293125),(43.0808063478749,-104.3218824375),(42.04509240185096,-104.1461011875),(41.71790878546075,-104.9371168125),(41.52079484930571,-106.8707105625),(42.629797361046904,-106.1675855625),(42.629797361046904,-108.1890699375),(41.586566434038176,-108.1890699375),(39.78730862624461,-107.9253980625),(40.79289540013047,-106.8707105625),(38.835301864188025,-106.5191480625),(37.87038853773907,-107.2222730625),(37.59234234764633,-109.0679761875),(35.40235069523983,-108.3648511875),(35.11528197375429,-110.1226636875),(36.32827140375537,-109.9468824375),(36.257431381339934,-110.8257886875),(34.827198251710534,-111.0894605625),(34.827198251710534,-111.7046949375),(37.31325344368629,-112.0562574375),(36.822347123603215,-113.7261793125),(35.90226783511205,-113.4625074375),(35.75975377195103,-115.5718824375),(37.52266777105604,-115.6597730625),(37.52266777105604,-116.9781324375),(38.62961799890348,-116.5386793125),(38.560925160532605,-118.8238355625),(39.854812851540274,-119.7906324375),(39.92225074990976,-118.6480543125),(41.19093358956276,-118.7359449375),(41.257039554634666,-116.2750074375),(42.56509868755893,-117.6812574375)])#45.962492910366194,-119.7906324375,45
min_x, min_y, max_x, max_y = poly.bounds
print [min_x, min_y, max_x, max_y]
points = [random_point_within(poly) for i in range(20000)]
start_time = time.time()
checks = [int(point.within(poly)) for point in points]
print("--- %s seconds ---" % (time.time() - start_time))
#print (checks[2])

