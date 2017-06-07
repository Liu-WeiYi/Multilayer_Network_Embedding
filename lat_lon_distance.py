from math import *
import math


'''
# 距离的第一种计算方法 --- 可能出现分母为 0 的情况！
def calcDistance(Lat_A, Lng_A, Lat_B, Lng_B):
    # input Lat_A 纬度A
    # input Lng_A 经度A
    # input Lat_B 纬度B
    # input Lng_B 经度B
    # output distance 距离(km)

    ra = 6378.140  # 赤道半径 (km)
    rb = 6356.755  # 极半径 (km)
    flatten = (ra - rb) / ra  # 地球扁率
    rad_lat_A = radians(Lat_A)
    rad_lng_A = radians(Lng_A)
    rad_lat_B = radians(Lat_B)
    rad_lng_B = radians(Lng_B)
    pA = atan(rb / ra * tan(rad_lat_A))
    pB = atan(rb / ra * tan(rad_lat_B))
    xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
    c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
    c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (xx + dr)
    return distance
'''
def calcDistance(Lat_A, Lng_A, Lat_B, Lng_B):
    # input Lat_A 纬度A
    # input Lng_A 经度A
    # input Lat_B 纬度B
    # input Lng_B 经度B
    # output distance 距离(km)
    radlat1=radians(Lat_A)
    radlat2=radians(Lat_B)
    a=radlat1-radlat2
    b=radians(Lng_A)-radians(Lng_B)
    s=2*asin(sqrt(pow(sin(a/2),2)+cos(radlat1)*cos(radlat2)*pow(sin(b/2),2)))
    earth_radius=6378.137
    s=s*earth_radius
    if s<0:
        return -s
    else:
        return s
