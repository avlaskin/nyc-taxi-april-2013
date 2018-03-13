import os
import time
import json
from collections import namedtuple, deque
import numpy as np
import pandas as pd
from scipy import stats
import scipy
from operator import itemgetter, attrgetter
from scipy.special import factorial
import math
#from gmplot import gmplot

Point = namedtuple('Point', ('x', 'y'))

class TaxiZone:
    def __init__(self, locID, polygons, name="", boro=""):
        self.locID = locID
        self.polygons = polygons
        self.name = name
        self.boro = boro
        self.count = 0

def haversine_distance(lat1, lng1, lat2, lng2):
    """function to calculate haversine distance between two co-ordinates"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    earth_radius = 6371  #  km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * earth_radius * np.arcsin(np.sqrt(d))
    return h

def manhattan_distance(lat1, lng1, lat2, lng2):
    """function to calculate manhatten distance between pick->drop"""
    a = haversine_distance(lat1, lng1, lat1, lng2)
    b = haversine_distance(lat1, lng1, lat2, lng1)
    return a + b

def is_point_in_zone(point, zone):
    """Checks if coordinates of a Point are in the zone.
        Very slow and not efficient algo, may be we can use code from OpenCV for that."""
    for poly in zone.polygons:
        inside = False
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[i - 1]
            if ((p1.y > point.y) != (p2.y > point.y)) and (point.x < (p2.x - p1.x) * (point.y - p1.y) / (p2.y - p1.y) + p1.x):
                inside = not inside
        if inside:
            return True
    return False

def get_zone_index(zones, point):
    for i in range(len(zones)):
        zone = zones[i]
        if is_point_in_zone(point, zone):
            return i
    return -1

def get_zone_locid(zones, point):
    for i in range(len(zones)):
        zone = zones[i]
        if is_point_in_zone(point, zone):
            return zone.locID
    return 400

def get_zone_index_by_name(zones, name):
    for i in range(len(zones)):
        zone = zones[i]
        if name == zone.name:
            return i
    return -1

def process_zones(zones_path):
    with open(zones_path, 'r') as f:
        zones_json = json.load(f)

    zones = []
    for feature in zones_json['features']:
        geom_type = feature['geometry']['type']
        zone_name = feature['properties']['zone']
        area_name = feature['properties']['borough']
        if geom_type == 'Polygon':
            polygons_json = [feature['geometry']['coordinates']]
        elif geom_type == 'MultiPolygon':
            polygons_json = feature['geometry']['coordinates']
        else:
            raise ValueError('unknown geometry type {}'.format(geom_type))

        polygons = [[Point(*p) for p in poly[0]] for poly in polygons_json]
        zones.append(TaxiZone(feature['properties']['LocationID'], polygons, zone_name, area_name))
    return zones

def getCleanTripData():
    trip = pd.read_csv('./data/clean_1_trip_data_4_2013.csv', skipinitialspace=True, dtype={"rate_code": int, "passenger_count": int,"store_and_fwd_flag": str})
    lat_min = 40.5
    lat_max = 41
    long_min = -74.25
    long_max = -73.70
    location_p_long_filter = ((trip.pickup_longitude >= long_min) & (trip.pickup_longitude <= long_max))
    location_p_lat_filter = ((trip.pickup_latitude >= lat_min) & (trip.pickup_latitude <= lat_max))
    location_d_lat_filter = ((trip.dropoff_latitude >= lat_min) & (trip.dropoff_latitude <= lat_max))
    location_d_long_filter = ((trip.dropoff_longitude >= long_min) & (trip.dropoff_longitude <= long_max))
    filtered_trips = trip[location_p_long_filter & location_p_lat_filter & location_d_long_filter & location_d_lat_filter]
    return filtered_trips

if __name__ == "__main__":
    """Here we just map trips to Taxi zones"""
    filtered_trips = getCleanTripData()
    zones = process_zones('./data/taxi_zones/taxi_zones.json')

    pickups_lat = np.array(filtered_trips.pickup_latitude)
    pickups_long = np.array(filtered_trips.pickup_longitude)
    dropoff_lat = np.array(filtered_trips.dropoff_latitude)
    dropoff_long = np.array(filtered_trips.dropoff_longitude)
    pickups = []
    dropoffs = []

    l = len(filtered_trips.index)
    print("Total: {}".format(l))
    s = time.time()
    for i in range(l):
        pp = Point(pickups_long[i], pickups_lat[i])
        dp = Point(dropoff_long[i], dropoff_lat[i])
        pickup_locid = get_zone_locid(zones, pp)
        dropoff_locid = get_zone_locid(zones, dp)
        pickups.append(pickup_locid)
        dropoffs.append(dropoff_locid)
        if i % 1000 == 0:
            if i > 0:
                e = time.time()
                d = (e - s)
                speed = i / d
                print("Done: {} / {} Time Spent: {} mins. Left Time: {} mins. Speed: {}".format(i,l,d/60,(l-i)/speed/60,speed))

    trips = filtered_trips.assign(pickup_locid=pd.Series(pickups).values)
    trips = trips.assign(dropoff_locid=pd.Series(dropoffs).values)
    trips.to_csv('./data/1_clean_trip_data_plus_locations_4_2013.csv', sep=',')
