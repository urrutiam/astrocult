# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:05:20 2026

@author: USER
"""

import datetime
import math



import sys
sys.path.append("C:/Users/USER/Downloads/archaeoastropy/src/archeaoastropy")



from JulianHelper import JulianHelper
import Declination as Declination
import Azimuth as Azimuth
import GeoUtils as GeoUtils

def test_declination():
    """Unit tests for the Declination class"""
    
    dec = Declination.Declination()
    
    dec.calculate_declinations_for_entire_year(2000, 45, 23, 100)
    dec.plot_decl_adaptive([1,2,3], 45, [0.1,0.2,0.3])

def test_julianhelper():
    """Unit tests for the JulianHelper class"""
    
    jh = JulianHelper()
    
    julian_dates = [None, datetime.datetime(1500,1,1), datetime.datetime(2000,1,1)]

    for julian_date in julian_dates:
        print("Testing Julian date:", str(julian_date))
        try:
            print(jh.julian_to_gregorian_datetime(julian_date))
            print(jh.julian_gregorian_offset_days(julian_date))
            print(jh.julian_gregorian_offset_days(julian_date))
            print(jh.estimate_sunrise_noaa(julian_date, 45, 23))
            print(jh.estimate_sunrise_noaa(julian_date, 45, 200))
        except Exception as e:
            print(e)
        print("---")
    try:
        print("Testing Julian date: 2000,1,1")
        print(jh.julian_gregorian_offset_days(2000,1,1))
        print("---")
        print("Testing Julian date: 2000,1,-1")
        print(jh.julian_gregorian_offset_days(2000,1,-1))
    except Exception as e:
        print(e)
        
def test_azimuth():
    """Unit tests for the Azimuth class"""
    
    az = Azimuth.Azimuth()
    az.plot_az_adaptive([1,2,3], 45, [0.1,0.2,0.3])
    
def test_geoutils():
    sunDec = -24.1552
    moonDec = -30.1094
    
    vatinLat = 45.210
    vatinLon = 21.287
    
    # summer solstice point 6a
    target1Lat = 45.141687
    target1Lon = 21.425
    target1Alt = 1.9
    dist1FromVatin = 13380
    
    # southern lunastice point 4a
    target2Lat = 45.147294
    target2Lon = 21.371814
    target2Alt = 1.42
    dist2FromVatin = 9678
    
    gu = GeoUtils.GeoUtils()
    dec = Declination.Declination()

    az1 = gu.get_bearing(vatinLat, target1Lat, vatinLon, target1Lon)
    dec1 = dec.calc_decl(target1Lat, target1Alt, az1)
    
    print("Az1", az1, "Dec1", dec1)
    
    az2 = gu.get_bearing(vatinLat, target2Lat, vatinLon, target2Lon)
    dec2 = dec.calc_decl(target2Lat, target2Alt, az2)
    
    print("Az2", az2, "Dec2", dec2)
    
    vatinError = math.sqrt((sunDec-dec1) * (sunDec-dec1) + (moonDec-dec2) * (moonDec-dec2))
    
    print("Error", vatinError)
    
    target1HeightMeters = gu.horizon_height_meters(dist1FromVatin, target1Alt)
    target2HeightMeters = gu.horizon_height_meters(dist2FromVatin, target2Alt)
    
    d1 = gu.distance_between(vatinLat, vatinLon, target1Lat, target1Lon)
    d2 = gu.distance_between(vatinLat, vatinLon, target2Lat, target2Lon)
    
    print("Dis1", d1, "Dist2", d2)
    
    latRange = list(range(44968, 45500, 1))
    lonRange = list(range(20871, 21468, 1))
    
    latRange = [x/1000 for x in latRange]
    lonRange = [x/1000 for x in lonRange]
    
   
    targetGeoLocations = []
    targetGeoLocations.append([target1Lat, target1Lon, target1HeightMeters])
    targetGeoLocations.append([target2Lat, target2Lon, target2HeightMeters])
    
    targetAstroBodies = []
    targetAstroBodies.append(['sun', sunDec])
    targetAstroBodies.append(['moon', moonDec])
    
    coordsRange = [latRange, lonRange]
    
    df = gu.compute_error_heatmap(targetGeoLocations, targetAstroBodies, coordsRange)
    print("Location probability", gu.compute_location_probability(df, 2))
    
    gu.plot_error_heatmap_histogram(df)
    
    json_heatmap, json_targets = gu.build_geojson_from_heatmap(df, [[vatinLat, vatinLon, vatinError]], targetGeoLocations)
    gu.show_heatmap(df, [vatinLat, vatinLon, vatinError], targetGeoLocations, [vatinLat, vatinLon], 11, 4)    
        
if __name__ == "__main__":
    test_julianhelper()
    test_declination()
    test_azimuth()
    test_geoutils()
