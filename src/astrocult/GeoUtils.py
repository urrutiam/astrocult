# -*- coding: utf-8 -*-
#pip install geographiclib
from geographiclib.geodesic import Geodesic
import math
import pandas as pd
import seaborn as sns
#pip install geojson
import geojson
from geojson import Feature, Point, FeatureCollection
#pip install "leafmap[raster]"
import leafmap.foliumap as leafmap
import matplotlib.pyplot as plt
from typing import Tuple
import datetime
import os
import webbrowser

from urllib.request import urlopen


import Declination as Declination

class GeoUtils:
  """ Contains a set of methods for dealing with astronomical computations
  based on geographical coordinates.
  """
  
  def get_magnetic_declination(self, 
                           lat: float,
                           lon: float,
                           year: int=2026,
                           month: int=1,
                           day: int=1) -> tuple[float, float]:
      """      
      Uses the NOAA services to automatically retrieve the magnetic declination
      for a given location. It uses the WMM model (2024-2029). The service has 
      an access limit so it is advised not to overuse it.
      The returned magnetic declination can be added to the magnetic azimuth
      to obtain the true azimuth which can than be used for plotting or for 
      computing the astronomical declination.
      
      Parameters   
      lat : float
          The latitude of the location
      lon : float
          The longitude of the location
      year : int
          The year for which the magnetic declination will be retrieved
      month : int
          The month for which the magnetic declination will be retrieved
      day : int
          The day for which the magnetic declination will be retrieved
          
      Returns
      tuple 
          The magnetic declination
          The magnetic declination uncertainty
          
      Raises
          ValueError if any of the arguments is None or outside their range
      """  
      
      if None in [lat, lon]:
          raise ValueError("One of the arguments is None.") 
      if not (-90 <= lat <= 90):
          raise ValueError("The latitude should be between -90 and 90.")   
      if not (-180 <= lon <= 180):
          raise ValueError("The longitude should be between -180 and 180.") 
      if not (2024 <= year <= 2029):
          raise ValueError("The year should be between 2024 and 2029.")
      if not (1 <= month <= 12):
          raise ValueError("The month should be between 1 and 12")
      if not (1 <= day <= 31):
          raise ValueError("The day should be between 1 and 31")
              
      url = "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination?startYear=" + str(year) + "&startMonth=" + str(month) + "&startDay=" + str(day) + "&lat1=" + str(lat) + "&lon1=" + str(lon) + "&key=zNEw7&resultFormat=csv" 
      
      f = urlopen(url)
      response = f.read().decode('utf-8')
      
      print(response)
      
      data_line = response.strip().split("\n")[-1]
                 
      try:
          md = float(data_line.split(",")[4])
          md_unc = float(data_line.split(",")[6])
       
          return md, md_unc
      except Exception:
          raise Exception("Error parsing response from https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination. The server might be busy or the response format is different to what is accepted.")
          
  def get_bearing(self, lat1:float, lat2:float, lon1:float, lon2:float) -> float:
    """ Computes the azimuth in degrees from (lat1, lon1) towards (lat2, lon2).

    Parameters
      lat1 : float
        The latitude of the first point in degrees
      lat2 : float
        The latitude of the second point in degrees
      lon1 : float
        The longitude of the first point in degrees
      lon2 : float
        The longitude of the second point in degrees

    Returns
      The azimuth between two points in degrees

    Raises
      ValueError: If at least one argument is None

    """

    if None in [lat1, lat2, lon1, lon2]:
      raise ValueError("At least one argument is None.")

    if not (-90 <= lat1 <= 90) or not (-180 <= lon1 <= 180):
      raise ValueError("`lat1```should be between [-90,90] and 'lon1' between [-180,180].")

    if not (-90 <= lat2 <= 90) or not (-180 <= lon2 <= 180):
      raise ValueError("`lat2```should be between [-90,90] and 'lon2' between [-180,180].")

    brng = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']

    return brng

  def distance_between(self, lat1:float, lon1:float, lat2:float, lon2:float) -> float:
    """ Calculates the distance in meters between 2 points on the earth by using
    the haversine formula.

    Parameters
      lat1 : float
        The latitude of the first point in degrees
      lon1 : float
        The longitude of the first point in degrees
      lat2 : float
        The latitude of the second point in degrees
      lon2 : float
        The longitude of the second point in

    Returns
      The distance between two points in meters

    Raises
      ValueError: If at least one argument is None
    """

    if None in [lat1, lat2, lon1, lon2]:
      raise ValueError("At least one argument is None.")

    if not (-90 <= lat1 <= 90) or not (-180 <= lon1 <= 180):
      raise ValueError("`lat1```should be between [-90,90] and 'lon1' between [-180,180].")

    if not (-90 <= lat2 <= 90) or not (-180 <= lon2 <= 180):
      raise ValueError("`lat2```should be between [-90,90] and 'lon2' between [-180,180].")

    earthRadiusKm = 6371; #km

    dLat = math.radians(lat2 - lat1);
    dLon = math.radians(lon2 - lon1);

    lat1 = math.radians(lat1);
    lat2 = math.radians(lat2);

    a = math.sin(dLat/2) * math.sin(dLat/2) + math.sin(dLon/2) * math.sin(dLon/2) * math.cos(lat1) * math.cos(lat2);
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a));

    return earthRadiusKm * c * 1000;

  def horizon_height_degrees(self, dist:float, heightM:float) -> float:
    """ Calculates the height of the horizon in degrees.

    Parameters
      dist : float
        The distance between the observer and the target in meters
      heightM : float
        The height of the horizon in meters

    Returns
      The height of the horizon in degrees

    Raises
      ValueError: If at least one argument is None

    """

    if None in [dist, heightM]:
      raise ValueError("At least one argument is None")

    return math.degrees(math.atan(heightM / dist))

  def horizon_height_meters(self, dist:float, heightD:float) -> float:
    """ Calculates the height of the horizon in meters.

    Parameters
      dist : float
        The distance between the observer and the target in meters
      height

    Returns
      The height of the horizon in meters

    Raises
      ValueError: If at least one argument is None
    """

    if None in [dist, heightD]:
      raise ValueError("At least one argument is None.")

    if heightD > 90 or heightD < -90:
      raise ValueError("`heightD` should be within [-90,90] degrees.")

    return math.tan(math.radians(heightD)) * dist

  def compute_error_heatmap(self, targetGeoLocations:list, targetAstroBodies:list,
                            coordsRange:list) -> pd.DataFrame:
    """ Computes the error heatmap for a given set of target locations and
    astronomical bodies. The heatmap will be computed within a defined
    geographical range.

    Parameters
      targetGeoLocations : list
        Format: [[lat, lon, heightMeters], [], ...]
      targetAstroBodies : list
        Format: [['sun', dec], [], ...]
      coordsRange : list
        Format [latRange, lonRange]
    Returns
      A pandas dataframe containing the error heatmap with columns 'lat', 'lon', 'err_val'
      where `val` represents the error between the target and the astronomical
      body declinations' computed as the square root of the square sum of absolute
      declination differences (the astro body declination and the declination provided by
      the bearing from a given (lat,lon) to (targetLat, targetLon))).

    Raises
      ValueError: If at least one argument is None
    """

    if None in [targetGeoLocations, targetAstroBodies, coordsRange]:
      raise ValueError("At least one argument is None.")

    pixelValues = []
    for lat in coordsRange[0]:
      for lon in coordsRange[1]:

        dec = []
        for targetGeoLocation in targetGeoLocations:
          targetLat = targetGeoLocation[0]
          targetLon = targetGeoLocation[1]
          targetHeightMeters = targetGeoLocation[2]

          az = self.get_bearing(lat, targetLat, lon, targetLon)
          
          az = az % 360 # bring angles in 0,360 range
          targetAlt = self.horizon_height_degrees(self.distance_between(
              lat, lon, targetLat, targetLon), targetHeightMeters)
          
          decl = Declination.Declination()
          
          dec.append(decl.calc_decl(targetLat, targetAlt, az))

        errAlignment = []
        for i in range(len(targetAstroBodies)):
          dec1 = dec[i]
          dec2 = targetAstroBodies[i][1]
          errAlignment.append(math.fabs(dec1-dec2))

        pixelValues.append([lat, lon, round(math.sqrt(sum([i ** 2 for i in errAlignment])), 1)])

    return pd.DataFrame(pixelValues, columns=['lat', 'lon', 'err_val'])

  def compute_location_probability(self, df:pd.DataFrame, threshold:float=2, number_of_random_samples:int=20000) -> float:
    """Calculates the probability of randomly picking a location with less than
    the error threshold.  It uses a Monte Carlo approach by generating `number_of_random_samples` random positions.

    Parameters
      df : pd.DataFrame
        The error heatmap dataframe. Generated by `self.compute_error_heatmap`.
      threshold : float
        The error threshold

    Returns
      The probability of randomly picking a location with less than the error threshold.

    Raises
      ValueError: If `df` is None
    """

    if df is None:
      raise ValueError("The error heatmap dataframe cannot be None.")

    count = 0
    sample = df.sample(n=min(20000, int(df.size * 0.2)))
    #print(sample)
    #print(sample.size)
    for index, row in sample.iterrows():
      if row['err_val'] < threshold: # Changed 'val' to 'err_val'
        count = count + 1

    return count/sample.size

  def build_geojson_from_heatmap(self, df:pd.DataFrame, observatories = [],
                                 targetGeoLocations = [], filter_threshold = 0.5) -> Tuple[str, str]:
    """ Generates a geojson feature collection from a pandas dataframe. It can
    be used to plot the heatap, markers and target locations on a map.

    Parameters
      df : pd.DataFrame
        The error heatmap dataframe.
      observatory : list
        The observation point to be added to the geojson feature collection. Format: [lat, lon, error], ...]
      targetGeoLocations : list
        The list of target locations (horizon points) used by the `self.compute_error_heatmap`
      filter_threshold : float
        The threshold for filtering the `df`. Default: 0.5

    Returns
      Tuple[str, str]
        Contains 2 GeoJSON strings, one for the heatmap and another for the targets (incl. observation points and horizon points)

    Raises
      ValueError: If `df` is None
    """

    if pd is None:
      raise ValueError("The error heatmap dataframe cannot be None.")

    geojson_feature_list = []
    for index, row in df.iterrows():
        geojson_feature_list.append(Feature(geometry=Point((row['lon'], row['lat'])), properties={'value': row['err_val']})) # Changed 'val' to 'err_val'
    geojson_feature_collection = FeatureCollection(geojson_feature_list)

    json_heatmap = geojson.dumps(geojson_feature_collection)

    geojson_feature_list_targets = []
    for marker in observatories:
        geojson_feature_list_targets.append(Feature(geometry=Point((marker[1], marker[0])), properties={'value': marker[2]}))

    for target in targetGeoLocations:
        geojson_feature_list_targets.append(Feature(geometry=Point((target[1], target[0])), properties={'value': 'n/a'}))

    geojson_feature_collection_targets = FeatureCollection(geojson_feature_list_targets)

    json_targets = geojson.dumps(geojson_feature_collection_targets)

    return json_heatmap, json_targets

  def plot_error_heatmap_histogram(self, df:pd.DataFrame):
    """ Plots a histogram of the error heatmap generated by `self.compute_error_heatmap`.

    Parameters
      df : pd.DataFrame
        The error heatmap dataframe.

    Raises
      ValueError: If the argument is None
    """

    if df is None:
      raise ValueError("The argument cannot be None.")

    # Plotting a basic histogram
    sns.histplot(df['err_val'], bins=100, kde=True, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Error values')
    plt.ylabel('Frequency')
    plt.title('Error values distribution')

    # Display the plot
    plt.show()

  def show_heatmap(self, df:pd.DataFrame, observatory = [], targetGeoLocations = [],
                   center = [], zoom=11, filter_threshold=0.5):
    """ Shows the error heatmap generated by `self.compute_error_heatmap`. The
    json and json_targets are generated by `self.build_geojson_from_heatmap`.

    Parameters
      df : pd.DataFrame
        The error heatmap dataframe generated by `self.compute_error_heatmap`
      observatory : list
        The observation point to be added to the geojson feature collection. Format: [[lat, lon, error], ...]
      targetGeoLocations : list
        The list of target locations used by the `self.compute_error_heatmap`
      center : list
        The center of the map. Format: [lat, lon] in degrees
      zoom : int
        The zoom level of the map. Default: 11
      filter_threshold: int
        The threshold for filtering the error heatmap before generating the
        geojsons. Default: 0.5

    Raises
      ValueError: If `json` or `json_targets` is None
    """

    if df is None or observatory is None or targetGeoLocations is None:
        raise ValueError("`df`, `markers` and `targetGeoLocation` cannot be None.")

    
    #json, json_targets = self.build_geojson_from_heatmap(df, observatories, targetGeoLocations, filter_threshold)

    m = leafmap.Map(center=(center[0], center[1]), zoom=zoom)
    m.add_basemap("OpenTopoMap")
    m.add_heatmap(
        df[df['err_val']<filter_threshold], # Changed 'val' to 'err_val'
        latitude="lat",
        longitude="lon",
        value="err_val", # Changed 'val' to 'err_val'
        name="Heat map",
        radius=25,
    )
    
    
    m.add_marker(observatory[:2], 
                 icon_shape='circle-dot',
                 popup = "Observatory error: " + str(observatory[2]))
    
    for geoLoc in targetGeoLocations:
        m.add_marker([geoLoc[0], geoLoc[1]], 
                     icon_shape='rectangle-dot',
                     popup = "Target elevation: " + str(geoLoc[2]))

    time = datetime.datetime.now()
    out_html = f"heatmap_{str(time)[:19]}.html"
    out_html = out_html.replace(":", "_")
    m.to_html(out_html)
    
    out_html = os.path.abspath(out_html)
    webbrowser.open_new_tab(f"file://{out_html}")
    
  def save_heatmap(self, df:pd.DataFrame):
    """ Saves the error heatmap generated by `self.compute_error_heatmap` to a csv file.
    For the moment markers and targets are not saved. Only error values are saved.

    Parameters
      df : pd.DataFrame

    Raises
      ValueError: If `df` is None
    """

    if df is None:
      raise ValueError("The argument cannot be None.")

    time = datetime.datetime.now()
    out_csv = f"heatmap_errors_{str(time)[:19]}.csv"
    out_csv = out_csv.replace(":", "_")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    
