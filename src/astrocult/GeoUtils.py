# -*- coding: utf-8 -*-
#pip install geographiclib
from geographiclib.geodesic import Geodesic
import math
import numpy as np
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
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
import socket
import xml.etree.ElementTree as ET


import HelperFunctions

class GeoUtils:
  """ Contains a set of methods for dealing with astronomical computations
  based on geographical coordinates.

  Overview
  --------
  - `get_magnetic_declination`: magnetic declination from the NOAA
    calculator (models WMM and WMMHR for 2024-2029, IGRF for 1590-2029),
    to convert compass azimuths into true azimuths.
  - `get_bearing`, `distance_between`: azimuth (WGS84 geodesic) and
    distance between two points.
  - `horizon_height_degrees`, `horizon_height_meters`: altitude of a
    horizon point and its inverse, including the Earth's curvature and,
    optionally, terrestrial refraction.
  - `compute_error_heatmap`: for a grid of candidate observer positions,
    the error between the declinations towards one or more targets and
    given astronomical declinations, combined as RSS, RMS or maximum.
    Optionally marks positions from which a target is hidden by the
    Earth's curvature or too far away. Computed for the whole grid at once
    with NumPy (WGS84 bearings, Vincenty's formula).
  - `compute_location_probability`, `compute_location_area`: share and
    area (km²) of the map below an error threshold.
  - `plot_error_map`: static map of the heatmap (matplotlib only);
    `show_heatmap`: interactive web map with a topographic basemap.
  - `plot_error_heatmap_histogram`, `save_heatmap`,
    `build_geojson_from_heatmap`: analyze and export a heatmap.

  Conventions: latitudes and longitudes in degrees (north and east
  positive), distances in meters along the Earth's surface, target
  heights in meters relative to the observer's eye level.
  """
  
  # Year ranges accepted by the NOAA declination calculator for each model
  # (https://www.ngdc.noaa.gov/geomag/CalcSurveyFin.shtml). The WMM model
  # in use since 2025 is WMM-2025; update these when NOAA releases a new model.
  MAGNETIC_MODEL_YEARS = {"WMM": (2024, 2029), "WMMHR": (2024, 2029), "IGRF": (1590, 2029)}
  NOAA_DECLINATION_URL = "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination"

  def get_magnetic_declination(self, 
                           lat: float,
                           lon: float,
                           year: int | None = None,
                           month: int | None = None,
                           day: int | None = None,
                           model: str = "WMM",
                           timeout: float = 15.0,
                           api_key: str = "zNEw7") -> tuple[float, float]:
      """      
      Uses the NOAA magnetic declination calculator to retrieve the magnetic
      declination for a given location and date. The returned magnetic
      declination can be added to the magnetic azimuth to obtain the true
      azimuth, which can then be used for plotting or for computing the
      astronomical declination.

      Models and valid years (as documented by NOAA):
        - 'WMM'   World Magnetic Model (currently WMM-2025): 2024-2029
        - 'WMMHR' WMM High Resolution: 2024-2029
        - 'IGRF'  International Geomagnetic Reference Field: 1590-2029
                  (before 1900 NOAA uses the gufm1 historical model). Useful
                  for compass readings taken in past surveys.
      Typical accuracy is about 0.5°; local magnetic anomalies can be larger.

      Usage limits: NOAA allows at most 50 requests per second across all
      users. Make bulk requests one after another, not in parallel; if
      requests start failing, wait 5-10 minutes before retrying.
      
      Parameters   
      lat : float
          The latitude of the location
      lon : float
          The longitude of the location
      year, month, day : int | None
          The date of the compass readings. Default: today (each value that
          is None is taken from today's date)
      model : str
          'WMM' (default), 'WMMHR' or 'IGRF' (case-insensitive)
      timeout : float
          Seconds to wait for NOAA's response. Default: 15
      api_key : str
          NOAA API key. Default: the public key published by NOAA ('zNEw7');
          NOAA announces key changes to registered users.
          
      Returns
      tuple 
          The magnetic declination in degrees (positive east)
          The magnetic declination uncertainty in degrees (NaN if NOAA does
          not provide one for the chosen model)
          
      Raises
          ValueError if an argument is None, out of range, or not a valid date
          RuntimeError if NOAA cannot be reached, returns an error, or sends
          a response without a declination value
      """  
      
      if None in [lat, lon]:
          raise ValueError("`lat` and `lon` cannot be None.") 
      if not (-90 <= lat <= 90):
          raise ValueError("The latitude should be between -90 and 90.")   
      if not (-180 <= lon <= 180):
          raise ValueError("The longitude should be between -180 and 180.") 

      model = str(model).upper()
      if model not in self.MAGNETIC_MODEL_YEARS:
          raise ValueError(f"`model` should be one of {list(self.MAGNETIC_MODEL_YEARS)}.")

      today = datetime.date.today()
      year = today.year if year is None else year
      month = today.month if month is None else month
      day = today.day if day is None else day
      y_min, y_max = self.MAGNETIC_MODEL_YEARS[model]
      if not (y_min <= year <= y_max):
          raise ValueError(f"For model {model} the year should be between {y_min} and {y_max}.")
      try:
          datetime.date(year, month, day)
      except (TypeError, ValueError) as e:
          raise ValueError(f"Invalid date {year}-{month}-{day}: {e}") from None

      query = urlencode({"lat1": lat, "lon1": lon, "model": model,
                         "startYear": year, "startMonth": month, "startDay": day,
                         "key": api_key, "resultFormat": "xml"})
      url = f"{self.NOAA_DECLINATION_URL}?{query}"

      try:
          with urlopen(url, timeout=timeout) as f:
              body = f.read().decode("utf-8", errors="replace")
      except HTTPError as e:
          raise RuntimeError(f"NOAA returned HTTP {e.code} ({e.reason}). If you are sending "
                             "many requests, wait 5-10 minutes before retrying.") from e
      except (URLError, socket.timeout, TimeoutError) as e:
          raise RuntimeError(f"Could not reach the NOAA declination service within {timeout} s "
                             f"({getattr(e, 'reason', e)}). Check your internet connection.") from e

      try:
          root = ET.fromstring(body)
          decl_node = root.find(".//declination")
          unc_node = root.find(".//declination_uncertainty")
          if decl_node is None or decl_node.text is None:
              raise ValueError("no <declination> element")
          md = float(decl_node.text.strip())
          md_unc = float(unc_node.text.strip()) if (unc_node is not None and unc_node.text) else float("nan")
      except (ET.ParseError, ValueError) as e:
          snippet = " ".join(body.split())[:200]
          raise RuntimeError(f"Unexpected response from the NOAA declination service ({e}). "
                             f"Response starts with: {snippet!r}") from None

      return md, md_unc
          
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

  EARTH_RADIUS_M = 6371000.0   # mean Earth radius, as in `distance_between`

  @staticmethod
  def _horizon_altitude_deg(dist, height, curvature: bool = True,
                            refraction_k: float = 0.0) -> np.ndarray:
    """ Vectorized altitude (degrees) of a point `height` meters above the
    observer's eye at surface distance `dist` meters. See
    `horizon_height_degrees` for the meaning of the options. """

    dist = np.asarray(dist, dtype=float)
    height = np.asarray(height, dtype=float)
    if not curvature:
      return np.degrees(np.arctan2(height, dist))
    R = GeoUtils.EARTH_RADIUS_M / (1.0 - refraction_k)   # effective radius
    theta = dist / R                                      # central angle
    r = R + height
    return np.degrees(np.arctan2(r * np.cos(theta) - R, r * np.sin(theta)))

  def horizon_height_degrees(self, dist:float, heightM:float, curvature:bool=True,
                             refraction_k:float=0.0) -> float:
    """ Calculates the altitude (angular height) in degrees of a horizon
    point, e.g. a mountain top, seen by the observer.

    With `curvature=True` (default) the Earth's curvature is taken into
    account: a distant point appears lower than atan(height / dist)
    because the surface drops away by about dist^2 / (2R), e.g. ~8 m at
    10 km and ~200 m at 50 km (0.04° and 0.22° in altitude). The exact
    spherical geometry is used, not this approximation.

    Parameters
      dist : float
        Distance between the observer and the horizon point along the
        Earth's surface, in meters (e.g. from `distance_between`)
      heightM : float
        Height of the horizon point above the observer's eye level, in
        meters (negative if it is lower)
      curvature : bool
        Account for the Earth's curvature. False gives the flat-Earth
        value atan(heightM / dist) of earlier versions. Default: True
      refraction_k : float
        Terrestrial refraction coefficient (0 <= k < 1). Light from the
        horizon point bends with the atmosphere, which raises it slightly;
        this is modelled with an effective Earth radius R / (1 - k). 0
        means pure geometry; ~0.13 is a typical daytime value. This is
        the refraction of the terrain point itself, not the astronomical
        refraction of the celestial body (see
        `Altitude.update_alt_by_atmospheric_refraction`). Default: 0.0

    Returns
      The altitude of the horizon point in degrees

    Raises
      ValueError: If an argument is None, `dist` is not positive, or
      `refraction_k` is outside [0, 1)
    """

    if None in [dist, heightM]:
      raise ValueError("At least one argument is None")
    if dist <= 0:
      raise ValueError("`dist` should be positive.")
    if not (0 <= refraction_k < 1):
      raise ValueError("`refraction_k` should be within [0, 1).")

    return float(self._horizon_altitude_deg(dist, heightM, curvature, refraction_k))

  def horizon_height_meters(self, dist:float, heightD:float, curvature:bool=True,
                            refraction_k:float=0.0) -> float:
    """ Calculates the height in meters, relative to the observer's eye
    level, of a horizon point seen at altitude `heightD` degrees at
    surface distance `dist`. Inverse of `horizon_height_degrees`, with the
    same options.

    Parameters
      dist : float
        Distance between the observer and the horizon point along the
        Earth's surface, in meters
      heightD : float
        Altitude of the horizon point in degrees
      curvature : bool
        Account for the Earth's curvature (see `horizon_height_degrees`).
        False gives the flat-Earth value tan(heightD) * dist. Default: True
      refraction_k : float
        Terrestrial refraction coefficient (see `horizon_height_degrees`).
        Default: 0.0

    Returns
      The height of the horizon point above the observer's eye, in meters

    Raises
      ValueError: If an argument is None or out of range, or if no point at
      that distance can be seen at that altitude
    """

    if None in [dist, heightD]:
      raise ValueError("At least one argument is None.")
    if heightD > 90 or heightD < -90:
      raise ValueError("`heightD` should be within [-90,90] degrees.")
    if dist < 0:
      raise ValueError("`dist` should not be negative.")
    if not (0 <= refraction_k < 1):
      raise ValueError("`refraction_k` should be within [0, 1).")

    if not curvature:
      return math.tan(math.radians(heightD)) * dist

    R = self.EARTH_RADIUS_M / (1.0 - refraction_k)
    theta = dist / R
    a = math.radians(heightD)
    denom = math.cos(a + theta)
    if denom <= 0:
      raise ValueError("No point at this distance can be seen at this altitude.")
    # law of sines in the triangle Earth centre - observer - horizon point
    return R * math.cos(a) / denom - R

  @staticmethod
  def _inverse_azimuth_wgs84(lat1, lon1, lat2, lon2, tol: float = 1e-12,
                             max_iter: int = 200) -> np.ndarray:
    """ Vectorized initial azimuth (degrees, -180..180] of the geodesic from
    (lat1, lon1) to (lat2, lon2) on the WGS84 ellipsoid, using Vincenty's
    inverse formula. Agrees with geographiclib (`get_bearing`) far below any
    measurement precision. The rare nearly antipodal pairs where Vincenty
    does not converge are computed with geographiclib instead.

    Parameters
      lat1, lon1, lat2, lon2 : float | np.ndarray
        Coordinates in degrees (broadcast against each other)

    Returns
      np.ndarray of azimuths in degrees
    """

    a_f = 1 / 298.257223563
    lat1, lon1, lat2, lon2 = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (lat1, lon1, lat2, lon2)))
    L = np.radians((lon2 - lon1 + 180.0) % 360.0 - 180.0)
    U1 = np.arctan((1 - a_f) * np.tan(np.radians(lat1)))
    U2 = np.arctan((1 - a_f) * np.tan(np.radians(lat2)))
    sinU1, cosU1, sinU2, cosU2 = np.sin(U1), np.cos(U1), np.sin(U2), np.cos(U2)

    lam = L.copy()
    converged = np.zeros(L.shape, dtype=bool)
    for _ in range(max_iter):
      sin_lam, cos_lam = np.sin(lam), np.cos(lam)
      sin_sigma = np.hypot(cosU2 * sin_lam, cosU1 * sinU2 - sinU1 * cosU2 * cos_lam)
      cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lam
      sigma = np.arctan2(sin_sigma, cos_sigma)
      with np.errstate(divide='ignore', invalid='ignore'):
        sin_alpha = np.where(sin_sigma == 0, 0.0, cosU1 * cosU2 * sin_lam / sin_sigma)
        cos2_alpha = 1.0 - sin_alpha ** 2
        cos_2sm = np.where(cos2_alpha == 0, 0.0, cos_sigma - 2 * sinU1 * sinU2 / cos2_alpha)
      C = a_f / 16 * cos2_alpha * (4 + a_f * (4 - 3 * cos2_alpha))
      lam_new = L + (1 - C) * a_f * sin_alpha * (
          sigma + C * sin_sigma * (cos_2sm + C * cos_sigma * (-1 + 2 * cos_2sm ** 2)))
      converged = np.abs(lam_new - lam) < tol
      lam = lam_new
      if converged.all():
        break

    sin_lam, cos_lam = np.sin(lam), np.cos(lam)
    azi = np.degrees(np.arctan2(cosU2 * sin_lam, cosU1 * sinU2 - sinU1 * cosU2 * cos_lam))

    if not converged.all():
      azi = np.array(azi, dtype=float)   # writable copy
      for idx in zip(*np.nonzero(~converged)):
        azi[idx] = Geodesic.WGS84.Inverse(float(lat1[idx]), float(lon1[idx]),
                                          float(lat2[idx]), float(lon2[idx]))['azi1']
    return azi

  @staticmethod
  def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """ Vectorized haversine distance in meters (same formula and Earth
    radius, 6371 km, as `distance_between`). """

    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 1000.0 * 2 * np.arctan2(np.sqrt(h), np.sqrt(1 - h))

  HEATMAP_COMBINE = ("rss", "rms", "max")

  def compute_error_heatmap(self, targetGeoLocations:list, targetAstroBodies:list,
                            coordsRange:list, curvature:bool=True,
                            refraction_k:float=0.0, combine:str="rss",
                            observer_height_m:float|None=None,
                            max_distance_km:float|None=None) -> pd.DataFrame:
    """ Computes the error heatmap for a given set of target locations and
    astronomical bodies. The heatmap will be computed within a defined
    geographical range.

    For every candidate observer position (lat, lon) of the grid and every
    target i, the azimuth (WGS84 geodesic) and horizon altitude towards the
    target are converted into a declination for the observer's latitude.
    The errors e_i = dec_i - body_dec_i of all targets are combined into
    one value per cell (see `combine`), rounded to 0.1°. The whole grid is
    computed at once with NumPy. Any number of targets (1 or more) can be
    used; each extra target narrows the area where all alignments work.

    Distance: the altitude of a far target is lowered by the Earth's
    curvature (`curvature`). Optionally, cells from which a target cannot
    be used at all are marked unusable (err_val = inf): with
    `observer_height_m`, targets hidden behind the Earth's curvature; with
    `max_distance_km`, targets farther than a chosen limit. A target is
    hidden when its distance exceeds sqrt(2 R h_obs) + sqrt(2 R h_target),
    the sum of the observer's and the target's horizon distances, where
    h_obs = `observer_height_m` and h_target = heightMeters +
    `observer_height_m` are heights above the terrain in between, and R is
    the (refraction-corrected) Earth radius. This assumes that the terrain
    between observer and target is a smooth surface at the observer's
    ground level: an observer on a hill above a valley sees farther, and a
    ridge in between can hide a target this check considers visible.

    Parameters
      targetGeoLocations : list
        Format: [[lat, lon, heightMeters], [], ...]. heightMeters is the
        height of the target above the observer's eye level. The grid
        cells have no elevation of their own, so the same height
        difference is used for every cell.
      targetAstroBodies : list
        Format: [['sun', dec], [], ...]. Must have the same length as
        `targetGeoLocations`: body i is matched with target i.
      coordsRange : list
        Format [latRange, lonRange], each a sequence of values in degrees
      curvature : bool
        Account for the Earth's curvature in the altitude of each target
        (see `horizon_height_degrees`). Default: True
      refraction_k : float
        Terrestrial refraction coefficient (see `horizon_height_degrees`);
        also used by the visibility check. Default: 0.0
      combine : str
        How the errors of the targets are combined:
        'rss' sqrt(sum_i e_i^2), the original method; it grows with the
        number of targets (n targets each off by e give e*sqrt(n)), so a
        fixed threshold is stricter for more targets.
        'rms' sqrt(mean_i e_i^2), the typical error per target; a threshold
        means "each alignment within about this", whatever the number of
        targets.
        'max' max_i |e_i|, the worst target; a threshold means "every
        alignment within this".
        With one target all three are the same. Default: 'rss'
      observer_height_m : float | None
        Height of the observer's eye above the terrain between observer and
        targets, in meters (e.g. 1.6 for a standing person). If given,
        cells from which a target is hidden by the Earth's curvature are
        marked unusable. Default: None (no visibility check)
      max_distance_km : float | None
        If given, cells farther than this from any target are marked
        unusable, e.g. the distance at which a peak can still be identified
        as a marker. Default: None (no limit)
    Returns
      A pandas dataframe with columns 'lat', 'lon', 'err_val', one row per
      grid cell (latitude-major order). err_val is NaN where a cell
      coincides with a target (the altitude is undefined) and inf where a
      target is hidden or too far; NaN cells are left out of
      `compute_location_probability` and `compute_location_area`, inf
      cells count as unsuitable. The settings are stored in `df.attrs`
      ('combine', 'observer_height_m', 'max_distance_km') for
      `plot_error_map`.

    Raises
      ValueError: If at least one argument is None, there are no targets,
      the target and body lists differ in length, or values are out of
      range
    """

    if None in [targetGeoLocations, targetAstroBodies, coordsRange]:
      raise ValueError("At least one argument is None.")
    if not (0 <= refraction_k < 1):
      raise ValueError("`refraction_k` should be within [0, 1).")
    if len(targetGeoLocations) != len(targetAstroBodies):
      raise ValueError("`targetGeoLocations` and `targetAstroBodies` must have the same length.")
    if len(targetGeoLocations) == 0:
      raise ValueError("At least one target (and astronomical body) is required.")
    combine = str(combine).lower()
    if combine not in self.HEATMAP_COMBINE:
      raise ValueError(f"`combine` must be one of {list(self.HEATMAP_COMBINE)}.")
    if observer_height_m is not None and observer_height_m <= 0:
      raise ValueError("`observer_height_m` must be positive.")
    if max_distance_km is not None and max_distance_km <= 0:
      raise ValueError("`max_distance_km` must be positive.")
    if len(coordsRange) != 2:
      raise ValueError("`coordsRange` must be [latRange, lonRange].")

    lats = np.asarray(coordsRange[0], dtype=float).ravel()
    lons = np.asarray(coordsRange[1], dtype=float).ravel()
    if np.any(np.abs(lats) > 90) or np.any(np.abs(lons) > 180):
      raise ValueError("Grid latitudes should be within [-90,90] and longitudes within [-180,180].")

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_f, lon_f = lat_grid.ravel(), lon_grid.ravel()
    lat_r = np.radians(lat_f)

    R_eff = self.EARTH_RADIUS_M / (1.0 - refraction_k)
    usable = np.ones(lat_f.size, dtype=bool)
    errors = []
    for target, body in zip(targetGeoLocations, targetAstroBodies):
      t_lat, t_lon, t_height = float(target[0]), float(target[1]), float(target[2])
      if not (-90 <= t_lat <= 90) or not (-180 <= t_lon <= 180):
        raise ValueError("Target latitudes should be within [-90,90] and longitudes within [-180,180].")

      az = self._inverse_azimuth_wgs84(lat_f, lon_f, t_lat, t_lon) % 360.0
      dist = self._haversine_m(lat_f, lon_f, t_lat, t_lon)
      alt = np.where(dist > 0,
                     self._horizon_altitude_deg(dist, t_height, curvature, refraction_k),
                     np.nan)

      alt_r, az_r = np.radians(alt), np.radians(az)
      sin_dec = np.sin(alt_r) * np.sin(lat_r) + np.cos(alt_r) * np.cos(lat_r) * np.cos(az_r)
      dec = np.degrees(np.arcsin(np.clip(sin_dec, -1.0, 1.0)))
      errors.append(dec - float(body[1]))

      if max_distance_km is not None:
        usable &= dist <= max_distance_km * 1000.0
      if observer_height_m is not None:
        # hidden behind the Earth's curvature beyond the sum of both horizon distances
        h_target = max(t_height + observer_height_m, 0.0)
        reach = np.sqrt(2 * R_eff * observer_height_m) + np.sqrt(2 * R_eff * h_target)
        usable &= dist <= reach

    errors = np.asarray(errors)                          # (targets, cells)
    if combine == "rss":
      err = np.sqrt(np.sum(errors ** 2, axis=0))
    elif combine == "rms":
      err = np.sqrt(np.mean(errors ** 2, axis=0))
    else:
      err = np.max(np.abs(errors), axis=0)
    err = np.round(err, 1)
    err[~usable & ~np.isnan(err)] = np.inf

    df = pd.DataFrame({"lat": lat_f, "lon": lon_f, "err_val": err})
    df.attrs.update(combine=combine, observer_height_m=observer_height_m,
                    max_distance_km=max_distance_km)
    return df

  def compute_location_probability(self, df:pd.DataFrame, threshold:float=2,
                                   number_of_random_samples:int|None=None,
                                   rng:np.random.Generator|None=None) -> float:
    """Calculates the probability of randomly picking a location whose error
    is below `threshold`, i.e. the fraction of heatmap cells with
    `err_val < threshold`.

    By default the fraction is computed exactly over all cells. If
    `number_of_random_samples` is given (and smaller than the number of
    cells), a Monte Carlo estimate from that many randomly drawn cells is
    returned instead.

    Parameters
      df : pd.DataFrame
        The error heatmap dataframe. Generated by `self.compute_error_heatmap`.
      threshold : float
        The error threshold in degrees. Default: 2
      number_of_random_samples : int | None
        Number of cells to sample for a Monte Carlo estimate. Default: None
        (exact computation over all cells)
      rng : np.random.Generator | None
        Random generator for reproducible sampling. Default: None

    Returns
      The probability of randomly picking a location with an error below the threshold.

    Raises
      ValueError: If `df` is None, has no `err_val` column, contains no valid
      error values, or `number_of_random_samples` is not positive
    """

    if df is None:
      raise ValueError("The error heatmap dataframe cannot be None.")
    if 'err_val' not in df.columns:
      raise ValueError("`df` must contain an 'err_val' column.")

    errs = df['err_val'].to_numpy(dtype=float)
    errs = errs[~np.isnan(errs)]
    if errs.size == 0:
      raise ValueError("`df` contains no valid error values.")

    if number_of_random_samples is None or number_of_random_samples >= errs.size:
      return float(np.mean(errs < threshold))

    if number_of_random_samples <= 0:
      raise ValueError("`number_of_random_samples` must be positive.")

    rng = np.random.default_rng() if rng is None else rng
    sample = rng.choice(errs, size=int(number_of_random_samples), replace=False)
    return float(np.mean(sample < threshold))

  def compute_location_area(self, df:pd.DataFrame, threshold:float=2) -> tuple[float, float]:
    """Calculates the area in square kilometres of the heatmap cells whose
    error is below `threshold`, and the total area of the mapped cells.
    Each cell's area is computed from the grid spacing at its own latitude
    (cells shrink east-west towards the poles), on a sphere of radius
    6371 km. Cells with an undefined error (NaN) are left out of both.

    Parameters
      df : pd.DataFrame
        The error heatmap from `compute_error_heatmap`, on a regular grid
      threshold : float
        The error threshold in degrees. Default: 2

    Returns
      tuple
        (area below the threshold, total mapped area), both in km²

    Raises
      ValueError: If `df` is None, lacks the 'lat', 'lon' and 'err_val'
      columns, or has fewer than 2 distinct latitudes or longitudes
    """

    if df is None or not {"lat", "lon", "err_val"}.issubset(df.columns):
      raise ValueError("`df` must be a heatmap with 'lat', 'lon' and 'err_val' columns.")
    lats = np.unique(df["lat"].to_numpy(float))
    lons = np.unique(df["lon"].to_numpy(float))
    if lats.size < 2 or lons.size < 2:
      raise ValueError("The heatmap needs at least 2 distinct latitudes and longitudes.")

    km_per_deg = np.pi * 6371.0 / 180.0
    dlat_km = float(np.median(np.diff(lats))) * km_per_deg
    dlon_km = float(np.median(np.diff(lons))) * km_per_deg
    cell_km2 = dlat_km * dlon_km * np.cos(np.radians(df["lat"].to_numpy(float)))
    err = df["err_val"].to_numpy(float)
    valid = ~np.isnan(err)
    return float(cell_km2[valid & (err < threshold)].sum()), float(cell_km2[valid].sum())

  def build_geojson_from_heatmap(self, df:pd.DataFrame, observatories:list|None = None,
                                 targetGeoLocations:list|None = None, filter_threshold = 0.5) -> Tuple[str, str]:
    """ Generates a geojson feature collection from a pandas dataframe. It can
    be used to plot the heatap, markers and target locations on a map.

    Parameters
      df : pd.DataFrame
        The error heatmap dataframe.
      observatories : list | None
        The observation points to add to the targets collection.
        Format: [[lat, lon, error], ...]. Default: None (none)
      targetGeoLocations : list
        The list of target locations (horizon points) used by the `self.compute_error_heatmap`
      filter_threshold : float
        Only heatmap cells with `err_val` below this value are exported.
        Default: 0.5

    Returns
      Tuple[str, str]
        Contains 2 GeoJSON strings, one for the heatmap and another for the targets (incl. observation points and horizon points)

    Raises
      ValueError: If `df` is None
    """

    if df is None:
      raise ValueError("The error heatmap dataframe cannot be None.")

    # Keep only cells below the error threshold (as in `show_heatmap`)
    df = df[df['err_val'] < filter_threshold]

    observatories = [] if observatories is None else observatories
    targetGeoLocations = [] if targetGeoLocations is None else targetGeoLocations

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

    # Plotting a basic histogram (NaN and inf cells, i.e. undefined or
    # unusable positions, are left out)
    errs = df['err_val'][np.isfinite(df['err_val'])]
    sns.histplot(errs, bins=100, kde=True, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Error values')
    plt.ylabel('Frequency')
    plt.title('Error values distribution')

    # Display the plot
    plt.show()

  def plot_error_map(self, df:pd.DataFrame, observatory:list|None=None,
                     targetGeoLocations:list|None=None, threshold:float=2.0,
                     max_error:float|None=None, title:str="", cmap:str|None=None,
                     ax:plt.Axes|None=None, show:bool=True,
                     savepath:str|None=None) -> tuple[plt.Figure, plt.Axes]:
    """ Draws the error heatmap of `compute_error_heatmap` as a static map
    with matplotlib (no web map or browser needed).

    Each grid cell is coloured by its combined declination error: light
    blue colours are good candidate observer positions, light gray ones
    poor (colours saturate at `max_error`); cells where a target is hidden
    or too far (err_val = inf, see `compute_error_heatmap`) are drawn in a
    darker gray. An orange contour marks where the error is below
    `threshold`; the legend gives that area in km² and
    as a share of the mapped area (`compute_location_area`). The share
    depends on the extent of the grid, so choose and report the extent
    deliberately. The map is
    drawn with the correct east-west / north-south proportions for its
    latitude, with a kilometre scale bar and a north arrow.

    Parameters
      df : pd.DataFrame
        The error heatmap from `compute_error_heatmap` (columns 'lat',
        'lon', 'err_val'), computed on a regular grid
      observatory : list | None
        [lat, lon] or [lat, lon, error] of the studied site, drawn as a
        star. Default: None
      targetGeoLocations : list | None
        The targets used for the heatmap, [[lat, lon, heightMeters], ...],
        drawn as triangles labelled T1, T2, ... Default: None
      threshold : float
        Error threshold in degrees for the contour and the probability.
        Default: 2.0
      max_error : float | None
        Error at which the colours saturate. Default: None (3 x threshold)
      title : str
        Plot title. Default: '' (no title)
      cmap : str | None
        Matplotlib colormap name. Default: None, a built-in map from dark
        blue (low error, good) to very light gray (high error), so good
        positions stand out and the map prints well
      ax : plt.Axes | None
        Axes to draw into; a new figure is created if None. Default: None
      show : bool
        Call plt.show() if the function created the figure. Default: True
      savepath : str | None
        If given, the figure is also saved there at 300 dpi. Default: None

    Returns
      tuple
        The matplotlib (Figure, Axes)

    Raises
      ValueError: If `df` is None or lacks the 'lat', 'lon' and 'err_val'
      columns, or `threshold` is not positive
    """

    if df is None or not {"lat", "lon", "err_val"}.issubset(df.columns):
      raise ValueError("`df` must be a heatmap with 'lat', 'lon' and 'err_val' columns.")
    if threshold <= 0:
      raise ValueError("`threshold` must be positive.")

    from matplotlib.lines import Line2D
    text_color = "0.15"

    grid = df.pivot_table(index="lat", columns="lon", values="err_val", aggfunc="mean")
    lats, lons = grid.index.to_numpy(float), grid.columns.to_numpy(float)
    Z = np.ma.masked_invalid(grid.to_numpy(float))
    vmax = float(max_error) if max_error is not None else 3.0 * threshold
    mean_lat = float(np.mean(lats))

    created_fig = ax is None
    if created_fig:
      height = 7.0
      width = min(14.0, max(6.0, height * (np.ptp(lons) * np.cos(np.radians(mean_lat))) /
                                     max(np.ptp(lats), 1e-9) + 2.5))
      fig, ax = plt.subplots(figsize=(width, height))
    else:
      fig = ax.figure

    if cmap is None:
      from matplotlib.colors import LinearSegmentedColormap
      cmap = LinearSegmentedColormap.from_list(
          "astrocult_error", ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f4f4f4"])
    mesh = ax.pcolormesh(lons, lats, Z, shading="nearest", cmap=cmap, vmin=0.0, vmax=vmax,
                         rasterized=True)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.04, extend="max")
    cb_label = {"rss": "Combined declination error (°)",
                "rms": "Declination error per target, RMS (°)",
                "max": "Largest declination error of any target (°)"}.get(df.attrs.get("combine", "rss"),
                                                                          "Declination error (°)")
    cb.set_label(cb_label, fontsize=12, color=text_color)
    cb.ax.tick_params(labelsize=10, colors=text_color)
    cb.outline.set_edgecolor("0.6")
    if np.ma.count(Z) and float(Z.min()) < threshold < float(Z.max()):
      ax.contour(lons, lats, Z.filled(np.nan), levels=[threshold], colors="#e76f51",
                 linewidths=1.8, zorder=3)

    area_below, area_total = self.compute_location_area(df, threshold)
    fmt_km2 = lambda a: f"{a:,.2f}" if a < 10 else (f"{a:,.1f}" if a < 1000 else f"{a:,.0f}")
    share = 100 * area_below / area_total if area_total > 0 else 0.0
    handles = [Line2D([0], [0], color="#e76f51", lw=1.8,
                      label=f"Error < {threshold:g}°: {fmt_km2(area_below)} km² "
                            f"({share:.1f}% of {fmt_km2(area_total)} km² mapped)")]

    # cells where a target is hidden by the Earth's curvature or too far (err_val = inf)
    unusable = np.isinf(grid.to_numpy(float))
    if unusable.any():
      from matplotlib.colors import ListedColormap
      from matplotlib.patches import Patch
      ax.pcolormesh(lons, lats, np.ma.masked_where(~unusable, np.ones_like(unusable, dtype=float)),
                    shading="nearest", cmap=ListedColormap(["#c9c9c9"]), vmin=0, vmax=1,
                    rasterized=True, zorder=1.5)
      reasons = []
      if df.attrs.get("observer_height_m") is not None:
        reasons.append(f"hidden by the Earth's curvature (eye height {df.attrs['observer_height_m']:g} m)")
      if df.attrs.get("max_distance_km") is not None:
        reasons.append(f"farther than {df.attrs['max_distance_km']:g} km")
      handles.append(Patch(facecolor="#c9c9c9", edgecolor="0.6",
                           label="Not usable: a target " + (" or ".join(reasons) if reasons else
                                                            "is not visible")))

    if targetGeoLocations:
      for i, t in enumerate(targetGeoLocations, start=1):
        ax.plot(t[1], t[0], marker="^", ms=11, mfc="#2a9d8f", mec="white", mew=1.4, zorder=5,
                ls="none")
        ax.annotate(f"T{i}", (t[1], t[0]), xytext=(7, 5), textcoords="offset points",
                    fontsize=10, fontweight="semibold", color=text_color, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5, alpha=0.9))
      handles.append(Line2D([0], [0], marker="^", ms=10, mfc="#2a9d8f", mec="white", ls="none",
                            label="Targets: " + ", ".join(f"T{i} {t[2]:+.0f} m" for i, t in
                                                           enumerate(targetGeoLocations, start=1))))
    if observatory is not None:
      ax.plot(observatory[1], observatory[0], marker="*", ms=19, mfc="white", mec="black",
              mew=1.2, zorder=6, ls="none")
      obs_label = "Observatory" + (f" (error {observatory[2]:.1f}°)" if len(observatory) > 2 else "")
      handles.append(Line2D([0], [0], marker="*", ms=15, mfc="white", mec="black", ls="none",
                            label=obs_label))

    # true proportions: 1° of longitude is cos(latitude) times shorter than 1° of latitude
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    ax.set_xlabel("Longitude (°E)", fontsize=12, color=text_color)
    ax.set_ylabel("Latitude (°N)", fontsize=12, color=text_color)
    HelperFunctions.format_map_ticks(ax, (lons.min(), lons.max()), (lats.min(), lats.max()))
    ax.tick_params(labelsize=10, colors=text_color)
    for spine in ax.spines.values():
      spine.set_color("0.4")
    if title:
      ax.set_title(title, fontsize=14, fontweight="semibold", color=text_color, loc="left")

    # scale bar (bottom left) and north arrow (top left)
    HelperFunctions.add_map_scale_bar(ax, (lons.min(), lons.max()), (lats.min(), lats.max()))
    HelperFunctions.add_north_arrow(ax)

    ax.legend(handles=handles, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=1, frameon=True, framealpha=0.95, edgecolor="0.85", fancybox=False)
    HelperFunctions.add_version_watermark(ax)

    if created_fig:
      fig.tight_layout()
    if savepath:
      fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show and created_fig:
      plt.show()
    return fig, ax

  def show_heatmap(self, df:pd.DataFrame, observatory:list|None = None,
                   targetGeoLocations:list|None = None, center:list|None = None,
                   zoom=11, filter_threshold=0.5):
    """ Shows the error heatmap generated by `self.compute_error_heatmap` on
    an interactive web map (OpenTopoMap basemap), saved as an HTML file and
    opened in the browser. For a static figure see `plot_error_map`.

    Only cells with an error below `filter_threshold` are shown. Their heat
    weight is `filter_threshold - err_val`, so the best locations (error
    near 0) glow brightest and cells near the threshold fade out. The heat
    layer blurs each cell by a fixed number of screen pixels, so its look
    changes with the zoom level.

    Parameters
      df : pd.DataFrame
        The error heatmap dataframe generated by `self.compute_error_heatmap`
      observatory : list
        The observation point, format [lat, lon, error]
      targetGeoLocations : list
        The list of target locations used by the `self.compute_error_heatmap`
      center : list | None
        The center of the map, format [lat, lon] in degrees. Default: None,
        the centre of the heatmap grid
      zoom : int
        The zoom level of the map. Default: 11
      filter_threshold: float
        Only cells with an error below this value (degrees) are shown.
        Default: 0.5

    Raises
      ValueError: If `df`, `observatory` or `targetGeoLocations` is None
    """

    if df is None or observatory is None or targetGeoLocations is None:
        raise ValueError("`df`, `observatory` and `targetGeoLocations` cannot be None.")

    if not center:
        center = [float(df['lat'].mean()), float(df['lon'].mean())]

    # Heat weight: higher for lower error, so the best locations are brightest
    shown = df[df['err_val'] < filter_threshold].copy()
    shown['weight'] = filter_threshold - shown['err_val']

    
    #json, json_targets = self.build_geojson_from_heatmap(df, observatories, targetGeoLocations, filter_threshold)

    m = leafmap.Map(center=(center[0], center[1]), zoom=zoom)
    m.add_basemap("OpenTopoMap")
    m.add_heatmap(
        shown,
        latitude="lat",
        longitude="lon",
        value="weight",
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
    
