# -*- coding: utf-8 -*-
import numpy as np
import math

class Altitude:
  """ Class containing altitude related utility functions.
    
  Requirements: numpy.

  Atmospheric refraction uses Bennett's formula,
  R = cot(h + 7.31 / (h + 4.4)) arcminutes for an apparent altitude h in
  degrees (about 34.5' at the horizon). `update_alt_by_atmospheric_refraction`
  converts an apparent (observed) altitude into the true geometric
  altitude, and `err_alt` propagates a reading error through this
  correction; the error grows near the horizon (0.5° becomes about 0.61°
  at 0°). This is the astronomical refraction of the celestial body. The
  terrestrial refraction of a horizon point is a separate effect, handled
  by `GeoUtils.horizon_height_degrees(refraction_k=...)`.
  """
  
  def err_alt(self, 
              alt:float,
              err_alt:float=0.5) -> float:
    """ Calculates the altitude error based on the systematic error `err_alt` and 
    the uncertainty propagated through the atmospheric refraction correction
    (Bennett formula, see `update_alt_by_atmospheric_refraction`).
           
    Parameters   
    alt:float
        The object's apparent altitude in degrees
    err_alt:float
        The error in the altitude reading. Can be the error of the instrument, 
        the reading error, or both. Default=0.5
    Returns
        The altitude error in degrees
        
    Raises
        ValueError if any of the arguments is None or `alt` is outside the
        range where the refraction formula is valid
    """    

    if alt is None or err_alt is None:
      raise ValueError("Arguments cannot be None.")
    if not (-2 <= alt <= 90):
      raise ValueError("`alt` should be between -2 and 90 degrees for the refraction model.")

    x = alt + 7.31 / (alt + 4.4)                      # Bennett argument, degrees
    dx_dh = 1.0 - 7.31 / (alt + 4.4) ** 2
    # d/dh [0.0167 * cot(x deg)] = -0.0167 * (pi/180) * csc^2(x) * dx/dh
    der_alt = -0.0167 * math.radians(1.0) * dx_dh / math.sin(math.radians(x)) ** 2

    return err_alt * (abs(der_alt) + 1.0)

  def update_alt_by_atmospheric_refraction(self, 
                                           alt:float) -> float:
    """ Corrects a measured altitude for atmospheric refraction (Bennett's
    formula): refraction lifts objects near the horizon, so the true
    (geometric) altitude is lower than the apparent (observed) one, by about
    0.57° at the horizon.

    Parameters
      alt : float
        The apparent (observed) altitude in degrees, between -2 and 90
        
    Returns
      The true (geometric) altitude in degrees

    Raises
      ValueError: If the argument is None or outside the range where the
      refraction formula is valid
    """ 
    
    if alt is None:
      raise ValueError("Argument is None.")  
    if not (-2 <= alt <= 90):
      raise ValueError("`alt` should be between -2 and 90 degrees for the refraction model.")
      
    rr = 0.0167 / np.tan(np.radians(alt + (7.31 / (alt + 4.4))))
    
    return alt - rr