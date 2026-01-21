# -*- coding: utf-8 -*-
import numpy as np
import math

class Altitude:
  """ Class containing altitude related utility functions.
    
  Requirements: numpy.
  """
  
  def err_alt(self, 
              alt:float,
              err_alt:float=0.5) -> float:
    """ Calculates the altitude error based on the systematic error `err_alt` and 
    the error from the atmospheric refraction.
           
    Parameters   
    alt:float
        The object's altitude in degrees
    err_alt:float
        The error in the altitude reading. Can be the error of the instrument, 
        the reading error, or both. Default=0.5
    Returns
        The declination error in degrees
        
    Raises
        ValueError if any of the arguments is None or outside their value range
    """    

    cociente = 7.31 / (alt + 4.4)
    err1 = (1 - cociente / (alt + 4.4)) / (math.degrees(np.cos(np.radians(cociente + alt))))**2
    err2 = (math.degrees(np.tan(np.radians(cociente + alt))))**2
    der_alt = -0.0167 * err1 / err2
    #err = 0.25*np.sqrt((der_alt)**2+1.0)

    return err_alt * ((np.abs(der_alt)) + 1)

  def update_alt_by_atmospheric_refraction(self, 
                                           alt:float) -> float:
    """ Updates the altitude by taking into account the atmospheric refraction.

    Parameters
      alt : float
        The altitude without atmospheric refraction
        
    Returns
      The updated altitude

    Raises
      ValueError: If the argument is None
    """ 
    
    if alt is None:
      raise ValueError("Argument is None.")  
      
    rr = 0.0167 / np.tan(np.radians(alt + (7.31 / (alt + 4.4))))
    
    return alt - rr