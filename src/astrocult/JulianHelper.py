# -*- coding: utf-8 -*-
import datetime
import math

from multipledispatch import dispatch

class JulianHelper:
    """
    This class provides helper methods for handling Julian dates. You will find
    here methods that:
    - Convert Julian dates to Gregorian dates.
    - Calculate the difference in days between Gregorian and Julian dates.
    """

    def __jdn_from_julian(self, y: int, m: int, d: int) -> int:
        """Transforms the Julian date into Julian Day Number (JDN), i.e., the 
        number of days since the beginning of the Julian period.
        
        Parameters
        ----------
        y : int
            The Julian year
        m : int
            The Julian month
        d : int
            The Julian day
        
        Returns
        -------
        int
            The JDN

        Raises
        ------
        ValueError
            If a None value has been provided in any of the arguments, or the values are negative.
        """
        
        if None in [y, m, d]:
            raise ValueError("All arguments should be integers.") 
        if m <=0 or d <=0:
            raise ValueError("Month and Day should be positive integers.") 
            
        a = (14 - m) // 12
        y2 = y + 4800 - a
        m2 = m + 12 * a - 3
        
        return d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - 32083

    def __gregorian_from_jdn(self, y: int) -> tuple[int, int, int]:
        """Calculates a Gregorian date from a Julian Day Number (JDN)
        
        Parameters
        ----------
        y : int
            The Julian year
        
        Returns
        -------
        a tuple containing 3 values: year, month, day
        
        Raises
        ------
        ValueError
            If a None value has been provided.
        """

        if y is None:
            raise ValueError("Year  should be an integer.")
            
        # based on Fliegel & Van Flandern
        f = y + 1401 + (((4 * y + 274277) // 146097) * 3) // 4 - 38
        e = 4 * f + 3
        g = (e % 1461) // 4
        h = 5 * g + 2
        gD = (h % 153) // 5 + 1
        gM = ((h // 153 + 2) % 12) + 1
        gY = e // 1461 - 4716 + (12 + 2 - gM) // 12
        
        return gY, gM, gD

    def is_julian_leap_year(self, y: int) -> bool:
        """Calculates whether or not a Julian year is leap
        
        Parameters
        ----------
        y : int
            The Julian year
    
        Returns
        -------
        bool
            true if the year is leap, false otherwise
    
        Raises
        ------
        ValueError
            If a None value has been provided.
        """
        
        if y is None:
            raise ValueError("All arguments should be integers.") 
        
        return (y % 4) == 0

    def julian_to_gregorian_datetime(self, dt_jul: datetime.datetime) -> datetime.datetime:
        """Calculates a Gregorian date from a Julian date
        
        Parameters
        ----------
        dt_jul : datetime.datetime
            The Julian date
        
        Returns
        -------
        datetime.datetime
            The Gregorian date
        
        Raises
        ------
        ValueError
            If a None value has been provided.
        """
        
        if dt_jul is None:
            raise ValueError("The date should not be None.")
            
        J = self.__jdn_from_julian(dt_jul.year, dt_jul.month, dt_jul.day)
        Yg, Mg, Dg = self.__gregorian_from_jdn(J)
        
        return datetime.datetime(Yg, Mg, Dg, dt_jul.hour, dt_jul.minute, dt_jul.second)
    
    @dispatch(int, int, int)
    def julian_gregorian_offset_days(self, y: int, m: int, d: int) -> int:
        """Calculates the differences between the provided Julian date and the 
        corresponding Gregorian date.
        
        Parameters
        ----------
        y : int
            The Julian year
        m : int
            The Julian month
        d : int
            The Julian day
        
        Returns
        -------
        int
            The difference in days
        
        Raises
        ------
        ValueError
            If a None value has been provided in any of the arguments, or the 
            values are negative, or the Julian year is before 1582 CE.
        """
        
        if None in [y, m, d]:
            raise ValueError("All arguments should be integers.") 
        if m <=0 or d <=0:
            raise ValueError("Month and Day should be positive integers.") 
        if y < 1582:
            raise ValueError("Cannot compute a Gregorian day before the 1582 reform.")
        
        Yg, Mg, Dg = self.__gregorian_from_jdn(self.__jdn_from_julian(y, m, d))
        djul = datetime.date(y, m, d)
        dgre = datetime.date(Yg, Mg, Dg)
        
        return (dgre - djul).days
    
    @dispatch(datetime.datetime)
    def julian_gregorian_offset_days(self, julian_date: datetime.datetime) -> int:
        """Calculates the differences between the provided Julian date and the 
        corresponding Gregorian date.
        
        Parameters
        ----------
        julian_date: datetime.datetime
            The Julian year
        
        Returns
        -------
        int
            The difference in days
        
        Raises
        ------
        ValueError
            If a None value has been provided the argument, or the Julian year
            is before 1582 CE.
        """
        
        return self.julian_gregorian_offset_days(julian_date.year,
                                                 julian_date.month,
                                                 julian_date.day)
    
    def estimate_sunrise_noaa(self, greg_date: datetime.datetime, lat: float, 
                              lon_east: float) -> datetime.datetime:
        """Calculates the UTC sunrise (NOAA) for a given Gregorian date, 
        latitude (North is positive) and longitude (East is positive).
        It assumes  a zenith = 90.833° considering the standard refraction + 
        solar radius
      
        Parameters
        ----------
        greg_date : datetime.datetime
            The Gregorian date
        lat : float
            The latitude of the location
        lon_east : float
            The longitude of the location
          
        Returns
        -------
        datetime.datetime
            The estimated sunrise time
          
        Raises
        ------
        ValueError
           If a None value has been provided in any of the arguments, or if the 
           latitude and longitude values are outside their ranges.
        """
      
        if None in [greg_date, lat, lon_east]:
            raise ValueError("One of the arguments is None.") 
        
        if not (-90 <= lat <= 90) or not (-180 <= lon_east <= 180):
            raise ValueError("`lat```should be between [-90,90] and 'lon_east' between [-180,180].")
            
        ZENITH = 90.833  # refracción estándar + radio solar (borde superior visible)

        N = greg_date.timetuple().tm_yday
        lon_w = -lon_east
        lng_hour = lon_w / 15.0
    
        t_rise = N + ((6.0 - lng_hour) / 24.0)
    
        M = (0.9856 * t_rise) - 3.289
        L = M + (1.916 * math.sin(math.radians(M))) + (0.020 * math.sin(math.radians(2*M))) + 282.634
        L %= 360.0
    
        RA = math.degrees(math.atan(0.91764 * math.tan(math.radians(L))))
        RA %= 360.0
        Lq = (math.floor(L/90.0)) * 90.0
        RAq = (math.floor(RA/90.0)) * 90.0
        RA = (RA + (Lq - RAq)) / 15.0  # h
    
        sin_dec = 0.39782 * math.sin(math.radians(L))
        cos_dec = math.cos(math.asin(sin_dec))
    
        cosH = (math.cos(math.radians(ZENITH)) - (sin_dec * math.sin(math.radians(lat)))) / (cos_dec * math.cos(math.radians(lat)))
        if cosH > 1.0 or cosH < -1.0:
            return None
    
        H = 360.0 - math.degrees(math.acos(cosH))
        H /= 15.0  # h
    
        T = H + RA - (0.06571 * t_rise) - 6.622
        UT = (T - lng_hour) % 24.0
    
        hour   = int(UT)
        minute = int((UT - hour) * 60.0)
        second = int((((UT - hour) * 60.0) - minute) * 60.0)
    
        return datetime.datetime(greg_date.year, greg_date.month, greg_date.day, hour, minute, second)