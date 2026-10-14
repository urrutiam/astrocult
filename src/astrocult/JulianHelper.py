# -*- coding: utf-8 -*-
import datetime
import math
import operator

class JulianHelper:
    """
    This class provides helper methods for handling Julian dates. You will find
    here methods that:
    - Convert Julian dates to Gregorian dates and to Julian Day Numbers (JDN).
    - Calculate the difference in days between Gregorian and Julian dates.
    - Estimate sunrise times.

    All calendar arithmetic is done with integer Julian Day Numbers, so it works
    for any year from -4712 onwards (astronomical year numbering: year 0 is
    1 BCE, year -1 is 2 BCE, ...), including Julian-only leap days such as
    1700-02-29 that do not exist in the Gregorian calendar. Gregorian dates
    before 1582-10-15 are proleptic Gregorian, which is the calendar used by
    Skyfield's `Timescale.utc()`.

    The -4712 limit (4713 BCE, the start of the Julian Day count) belongs to
    these calendar functions, not to the ephemerides. Some JPL ephemerides
    reach further back (e.g. de441.bsp to ~13200 BCE), but dates before -4712
    cannot be converted here, so `Declination.calculate_declinations_for_entire_year`
    is also limited to Julian years >= -4712, whatever ephemeris is used.
    """

    MIN_YEAR = -4712   # Julian year of JDN 0
    _JULIAN_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

    # ------------------------------------------------------------------
    # Private calendar primitives
    # ------------------------------------------------------------------

    def __validate_julian_date(self, y: int, m: int, d: int) -> None:
        """Raises ValueError unless (y, m, d) is a valid Julian calendar date."""

        if None in [y, m, d]:
            raise ValueError("All arguments should be integers.")
        if y < self.MIN_YEAR:
            raise ValueError(f"The year should be >= {self.MIN_YEAR}.")
        if not (1 <= m <= 12):
            raise ValueError("Month should be between 1 and 12.")
        dim = 29 if (m == 2 and self.is_julian_leap_year(y)) else self._JULIAN_MONTH_DAYS[m - 1]
        if not (1 <= d <= dim):
            raise ValueError(f"Day should be between 1 and {dim} for Julian {y}-{m:02d}.")

    def __is_gregorian_leap_year(self, y: int) -> bool:
        """True if `y` is a leap year in the (proleptic) Gregorian calendar."""

        return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

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
        """
            
        a = (14 - m) // 12
        y2 = y + 4800 - a
        m2 = m + 12 * a - 3
        
        return d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - 32083

    def __jdn_from_gregorian(self, y: int, m: int, d: int) -> int:
        """Transforms a (proleptic) Gregorian date into a Julian Day Number."""

        a = (14 - m) // 12
        y2 = y + 4800 - a
        m2 = m + 12 * a - 3

        return d + (153 * m2 + 2) // 5 + 365 * y2 + y2 // 4 - y2 // 100 + y2 // 400 - 32045

    def __gregorian_from_jdn(self, jdn: int) -> tuple[int, int, int]:
        """Calculates a (proleptic) Gregorian date from a Julian Day Number (JDN)
        
        Parameters
        ----------
        jdn : int
            The Julian Day Number
        
        Returns
        -------
        a tuple containing 3 values: year, month, day
        """
            
        # based on Fliegel & Van Flandern
        f = jdn + 1401 + (((4 * jdn + 274277) // 146097) * 3) // 4 - 38
        e = 4 * f + 3
        g = (e % 1461) // 4
        h = 5 * g + 2
        gD = (h % 153) // 5 + 1
        gM = ((h // 153 + 2) % 12) + 1
        gY = e // 1461 - 4716 + (12 + 2 - gM) // 12
        
        return gY, gM, gD

    def __julian_from_jdn(self, jdn: int) -> tuple[int, int, int]:
        """Calculates a Julian calendar date from a Julian Day Number (JDN)."""

        c = jdn + 32082
        d = (4 * c + 3) // 1461
        e = c - (1461 * d) // 4
        m = (5 * e + 2) // 153
        day = e - (153 * m + 2) // 5 + 1
        month = m + 3 - 12 * (m // 10)
        year = d - 4800 + m // 10

        return year, month, day

    # ------------------------------------------------------------------
    # Public calendar API
    # ------------------------------------------------------------------

    def is_julian_leap_year(self, y: int) -> bool:
        """Calculates whether or not a Julian year is leap
        
        Parameters
        ----------
        y : int
            The Julian year (astronomical numbering, 0 = 1 BCE)
    
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

    def julian_to_jdn(self, y: int, m: int, d: int) -> int:
        """Converts a Julian calendar date into a Julian Day Number.

        Parameters
        ----------
        y, m, d : int
            The Julian year (astronomical numbering, >= -4712), month and day

        Returns
        -------
        int
            The Julian Day Number

        Raises
        ------
        ValueError
            If the date is not a valid Julian calendar date.
        """

        self.__validate_julian_date(y, m, d)
        return self.__jdn_from_julian(y, m, d)

    def jdn_to_julian(self, jdn: int) -> tuple[int, int, int]:
        """Converts a Julian Day Number (>= 0) into a Julian calendar date (y, m, d)."""

        if jdn is None or jdn < 0:
            raise ValueError("`jdn` should be a non-negative integer.")
        return self.__julian_from_jdn(int(jdn))

    def jdn_to_gregorian(self, jdn: int) -> tuple[int, int, int]:
        """Converts a Julian Day Number (>= 0) into a (proleptic) Gregorian date (y, m, d)."""

        if jdn is None or jdn < 0:
            raise ValueError("`jdn` should be a non-negative integer.")
        return self.__gregorian_from_jdn(int(jdn))

    def julian_to_gregorian(self, y: int, m: int, d: int) -> tuple[int, int, int]:
        """Converts a Julian calendar date into the (proleptic) Gregorian date of
        the same day. Works for any year >= -4712, including Julian-only leap
        days (e.g. Julian 1900-02-29 -> Gregorian 1900-03-13).

        Parameters
        ----------
        y, m, d : int
            The Julian year (astronomical numbering), month and day

        Returns
        -------
        tuple
            The Gregorian (year, month, day)

        Raises
        ------
        ValueError
            If the date is not a valid Julian calendar date.
        """

        return self.__gregorian_from_jdn(self.julian_to_jdn(y, m, d))

    def gregorian_day_of_year(self, y: int, m: int, d: int) -> int:
        """Day of the year (1-366) of a (proleptic) Gregorian date."""

        return self.__jdn_from_gregorian(y, m, d) - self.__jdn_from_gregorian(y, 1, 1) + 1

    def julian_to_gregorian_datetime(self, dt_jul: datetime.datetime) -> datetime.datetime:
        """Calculates a Gregorian date from a Julian date. The time of day is kept.

        Note: `datetime` only supports years 1-9999 and cannot hold Julian-only
        leap days such as 1700-02-29. Use `julian_to_gregorian(y, m, d)` for
        those dates or for years before 1 CE.
        
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
            
        Yg, Mg, Dg = self.julian_to_gregorian(dt_jul.year, dt_jul.month, dt_jul.day)
        
        return datetime.datetime(Yg, Mg, Dg, dt_jul.hour, dt_jul.minute, dt_jul.second)
    
    def julian_gregorian_offset_days(self, y, m: int | None = None,
                                     d: int | None = None) -> int:
        """Calculates the number of days to add to a Julian date to obtain the
        (proleptic) Gregorian date of the same day, e.g. 10 for 1582-10-05,
        13 for 2000-01-01, and negative values before 200 CE.

        On a Julian-only leap day (29 February of 1700, 1800, 1900, ...) the
        new offset already applies: Julian 1900-02-29 = Gregorian 1900-03-13,
        offset 13.

        Can be called as `julian_gregorian_offset_days(y, m, d)` with any
        integers (Python or numpy), or with a single `datetime.date` /
        `datetime.datetime` holding the Julian date.
        
        Parameters
        ----------
        y : int | datetime.date
            The Julian year (astronomical numbering, >= -4712), or a date
            object holding the whole Julian date
        m : int | None
            The Julian month (omit when `y` is a date)
        d : int | None
            The Julian day (omit when `y` is a date)
        
        Returns
        -------
        int
            The difference in days
        
        Raises
        ------
        ValueError
            If an argument is None or not an integer, or the date is not a
            valid Julian calendar date.
        """
        
        if isinstance(y, datetime.date):        # also covers datetime.datetime
            if m is not None or d is not None:
                raise ValueError("Pass either a date object or (y, m, d), not both.")
            y, m, d = y.year, y.month, y.day
        if None in [y, m, d]:
            raise ValueError("All arguments should be integers.")
        try:
            y, m, d = operator.index(y), operator.index(m), operator.index(d)
        except TypeError:
            raise ValueError("Year, month and day should be integers.") from None

        self.__validate_julian_date(y, m, d)
        if m == 2 and d == 29 and not self.__is_gregorian_leap_year(y):
            m, d = 3, 1
        
        return self.__jdn_from_julian(y, m, d) - self.__jdn_from_gregorian(y, m, d)
    
    # ------------------------------------------------------------------
    # Sunrise
    # ------------------------------------------------------------------

    def sunrise_ut_hours_noaa(self, gy: int, gm: int, gd: int, lat: float,
                              lon_east: float) -> float | None:
        """Calculates the sunrise (NOAA almanac algorithm) for the local
        sunrise on the (proleptic) Gregorian date (gy, gm, gd), as hours of UT
        counted from 00:00 UTC of that date. The value can be negative or
        >= 24 when, far from Greenwich, the sunrise falls on the previous or
        next UTC day. Works for any year.
        It assumes a zenith = 90.833° considering the standard refraction +
        solar radius.

        Parameters
        ----------
        gy, gm, gd : int
            The Gregorian year, month and day (local date of the sunrise)
        lat : float
            The latitude of the location (North is positive)
        lon_east : float
            The longitude of the location (East is positive)

        Returns
        -------
        float | None
            The UT hours of sunrise, or None if the sun does not rise or set
            on that day (polar day/night)

        Raises
        ------
        ValueError
           If a None value has been provided in any of the arguments, or if the 
           latitude and longitude values are outside their ranges.
        """

        if None in [gy, gm, gd, lat, lon_east]:
            raise ValueError("One of the arguments is None.")
        if not (-90 <= lat <= 90) or not (-180 <= lon_east <= 180):
            raise ValueError("`lat` should be between [-90,90] and 'lon_east' between [-180,180].")

        ZENITH = 90.833  # refracción estándar + radio solar (borde superior visible)

        N = self.gregorian_day_of_year(gy, gm, gd)
        # NOAA almanac convention: longitude positive East, in hours
        lng_hour = lon_east / 15.0
    
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
    
        # Local mean time of sunrise on the local date, then shift to UT.
        # UT is NOT wrapped into [0, 24): values outside it belong to the
        # previous/next UTC day.
        T = (H + RA - (0.06571 * t_rise) - 6.622) % 24.0
        
        return T - lng_hour

    def estimate_sunrise_noaa(self, greg_date: datetime.datetime, lat: float, 
                              lon_east: float) -> datetime.datetime:
        """Calculates the UTC sunrise (NOAA) for a given Gregorian date, 
        latitude (North is positive) and longitude (East is positive).
        It assumes  a zenith = 90.833° considering the standard refraction + 
        solar radius. The returned datetime carries the correct UTC date, which
        can be the day before or after `greg_date` far from Greenwich.
        For years before 1 CE use `sunrise_ut_hours_noaa`.
      
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
            The estimated sunrise time (UTC), or None during polar day/night
          
        Raises
        ------
        ValueError
           If a None value has been provided in any of the arguments, or if the 
           latitude and longitude values are outside their ranges.
        """
      
        if greg_date is None:
            raise ValueError("One of the arguments is None.") 
            
        ut = self.sunrise_ut_hours_noaa(greg_date.year, greg_date.month,
                                        greg_date.day, lat, lon_east)
        if ut is None:
            return None
    
        midnight = datetime.datetime(greg_date.year, greg_date.month, greg_date.day)
        
        return midnight + datetime.timedelta(seconds=math.floor(ut * 3600.0))
