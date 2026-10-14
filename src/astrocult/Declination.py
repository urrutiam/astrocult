# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy.random import Generator

from scipy import integrate
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D


from skyfield.api import load, wgs84

import JulianHelper as JulianHelper
import HelperFunctions as HelperFunctions


class Declination:
    """ Class containing declination related utility functions.
    
    Requirements: skyfield, numpy, pandas, scipy, matplotlib.

    Overview
    --------
    - `calc_decl`, `err_decl`: declination of an orientation from its
      azimuth, horizon altitude and the site latitude, and its error.
    - `calculate_declinations_for_entire_year`: the Sun's declination at
      sunrise for every day of a Julian year at a site, from a JPL
      ephemeris through skyfield. Works for Julian years >= -4712
      (astronomical numbering: 0 = 1 BCE). The ephemeris file is chosen
      automatically (de421 for 1900-2049, de422 otherwise) or with
      `ephemeris=`, and downloaded on first use.
    - `curvigram_decl_adaptive`, `distribution_decl_adaptive`,
      `plot_decl_adaptive`: adaptive-kernel density ("curvigram") of
      declinations, renormalized against a null model, with the peaks
      above the null-model significance threshold marked.
    - `make_null_sampler`: the null models. 'orientations' (default):
      random azimuths converted at the sites' latitude and horizon, i.e.
      no astronomical intent. 'solar': the Sun's declination on random
      days, to test for preferred dates. 'uniform': only to reproduce
      earlier results (it overstates significance).
    - `solar_declination_approx`: quick approximate solar declination for
      a day of the year.
    - `plot_fmap`: 2-D density map of declinations over time.

    Conventions
    -----------
    - Declinations are linear (-90° to +90°), unlike azimuths, which are
      circular (see `Azimuth`).
    - Binwidths follow bw = 2*sigma; twice the measurement error of each
      point is recommended.
    - Reference lines: equinoxes (declination 0°) dotted, solstices
      (±`obliquity_deg`) solid. The default obliquity, 23.56°, is close to
      the present value; pass the value for the epoch of the sites (about
      23.9° around 2000 BCE).
    - Plots carry an "astrocult v<version>" label above their top-right
      corner, outside the plot area.
    """
    
    
    def err_decl(self, 
                 lat:float, 
                 az:float, 
                 alt:float, 
                 err_az:float=0.5, 
                 err_alt:float=0.5) -> float:
        """ Calculates the declination error based on the error considered when
        calculating the azimuth and horizon altitude. The two contributions
        are added linearly (|d dec/d alt| * err_alt + |d dec/d az| * err_az),
        a conservative, worst-case combination.

        Note the argument order (lat, az, alt), which differs from
        `calc_decl(lat, alt, az)`; passing keyword arguments avoids mix-ups.
               
        Parameters
        lat:float
            The observer's latitude in degrees
        az:float
            The object's azimuth in degrees as observed from `lat` (any value;
            it is wrapped into [0, 360), so bearings from
            `GeoUtils.get_bearing`, which may be negative, can be used directly)
        alt:float
            The object's altitude in degrees
        err_az:float
            The error in the azimuth reading. Can be the error of the instrument, 
            the reading error, or both. Default=0.5 degrees
        err_alt:float
            The error in the altitude reading. Can be the error of the instrument, 
            the reading error, or both. Default=0.5 degrees. See `Altitude.err_alt()` for 
            details on how you can compute the value automatically from `alt`
            
        Returns
            The declination error in degrees
            
        Raises
            ValueError if any of the arguments is None or outside their value range
        """        
        
        if None in [lat, az, alt, err_az, err_alt]:
            raise ValueError("Arguments should not be None.")
        if not (-90 <= lat <= 90):
            raise ValueError("The `lat` should be between -90 and 90.")   
        if not (-90 <= alt <= 90):
            raise ValueError("The `alt` should be between -90 and 90")        
        if not math.isfinite(az):
            raise ValueError("The `az` should be a finite number")
        az = az % 360.0
        
        decl = self.calc_decl(lat, alt, az)
        
        az = np.radians(az)
        alt = np.radians(alt)
        lat = np.radians(lat)
        
        der_alt = (np.sin(lat) * np.cos(alt) - np.cos(lat) * np.cos(az) * np.sin(alt)) / np.sqrt(1 - ((np.sin(np.radians(decl)))**2))
        der_az = -np.cos(lat) * np.sin(az) * np.cos(alt) / np.sqrt(1 - ((np.sin(np.radians(decl)))**2))
        err_decl = np.abs(der_alt)*err_alt + np.abs(der_az) * err_az
        
        return err_decl
    
    def calc_decl(self, lat:float, alt:float, az:float) -> float:
        """ Calculates the declination based on altitude and azimuth of an object.
        The observer's latitude should be specified.

        Note the argument order (lat, alt, az), which differs from
        `err_decl(lat, az, alt)`; passing keyword arguments avoids mix-ups.
        
        Parameters
        lat:float
            The observer's latitude in degrees
        alt:float
            The object's altitude in degrees
        az:float
            The object's azimuth in degrees as observed from `lat` (any value;
            it is wrapped into [0, 360), so bearings from
            `GeoUtils.get_bearing`, which may be negative, can be used directly)
            
        Returns
            The declination in degrees
            
        Raises
            ValueError if any of the arguments is None
        """
        
        if None in [lat, az, alt]:
            raise ValueError("Arguments should not be None.")
        if not (-90 <= lat <= 90):
            raise ValueError("The `lat` should be between -90 and 90.")   
        if not (-90 <= alt <= 90):
            raise ValueError("The `alt` should be between -90 and 90")        
        if not math.isfinite(az):
            raise ValueError("The `az` should be a finite number")
        az = az % 360.0
            
        sin_dec = (math.sin(math.radians(alt)) * math.sin(math.radians(lat))
                   + math.cos(math.radians(alt)) * math.cos(math.radians(lat))
                   * math.cos(math.radians(az)))
        return math.degrees(math.asin(max(-1.0, min(1.0, sin_dec))))   # guard against rounding
    
    def calculate_declinations_for_entire_year(self, julian_year : int, 
                                               latitude : float, 
                                               longitude : float,
                                               elevation : float, write_to_file = True,
                                               ephemeris : str | None = None,
                                               verbose : bool = True) -> pd.DataFrame:
        """Calculates the solar declinations for a given year and geographical position.
        
        The Sun's position is computed at the estimated sunrise of every day of
        the Julian year. For very old dates the uncertainty of Earth's rotation
        (Delta T, up to a few hours around 2000 BCE) shifts the sunrise instant
        slightly; the resulting declination error stays below about 0.05 deg.
        
        Parameters
        ----------
        julian_year : int
            The Julian year (astronomical numbering: 0 = 1 BCE, -1 = 2 BCE, ...).
            Must be >= -4712, the limit of the calendar functions in
            `JulianHelper` (see below)
        latitude : float
            The latitude in degrees
        longitude : float
            The longitude in degrees
        elevation : float
            The elevation in meters
        write_to_file : bool, optional
            write results to a file in the current folder (default=True)
        verbose : bool, optional
            print progress messages (default=True)
        ephemeris : str | None, optional
            JPL ephemeris file loaded with skyfield's `load` (downloaded on first
            use). Default: 'de421.bsp' (17 MB) for Julian years 1900-2049,
            otherwise 'de422.bsp' (623 MB, ~3000 BCE - 3000 CE). For earlier
            years pass e.g. 'de441.bsp' (~3 GB).
            Note: the usable range is the overlap of the ephemeris coverage
            and the calendar functions in `JulianHelper`, which start at
            Julian year -4712 (4713 BCE). So even with de441.bsp (which
            reaches ~13200 BCE), years before -4712 are rejected.
            Download: skyfield downloads the file automatically on first use
            and saves it in the current working directory; later runs reuse
            it. To download manually (e.g. for offline machines), get the
            SPICE `.bsp` file from JPL:
            https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/
            (mirror: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/)
            and place it, with its original file name, in the working
            directory. Only the `.bsp` files work with skyfield; the files in
            JPL's `Linux/`, `SunOS/` and `ascii/` folders use other formats.
        
        Returns
        -------
        pd.DataFrame
            A dataframe containing the information for the entire year. 

        Raises
        ------
        ValueError
            If a None value has been provided in any of the arguments,
            the values are out of range, or the year is outside the coverage
            of the ephemeris
        """
        
        if None in [julian_year, latitude, longitude, elevation]:
            raise ValueError("No None accepted in any argument.") 
        if not (-90 <= latitude <= 90):
            raise ValueError("The latitude should be between -90 and 90.")   
        if not (-180 <= longitude <= 180):
            raise ValueError("The longitude should be between -180 and 180.")  
        
        jh = JulianHelper.JulianHelper()
        
        days_in_year = 366 if jh.is_julian_leap_year(julian_year) else 365
        jdn_start = jh.julian_to_jdn(julian_year, 1, 1)

        if ephemeris is None:
            ephemeris = 'de421.bsp' if 1900 <= julian_year <= 2049 else 'de422.bsp'
        eph = load(ephemeris)

        # Check coverage (with one day of margin for sunrises on the
        # previous/next UTC day) before running the loop
        cov_start, cov_end = self._ephemeris_coverage(eph)
        if jdn_start - 1 < cov_start or jdn_start + days_in_year + 1 > cov_end:
            raise ValueError(
                f"Julian year {julian_year} is outside the coverage of {ephemeris} "
                f"(JD {cov_start:.1f} - {cov_end:.1f}). Pass a wider ephemeris, "
                f"e.g. ephemeris='de441.bsp'.")
        if verbose:
            print(f"{ephemeris} ephemeris loaded.")

        ts  = load.timescale()
        sun = eph['sun']
        earth = eph['earth']
        observer = earth + wgs84.latlon(latitude, longitude, 
                                        elevation_m = elevation)
        
        def fmt_date(y: int, m: int, d: int) -> str:
            # ISO-like date; astronomical year numbering (0 = 1 BCE, -1 = 2 BCE)
            return f"{y:04d}-{m:02d}-{d:02d}" if y >= 0 else f"-{-y:04d}-{m:02d}-{d:02d}"

        rows = []
        if verbose:
            print(f"Starting computations for Julian year {julian_year} ...")

        for k in range(days_in_year):
            # Calendar arithmetic on Julian Day Numbers: handles Julian-only
            # leap days (e.g. 1700-02-29) and years before 1 CE.
            jdn = jdn_start + k
            jy, jm, jd = jh.jdn_to_julian(jdn)
            gy, gm, gd = jh.jdn_to_gregorian(jdn)

            sunrise_ut = jh.sunrise_ut_hours_noaa(gy, gm, gd, latitude, longitude)

            if sunrise_ut is None:
                dec_date = np.nan
                dec_j2000 = np.nan
                amanecer_fecha = ""
                amanecer_txt = ""
            else:
                # Split into UTC date (proleptic Gregorian, as used by Skyfield)
                # and time of day; far from Greenwich the sunrise can fall on
                # the previous or next UTC day.
                day_shift = math.floor(sunrise_ut / 24.0)
                uy, um, ud = jh.jdn_to_gregorian(jdn + day_shift)
                seconds = math.floor((sunrise_ut - 24.0 * day_shift) * 3600.0)
                hh, rem = divmod(seconds, 3600)
                mm, ss = divmod(rem, 60)

                # Tiempo Skyfield
                t = ts.utc(uy, um, ud, hh, mm, ss)

                # Vector aparente (incluye aberración, efecto luz, etc.)
                app = observer.at(t).observe(sun).apparent()

                # 1) COORDENADAS EN MARCO DE LA FECHA (true equator/equinox of date)
                ra_date_ang, dec_date_ang, _ = app.radec(epoch='date')
                dec_date   = dec_date_ang.degrees   # DEC en grados

                # 2) COORDENADAS EN J2000 (por defecto)
                ra_j2k_ang, dec_j2k_ang, _ = app.radec()
                dec_j2000  = dec_j2k_ang.degrees

                amanecer_fecha = fmt_date(uy, um, ud)
                amanecer_txt = f"{hh:02d}:{mm:02d}:{ss:02d}"

            rows.append({
                "Historical Julian Date": fmt_date(jy, jm, jd),
                "Current Gregorian date": fmt_date(gy, gm, gd),
                "Difference between them (days)": jh.julian_gregorian_offset_days(jy, jm, jd),
                "Sunrise date (UTC)": amanecer_fecha,
                "Sunrise time (UTC)": amanecer_txt,
                "Sun Declination value on date": (np.round(dec_date, 6) if dec_date == dec_date else np.nan),
                "Sun Declination value J2000": (np.round(dec_j2000, 6) if dec_j2000 == dec_j2000 else np.nan)
            })

            if verbose and (k + 1) % 20 == 0:
                print(f"{k + 1}/{days_in_year} processed days…")

        df = pd.DataFrame(rows)

        if write_to_file:
            time = datetime.datetime.now()
            out_csv = f"sunrise_declinations_{julian_year}_julian_{str(time)[:19]}.csv"
            out_csv = out_csv.replace(":", "_")
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")  # BOM para Excel
            if verbose:
                print(f"✅ Saved: {out_csv}")
        
        return df
    
    TROPICAL_YEAR_DAYS = 365.2422
    NULL_MODELS = ("orientations", "solar", "uniform")

    @staticmethod
    def _declination_from(altitude, latitude, azimuth) -> np.ndarray:
        """ Vectorized `calc_decl` (degrees in, degrees out, no validation). """
        h, p, a = np.radians(altitude), np.radians(latitude), np.radians(azimuth)
        s = np.sin(h) * np.sin(p) + np.cos(h) * np.cos(p) * np.cos(a)
        return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))

    @staticmethod
    def solar_declination_approx(day, obliquity_deg: float = 23.44) -> np.ndarray:
        """ Approximate declination of the Sun on day `day` of the year
        (0 = 1 January, may be fractional), including the effect of the
        Earth's orbital eccentricity:
        dec = -asin(sin(eps) * cos(0.98565 (d + 10) + 1.914 sin(0.98565 (d - 2))))
        (angles in degrees). Accurate to about 0.5°, which is enough for a
        null model; for precise values use
        `calculate_declinations_for_entire_year`.

        Parameters
        ----------
        day : float | np.ndarray
            Day of the year, 0 to 365.24
        obliquity_deg : float
            Obliquity of the ecliptic; pass the value for the epoch of the
            sites. Default: 23.44 (present value)

        Returns
        -------
        np.ndarray
            Solar declination in degrees
        """
        d = np.asarray(day, dtype=float)
        lam = 0.98565 * (d + 10.0) + 1.914 * np.sin(np.radians(0.98565 * (d - 2.0)))
        return -np.degrees(np.arcsin(np.sin(np.radians(obliquity_deg)) * np.cos(np.radians(lam))))

    def make_null_sampler(self,
                          null_model: str = "orientations",
                          latitude=None,
                          null_altitudes=None,
                          solar_declinations=None,
                          obliquity_deg: float = 23.56,
                          loc: float = -90.0,
                          scale: float = 180.0):
        """ Builds the null model for declination curvigrams: a function
        `sampler(n, rng)` that returns n random declinations representing
        "no astronomical preference" (or, for 'solar', "no preference for
        particular dates").

        Null models
        -----------
        'orientations' (default)
            Random orientations: azimuths uniform in 0-360°, converted to
            declinations with the sites' latitude(s) and horizon
            altitude(s). This is what structures built without any
            astronomical intent would produce. Random orientations give
            declinations only within about ±(90° - latitude) and pile up at
            the edges of that range; a uniform null would wrongly make
            those edges look significant. With `null_altitudes=None` a flat
            horizon (0°) is assumed, as in the original astrocult code.
        'solar'
            Structures oriented to the rising (or setting) Sun on a random
            day of the year: declinations of the Sun on uniformly random
            days. Use it to ask whether specific dates (solstices,
            equinoxes, feast days) are preferred beyond a general solar
            orientation. Solar declinations pile up near the solstices, so
            solstice peaks are only significant if they exceed that.
        'uniform'
            Declinations uniform over [loc, loc + scale]. Only for
            reproducing results of astrocult versions before this option
            existed; it overstates significance (see 'orientations').

        Parameters
        ----------
        null_model : str
            'orientations', 'solar' or 'uniform' (case-insensitive).
            Default: 'orientations'
        latitude : float | array-like | None
            Site latitude in degrees, or one latitude per site. Required
            for 'orientations'.
        null_altitudes : float | array-like | None
            Horizon altitude(s) in degrees used for the random
            orientations, typically the measured horizon altitude of each
            site. If `latitude` and `null_altitudes` have the same length
            (> 1), they are resampled together as (latitude, altitude)
            pairs of the same site. Default: None (flat horizon, 0°)
        solar_declinations : array-like | None
            For 'solar': a pool of solar declinations to draw from instead
            of the built-in approximation, e.g. the column "Sun Declination
            value on date" of `calculate_declinations_for_entire_year` for
            the epoch and site. Default: None
        obliquity_deg : float
            Obliquity used by the built-in solar approximation. Default: 23.56
        loc, scale : float
            Range of the 'uniform' model. Default: -90.0, 180.0

        Returns
        -------
        Callable | None
            The sampler, or None for 'uniform' (the generic functions then
            draw uniformly over [loc, loc + scale])

        Raises
        ------
        ValueError
            If the model is unknown, `latitude` is missing for
            'orientations', or values are out of range
        """

        model = str(null_model).lower()
        if model not in self.NULL_MODELS:
            raise ValueError(f"`null_model` must be one of {list(self.NULL_MODELS)}.")

        if model == "uniform":
            return None

        if model == "solar":
            if solar_declinations is not None:
                pool = np.asarray(solar_declinations, dtype=float).ravel()
                pool = pool[np.isfinite(pool)]
                if pool.size == 0:
                    raise ValueError("`solar_declinations` contains no valid values.")
                return lambda n, rng: rng.choice(pool, size=n, replace=True)
            eps = float(obliquity_deg)
            return lambda n, rng: self.solar_declination_approx(
                rng.uniform(0.0, self.TROPICAL_YEAR_DAYS, n), eps)

        # orientations
        if latitude is None:
            raise ValueError("`latitude` is required for null_model='orientations' "
                             "(one value, or one per site).")
        lats = np.atleast_1d(np.asarray(latitude, dtype=float)).ravel()
        alts = np.atleast_1d(np.asarray(0.0 if null_altitudes is None else null_altitudes,
                                        dtype=float)).ravel()
        if lats.size == 0 or alts.size == 0:
            raise ValueError("`latitude` and `null_altitudes` must not be empty.")
        if np.any(np.abs(lats) > 90) or np.any(np.abs(alts) > 90):
            raise ValueError("Latitudes and altitudes must be within [-90, 90].")
        paired = lats.size == alts.size and lats.size > 1

        def sampler(n, rng):
            if paired:
                i = rng.integers(0, lats.size, n)
                la, al = lats[i], alts[i]
            else:
                la = lats[rng.integers(0, lats.size, n)]
                al = alts[rng.integers(0, alts.size, n)]
            return self._declination_from(al, la, rng.uniform(0.0, 360.0, n))

        return sampler

    @staticmethod
    def _ephemeris_coverage(eph) -> tuple[float, float]:
        """Julian Date range covered by every segment of a skyfield SpiceKernel.

        Parameters
        ----------
        eph : skyfield.jpllib.SpiceKernel
            The loaded ephemeris

        Returns
        -------
        tuple[float, float]
            The first and last Julian Date covered by all segments
        """
        segments = [getattr(seg, 'spk_segment', seg) for seg in eph.segments]
        return max(seg.start_jd for seg in segments), min(seg.end_jd for seg in segments)
    
    def curvigram_decl_adaptive(self,
        declinations: np.ndarray[float],
        binwidths: np.ndarray[float],                # per-sample bandwidth (degrees)
        *,
        grid_step: float = 0.5,
        xlim: tuple[float, float] = (-90.0, 90.0), # evaluation range in degrees
        left_pad: float = 0.0,
        right_pad: float = 0.0,
        weights: np.ndarray[float] | None = None,
        normalize: str = "pdf",                 # {"pdf","none","max1"}
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adaptive Epanechnikov curvigram (kernel density) of declinations on a
        linear domain. Each declination has its own kernel half-width.
        Unlike azimuths, declinations are not circular: kernels are not wrapped
        at ±90°, and with `normalize='pdf'` the density is rescaled to unit
        area within the evaluation range.

        Parameters
        ----------
        declinations : np.ndarray
            The declinations in degrees
        binwidths : np.ndarray
            Per-sample Epanechnikov half-widths h in degrees (kernel support
            is [x - h, x + h]). These are NOT the user binwidths: convert
            2*error-style binwidths first with
            `HelperFunctions.gaussian_bw_to_epanechnikov_h`, as
            `plot_*_adaptive` does automatically.
        grid_step : float
            Spacing of the evaluation grid in degrees. Default: 0.5
        xlim : tuple[float, float]
            The evaluation range in degrees. Default: (-90.0, 90.0)
        left_pad : float
            Extra degrees added below xlim[0]. Default: 0.0
        right_pad : float
            Extra degrees added above xlim[1]. Default: 0.0
        weights : np.ndarray[float] | None
            Non-negative weight per declination (rescaled to sum 1). Default:
            None (equal weights)
        normalize : str
            'pdf' (area 1), 'max1' (peak 1) or 'none' (no normalization);
            case-insensitive, None also means 'none'. Default: "pdf"

        Returns
        -------
        tuple
            grid : np.ndarray, shape (M, 1)
                Evaluation grid in degrees (column vector)
            density : np.ndarray, shape (M,)
                Density evaluated on `grid`

        Raises
        ------
        ValueError
            If a None value has been provided in any of the arguments,
            or the values are out of range
        """
        if declinations is None or binwidths is None:
            raise ValueError("`declinations` and `binwidths` cannot be None.")

        # Accept None and any capitalization ("none", "None", "PDF", ...)
        normalize = "none" if normalize is None else str(normalize).lower()
        if normalize not in ("pdf", "max1", "none"):
            raise ValueError("`normalize` must be 'pdf', 'max1', or 'none'.")

        dec = np.asarray(declinations, dtype=float).ravel()
        h = np.asarray(binwidths, dtype=float).ravel()

        if dec.size == 0:
            raise ValueError("`declination` must contain at least one value.")
        if h.size != dec.size:
            raise ValueError("`binwidths` must have the same length as `declination`.")
        if not np.all(h > 0):
            raise ValueError("All entries in `binwidths` must be positive.")
        if grid_step <= 0:
            raise ValueError("`grid_step` must be positive.")

        x0, x1 = float(xlim[0]), float(xlim[1])
        start = x0 - float(left_pad)
        stop = x1 + float(right_pad)
        grid = np.arange(start, stop + 1e-12, grid_step)
        new_dec = grid[:, None]

        if weights is None:
            w = np.ones_like(dec) / dec.size
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.size != dec.size:
                raise ValueError("`weights` must have same length as `declination`.")
            if np.any(w < 0):
                raise ValueError("`weights` must be nonnegative.")
            s = w.sum()
            if s == 0:
                raise ValueError("Sum of `weights` must be > 0.")
            w = w / s

        # Pairwise linear distances |x - xi|
        d_abs = np.abs(new_dec - dec[None, :])  # (M,N)
        u = d_abs / h[None, :]

        inside = (u <= 1.0)
        Ku = np.zeros_like(u, dtype=float)
        Ku[inside] = 0.75 * (1.0 - u[inside] ** 2)

        dens_mat = (Ku / h[None, :]) * w[None, :]
        density = dens_mat.sum(axis=1)

        if normalize == "pdf":
            area = integrate.trapezoid(density, grid)
            if area > 0:
                density = density / area
        elif normalize == "max1":
            m = density.max()
            if m > 0:
                density = density / m
        elif normalize == "none":
            pass

        return new_dec, density
    
    def distribution_decl_adaptive(self,
        declinations: np.ndarray[float],
        binwidth: np.ndarray[float],
        *,
        loc: float = -90.0,
        scale: float = 180.0,
        iterations: int = 100,
        title: str = "",
        n_uniform: int = 80_000,
        rng: Generator | None = None,
        grid_step: float = 0.5,
        xlim: tuple[float, float] = (-90.0, 90.0),
        normalize: str = "pdf",
        null_model: str = "orientations",
        latitude=None,
        null_altitudes=None,
        solar_declinations=None,
        obliquity_deg: float = 23.56,
        renormalization: str = "auto",
        return_null_expectation: bool = False,
        return_null_stats: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adaptive density of declinations plus null-model renormalization.

        Wraps `HelperFunctions.distribution_generic_adaptive` with
        `curvigram_fn=self.curvigram_decl_adaptive` (linear domain) and the
        null model built by `self.make_null_sampler` (random orientations
        at the site latitude by default).

        Parameters
        ----------
        declinations : np.ndarray
            The declinations in degrees
        binwidth : np.ndarray
            Per-sample Epanechnikov half-widths h in degrees (kernel support
            is [x - h, x + h]). These are NOT the user binwidths: convert
            2*error-style binwidths first with
            `HelperFunctions.gaussian_bw_to_epanechnikov_h`, as
            `plot_*_adaptive` does automatically.
        loc : float
            Start of the uniform null-model range in degrees. Default: -90.0
        scale : float
            Width of the uniform null-model range in degrees. Default: 180.0
        iterations : int
            Number of null-model samples. Default: 100
        title : str
            Not used in the calculation. Default: ''
        n_uniform : int
            Size of the grid the null-model samples are drawn from. Default: 80_000
        rng : Generator | None
            Random generator; pass one for reproducible results. Default: None
        grid_step : float
            Spacing of the evaluation grid in degrees. Default: 0.5
        xlim : tuple[float, float]
            The evaluation range in degrees. Default: (-90.0, 90.0)
        normalize : str
            'pdf' (area 1), 'max1' (peak 1) or 'none' (no normalization);
            case-insensitive, None also means 'none'. Default: "pdf"
        null_model : str
            'orientations', 'solar' or 'uniform'; see `make_null_sampler`.
            Default: "orientations"
        latitude : float | array-like | None
            Site latitude(s); required for 'orientations'. Default: None
        null_altitudes : float | array-like | None
            Horizon altitude(s) for 'orientations'. Default: None (0°)
        solar_declinations : array-like | None
            Pool of solar declinations for 'solar'. Default: None
        obliquity_deg : float
            Obliquity for the built-in solar model. Default: 23.56
        renormalization : str
            How the curve is centred before dividing by the null-model scale.
            'expected' subtracts the null model's expected density at each
            position, so the curve shows the excess over what the null model
            predicts there (scale: typical point-wise null standard deviation).
            'mean' subtracts the curve's own average (the original method; scale:
            mean standard deviation of the null curves). 'auto' uses 'expected'
            when a `null_sampler` is given (declination null models
            'orientations' and 'solar') and 'mean' otherwise (azimuths,
            'uniform'), so earlier results are reproduced. 'expected' is needed
            for the 'solar' null model, whose expected curve has strong peaks at
            the solstices. Default: 'auto'
        return_null_expectation : bool
            Also return the null model's expected density on the grid (the mean
            of the null-model densities) as a 4th value. Default: False
        return_null_stats : bool
            Also return a dict with the null model's expected density
            ('expectation'), the scale used for renormalization ('scale'), the
            renormalization mode ('mode'), and the point-wise 1st and 99th
            percentiles of the null densities ('density_lo', 'density_hi') and of
            the renormalized null curves ('excess_lo', 'excess_hi'), as a 4th
            value. Takes precedence over `return_null_expectation`. Default: False

        Returns
        -------
        tuple
            grid : np.ndarray, shape (M, 1)
                Declination grid in degrees
            normed_density : np.ndarray, shape (M,)
                Density of the data
            renorm_density : np.ndarray, shape (M,)
                Mean-centered density divided by the mean standard deviation of
                the null-model densities

        Raises
        ------
        ValueError
            If a None value has been provided as an argument
        """
        
        null_sampler = self.make_null_sampler(null_model, latitude, null_altitudes,
                                              solar_declinations, obliquity_deg, loc, scale)
        return HelperFunctions.distribution_generic_adaptive(
            data=declinations,
            binwidths=binwidth,
            null_sampler=null_sampler,
            renormalization=renormalization,
            return_null_expectation=return_null_expectation,
            return_null_stats=return_null_stats,
            curvigram_fn=self.curvigram_decl_adaptive,
            iterations=iterations,
            title=title,
            n_uniform=n_uniform,
            rng=rng,
            grid_step=grid_step,
            xlim=xlim,
            wrap_inputs=None,
            normalize=normalize,
            loc=loc,
            scale=scale,
        )
    
    def plot_decl_adaptive(self,
        declinations,
        latitude: float,
        binwidths: np.ndarray[float],
        *,
        title: str = "",
        iterations: int = 100,
        label: str | None = None,
        xlim: tuple[float, float] | None = None,
        show: bool = True,
        savepath: str | None = None,
        ax: plt.Axes | None = None,
        line_color: str = "darkcyan",
        fill_color: str = "cadetblue",
        fill_alpha: float = 0.35,
        three_sigma_level: float = 3.0,
        obliquity_deg: float = 23.56,
        draw_reference_lines: bool = True,
        # pass-through
        grid_step: float = 0.5,
        normalize: str = "pdf",
        loc: float = -90.0,
        scale: float = 180.0,
        n_uniform: int = 80_000,
        rng: Generator | None = None,
        # thresholds & peaks
        draw_thresholds: bool = True,
        null_percentile: float = 99.0,
        null_iterations: int | None = None,
        mark_significant_peaks: bool = True,
        min_peak_separation_deg: float = 2.5,
        peak_prominence: float | None = 1.0,
        xtick_step: float = 10.0,
        null_model: str = "orientations",
        null_altitudes=None,
        solar_declinations=None,
        renormalization: str = "auto",
        show_null_model: bool = False,
        show_null_range: bool = False,
        view: str = "excess",
        show_three_sigma: bool = False,
        xpad_deg: float = 10.0,
        draw_lunar_lines: bool = True,
        lunar_inclination_deg: float = 5.145,
    ) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Plots the curvigram of declinations with per-point bin widths.
        See also: `self.curvigram_decl_adaptive` and `self.distribution_decl_adaptive`.

        The reference lines mark the declinations of the equinoxes (0°) and
        solstices (±`obliquity_deg`). They do not depend on the horizon.

        Significance is judged against a null model (see
        `make_null_sampler`): by default random orientations at the site
        latitude(s), i.e. what structures without astronomical intent would
        produce. With `null_model='solar'` the question becomes whether
        particular dates are preferred among solar orientations. The curve
        and the significance threshold always use the same null model. With
        these null models the curve shows the excess over what the null
        model expects at each declination (`renormalization='expected'`),
        so e.g. the natural pile-up of random orientations at the edges of
        the possible range, or of solar declinations at the solstices, is
        not mistaken for a preference.

        Parameters
        ----------
        declinations : np.ndarray[float]
            The declinations in degrees
        latitude : float | array-like | None
            Site latitude in degrees, or one latitude per site. Required for
            the default null model ('orientations').
        binwidths : np.ndarray[float]
            Per-point binwidths in degrees, interpreted as bw = 2*sigma.
            Recommended: twice the measurement error of each point, e.g.
            ``2 * Declination.err_decl(...)``. Any positive values are
            accepted; they are converted internally with
            `HelperFunctions.gaussian_bw_to_epanechnikov_h`.
        title : str
            Plot title. Default: '' (no title)
        iterations : int
            Number of null-model samples used to renormalize the curve. Default: 100
        label : str | None
            Legend label for the curve. Default: None
        xlim : tuple[float, float]
            Visible x-axis range in degrees. The curve, the threshold and the
            peaks are always computed over the full range, so zooming in does
            not change the results. Default: None, automatic: the outermost
            structures ± `xpad_deg`, rounded outwards to `xtick_step`, within
            -90..90°
        show : bool
            Call plt.show() if the function created the figure. Default: True
        savepath : str | None
            If given, the figure is also saved there at 300 dpi. Default: None
        ax : plt.Axes | None
            Axes to draw into; a new figure is created if None. Default: None
        line_color : str
            Color of the curve. Default: "darkcyan"
        fill_color : str
            Color of the fill above the mean. Default: "cadetblue"
        fill_alpha : float
            Opacity of the fill of significant regions is twice this value;
            the rest of the curve gets a faint fill (about a third). Default: 0.35
        three_sigma_level : float
            Height of the optional 3-sigma line (see `show_three_sigma`), in
            units of the null model's typical variation. It is also the level
            used for peak significance when `draw_thresholds=False`. Default: 3.0
        obliquity_deg : float
            Obliquity of the ecliptic used for the solstice lines; pass the
            value for the epoch of the site if needed. Default: 23.56
        draw_reference_lines : bool
            Draws the declinations of the equinoxes (0°) and solstices (±`obliquity_deg`). Default: True
        show_three_sigma : bool
            Draw a dashed line at `three_sigma_level` (only in the 'excess' view
            and the polar plot). It is NOT a significance test: it checks one
            position at a time, so random datasets usually cross it somewhere.
            Significance is shown by the strong fill and the peak markers.
            Default: False
        xpad_deg : float
            Padding in degrees around the outermost structures for the automatic
            x-axis range (`xlim=None`). Default: 10.0
        draw_lunar_lines : bool
            Also draw the Moon's declination limits at the major and minor
            standstills, ±(obliquity + inclination) and ±(obliquity -
            inclination), about ±28.7° and ±18.4° with the defaults (on
            azimuth plots, their flat-horizon rising and setting azimuths). These are
            geocentric values: the Moon's parallax lowers its apparent position
            near the horizon by about 0.9°, so treat them as guides. Default: True
        lunar_inclination_deg : float
            Inclination of the lunar orbit to the ecliptic. Default: 5.145
        grid_step : float
            Spacing of the evaluation grid in degrees. Default: 0.5
        normalize : str
            'pdf' (area 1), 'max1' (peak 1) or 'none' (no normalization);
            case-insensitive, None also means 'none'. Default: "pdf"
        loc : float
            Start of the uniform null-model range in degrees. Default: -90.0
        scale : float
            Width of the uniform null-model range in degrees. Default: 180.0
        n_uniform : int
            Size of the grid the null-model samples are drawn from. Default: 80_000
        rng : Generator | None
            Random generator; pass one for reproducible results. Default: None
        draw_thresholds : bool
            Computes the null-model percentile threshold (`null_percentile`)
            used to decide which peaks are significant. The threshold is
            not drawn on the plot; it is returned in the info dict as
            `global_threshold`. If False, the `three_sigma_level` is used
            for peak significance instead. Default: True
        null_percentile : float
            Percentile of the null-model maxima used as the significance
            threshold. Default: 99.0
        null_iterations : int | None
            Iterations for the threshold estimate; None uses `iterations`.
            Use at least 1000 for final results: the 99th percentile of
            fewer random curves is noisy. Default: None
        mark_significant_peaks : bool
            Mark and label the peaks above the significance threshold.
            Default: True
        min_peak_separation_deg : float
            Minimum distance between reported peaks in degrees. Default: 2.5
        peak_prominence : float | None
            Minimum peak prominence (see scipy.signal.find_peaks). Default: 1.0
        xtick_step : float
            Spacing of the x-axis ticks in degrees. Default: 10.0
        null_model : str
            'orientations' (random azimuths -> declinations at the sites'
            latitude and horizon), 'solar' (Sun's declination on random days)
            or 'uniform' (uniform declinations, only to reproduce results of
            earlier versions). See `make_null_sampler`. Default: "orientations"
        null_altitudes : float | array-like | None
            Horizon altitude(s) for 'orientations', typically the measured
            horizon altitude of each site; paired with `latitude` when both
            have one value per site. Default: None (flat horizon, 0°)
        solar_declinations : array-like | None
            Pool of solar declinations for 'solar', e.g. from
            `calculate_declinations_for_entire_year`. Default: None (built-in
            approximation with `obliquity_deg`)
        renormalization : str
            How the curve is centred before dividing by the null-model scale.
            'expected' subtracts the null model's expected density at each
            position, so the curve shows the excess over what the null model
            predicts there (scale: typical point-wise null standard deviation).
            'mean' subtracts the curve's own average (the original method; scale:
            mean standard deviation of the null curves). 'auto' uses 'expected'
            when a `null_sampler` is given (declination null models
            'orientations' and 'solar') and 'mean' otherwise (azimuths,
            'uniform'), so earlier results are reproduced. 'expected' is needed
            for the 'solar' null model, whose expected curve has strong peaks at
            the solstices. Default: 'auto'
        show_null_model : bool
            Also draw the null model's expected curvigram (the average density of
            the random datasets used for the significance test) as a gray dashed
            curve on a second y-axis on the right, in density units. It is the
            average of the `iterations` random datasets, so it becomes smoother
            with more iterations. It shows the
            shape the null model predicts, e.g. the pile-up of random orientations
            at the edges of the possible range, or of solar declinations at the
            solstices, behind the data curve, which shows the excess over it.
            Not available with `polar=True`. Default: False
        show_null_range : bool
            Draw a light gray band from the null model's expected curve up to the
            99th percentile of its random datasets at each position: random data
            stays below the top of the band 99% of the time at that position.
            The band is one-sided, like the significance test, and in the
            'density' view it starts at the dashed null-model curve. Whether a
            peak is significant overall (accounting for all positions) is shown
            by the fill and the markers. Default: False
        view : str
            'excess' (default): the renormalized curve, i.e. the excess over the
            null model in units of its typical variation; 0 means "as the null
            model expects". 'density': the data density and the null model's
            expected density on one axis, both divided by the null model's
            highest density, so 1 is the null model's peak and 0 means no
            structures for both curves; `show_three_sigma` has no effect in
            this view. Significance is the same in both views.
            Not available with `polar=True`. Default: "excess"

        Returns
        -------
        tuple
            fig : plt.Figure
            ax : plt.Axes
            grid : np.ndarray, shape (M, 1)
                Evaluation grid in degrees
            normed_density : np.ndarray, shape (M,)
                Density of the data
            renorm_density : np.ndarray, shape (M,)
                The plotted curve: mean-centered density divided by the mean
                standard deviation of the null-model densities
            info : dict
                'three_sigma_level', 'global_threshold',
                'global_threshold_percentile', 'mean_null_std', 'null_model',
                and the peak lists 'peaks_all' and 'peaks_significant' (dicts
                with 'angle', 'height', 'prominence', 'index')

        Raises
        ------
        ValueError
            If mandatory arguments are None, `data` and `binwidths` differ in
            length, values are out of range, or `latitude` is missing for
            the 'orientations' null model
        """
        
        null_sampler = self.make_null_sampler(null_model, latitude, null_altitudes,
                                              solar_declinations, obliquity_deg, loc, scale)
        null_settings = dict(null_model=null_model, latitude=latitude,
                             null_altitudes=null_altitudes,
                             solar_declinations=solar_declinations,
                             obliquity_deg=obliquity_deg)

        # Reference verticals at solstitial declinations ±ε and equatorial 0°
        ref_lines = []
        for dec in (0.0, obliquity_deg, -obliquity_deg):
            ref_lines.append(dec)

        # Lunar standstill limits: +/-(obliquity + inclination) and +/-(obliquity - inclination)
        lunar_lines = {}
        if draw_lunar_lines:
            major = obliquity_deg + lunar_inclination_deg
            minor = obliquity_deg - lunar_inclination_deg
            lunar_lines = {"major": [major, -major], "minor": [minor, -minor]}
            
            
        result = HelperFunctions.plot_curvigram(
                                       declinations,
                                       None,
                                       binwidths, 
                                       self.curvigram_decl_adaptive,
                                       self.distribution_decl_adaptive,
                                       ref_lines=ref_lines,
                                       title=title,
                                       iterations=iterations,
                                       label=label,
                                       xlim=xlim,
                                       show=show,
                                       savepath=savepath,
                                       ax=ax,
                                       line_color=line_color,
                                       fill_color=fill_color,
                                       fill_alpha=fill_alpha,
                                       three_sigma_level=three_sigma_level,
                                       obliquity_deg=obliquity_deg,
                                       draw_reference_lines=draw_reference_lines,
                                       grid_step=grid_step,
                                       wrap_inputs=False,
                                       normalize=normalize,
                                       loc=loc,
                                       scale=scale,
                                       n_uniform=n_uniform,
                                       rng=rng,
                                       draw_thresholds=draw_thresholds,
                                       null_percentile=null_percentile,
                                       null_iterations=null_iterations,
                                       mark_significant_peaks=mark_significant_peaks,
                                       min_peak_separation_deg=min_peak_separation_deg,
                                       peak_prominence=peak_prominence,
                                       xtick_step=xtick_step,
                                       null_sampler=null_sampler,
                                       distribution_kwargs=null_settings,
                                       renormalization=renormalization,
                                       show_null_model=show_null_model,
                                       null_model_label=f"Null model '{str(null_model).lower()}'",
                                       show_null_range=show_null_range,
                                       view=view,
                                       show_three_sigma=show_three_sigma,
                                       xpad_deg=xpad_deg,
                                       lunar_lines=lunar_lines
                                       )
        result[-1]["null_model"] = str(null_model).lower()
        return result
    
    def plot_fmap(self, 
                  xaxis:list, 
                  yaxis:list, 
                  bw:float, 
                  xgrid1:int, 
                  xgrid2:int, 
                  ygrid1:int, 
                  ygrid2:int, 
                  grid:int=250, 
                  xlabel:str='', 
                  ylabel:str='', 
                  cmap:str='nipy_spectral', 
                  norm:bool=True, 
                  ymarkers:bool=True,
                  show:bool=True,
                  bw_x:float|None=None,
                  title:str='',
                  obliquity_deg:float=23.56,
                  ax:plt.Axes|None=None,
                  savepath:str|None=None) -> plt.Figure:
        """
        Plots a density diagram / frequency map. It can be used to visualize
        for instance the change in orientation tendency (declinations) (Y axis)
        over time (X axis).

        The density is a sum of 2-D Gaussian kernels, one per data point,
        with independent widths along the two axes: `bw` along declination
        (in degrees) and `bw_x` along the X axis (in its own units). Each
        width therefore keeps its meaning whatever the units of the other
        axis (years, centuries, ...).
        
        Parameters
        ----------
        xaxis : list
            A list of temporal values
        yaxis : list
            A list of declination values
        bw : float
            Kernel width along declination in degrees, interpreted as
            bw = 2*sigma like the curvigram binwidths. Recommended: twice the
            declination error, e.g. 2*self.err_decl(...)
        xgrid1 : int
            The left limit of the plot. Can be the smallest value in `xaxis`.
            The xgrid[1,2] and ygrid[1,2] can be used to plot parts of the 
            `xaxis` and `yaxis` data.
        xgrid2 : int
            The right limit of the plot. Can be the largest value in `xaxis`
        ygrid1 : int
            The lower limit of the plot. Can be the smallest value in `yaxis`
        ygrid2 : int
            The upper limit of the plot. Can be the largest value in `yaxis`
        grid : int
            Default=250
        xlabel : str
            The label for the x axis. Default=''            
        ylabel : str
            The label for the y axis. Default=''            
        cmap : str
            The color map used to show the plot. Default='nipy_spectral'            
        norm : bool
            Show the density standardized as (density - mean) / std, in the
            same units as the curvigrams. Default=True                     
        ymarkers : bool
            Draw horizontal reference lines at the declinations of the
            equinoxes (0°, dotted) and solstices (±`obliquity_deg`, solid),
            outlined in black so they stay visible over bright bands, plus
            markers on the y axis (white diamond: equinoxes, yellow
            triangles: solstices) and a legend. Default=True
        show : bool
            Call plt.show() if the function created the figure. Default: True
        bw_x : float | None
            Kernel width along the X axis in its own units (bw = 2*sigma),
            e.g. twice the dating uncertainty. Default: None, which uses
            Scott's rule for 2-D data: sigma = std(xaxis) * N^(-1/6)
        title : str
            Plot title. Default: '' (no title)
        obliquity_deg : float
            Obliquity of the ecliptic for the solstice lines; pass the value
            for the epoch of the sites if needed. Default: 23.56
        ax : plt.Axes | None
            Axes to draw into; a new 11 x 4.2 inch figure is created if None.
            Default: None
        savepath : str | None
            If given, the figure is also saved there at 300 dpi. Default: None
            
        Returns
        -------
            plt.Figure
            
        Raises
        ------
        ValueError
            If `xaxis` or `yaxis` is None or empty, they differ in length,
            or a bandwidth is not positive.
            
        """

        if xaxis is None or yaxis is None:
            raise ValueError("`xaxis` and `yaxis` cannot be None.")
            

        x = np.asarray(xaxis, dtype=float).ravel()
        y = np.asarray(yaxis, dtype=float).ravel()
        if x.size == 0 or x.size != y.size:
            raise ValueError("`xaxis` and `yaxis` must be non-empty and of equal length.")
        if bw is None or bw <= 0:
            raise ValueError("`bw` must be positive.")

        # Kernel standard deviations (bw = 2*sigma convention)
        sigma_y = bw / 2.0
        if bw_x is None:
            sigma_x = float(np.std(x)) * x.size ** (-1.0 / 6.0)   # Scott's rule, 2-D
            if sigma_x <= 0:                                       # all X values equal
                sigma_x = abs(xgrid2 - xgrid1) / grid
        else:
            if bw_x <= 0:
                raise ValueError("`bw_x` must be positive.")
            sigma_x = bw_x / 2.0
        
        # evaluate on a regular grid
        xgrid = np.linspace(xgrid1, xgrid2, grid)
        ygrid = np.linspace(ygrid1, ygrid2, grid)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

        # density(X, Y) = 1/N * sum_i N(X; x_i, sigma_x) * N(Y; y_i, sigma_y).
        # The kernel is separable, so the whole grid is one matrix product.
        gx = np.exp(-0.5 * ((xgrid[None, :] - x[:, None]) / sigma_x) ** 2) / (sigma_x * np.sqrt(2 * np.pi))
        gy = np.exp(-0.5 * ((ygrid[None, :] - y[:, None]) / sigma_y) ** 2) / (sigma_y * np.sqrt(2 * np.pi))
        ZZ = (gy.T @ gx).ravel() / x.size
        if norm: 
            ZZ = (ZZ-np.mean(ZZ))/np.std(ZZ)

        # Styling consistent with the curvigram plots (HelperFunctions.plot_curvigram)
        text_color = "0.15"
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(1, 1, figsize=(11, 4.2))
        else:
            fig = ax.figure

        im = ax.imshow(ZZ.reshape(Xgrid.shape),
                       origin='lower', aspect='auto',
                       extent=[xgrid1, xgrid2, ygrid1, ygrid2],
                       cmap=cmap)
        
        cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
        cb.set_label("Normalized relative frequency" if norm else "Density",
                     fontsize=12, color=text_color)
        cb.ax.tick_params(labelsize=10, colors=text_color)
        cb.outline.set_edgecolor("0.6")

        ax.set_xlabel(xlabel, fontsize=13, color=text_color)
        ax.set_ylabel(ylabel, fontsize=13, color=text_color)
        if title:
            ax.set_title(title, fontsize=14, fontweight="semibold", color=text_color, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="major", labelsize=11, colors=text_color, length=4)
        ax.tick_params(axis="both", which="minor", length=2, color="0.5")
        for spine in ax.spines.values():
            spine.set_color("0.4")
        
        if ymarkers:
            # Same line styles as the curvigrams: equinoxes dotted, solstices
            # solid. A black outline keeps the white lines visible over
            # bright density bands, and markers on the y axis show the
            # positions even where the lines are hidden.
            outline = [patheffects.withStroke(linewidth=3.2, foreground="black")]
            eq_style = dict(color="white", ls=(0, (2, 2)), lw=1.4, path_effects=outline)
            sol_style = dict(color="white", ls="-", lw=1.3, path_effects=outline)
            eq_marker = dict(marker="D", ms=7, mfc="white", mec="black", mew=1.0)
            sol_marker = dict(marker=">", ms=10, mfc="#ffd23f", mec="black", mew=1.0)

            ax.axhline(y=0, zorder=3, **eq_style)
            for dec in (obliquity_deg, -obliquity_deg):
                ax.axhline(y=dec, zorder=3, **sol_style)

            # Markers on the y-axis line (x in axes coordinates, y in degrees)
            y_lo, y_hi = sorted((ygrid1, ygrid2))
            for dec, mstyle in [(0.0, eq_marker), (obliquity_deg, sol_marker), (-obliquity_deg, sol_marker)]:
                if y_lo <= dec <= y_hi:
                    ax.plot([0.0], [dec], transform=ax.get_yaxis_transform(), ls="none",
                            clip_on=False, zorder=6, **mstyle)

            handles = [Line2D([0], [0], **eq_style, **eq_marker),
                       Line2D([0], [0], **sol_style, **sol_marker)]
            labels = [r"Equinoxes ($\delta$ = 0$^\circ$)",
                      rf"Solstices ($\delta$ = $\pm${obliquity_deg:.2f}$^\circ$)"]
            leg = ax.legend(handles, labels, fontsize=10, loc="upper right", frameon=True,
                            framealpha=0.85, facecolor="0.15", edgecolor="0.5", fancybox=False,
                            borderpad=0.6, handlelength=3.0)
            for t in leg.get_texts():
                t.set_color("white")

        HelperFunctions.add_version_watermark(ax)

        if created_fig:
            fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
        if show and created_fig:
            plt.show()
            
        return fig