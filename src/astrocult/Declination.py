# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy.random import Generator

from scipy import integrate
from scipy import stats
from scipy.stats import gaussian_kde
import datetime
from  datetime import timedelta
import math
import matplotlib.pyplot as plt


from skyfield.api import load, wgs84

import JulianHelper as JulianHelper
import HelperFunctions as HelperFunctions

import subprocess

class Declination:
    """ Class containing declination related utility functions.
    
    Requirements: skyfield, pandas, numpy (conda-forge), matlibplot, scipy, typing.
    """
    
    EPHEM_FILES = ['de431_part-1.bsp', 'de431_part-2.bsp']
    
    def err_decl(self, 
                 lat:float, 
                 az:float, 
                 alt:float, 
                 err_az:float=0.5, 
                 err_alt:float=0.5) -> float:
        """ Calculates the declination error based on the error considered when
        calculating the azimuth and horizon altitude.
               
        Parameters
        lat:float
            The observer's latitude in degrees
        az:float
            The object's azimuth as observed from `lat`            
        alt:float
            The object's altitude in degrees
        err_az:float
            The error in the azimuth reading. Can be the error of the instrument, 
            the reading error, or both. Default=0.5 degrees
        err_alt:float
            The error in the altitude reading. Can be the error of the instrument, 
            the reading error, or both. Default=0.5 degrees. See `GeoUtils.err_alt()` for 
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
        if not (0 <= az <= 360):
            raise ValueError("The `az` should be between 0 and 360")    
        
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
        
        Parameters
        lat:float
            The observer's latitude in degrees
        alt:float
            The object's altitude in degrees
        az:float
            The object's azimuth as observed from `lat`
            
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
        if not (0 <= az <= 360):
            raise ValueError("The `az` should be between 0 and 360")    
            
        return math.degrees(math.asin(math.sin(math.radians(alt)) * math.sin(math.radians(lat))
                         + math.cos(math.radians(alt)) * math.cos(math.radians(lat)) 
                                    * math.cos(math.radians(az))))
    
    def calculate_declinations_for_entire_year(self, julian_year : int, 
                                               latitude : float, 
                                               longitude : float,
                                               elevation : float, write_to_file = True) -> pd.DataFrame:
        """Calculates the solar declinations for a given year and geographical position.
        
        Parameters
        ----------
        julian_year : int
            The Julian year
        latitude : float
            The latitude in degrees
        longitude : float
            The longitude in degrees
        elevation : float
            The elevation in meters
        write_to_file : bool, optional
            write results to a file in the current folder (default=True)
        
        Returns
        -------
        pd.DataFrame
            A dataframe containing the information for the entire year. 

        Raises
        ------
        ValueError
            If a None value has been provided in any of the arguments,
            or the values are out of range
        """
        
        if None in [julian_year, latitude, longitude, elevation]:
            raise ValueError("No None accepted in any argument.") 
        if latitude not in range(-90, 90):
            raise ValueError("The latitude should be between -90 and 90.")   
        if longitude not in range(-180, 180):
            raise ValueError("The longitude should be between -180 and 180.")  
        
        #if (julian_year < 1):
        #    eph = load(self.EPHEM_FILES[0])
        #else:
        #    eph = load(self.EPHEM_FILES[1])
        eph = load('de421.bsp') # suficiente para 1500–2500 aprox.
        #eph = load('de431t.bsp') # suficiente para 1500–2500 aprox.
        print("DE431 ephemerides file loaded.")
    
        sun = eph['sun']
        earth = eph['earth']
    
        ts  = load.timescale()
        sun = eph['sun']
        earth = eph['earth']
        observer = earth + wgs84.latlon(latitude, longitude, 
                                        elevation_m = elevation)
        
        jh = JulianHelper.JulianHelper()
        
        days_in_year = 366 if jh.is_julian_leap_year(julian_year) else 365
        start_jul = datetime.datetime(julian_year, 1, 1)

        rows = []
        print(f"Starting computations for Julian year {julian_year} ...")

        # Época J2000 para RA/Dec en ese marco
        t_J2000 = ts.tt(2000, 1, 1, 12)  # 2000-01-01 12:00 TT

        for k in range(days_in_year):
            jul_date  = start_jul + timedelta(days=k)
            greg_date = jh.julian_to_gregorian_datetime(jul_date)

            sunrise_utc = jh.estimate_sunrise_noaa(greg_date, latitude, longitude)

            if sunrise_utc is None:
                dec_date = np.nan
                dec_j2000 = np.nan
                amanecer_txt = ""
            else:
                # Tiempo Skyfield
                t = ts.utc(sunrise_utc.year, sunrise_utc.month, sunrise_utc.day,
                           sunrise_utc.hour, sunrise_utc.minute, sunrise_utc.second)

                # Vector aparente (incluye aberración, efecto luz, etc.)
                app = observer.at(t).observe(sun).apparent()

                # 1) COORDENADAS EN MARCO DE LA FECHA (true equator/equinox of date)
                ra_date_ang, dec_date_ang, _ = app.radec(epoch='date')
                dec_date   = dec_date_ang.degrees   # DEC en grados

                # 2) COORDENADAS EN J2000 (por defecto)
                ra_j2k_ang, dec_j2k_ang, _ = app.radec()
                dec_j2000  = dec_j2k_ang.degrees

                amanecer_txt = sunrise_utc.strftime("%H:%M:%S")

            rows.append({
                "Historical Julian Date": jul_date.strftime("%Y-%m-%d"),
                "Current Gregorian date": greg_date.strftime("%Y-%m-%d"),
                "Difference between them (days)": jh.julian_gregorian_offset_days(jul_date.year, jul_date.month, jul_date.day),
                "Sunrise time (UTC)": amanecer_txt,
                "Sun Declination value on date": (np.round(dec_date, 6) if dec_date == dec_date else np.nan),
                "Sun Declination value J2000": (np.round(dec_j2000, 6) if dec_j2000 == dec_j2000 else np.nan)
            })

            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{days_in_year} processed days…")

        df = pd.DataFrame(rows)

        if write_to_file:
            time = datetime.datetime.now()
            out_csv = f"sunrise_declinations_{julian_year}_julian_{str(time)[:19]}.csv"
            out_csv = out_csv.replace(":", "_")
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")  # BOM para Excel
            print(f"✅ Saved: {out_csv}")
        
        return df
    
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
        """Generate an Adaptive Epanechnikov curvigram on a circular (degree) domain.

         Parameters
         ----------
         declinations : np.ndarray
             The list of declinations
         binwidths : np.ndarray
             The list of per-sample bandwidths (degrees)
         grid_step : float, optional
             The the grid step. Default: 0.5
         xlim : Tuple, optional
             The evaluation range in degrees. Default: (-90.0, 90.0)
         left_pad : float, optional
             Default: 0
         right_pad : float, optional
             Default: 0
         weights : np.ndarray[float], optional
         normalize : str
             Possible values: pdf, none, max1. Default=pdf
         
         Returns
         -------
         Tuple
             Returns the new declinations processed with the Adaptive Epanechnicov kernel (M,1) and density (M,).
 
         Raises
         ------
         ValueError
             If a None value has been provided in any of the arguments,
             or the values are out of range
         """
        dec = np.asarray(declinations, dtype=float).ravel()
        h = np.asarray(binwidths, dtype=float).ravel()

        if declinations is None or binwidths is None:
            raise ValueError("`declinations` and `binwidths` cannot be None.")

        if normalize not in ["pdf", "max1", "None"]:
            raise ValueError("`normalize` must be 'pdf', 'max1', or 'none'.")

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generic adaptive density + null-model renormalization for azimuth values.

        It wraps the `HelperFunctions.distributions_dec_adaptive` method and 
        uses curvigram_fn=self.curvigram_dec_adaptive.
        
        Parameters
        ----------
        declinations : np.ndarray
            The declination input data
        binwidths : Sequence
            The bin np.ndarray
        loc : float
            Default: -90
        scale : float
            Default: 180
        iterations : int
            Default: 100
        title : str
        n_uniform : int
            Default: 80_000
        rng : Generator
        grid_step : float
            Default: 0.5
        xlim : Tupple
            Default: (-90, 90.0)
        normalize : str
            Possible values: pdf, none, max1. Default=pdf

        
        Returns
        -------
        Tupple
            decl : np.ndarray
                Declination grid returned by `curvigram_az` (shape: [N, ...]).
            normed_density : np.ndarray
                Density for the real sample.
            renorm_density : np.ndarray
                Mean-centered real density divided by the average std from the null model.

        Raises
        ------
        ValueError
            If a None value has been provided as an argument
        """
        
        return HelperFunctions.distribution_generic_adaptive(
            data=declinations,
            binwidths=binwidth,
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
        xlim: tuple[float, float] = (-90, 90),
        show: bool = True,
        savepath: str | None = None,
        ax: plt.Axes | None = None,
        line_color: str = "darkcyan",
        fill_color: str = "cadetblue",
        fill_alpha: float = 0.8,
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
    ) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray, np.ndarray, dict]:
        """ Plots the curvigram for the provided declinations and bin widths by using an adaptive approach.
        See also: `self.curvigram_decl_adaptive` and `self.distribution_decl_adaptive`.
        
        Parameters
        ----------
            declinations : np.ndarray[float]
            latitude : float
            binwidths : np.ndarray[float]
  
            title : str 
                Default: empty string
            iterations : int
                Default: 100
            label : str | None
                Default: None
            xlim : tuple[float, float] 
                Default: (0, 360)
            show : bool 
                Default: True
            savepath : str | None
                Default: None
            ax : plt.Axes | None 
                Default: None
            line_color : str 
                Default: "darkcyan"
            fill_color : str 
                Default: "cadetblue"
            fill_alpha : float 
                Default: 0.8
            three_sigma_level : float 
                Default: 3.0
            obliquity_deg : float 
                Default: 23.56
            draw_reference_lines : bool 
                Default: True
            grid_step : float 
                Default: 0.5
            wrap_inputs : bool 
                Default: True
            normalize : str 
                Default: "pdf"
            loc : float 
                Default: 0.0
            scale : float 
                Default: 359.0
            n_uniform : int 
                Default: 80_000
            rng : Generator | None 
                Default: None
            draw_thresholds : bool 
                Default: True
            null_percentile : float 
                Default: 99.0
            null_iterations : int | None
                Default: None
            mark_significant_peaks : bool 
                Default: True
            min_peak_separation_deg : float 
                Default: 5.0
            peak_prominence : float | None 
                Default: 1.0
            xtick_step : float  
                Default: 20.0
        
        Returns
        -------
        
        
        Raises
        ------
        """
        
        # Reference verticals at solstitial declinations ±ε and equatorial 0°
        ref_lines = []
        for dec in (0.0, obliquity_deg, -obliquity_deg):
            ref_lines.append(dec)
            
            
        return HelperFunctions.plot_curvigram(
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
                                       xtick_step=xtick_step
                                       )
    
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
                  show:bool=True) -> plt.Figure:
        """
        Plots a density diagram / frequency map. It can be used to visualize
        for instance the change in orientation tendency (declinations) (Y axis)
        over time (X axis).
        
        Parameters
        ----------
        xaxis : list
            A list of temporal values
        yaxis : list
            A list of declination values
        bw : float
            The declination uncertainty. Can be 2*self.err_decl(...)
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
            Variable indicating whether or not the data should be normalized. Default=True                     
        ymarkers : bool
            Variable indicating whether or not the markers on the X axis should be shown. 
            Default=True
        show : bool
            Default: True
            
        Returns
        -------
            plt.Figure
            
        Raises
        ------
        ValueError
            If any argument is None.
            
        """

        if xaxis is None or yaxis is None:
            raise ValueError("`xaxis` and `yaxis` cannot be None.")
            

        # fit an array of size [Ndim, Nsamples]
        #fig, ax = plt.subplots(1,1)
        
        data = np.vstack([xaxis, yaxis])
        bw = bw / np.std(data)
        kde = gaussian_kde(data, bw_method=bw)#'silverman')
        
        # evaluate on a regular grid
        xgrid = np.linspace(xgrid1, xgrid2, grid)
        ygrid = np.linspace(ygrid1, ygrid2, grid)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        ZZ = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        if norm == True: 
            ZZ = (ZZ-np.mean(ZZ))/np.std(ZZ)
    
        #ax = plt.axes(projection='3d')
        #ax.plot_surface(np.reshape(xaxis,(1,206)), np.reshape(yaxis,(1,206)), np.reshape(ZZ,(500,500)))
        
        fig, ax = plt.subplots(1,1)
        # Plot the result as an image
        plt.imshow(ZZ.reshape(Xgrid.shape),
                   origin='lower', aspect='auto',
                   extent=[xgrid1, xgrid2, ygrid1, ygrid2],
                   cmap=cmap)
        
        cb = plt.colorbar()
        if norm == True: 
            cb.set_label("Normalized relative frequency",fontsize=18)
        else: 
            cb.set_label("Frequency",fontsize=18)
        
        cb.ax.tick_params(labelsize=16)
        plt.xlabel(xlabel,fontsize=18 )
        plt.ylabel(ylabel,fontsize=18)
        plt.xticks(fontsize=16)
        ax.yaxis.set_ticks(np.arange(ygrid1, ygrid2, 10))
        ax.xaxis.set_ticks(np.arange(xgrid1, xgrid2+1, 1))
        plt.yticks(fontsize=16)
        fig.set_figwidth(6)
        fig.set_figheight(1)
        
        if ymarkers == True:
            ee = 23.56 #Ecliptic angle
            plt.axhline(y=-ee, color='white', ls=':', lw=1.3, alpha=0.7) #WS line
            plt.axhline(y=0, color='white',lw=1.3, alpha=0.7) #Equinox lines
            plt.axhline(y=ee, color='white',ls=':', lw=1.3, alpha=0.7) #SS lines
            
        if show:
            plt.show()
            
        return fig