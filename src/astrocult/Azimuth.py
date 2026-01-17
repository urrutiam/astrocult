# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate
from numpy.random import Generator
import matplotlib.pyplot as plt

import HelperFunctions

class Azimuth:
    """ Class containing azimuth related utility functions. You will find
    here methods that:
    - 
    
    Requirements: pandas, numpy (conda-forge), matlibplot, scipy, typing.
    """
    
    def calc_az(self, altitude_deg: float, 
                  latitude_deg: float, 
                  declination_deg: float, 
                  clip_normalized = False) -> float:
        """Solar azimuth angle in degrees (range [0, 360)).
        Note: simplified formula; used only to draw reference lines.
        
        Parameters
        ----------
        altitude_deg : float
            The altitude in degrees
        latitude_deg : float
            The latitude in degrees
        declination_deg : float
            The declination in degrees
        clip_normalized : bool
            Clips the values in the [-1,1] range. Default: false
        
        Returns
        -------
        float
            The azimuth in degrees, range [0, 360)

        Raises
        ------
        ValueError
            If a None value has been provided in any of the arguments,
            or the values are out of range
        """
        
        if None in [altitude_deg, latitude_deg, declination_deg]:
            raise ValueError("One of the arguments is None.") 
        if not (-90 <= latitude_deg <= 90):
            raise ValueError("The latitude should be between -90 and 90.")   
        if not (-90 <= altitude_deg <= 90):
            raise ValueError("The altitude should be between -90 and 90.") 
        if not (-90 <= declination_deg <= 90):
            raise ValueError("The declination should be between -90 and 90.") 
                
        altitude = np.radians(altitude_deg)
        latitude = np.radians(latitude_deg)
        declination = np.radians(declination_deg)

        numerator = np.sin(declination) - np.sin(altitude) * np.sin(latitude)
        denominator = np.cos(altitude) * np.cos(latitude)
        if clip_normalized:
            arg = np.clip(numerator / denominator, -1.0, 1.0)
        else:
            arg = numerator / denominator
        azimuth_rad = np.arccos(arg)

        # crude quadrant handling consistent with your earlier helper
        if np.sin(np.radians(altitude_deg)) < np.sin(declination):
            azimuth_rad = 2 * np.pi - azimuth_rad

        return float(np.degrees(azimuth_rad) % 360.0)
     
    def curvigram_az_adaptive(self,
        azimuths: np.ndarray[float],
        binwidths: np.ndarray[float],                # per-sample bandwidth (degrees)
        *,
        grid_step: float = 0.5,
        xlim: tuple[float, float] = (0.0, 360.0),  # evaluation range in degrees
        left_pad: float = 0.0,
        right_pad: float = 0.0,
        wrap_inputs: bool = True,
        weights: np.ndarray[float] | None = None,
        normalize: str = "pdf",                    # {"pdf","none","max1"}
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate an Adaptive Epanechnikov KDE on a circular (degree) domain. 
        Contrary to the standard approach where a single binwidth is provided here
        bin widths are provided for each azimuth point.
        For other ways of generating the KDE see the other `self.curvigram_az` methods.

         Parameters
         ----------
         azimuths : np.ndarray
             The list of azimuths
         binwidths : np.ndarray
             The list of per-sample bandwidths (degrees)
         grid_step : float, optional
             The grid step. Default: 0.5
         xlim : Tuple, optional
             The evaluation range in degrees. Default: (0.0, 360.0)
         left_pad : float, optional
             Default: 0
         right_pad : float, optional
             Default: 0
         weights : np.ndarray[float], optional
         normalize : str, optional
             Possible values: pdf, none, max1. Default=pdf
         
         Returns
         -------
         Tuple
            new_az : np.ndarray, shape (N, 1)
                Evaluation grid in degrees (column vector).
            density : np.ndarray, shape (N,)
                KDE density evaluated at `new_az`. 
         Raises
         ------
         ValueError
             If a None value has been provided in any of the arguments,
             or the values are out of range
         """

        if azimuths is None or binwidths is None:
            raise ValueError("`azimuths` and `binwidths` cannot be None.")

        if normalize not in ["pdf", "max1", "None"]:
            raise ValueError("`normalize` must be 'pdf', 'max1', or 'none'.")

        az = np.asarray(azimuths, dtype=float).ravel()
        h = np.asarray(binwidths, dtype=float).ravel()

        if az.size == 0:
            raise ValueError("`azimuth` must contain at least one value.")
        if h.size != az.size:
            raise ValueError("`binwidths` must have the same length as `azimuth`.")
        if not np.all(h > 0):
            raise ValueError("All entries in `binwidths` must be positive.")
        if grid_step <= 0:
            raise ValueError("`grid_step` must be positive.")

        if wrap_inputs:
            az = np.mod(az, 360.0)

        x0, x1 = float(xlim[0]), float(xlim[1])
        start = x0 - float(left_pad)
        stop = x1 + float(right_pad)
        grid = np.arange(start, stop + 1e-12, grid_step)
        new_az = grid[:, None]

        if weights is None:
            w = np.ones_like(az) / az.size
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.size != az.size:
                raise ValueError("`weights` must have same length as `azimuth`.")
            if np.any(w < 0):
                raise ValueError("`weights` must be nonnegative.")
            s = w.sum()
            if s == 0:
                raise ValueError("Sum of `weights` must be > 0.")
            w = w / s

        d_abs = np.abs(HelperFunctions.ang_diff_deg(new_az, az[None, :]))  # (M,N)
        u = d_abs / h[None, :]

        inside = (u <= 1.0)
        Ku = np.zeros_like(u, dtype=float)
        Ku[inside] = 0.75 * (1.0 - u[inside] ** 2)  # Epanechnikov

        dens_mat = (Ku / h[None, :]) * w[None, :]
        density = dens_mat.sum(axis=1)
        
        # Perform normalization
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
            
        return new_az, density
                         
    def distribution_az_adaptive(self,
        azimuths: np.ndarray[float],
        binwidths: np.ndarray[float],
        *,
        loc: float = 0.0,
        scale: float = 359.0,
        iterations: int = 100,
        title: str = "",
        n_uniform: int = 80_000,
        rng: Generator | None = None,
        grid_step: float = 0.5,
        xlim: tuple[float, float] = (0.0, 360.0),
        wrap_inputs: bool = True,
        normalize: str = "pdf",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generic adaptive density + null-model renormalization for azimuth values.
        Contrary to the standard approach where a single binwidth is provided here
        bin widths are provided for each azimuth point.

        It wraps the `HelperFunctions.distributions_az_adaptive` method and 
        uses curvigram_fn=self.curvigram_az_adaptive.
        
        Parameters
        ----------
        data : np.ndarray
            The azimuth input data
        binwidths : np.ndarray
            The bin bandwidths
        loc : float
            Default: 0
        scale : float
            Default: 359
        iterations : int
            Default: 100
        title : str
        n_uniform : int
            Default: 80_000
        rng : Generator
        grid_step : float
            Default: 0.5
        xlim : Tupple
            Default: (0.0, 360.0)
        wrap_inputs : bool
            Default: true
        normalize : str
            Possible values: pdf, none, max1. Default=pdf

        
        Returns
        -------
        Tupple
            new_az : np.ndarray
                Azimuth grid returned by `curvigram_az` (shape: [N, ...]).
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
            data=azimuths,
            binwidths=binwidths,
            curvigram_fn=self.curvigram_az_adaptive,
            iterations=iterations,
            title=title,
            n_uniform=n_uniform,
            rng=rng,
            grid_step=grid_step,
            xlim=xlim,
            wrap_inputs=wrap_inputs,
            normalize=normalize,
            loc=loc,
            scale=scale,
        )
    
    def plot_az_adaptive(self,
        azimuths: np.ndarray[float],
        latitude: float,
        binwidths: np.ndarray[float],
        *,
        title: str = "",
        iterations: int = 100,
        label: str | None = None,
        xlim: tuple[float, float] = (0, 360),
        show: bool = True,
        savepath: str | None = None,
        ax: plt.Axes | None = None,
        line_color: str = "darkcyan",
        fill_color: str = "cadetblue",
        fill_alpha: float = 0.8,
        three_sigma_level: float = 3.0,
        obliquity_deg: float = 23.56,
        draw_reference_lines: bool = True,
        # pass-through to adaptive density/null
        grid_step: float = 0.5,
        wrap_inputs: bool = True,
        normalize: str = "pdf",
        loc: float = 0.0,
        scale: float = 359.0,
        n_uniform: int = 80_000,
        rng: Generator | None = None,
        # thresholds & peaks
        draw_thresholds: bool = True,
        null_percentile: float = 99.0,
        null_iterations: int | None = None,
        mark_significant_peaks: bool = True,
        min_peak_separation_deg: float = 5.0,
        peak_prominence: float | None = 1.0,
        xtick_step: float = 20.0,
    ) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray, np.ndarray, dict]:
        """ Plots the curvigram for the provided azimuths and bin widths by using an adaptive approach.
        See also: `self.curvigram_az_adaptive` and `self.distribution_az_adaptive`.
        
        Parameters
        ----------
            azimuths : np.ndarray[float]
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
            ValueError
                If mandatory arguments are None, or values are out of range.
        """
        
        if None in [azimuths, binwidths, latitude]:
            raise ValueError("`azimuths`, `latitude` and `binwidths` cannot be None.")
            
        if not (-90 <= latitude <= 90):
            raise ValueError("`latitude``should be between [-90,90].")
            
        if len(azimuths) != len(binwidths):
            raise ValueError("`azimuths` and `binwidths` should be of equal lenght")
          
        
        ## Solstices ±ε and symmetric counterparts. Equinoxes
        ref_lines = []
        
        for eps in (obliquity_deg, -obliquity_deg):
            a = self.calc_az(0, latitude, obliquity_deg, True)
            ref_lines.append(a)
        ref_lines.append(90)
        ref_lines.append(270)
                
        return HelperFunctions.plot_curvigram(azimuths, 
                                       latitude, 
                                       binwidths, 
                                       self.curvigram_az_adaptive,
                                       self.distribution_az_adaptive,
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
                                       wrap_inputs=wrap_inputs,
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
