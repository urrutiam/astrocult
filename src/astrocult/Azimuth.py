# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate
from numpy.random import Generator
import matplotlib.pyplot as plt
import math

import HelperFunctions as HelperFunctions

class Azimuth:
    """ Class containing azimuth related utility functions.
    
    Requirements: numpy, scipy, matplotlib.

    Azimuths are in degrees, measured from north through east (0-360).

    Circular statistics
    -------------------
    Azimuths are angles on a circle: 0° and 360° are the same direction, and
    359° and 1° are only 2° apart. Ordinary (linear) statistics give wrong
    results for readings on either side of north: the arithmetic mean of 359°
    and 1° is 180° and their standard deviation is about 253°. The methods in
    this class therefore work on the circle:

    - Mean direction: atan2(mean(sin a_i), mean(cos a_i)).
    - Angular differences: the shortest signed difference, wrapped to
      (-180°, 180°] (see `HelperFunctions.ang_diff_deg`).
    - Spread (`err_az`): sample standard deviation of the wrapped differences
      from the mean direction. For readings that do not straddle north this
      equals the ordinary standard deviation.
    - Curvigrams (`curvigram_az_adaptive`) use the wrapped differences as
      kernel distances, so a cluster of orientations around north forms one
      peak instead of two half-peaks at 0° and 360°.
    - Peak detection (`HelperFunctions.find_circular_peaks`) wraps around
      0°/360°.
    - `plot_az_adaptive(..., polar=True)` draws the curvigram on a circle
      (compass view, north at the top, clockwise).
    - `plot_orientation_map` draws the sites on a lat/lon map with an arrow
      along each azimuth and a wedge for its error.
    - Clustering of azimuths should use
      `Clustering.create_circular_kmeans_clusters`, not ordinary KMeans.

    Declinations, unlike azimuths, are linear (-90° to +90°) and are handled
    with ordinary statistics in `Declination`.
    """
    
    def err_az(self, 
               azimuths: list,
               err_az: float=0.5) -> float:
        """ Calculates the error of an orientation from a set of azimuth
        readings, combining the systematic error `err_az` with the spread of
        the readings: sqrt(err_az^2 + s^2), where s is the circular sample
        standard deviation of the readings. Readings on either side of north
        are handled correctly (359° and 1° are 2° apart, not 358°). With a
        single reading s = 0.

        s is computed as sqrt(sum(d_i^2) / (n - 1)), where d_i are the
        shortest angular differences between each reading and the circular
        mean direction atan2(mean(sin a_i), mean(cos a_i)).

        Note: this "wrapped" standard deviation is used instead of Mardia's
        circular standard deviation sqrt(-2 ln R) (R = mean resultant length)
        because it reduces exactly to the ordinary sample standard deviation
        when the readings do not straddle north, so results stay comparable
        with linear calculations. For the small spreads typical of repeated
        field readings (a few degrees) the two definitions agree closely.
               
        Parameters   
        azimuths:list
            A set of azimuth readings in degrees. Can be magnetic or true azimuths
        err_az:float
            The error in each azimuth reading. Can be the error of the instrument, 
            the reading error, or both. Default=0.5 degrees
        Returns
            The azimuths' error in degrees
            
        Raises
            ValueError if `azimuths` is None or empty
        """    
        
        if azimuths is None:
            raise ValueError("`azimuths` cannot be None.")

        az = np.asarray(azimuths, dtype=float).ravel()
        if az.size == 0:
            raise ValueError("`azimuths` must contain at least one reading.")

        if az.size == 1:
            spread = 0.0
        else:
            rad = np.radians(az)
            mean = np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
            dev = HelperFunctions.ang_diff_deg(az, np.full_like(az, mean))
            spread = math.sqrt(float(np.sum(dev ** 2)) / (az.size - 1))

        return math.sqrt(err_az**2 + spread**2)
    
    def calc_az(self, altitude_deg: float, 
                latitude_deg: float, 
                declination_deg: float,
                clip_normalized: bool | None = None) -> tuple[float, float] | float:
        """Azimuth angle(s) in degrees (range [0, 360)) at which a body of
        declination `declination_deg` stands at altitude `altitude_deg`, seen
        from latitude `latitude_deg`. Accepts any real numbers (int, float,
        numpy scalars).

        Called with three arguments it returns both solutions as a tuple
        (east, west): the body reaches that altitude once while rising (east)
        and once while setting (west). The user should pick whichever is
        relevant in their case.

        Called with `clip_normalized` (True or False) it returns a single
        azimuth from a simplified formula kept for backward compatibility:
        the west solution for positive declinations and the east one
        otherwise. With True, impossible configurations are clipped instead
        of returning NaN. Prefer the three-argument form.
        
        Parameters
        ----------
        altitude_deg : float
            The altitude in degrees
        latitude_deg : float
            The latitude in degrees
        declination_deg : float
            The declination in degrees
        clip_normalized : bool | None
            None (default): return (east, west). True/False: return a single
            azimuth, clipping the cosine to [-1, 1] if True.
            
        Returns
        -------
        tuple | float
            (east, west) azimuths, or a single azimuth if `clip_normalized`
            is given

        Raises
        ------
        ValueError
            If a None value has been provided in any of the arguments,
            or the values are out of range
        """
        
        if None in [altitude_deg, latitude_deg, declination_deg]:
            raise ValueError("One of the arguments is None.") 
        altitude_deg, latitude_deg, declination_deg = (
            float(altitude_deg), float(latitude_deg), float(declination_deg))
        if not (-90 <= latitude_deg <= 90):
            raise ValueError("The latitude should be between -90 and 90.")   
        if not (-90 <= altitude_deg <= 90):
            raise ValueError("The altitude should be between -90 and 90.") 
        if not (-90 <= declination_deg <= 90):
            raise ValueError("The declination should be between -90 and 90.") 

        if clip_normalized is not None:
            # Simplified single-value formula (backward compatibility)
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

            # crude quadrant handling
            if np.sin(np.radians(altitude_deg)) < np.sin(declination):
                azimuth_rad = 2 * np.pi - azimuth_rad

            return float(np.degrees(azimuth_rad) % 360.0)

        lat = math.radians(latitude_deg)
        h   = math.radians(altitude_deg)
        dec = math.radians(declination_deg)
    
        # Hour angle
        cosH = (math.sin(h) - math.sin(lat) * math.sin(dec)) / \
               (math.cos(lat) * math.cos(dec))
    
        cosH = max(-1.0, min(1.0, cosH))
    
        H = math.acos(cosH)  # absolute value
    
        # azimuth cosine
        cosA = (math.sin(dec) - math.sin(lat) * math.sin(h)) / \
               (math.cos(lat) * math.cos(h))
    
        cosA = max(-1.0, min(1.0, cosA))
    
        # East (H negative → azimuth < 180°)
        sinA_E = -math.cos(dec) * math.sin(-H) / math.cos(h)
        A_E = math.degrees(math.atan2(sinA_E, cosA)) % 360
    
        # West (H positive → azimuth > 180°)
        sinA_W = -math.cos(dec) * math.sin(H) / math.cos(h)
        A_W = math.degrees(math.atan2(sinA_W, cosA)) % 360
    
        return A_E, A_W

    def plot_orientation_map(self,
        latitudes,
        longitudes,
        azimuths,
        errors=None,
        *,
        labels=None,
        arrow_length_km: float | None = None,
        point_color: str = "darkcyan",
        title: str = "",
        ax: plt.Axes | None = None,
        show: bool = True,
        savepath: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """ Draws a map of the sites with their orientations: each site is a
        dot at its latitude/longitude with an arrow pointing along its
        azimuth, and, if `errors` are given, a translucent wedge covering
        azimuth ± error showing the uncertainty of the orientation. The
        wedge reaches a little beyond the arrow tip so it stays visible for
        small errors; its angle is the true azimuth ± error.

        The map has the correct east-west / north-south proportions for its
        latitude, so arrow directions on the page are true directions, plus
        a kilometre scale bar and a north arrow. All arrows have the same
        length on the map; only their direction carries information.
        Azimuths should be true azimuths (magnetic readings corrected with
        `GeoUtils.get_magnetic_declination`).

        Parameters
        ----------
        latitudes, longitudes : array-like
            Site coordinates in degrees (north and east positive)
        azimuths : array-like
            True azimuths in degrees, measured from north through east
        errors : float | array-like | None
            Azimuth error(s) in degrees, one value for all sites or one per
            site (e.g. from `self.err_az`). Default: None (no wedges)
        labels : list[str] | None
            Optional site names written next to the dots. Default: None
        arrow_length_km : float | None
            Length of the arrows (and wedges) in km. Default: None (8% of
            the extent of the sites, or 1 km for a single site)
        point_color : str
            Colour of the dots, arrows and wedges. Default: "darkcyan"
        title : str
            Plot title. Default: '' (no title)
        ax : plt.Axes | None
            Axes to draw into; a new figure is created if None. Default: None
        show : bool
            Call plt.show() if the function created the figure. Default: True
        savepath : str | None
            If given, the figure is also saved there at 300 dpi. Default: None

        Returns
        -------
        tuple
            The matplotlib (Figure, Axes)

        Raises
        ------
        ValueError
            If the inputs are None, empty, of different lengths, or out of
            range
        """

        from matplotlib.patches import Polygon, Patch
        from matplotlib.lines import Line2D

        if latitudes is None or longitudes is None or azimuths is None:
            raise ValueError("`latitudes`, `longitudes` and `azimuths` cannot be None.")
        lat = np.asarray(latitudes, dtype=float).ravel()
        lon = np.asarray(longitudes, dtype=float).ravel()
        az = np.mod(np.asarray(azimuths, dtype=float).ravel(), 360.0)
        n = lat.size
        if n == 0 or lon.size != n or az.size != n:
            raise ValueError("`latitudes`, `longitudes` and `azimuths` must be non-empty and of equal length.")
        if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180):
            raise ValueError("Latitudes must be within [-90, 90] and longitudes within [-180, 180].")
        err = None
        if errors is not None:
            err = np.broadcast_to(np.asarray(errors, dtype=float), (n,)).copy()
            if np.any(err < 0) or np.any(err > 180):
                raise ValueError("`errors` must be within [0, 180] degrees.")
        if labels is not None and len(labels) != n:
            raise ValueError("`labels` must have one entry per site.")
        if arrow_length_km is not None and arrow_length_km <= 0:
            raise ValueError("`arrow_length_km` must be positive.")

        text_color = "0.15"
        km_per_deg = np.pi * 6371.0 / 180.0
        mean_lat = float(np.mean(lat))
        cos_lat = np.cos(np.radians(mean_lat))
        extent_km = max(np.ptp(lon) * km_per_deg * cos_lat, np.ptp(lat) * km_per_deg)
        L = float(arrow_length_km) if arrow_length_km is not None else (0.08 * extent_km if extent_km > 0 else 1.0)

        def point_at(lat0, lon0, az_deg, dist_km):
            """Point dist_km from (lat0, lon0) along az_deg (local flat approximation)."""
            a = np.radians(az_deg)
            return (lat0 + dist_km * np.cos(a) / km_per_deg,
                    lon0 + dist_km * np.sin(a) / (km_per_deg * np.cos(np.radians(lat0))))

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        xs, ys = [lon], [lat]
        for i in range(n):
            if err is not None and err[i] > 0:
                # wedge slightly longer than the arrow so the fan shows beyond the tip;
                # its angle is the true azimuth +/- error (not exaggerated)
                k = max(8, int(2 * err[i]) + 2)
                arc = np.linspace(az[i] - err[i], az[i] + err[i], k)
                plat, plon = point_at(lat[i], lon[i], arc, 1.25 * L)
                verts = np.column_stack([np.r_[lon[i], plon, lon[i]], np.r_[lat[i], plat, lat[i]]])
                ax.add_patch(Polygon(verts, closed=True, facecolor=point_color, alpha=0.35,
                                     edgecolor="none", zorder=2))
                ax.add_patch(Polygon(verts, closed=True, facecolor="none", edgecolor=point_color,
                                     alpha=0.8, lw=0.7, zorder=2.1))
                xs.append(plon); ys.append(plat)
            elat, elon = point_at(lat[i], lon[i], az[i], L)
            ax.annotate("", xy=(elon, elat), xytext=(lon[i], lat[i]),
                        arrowprops=dict(arrowstyle="-|>", color=point_color, lw=1.6,
                                        mutation_scale=12, shrinkA=0, shrinkB=0), zorder=3)
            xs.append([elon]); ys.append([elat])
        ax.scatter(lon, lat, s=34, color=point_color, edgecolors="white", linewidths=0.8, zorder=4)
        if labels is not None:
            for i in range(n):
                ax.annotate(str(labels[i]), (lon[i], lat[i]), xytext=(-5, -5), textcoords="offset points",
                            ha="right", va="top", fontsize=9, color=text_color, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

        # map extent: sites, arrow tips and wedges, with a margin
        x_all, y_all = np.concatenate([np.ravel(v) for v in xs]), np.concatenate([np.ravel(v) for v in ys])
        pad_x = max(0.08 * np.ptp(x_all), L / (km_per_deg * cos_lat) * 0.5)
        pad_y = max(0.08 * np.ptp(y_all), L / km_per_deg * 0.5)
        x0, x1 = x_all.min() - pad_x, x_all.max() + pad_x
        y0, y1 = y_all.min() - pad_y, y_all.max() + pad_y
        y0 -= 0.08 * (y1 - y0)                    # room for the scale bar below the sites
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(1.0 / cos_lat)

        ax.set_xlabel("Longitude (°E)", fontsize=12, color=text_color)
        ax.set_ylabel("Latitude (°N)", fontsize=12, color=text_color)
        HelperFunctions.format_map_ticks(ax, (x0, x1), (y0, y1))
        ax.tick_params(labelsize=10, colors=text_color)
        ax.grid(color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("0.4")
        if title:
            ax.set_title(title, fontsize=14, fontweight="semibold", color=text_color, loc="left")

        HelperFunctions.add_map_scale_bar(ax, (x0, x1), (y0, y1))
        HelperFunctions.add_north_arrow(ax)

        handles = [Line2D([0], [0], color=point_color, lw=1.6, marker=">", markersize=7,
                          markevery=[1], label="Orientation (true azimuth)")]
        if err is not None:
            handles.append(Patch(facecolor=point_color, alpha=0.35, edgecolor=point_color,
                                 lw=0.7, label="Azimuth ± error"))
        ax.legend(handles=handles, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.1),
                  ncol=len(handles), frameon=True, framealpha=0.95, edgecolor="0.85", fancybox=False)
        HelperFunctions.add_version_watermark(ax)

        if created_fig:
            fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
        if show and created_fig:
            plt.show()
        return fig, ax

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
        """Adaptive Epanechnikov curvigram (kernel density) of azimuths on a
        circular domain. Each azimuth has its own kernel half-width.
        Distances between grid points and azimuths are measured around the
        circle (shortest angular difference, see `HelperFunctions.ang_diff_deg`),
        so a kernel near 0°/360° continues on the other side and a cluster
        of orientations around north forms a single peak.

        Parameters
        ----------
        azimuths : np.ndarray
            The azimuths in degrees
        binwidths : np.ndarray
            Per-sample Epanechnikov half-widths h in degrees (kernel support
            is [x - h, x + h]). These are NOT the user binwidths: convert
            2*error-style binwidths first with
            `HelperFunctions.gaussian_bw_to_epanechnikov_h`, as
            `plot_*_adaptive` does automatically.
        grid_step : float
            Spacing of the evaluation grid in degrees. Default: 0.5
        xlim : tuple[float, float]
            The evaluation range in degrees. Default: (0.0, 360.0). When the
            range spans exactly 360°, the end point is omitted from the grid
            because it is the same direction as the start point.
        left_pad : float
            Extra degrees added below xlim[0]. Default: 0.0
        right_pad : float
            Extra degrees added above xlim[1]. Default: 0.0
        wrap_inputs : bool
            Map input azimuths into [0, 360) first. Default: True
        weights : np.ndarray[float] | None
            Non-negative weight per azimuth (rescaled to sum 1). Default: None
            (equal weights)
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

        if azimuths is None or binwidths is None:
            raise ValueError("`azimuths` and `binwidths` cannot be None.")

        # Accept None and any capitalization ("none", "None", "PDF", ...)
        normalize = "none" if normalize is None else str(normalize).lower()
        if normalize not in ("pdf", "max1", "none"):
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
        # On a full circle the end point (e.g. 360°) is the same direction as
        # the start point (0°), so it is left out to avoid counting it twice.
        full_circle = abs((stop - start) - 360.0) < 1e-9
        if full_circle:
            grid = np.arange(start, stop - 1e-9, grid_step)
        else:
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
            # periodic grid: every point has a full grid step around it
            area = density.sum() * grid_step if full_circle else integrate.trapezoid(density, grid)
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
        null_sampler=None,
        renormalization: str = "auto",
        return_null_expectation: bool = False,
        return_null_stats: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adaptive density of azimuths plus null-model renormalization.

        Wraps `HelperFunctions.distribution_generic_adaptive` with
        `curvigram_fn=self.curvigram_az_adaptive` (circular domain).

        Parameters
        ----------
        azimuths : np.ndarray
            The azimuths in degrees
        binwidths : np.ndarray
            Per-sample Epanechnikov half-widths h in degrees (kernel support
            is [x - h, x + h]). These are NOT the user binwidths: convert
            2*error-style binwidths first with
            `HelperFunctions.gaussian_bw_to_epanechnikov_h`, as
            `plot_*_adaptive` does automatically.
        loc : float
            Start of the uniform null-model range in degrees. Default: 0.0
        scale : float
            Width of the uniform null-model range in degrees. Default: 359.0
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
            The evaluation range in degrees. Default: (0.0, 360.0)
        wrap_inputs : bool
            Map input azimuths into [0, 360) first. Default: True
        normalize : str
            'pdf' (area 1), 'max1' (peak 1) or 'none' (no normalization);
            case-insensitive, None also means 'none'. Default: "pdf"
        null_sampler : Callable | None
            Optional custom null model `sampler(n, rng)` returning n random
            azimuths. Default: None (uniform over [loc, loc + scale])
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
                Azimuth grid in degrees
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
            null_sampler=null_sampler,
            renormalization=renormalization,
            return_null_expectation=return_null_expectation,
            return_null_stats=return_null_stats,
        )
    
    def plot_az_adaptive(self,
        azimuths: np.ndarray[float],
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
        polar: bool = False,
        show_null_model: bool = False,
        show_null_range: bool = False,
        view: str = "excess",
        show_three_sigma: bool = False,
        draw_lunar_lines: bool = True,
        lunar_inclination_deg: float = 5.145,
        test_deficits: bool = False,
        show_p_values: bool = True,
    ) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Plots the curvigram of azimuths with per-point bin widths, either on
        a linear 0-360° axis (default) or, with `polar=True`, on a circle
        (compass view), which shows the circular nature of azimuths: a
        cluster around north appears as one lobe.
        See also: `self.curvigram_az_adaptive` and `self.distribution_az_adaptive`.

        The reference lines (solstices ±ε for rising and setting, equinoxes at
        90°/270°) are computed for a flat horizon at altitude 0° without
        refraction, so they are visual guides only. A real horizon altitude
        shifts the solstice and equinox azimuths of a site. To test
        astronomical targets against a site-specific horizon, convert the
        measurements to declinations and use `Declination.plot_decl_adaptive`,
        whose reference lines (0°, ±ε) do not depend on the horizon.

        Parameters
        ----------
        azimuths : np.ndarray[float]
            The azimuths in degrees
        latitude : float
            Latitude of the site, used for the solstice reference lines
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
            not change the results. Default: None (the full 0-360°)
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
            Draws the flat-horizon (altitude 0°) solstice and equinox azimuths;
            each solstice and lunar standstill line is labelled with its
            azimuth. Default: True
        show_three_sigma : bool
            Draw a dashed line at `three_sigma_level` (only in the 'excess' view
            and the polar plot). It is NOT a significance test: it checks one
            position at a time, so random datasets usually cross it somewhere.
            Significance is shown by the strong fill and the peak markers.
            Default: False
        draw_lunar_lines : bool
            Also draw the Moon's declination limits at the major and minor
            standstills, ±(obliquity + inclination) and ±(obliquity -
            inclination), about ±28.7° and ±18.4° with the defaults (on
            azimuth plots, their flat-horizon rising and setting azimuths). These are
            geocentric values: the Moon's parallax lowers its apparent position
            near the horizon by about 0.9°, so treat them as guides. Default: True
        lunar_inclination_deg : float
            Inclination of the lunar orbit to the ecliptic. Default: 5.145
        test_deficits : bool
            Also test for significant deficits: dips where there are fewer
            structures than the null model explains. Random datasets record their
            lowest point anywhere on the curve, and dips below the
            (100 - `null_percentile`)th percentile of those minima are filled in
            a contrasting colour, marked with a downward triangle and listed in
            `info['deficits_significant']`. Testing both peaks and deficits, each
            at 1%, gives up to about 2% false alarms in total. Requires
            `draw_thresholds=True`. Default: False
        show_p_values : bool
            Write each labelled peak's (and deficit's) p-value under its angle.
            The p-value is the share of random datasets whose most extreme point,
            anywhere on the curve, reaches the observed value: a p-value for the
            whole curve, consistent with the significance test. Default: True
        grid_step : float
            Spacing of the evaluation grid in degrees. Default: 0.5
        wrap_inputs : bool
            Map input azimuths into [0, 360) first. Distances are circular
            anyway, so this does not change the curve. Default: True
        normalize : str
            'pdf' (area 1), 'max1' (peak 1) or 'none' (no normalization);
            case-insensitive, None also means 'none'. Default: "pdf"
        loc : float
            Start of the uniform null-model range in degrees. Default: 0.0
        scale : float
            Width of the uniform null-model range in degrees. Default: 359.0
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
            Minimum distance between reported peaks in degrees. Default: 5.0
        peak_prominence : float | None
            Minimum peak prominence (see scipy.signal.find_peaks). Default: 1.0
        xtick_step : float
            Spacing of the x-axis ticks in degrees. Default: 20.0
        polar : bool
            Azimuths only. Draw the curvigram on a circle (compass view: north
            at the top, azimuths clockwise) instead of a linear 0-360° axis.
            The radius shows the same renormalized frequency as the linear
            plot; the centre is left open, the gray circle marks 0 (the
            null-model mean, values below it lie inside), and the optional
            dashed red circle marks `three_sigma_level`. Reference lines become radial
            lines and significant peaks are labelled radially outside the
            curve. `xlim` and `xtick_step` are ignored (labels every 15°,
            marks every 5°). If
            `ax` is given it must be a polar axes, e.g.
            ``plt.subplot(projection="polar")``. Default: False
        show_null_model : bool
            Also draw the null model's expected curvigram (the average density of
            the random datasets used for the significance test) as a gray dashed
            curve on a second y-axis on the right, in density units. It is the
            average of the `iterations` random datasets, so it becomes smoother
            with more iterations. It shows the
            shape the null model predicts behind the data curve. For azimuths the
            null model is uniform, so this is a flat line.
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
                'three_sigma_level'; 'global_threshold' (significance threshold for
                peaks) and 'global_threshold_percentile'; 'mean_null_std';
                'null_expectation' (the null model's expected density on the grid
                with `show_null_model=True`, else None); 'n_null' (number of random
                datasets behind the threshold and the p-values); 'global_p_value'
                (p-value of the curve's highest point: how often random orientation
                produces a peak this high anywhere on the curve); the peak lists
                'peaks_all' and 'peaks_significant' (dicts with 'angle', 'height',
                'prominence', 'index', 'p_value'); and, with `test_deficits=True`,
                'deficit_threshold', 'deficits_all', 'deficits_significant' (same
                keys, 'height' being the depth of the dip) and
                'global_deficit_p_value' (otherwise None or empty)

        Raises
        ------
        ValueError
            If mandatory arguments are None, `data` and `binwidths` differ in
            length, or values are out of range
        """
        
        if None is azimuths or None is binwidths or None is latitude:
            raise ValueError("`azimuths`, `latitude` and `binwidths` cannot be None.")
            
        if not (-90 <= latitude <= 90):
            raise ValueError("`latitude``should be between [-90,90].")
            
        if len(azimuths) != len(binwidths):
            raise ValueError("`azimuths` and `binwidths` should be of equal lenght")
          
        
        ## Reference lines for a flat horizon (alt = 0), no refraction:
        ## solstices ±ε (rising and setting) and equinoxes.
        ref_lines = []
        
        for eps in (obliquity_deg, -obliquity_deg):
            # Skip solstices on which the sun never rises/sets at this latitude
            if abs(math.sin(math.radians(eps)) / math.cos(math.radians(latitude))) > 1.0:
                continue
            ref_lines.extend(self.calc_az(0.0, float(latitude), float(eps)))  # (east, west)
        ref_lines.extend([90.0, 270.0])

        # Lunar standstill limits: flat-horizon rising and setting azimuths
        lunar_lines = {}
        if draw_lunar_lines:
            for kind, dec_lim in (("major", obliquity_deg + lunar_inclination_deg),
                                  ("minor", obliquity_deg - lunar_inclination_deg)):
                pos = []
                for d in (dec_lim, -dec_lim):
                    if abs(math.sin(math.radians(d)) / math.cos(math.radians(latitude))) > 1.0:
                        continue        # never rises/sets at this latitude
                    pos.extend(self.calc_az(0.0, float(latitude), float(d)))
                lunar_lines[kind] = pos
                
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
                                       xtick_step=xtick_step,
                                       polar=polar,
                                       show_null_model=show_null_model,
                                       null_model_label="Null model (uniform azimuths)",
                                       show_null_range=show_null_range,
                                       view=view,
                                       show_three_sigma=show_three_sigma,
                                       lunar_lines=lunar_lines,
                                       test_deficits=test_deficits,
                                       show_p_values=show_p_values
                                       )
