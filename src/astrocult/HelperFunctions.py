# -*- coding: utf-8 -*-
"""Shared helpers for the curvigram analyses of `Azimuth` and `Declination`.

- Angles: `ang_diff_deg`, the shortest signed angular difference.
- Kernels: `gaussian_bw_to_epanechnikov_h` converts user binwidths
  (bw = 2*sigma) into Epanechnikov half-widths with the same variance.
- Null model: `distribution_generic_adaptive` renormalizes a curvigram
  against random datasets, and `estimate_null_max_threshold_adaptive`
  gives the percentile threshold used to call peaks significant. Both draw
  uniform samples by default or use a `null_sampler` (see
  `Declination.make_null_sampler` for declinations).
- Peaks: `find_circular_peaks` (azimuths, wraps at 0°/360°) and
  `find_linear_peaks` (declinations).
- Plotting: `plot_curvigram` draws the renormalized curve, fills the
  regions that pass the significance test, draws the reference lines
  (equinoxes dotted, solstices solid), labels the significant peaks,
  optionally shows the null model and its range, and adds the "astrocult v<version>"
  label above the top-right corner of the plot (`add_version_watermark`;
  the version comes from `get_astrocult_version`). The legend and the
  version label sit outside the plot area so they never hide data or peak
  labels; peak labels near the left/right edge are aligned inwards.
  Azimuth curvigrams can also be drawn on a circle (compass view) with
  `polar=True`.
- Reference lines: equinoxes dotted, solstices solid, lunar standstills
  dashed (major) and dash-dotted (minor), see `LUNAR_LINE_STYLES`.
- Maps: `add_map_scale_bar`, `add_north_arrow` and `format_map_ticks`, used by
  `GeoUtils.plot_error_map` and `Azimuth.plot_orientation_map`.
"""
import numpy as np
from typing import Callable
from numpy.random import default_rng, Generator
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import functools
import os
import tomllib
from importlib.metadata import version as _pkg_version, PackageNotFoundError


@functools.lru_cache(maxsize=1)
def get_astrocult_version() -> str:
    """Returns the astrocult version.

    Uses the metadata of the installed package (taken from `pyproject.toml`
    at build/install time). When running from a source checkout that is not
    installed, it falls back to reading `pyproject.toml` from this folder or
    up to two parent folders.

    Returns
    -------
    str
        The version string, e.g. "0.0.3", or "unknown" if it cannot be found.
    """
    try:
        return _pkg_version("astrocult")
    except PackageNotFoundError:
        pass

    folder = os.path.dirname(os.path.abspath(__file__))
    for _ in range(3):
        candidate = os.path.join(folder, "pyproject.toml")
        if os.path.isfile(candidate):
            try:
                with open(candidate, "rb") as f:
                    return str(tomllib.load(f)["project"]["version"])
            except (OSError, KeyError, tomllib.TOMLDecodeError):
                break
        folder = os.path.dirname(folder)

    return "unknown"


LUNAR_LINE_STYLES = {
    "major": dict(color="#4a6fa5", ls=(0, (6, 3)), lw=1.1),
    "minor": dict(color="#4a6fa5", ls=(0, (6, 2, 1.5, 2)), lw=1.0),
}


def _azimuth_line_labels(ref_lines, lunar_lines, ref_color: str) -> list:
    """(azimuth, colour) of the solstice and lunar standstill lines of an
    azimuth plot, i.e. every reference line except the equinoxes."""
    out = [(float(p), ref_color) for p in ref_lines
           if not (np.isclose(p, 90.0) or np.isclose(p, 270.0))]
    for kind, positions in (lunar_lines or {}).items():
        out += [(float(p), LUNAR_LINE_STYLES[kind]["color"]) for p in positions]
    return out


def add_map_scale_bar(ax: plt.Axes, lon_range: tuple[float, float],
                      lat_range: tuple[float, float]) -> float:
    """Draws a kilometre scale bar in the lower-left corner of a lat/lon map.
    The bar length is a round number (0.1 ... 500 km) of about a fifth of
    the map width.

    Parameters
    ----------
    ax : plt.Axes
        The map axes (x = longitude, y = latitude, in degrees)
    lon_range, lat_range : tuple[float, float]
        The longitude and latitude extent of the map in degrees

    Returns
    -------
    float
        The bar length in km
    """
    text_color = "0.15"
    lon0, lon1 = min(lon_range), max(lon_range)
    lat0, lat1 = min(lat_range), max(lat_range)
    mean_lat = (lat0 + lat1) / 2.0
    km_per_deg_lon = 111.32 * np.cos(np.radians(mean_lat))
    width_km = (lon1 - lon0) * km_per_deg_lon
    nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    bar_km = max([n for n in nice if n <= width_km / 5] or [nice[0]])
    x0 = lon0 + 0.04 * (lon1 - lon0)
    y0 = lat0 + 0.05 * (lat1 - lat0)
    x1 = x0 + bar_km / km_per_deg_lon
    ax.plot([x0, x1], [y0, y0], color="black", lw=5, solid_capstyle="butt", zorder=7)
    ax.plot([x0, x1], [y0, y0], color="white", lw=2.5, solid_capstyle="butt", zorder=8)
    ax.annotate(f"{bar_km:g} km", ((x0 + x1) / 2, y0), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, color=text_color, zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    return bar_km


def format_map_ticks(ax: plt.Axes, lon_range: tuple[float, float],
                     lat_range: tuple[float, float]) -> None:
    """Formats the longitude/latitude tick labels of a map with enough
    decimals for its extent (at least 2), so small maps do not show
    repeated labels.

    Parameters
    ----------
    ax : plt.Axes
        The map axes
    lon_range, lat_range : tuple[float, float]
        The longitude and latitude extent of the map in degrees
    """
    from matplotlib.ticker import FormatStrFormatter
    span = min(abs(lon_range[1] - lon_range[0]), abs(lat_range[1] - lat_range[0]))
    step = span / 5.0 if span > 0 else 0.01
    decimals = max(2, int(np.ceil(-np.log10(step))))
    ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


def add_north_arrow(ax: plt.Axes) -> None:
    """Draws a north arrow in the upper-left corner of a map (north = up).

    Parameters
    ----------
    ax : plt.Axes
        The map axes
    """
    import matplotlib.patheffects as patheffects
    text_color = "0.15"
    arrow = ax.annotate("", xy=(0.05, 0.96), xytext=(0.05, 0.86), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=text_color, lw=1.6,
                                        mutation_scale=14), zorder=8)
    arrow.arrow_patch.set_path_effects([patheffects.withStroke(linewidth=4, foreground="white")])
    ax.annotate("N", xy=(0.05, 0.835), xycoords="axes fraction", ha="center", va="center",
                fontsize=12, fontweight="bold", color=text_color, zorder=8,
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none", alpha=0.85))


def add_version_watermark(ax: plt.Axes) -> None:
    """Writes "astrocult v<version>" just above the top-right corner of
    `ax`, outside the plot area, so it never covers data, peak labels or
    the legend. Plot titles are left-aligned, so the two do not collide.
    The label is included when the figure is saved.

    Parameters
    ----------
    ax : plt.Axes
        The axes to annotate
    """
    ax.text(1.0, 1.01, f"astrocult v{get_astrocult_version()}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="0.45", clip_on=False, zorder=10)


def ang_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest signed angular difference a - b in degrees, wrapped to (-180, 180].
    Works with scalars and arrays (broadcasting).

    This is the basic operation of circular statistics: the difference is
    measured the short way around the circle, e.g. ang_diff_deg(1, 359) = 2
    and ang_diff_deg(359, 1) = -2, not -358 and 358. It is used for the
    azimuth kernel distances (`Azimuth.curvigram_az_adaptive`) and for the
    spread of azimuth readings (`Azimuth.err_az`).
    
    Parameters
    ----------
    a : np.ndarray
        
    b : np.ndarray
        
    
    Returns
    -------
    np.ndarray


    Raises
    ------
    ValueError
        If a None value has been provided in any of the arguments
    """
    
    if a is None or b is None:
        raise ValueError("Arguments cannot be None.")
    
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = (a - b + 180.0) % 360.0 - 180.0
    
    return np.where(d == -180.0, 180.0, d)


def gaussian_bw_to_epanechnikov_h(bw_gauss_like : np.ndarray[float]) -> np.ndarray[float]:
    """
    Convert user binwidths into the half-width ``h`` of an Epanechnikov kernel
    with the same variance as a Gaussian of standard deviation bw/2.

    A binwidth is interpreted as a full Gaussian width, bw = 2*sigma. The
    recommended choice is twice the measurement error of each point, e.g.
    ``bw = 2 * Declination.err_decl(...)`` or ``bw = 2 * Azimuth.err_az(...)``,
    but any positive binwidth can be used: larger values smooth more, smaller
    values resolve finer structure.

    An Epanechnikov kernel K(u) = 3/4 (1 - u^2) on [-h, h] has variance h^2/5,
    so matching sigma^2 = (bw/2)^2 gives h = sqrt(5) * sigma = (sqrt(5) / 2) * bw.
    
    Parameters
    ----------
    bw_gauss_like : np.ndarray
        The binwidths in degrees (bw = 2*sigma)
    
    Returns
    -------
    np.ndarray
        The Epanechnikov half-widths h in degrees

    Raises
    ------
    ValueError
        If a None value has been provided as an argument
    """
    
    if bw_gauss_like is None:
        raise ValueError("Argument cannot be None")
    
    return (np.sqrt(5.0) / 2.0) * np.asarray(bw_gauss_like, dtype=float)
 
def _resolve_renormalization(renormalization: str, null_sampler) -> str:
    """Returns 'expected' or 'mean' for a `renormalization` setting."""
    mode = str(renormalization).lower()
    if mode not in ("auto", "expected", "mean"):
        raise ValueError("`renormalization` must be 'auto', 'expected' or 'mean'.")
    if mode == "auto":
        mode = "expected" if null_sampler is not None else "mean"
    return mode


def distribution_generic_adaptive(
    data: np.ndarray[float],
    binwidths: np.ndarray[float],
    *,
    curvigram_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    iterations: int = 100,
    title: str = "",
    n_uniform: int = 80_000,
    rng: Generator | None = None,
    grid_step: float = 0.5,
    xlim: tuple[float, float] = (0.0, 360.0),
    wrap_inputs: bool | None = None,  # only relevant for azimuth
    normalize: str = "pdf",
    loc: float = 0.0,
    scale: float = 359.0,
    null_sampler: Callable[[int, Generator], np.ndarray] | None = None,
    renormalization: str = "auto",
    return_null_expectation: bool = False,
    return_null_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generic adaptive density + null-model renormalization.

    Computes the density of `data` with `curvigram_fn` and rescales it
    against `iterations` random datasets of the same size (the null model):
    (density - mean(density)) / s_null with `renormalization='mean'`, where
    s_null is the mean standard deviation of the null densities, or
    (density - E_null(x)) / s_null with `renormalization='expected'`, where
    E_null(x) is the null model's expected density at each position and
    s_null the typical point-wise null standard deviation. Works with either the circular (azimuth) or the
    linear (declination) curvigram.

    Null model: by default random samples are drawn uniformly over
    [loc, loc + scale]. For azimuths (loc=0, scale close to 360) this is the
    uniform distribution on the circle, i.e. no preferred orientation. For
    declinations a uniform distribution is NOT what random orientations
    produce (the conversion depends on latitude and horizon, which limits
    the range and piles declinations up at its edges), so `Declination`
    passes a `null_sampler` built by `Declination.make_null_sampler`
    instead (random azimuths converted to declinations by default).

    Parameters
    ----------
    data : Sequence
        The input data in degrees
    binwidths : Sequence
        Per-sample Epanechnikov half-widths h in degrees (kernel support
        is [x - h, x + h]). These are NOT the user binwidths: convert
        2*error-style binwidths first with
        `HelperFunctions.gaussian_bw_to_epanechnikov_h`, as
        `plot_*_adaptive` does automatically.
    curvigram_fn : Callable
        The curvigram function: `Azimuth.curvigram_az_adaptive` or
        `Declination.curvigram_decl_adaptive`
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
    wrap_inputs : bool | None
        Passed to `curvigram_fn` only if not None (azimuths only).
        Default: None
    normalize : str
        'pdf' (area 1), 'max1' (peak 1) or 'none' (no normalization);
        case-insensitive, None also means 'none'. Default: "pdf"
    loc : float
        Start of the uniform null-model range in degrees. Default: 0.0
    scale : float
        Width of the uniform null-model range in degrees. Default: 359.0
    null_sampler : Callable[[int, Generator], np.ndarray] | None
        Function returning one random dataset of the given size, used as the
        null model. None draws uniformly over [loc, loc + scale]. See
        `Declination.make_null_sampler` for the declination null models.
        Default: None
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
        the renormalized null curves ('excess_lo', 'excess_hi'), and the
        mean renormalized null curve ('excess_expected'), as a 4th
        value. Takes precedence over `return_null_expectation`. Default: False

    Returns
    -------
    tuple
        grid : np.ndarray, shape (M, 1)
            Evaluation grid returned by `curvigram_fn`
        normed_density : np.ndarray, shape (M,)
            Density of the data
        renorm_density : np.ndarray, shape (M,)
            Renormalized density (see above)
        null_expectation : np.ndarray, shape (M,)
            Only with `return_null_expectation=True`: the null model's
            expected density on the grid

    Raises
    ------
    ValueError
        If a None value has been provided as an argument, or values are
        out of range
    """
    
    if data is  None or binwidths is None or curvigram_fn is None:
        raise ValueError("`data`, `binwidths` and `curvigram_fn` cannot be None")
    
    x = np.asarray(data, dtype=float).ravel()
    h = np.asarray(binwidths, dtype=float).ravel()

    size = int(x.size)
    if size == 0:
        raise ValueError("`data` must contain at least one value.")
    if h.size != size:
        raise ValueError("`binwidth` must have the same length as `data`.")
    if not np.all(h > 0):
        raise ValueError("All entries in `binwidth` must be positive.")
    if scale <= 0:
        raise ValueError("`scale` must be positive.")
    if iterations <= 0:
        raise ValueError("`iterations` must be >= 1.")
    if grid_step <= 0:
        raise ValueError("`grid_step` must be positive.")

    rng = default_rng() if rng is None else rng

    n_uniform = max(int(n_uniform), size)
    uniform_grid = np.linspace(loc, loc + scale, n_uniform, endpoint=True)

    mode = _resolve_renormalization(renormalization, null_sampler)
    null_stds = np.empty(iterations, dtype=float)
    null_curves = []
    for i in range(iterations):
        if null_sampler is not None:
            dist_sample = np.asarray(null_sampler(size, rng), dtype=float)
        else:
            dist_sample = rng.choice(uniform_grid, size=size, replace=False)
        if wrap_inputs is None:
            _, density_i = curvigram_fn(
                dist_sample,
                h,
                grid_step=grid_step,
                xlim=xlim,
                normalize=normalize,
            )
        else:
            _, density_i = curvigram_fn(
                dist_sample,
                h,
                grid_step=grid_step,
                xlim=xlim,
                wrap_inputs=wrap_inputs,
                normalize=normalize,
            )
        null_stds[i] = float(np.std(density_i))
        null_curves.append(np.asarray(density_i, dtype=float))

    mean_null_std = float(np.mean(null_stds))
    null_curves = np.asarray(null_curves)
    null_expectation = null_curves.mean(axis=0)
    if mode == "expected":
        mean_null_std = float(np.sqrt(null_curves.var(axis=0).mean()))

    if wrap_inputs is None:
        new_x, normed_density = curvigram_fn(
            x,
            h,
            grid_step=grid_step,
            xlim=xlim,
            normalize=normalize,
        )
    else:
        new_x, normed_density = curvigram_fn(
            x,
            h,
            grid_step=grid_step,
            xlim=xlim,
            wrap_inputs=wrap_inputs,
            normalize=normalize,
        )
    normed_density = np.asarray(normed_density, dtype=float)

    eps = np.finfo(float).eps
    denom = mean_null_std if mean_null_std > eps else 1.0
    if mode == "expected":
        renorm_density = (normed_density - null_expectation) / denom
    else:
        renorm_density = (normed_density - np.mean(normed_density)) / denom

    if return_null_stats:
        if mode == "expected":
            z_null = (null_curves - null_expectation) / denom
        else:
            z_null = (null_curves - null_curves.mean(axis=1, keepdims=True)) / denom
        stats = {
            "expectation": null_expectation,
            "scale": denom,
            "mode": mode,
            "density_lo": np.percentile(null_curves, 1, axis=0),
            "density_hi": np.percentile(null_curves, 99, axis=0),
            "excess_lo": np.percentile(z_null, 1, axis=0),
            "excess_hi": np.percentile(z_null, 99, axis=0),
            "excess_expected": z_null.mean(axis=0),
        }
        return np.asarray(new_x), normed_density, renorm_density, stats
    if return_null_expectation:
        return np.asarray(new_x), normed_density, renorm_density, null_expectation
    return np.asarray(new_x), normed_density, renorm_density    

# ---------------------------------------------------------------------
# Global-null percentile threshold for maxima (adaptive)
# ---------------------------------------------------------------------

def estimate_null_max_threshold_adaptive(
    n_samples: int,
    h_epa: np.ndarray,
    *,
    #domain: str = "az",                   # "az" (circular) or "decl" (linear)
    curvigram_fn : Callable[..., tuple[np.ndarray, np.ndarray]],        
    percentile: float = 99.0,
    iterations: int = 500,
    n_uniform: int = 80_000,
    loc: float = 0.0,
    scale: float = 359.0,
    grid_step: float = 0.5,
    xlim: tuple[float, float] = (0.0, 360.0),
    wrap_inputs: bool = True,
    normalize: str = "pdf",
    rng: Generator | None = None,
    null_sampler: Callable[[int, Generator], np.ndarray] | None = None,
    renormalization: str = "auto",
):
    """Estimates the significance threshold for peaks of renormalized curves.

    Pass 1 computes the mean standard deviation of `iterations` null-model
    densities (random samples of size `n_samples`, drawn uniformly or with
    `null_sampler`). Pass 2 draws
    new null samples, renormalizes them with that value, and records the
    maximum of each curve. The threshold is the `percentile` of those
    maxima: a peak in the real data higher than this would appear in fewer
    than (100 - percentile)% of random datasets.

    Parameters
    ----------
    n_samples : int
        Number of data points (size of each null sample)
    h_epa : np.ndarray
        Per-sample Epanechnikov half-widths h in degrees (kernel support
        is [x - h, x + h]). These are NOT the user binwidths: convert
        2*error-style binwidths first with
        `HelperFunctions.gaussian_bw_to_epanechnikov_h`, as
        `plot_*_adaptive` does automatically.
    curvigram_fn : Callable
        The curvigram function: `Azimuth.curvigram_az_adaptive` or
        `Declination.curvigram_decl_adaptive`
    percentile : float
        Percentile of the null maxima used as threshold. Default: 99.0
    iterations : int
        Number of null samples in each pass. Default: 500
    n_uniform : int
        Size of the grid the null-model samples are drawn from. Default: 80_000
    loc : float
        Start of the uniform null-model range in degrees. Default: 0.0
    scale : float
        Width of the uniform null-model range in degrees. Default: 359.0
    grid_step : float
        Spacing of the evaluation grid in degrees. Default: 0.5
    xlim : tuple[float, float]
        The evaluation range in degrees. Default: (0.0, 360.0)
    wrap_inputs : bool
        Passed to the azimuth curvigram only. Default: True
    normalize : str
        'pdf' (area 1), 'max1' (peak 1) or 'none' (no normalization);
        case-insensitive, None also means 'none'. Default: "pdf"
    rng : Generator | None
        Random generator; pass one for reproducible results. Default: None
    null_sampler : Callable[[int, Generator], np.ndarray] | None
        Function returning one random dataset of the given size, used as the
        null model. None draws uniformly over [loc, loc + scale]. See
        `Declination.make_null_sampler` for the declination null models.
        Default: None
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

    Returns
    -------
    tuple
        threshold : float
            The `percentile` of the null maxima
        maxima : np.ndarray, shape (iterations,)
            Maximum of each renormalized null curve
        mean_null_std : float
            The null-model scale used for renormalization (pass 1): mean
            standard deviation of the null densities ('mean'), or typical
            point-wise standard deviation ('expected')

    Raises
    ------
    ValueError
        If a None value has been provided as an argument, or `h_epa` does
        not have `n_samples` values
    """
    
    if h_epa is None or curvigram_fn is None:
        raise ValueError("`h_epa` and `curvigram_h` cannot be None.")

    rng = default_rng() if rng is None else rng
    n_uniform = max(int(n_uniform), int(n_samples))
    uniform_grid = np.linspace(loc, loc + scale, n_uniform, endpoint=True)

    h_epa = np.asarray(h_epa, dtype=float).ravel()
    if h_epa.size != int(n_samples):
        raise ValueError("`h_epa` must have same length as `n_samples`.")

    curv = curvigram_fn


    mode = _resolve_renormalization(renormalization, null_sampler)

    # Pass 1: estimate mean std (and, for 'expected', the mean curve) of null densities
    stds = np.empty(iterations, dtype=float)
    curves1 = []
    for i in range(iterations):
        null_sample = (np.asarray(null_sampler(n_samples, rng), dtype=float) if null_sampler is not None
                       else rng.choice(uniform_grid, size=n_samples, replace=False))
        #if domain == "az":
        if curv.__name__ == "curvigram_az_adaptive":
            _, dens = curv(
                azimuths=null_sample,
                binwidths=h_epa,
                grid_step=grid_step,
                xlim=xlim,
                wrap_inputs=wrap_inputs,
                normalize=normalize,
            )
        else:
            _, dens = curv(
                declinations=null_sample,
                binwidths=h_epa,
                grid_step=grid_step,
                xlim=xlim,
                normalize=normalize,
            )
        stds[i] = float(np.std(dens))
        curves1.append(np.asarray(dens, dtype=float))
    mean_null_std = float(np.mean(stds))
    if mode == "expected":
        curves1 = np.asarray(curves1)
        null_expectation = curves1.mean(axis=0)
        mean_null_std = float(np.sqrt(curves1.var(axis=0).mean()))
    denom = mean_null_std if mean_null_std > np.finfo(float).eps else 1.0

    # Pass 2: maxima of renormalized null curves
    maxima = np.empty(iterations, dtype=float)
    for i in range(iterations):
        null_sample = (np.asarray(null_sampler(n_samples, rng), dtype=float) if null_sampler is not None
                       else rng.choice(uniform_grid, size=n_samples, replace=False))
        #if domain == "az":
        if curv.__name__ == "curvigram_az_adaptive":
            _, dens = curv(
                azimuths=null_sample,
                binwidths=h_epa,
                grid_step=grid_step,
                xlim=xlim,
                wrap_inputs=wrap_inputs,
                normalize=normalize,
            )
        else:
            _, dens = curv(
                declinations=null_sample,
                binwidths=h_epa,
                grid_step=grid_step,
                xlim=xlim,
                normalize=normalize,
            )
        z = ((dens - null_expectation) if mode == "expected" else (dens - np.mean(dens))) / denom
        maxima[i] = float(np.max(z))

    threshold = float(np.percentile(maxima, percentile))
    return threshold, maxima, mean_null_std

def find_circular_peaks(
    x_deg: np.ndarray,
    y: np.ndarray,
    *,
    min_separation_deg: float = 5.0,
    prominence: float | None = None,
    height: float | None = None,
    use_scipy: bool = True,
):
    """Peak detection on a circular domain [0,360). Used for azimuths.

    The curve is treated as periodic: it is tiled three times (x - 360, x,
    x + 360) before searching, so a peak at 0°/360° is found once, with its
    full height and prominence, instead of being cut in two at the edges.
            
    Parameters
    ----------
    x_deg : np.ndarray,
        Azimuth angles
    y : np.ndarray,
    min_separation_deg : float
        Default: 5
    prominence : float | None
    height : float | None
    use_scipy : bool
        Default: true    
        
    Returns
    -------
    List
        A list of dicts. Each dict has: angle, height, prominence, index.
        
    Raises
    ------
    ValueError
        If a None value has been provided as an argument
    """
    
    if x_deg is None or y is None:
        raise ValueError("`x_deg` and `y` must not be None.")
    
    x = np.asarray(x_deg, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("`x_deg` and `y` must have the same length.")
    if x.size < 3:
        return []

    step = np.median(np.diff(x))
    if step <= 0:
        raise ValueError("`x_deg` must be strictly increasing.")
    min_distance_samples = max(1, int(round(min_separation_deg / step)))

    x_ext = np.concatenate([x - 360.0, x, x + 360.0])
    y_ext = np.concatenate([y, y, y])

    peaks_out = []

    if use_scipy:
        try:
            from scipy.signal import find_peaks, peak_prominences
            idx_ext, props = find_peaks(
                y_ext,
                distance=min_distance_samples,
                prominence=prominence,
                height=height,
            )
            n = len(x)
            mask_mid = (idx_ext >= n) & (idx_ext < 2 * n)
            idx_mid = idx_ext[mask_mid]
            heights = props.get("peak_heights", y_ext[idx_ext])[mask_mid]
            if "prominences" in props:
                proms = props["prominences"][mask_mid]
            else:
                proms = peak_prominences(y_ext, idx_ext)[0][mask_mid]
            for i_ext, h, p in zip(idx_mid, heights, proms):
                i = i_ext - n
                angle = (x[i] % 360.0 + 360.0) % 360.0
                peaks_out.append({"angle": float(angle), "height": float(h), "prominence": float(p), "index": int(i)})
            peaks_out.sort(key=lambda d: d["angle"])
            return peaks_out
        except Exception:
            use_scipy = False

    # Fallback simple maxima with spacing filter
    y_pad = np.r_[y[-1], y, y[0]]
    loc_max = (y_pad[1:-1] > y_pad[:-2]) & (y_pad[1:-1] > y_pad[2:])
    cand_idx = np.where(loc_max)[0]
    if cand_idx.size:
        order = np.argsort(y[cand_idx])[::-1]
        chosen = []
        taken = np.zeros_like(y, dtype=bool)
        for j in order:
            i = cand_idx[j]
            if taken[max(0, i - min_distance_samples): i + min_distance_samples + 1].any():
                continue
            chosen.append(i)
            taken[i] = True
        cand_idx = np.array(sorted(chosen), dtype=int)
    for i in cand_idx:
        angle = (x[i] % 360.0 + 360.0) % 360.0
        peaks_out.append({"angle": float(angle), "height": float(y[i]), "prominence": None, "index": int(i)})
    peaks_out.sort(key=lambda d: d["angle"])
    
    return peaks_out


def find_linear_peaks(
    x_deg: np.ndarray,
    y: np.ndarray,
    *,
    min_separation_deg: float = 2.0,
    prominence: float | None = None,
    height: float | None = None,
    use_scipy: bool = True,
):
    """Peak detection on a linear axis. Used for declinations.

    Parameters
    ----------
    x_deg : np.ndarray
        Strictly increasing positions in degrees (e.g. declinations)
    y : np.ndarray
        Curve values at `x_deg`
    min_separation_deg : float
        Minimum distance between peaks in degrees. Default: 2.0
    prominence : float | None
        Minimum peak prominence (see scipy.signal.find_peaks). Default: None
    height : float | None
        Minimum peak height. Default: None
    use_scipy : bool
        Use scipy.signal.find_peaks; a simple local-maximum search is used
        if False or if scipy fails. Default: True

    Returns
    -------
    list
        One dict per peak, sorted by position, with keys 'angle' (position
        in degrees), 'height', 'prominence' (None with the simple search)
        and 'index'.

    Raises
    ------
    ValueError
        If `x_deg` and `y` differ in length or `x_deg` is not increasing
    """
    
    x = np.asarray(x_deg, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if x.size < 3:
        return []

    step = np.median(np.diff(x))
    if step <= 0:
        raise ValueError("x must be strictly increasing.")
    min_distance_samples = max(1, int(round(min_separation_deg / step)))

    peaks_out = []

    if use_scipy:
        try:
            from scipy.signal import find_peaks, peak_prominences
            idx, props = find_peaks(
                y,
                distance=min_distance_samples,
                prominence=prominence,
                height=height,
            )
            heights = props.get("peak_heights", y[idx])
            if "prominences" in props:
                proms = props["prominences"]
            else:
                proms = peak_prominences(y, idx)[0]
            for i, h, p in zip(idx, heights, proms):
                peaks_out.append({"angle": float(x[i]), "height": float(h), "prominence": float(p), "index": int(i)})
            peaks_out.sort(key=lambda d: d["angle"])
            return peaks_out
        except Exception:
            use_scipy = False

    # Fallback simple maxima
    loc_max = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    cand_idx = np.where(loc_max)[0] + 1
    if cand_idx.size:
        order = np.argsort(y[cand_idx])[::-1]
        chosen = []
        taken = np.zeros_like(y, dtype=bool)
        for j in order:
            i = cand_idx[j]
            if taken[max(0, i - min_distance_samples): i + min_distance_samples + 1].any():
                continue
            chosen.append(i)
            taken[i] = True
        cand_idx = np.array(sorted(chosen), dtype=int)
    for i in cand_idx:
        peaks_out.append({"angle": float(x[i]), "height": float(y[i]), "prominence": None, "index": int(i)})
    peaks_out.sort(key=lambda d: d["angle"])
    
    return peaks_out


def _draw_curvigram_polar(
    theta_deg: np.ndarray,
    y: np.ndarray,
    *,
    ax: plt.Axes | None,
    ref_lines,
    title: str,
    label: str | None,
    line_color: str,
    fill_color: str,
    fill_alpha: float,
    three_sigma_level: float,
    draw_reference_lines: bool,
    peaks_sig: list,
    significant: np.ndarray | None = None,
    null_band: tuple[np.ndarray, np.ndarray] | None = None,
    show_three_sigma: bool = False,
    lunar_lines: dict | None = None,
) -> tuple[plt.Figure, plt.Axes, bool]:
    """Draws an azimuth curvigram on a polar (compass) axes. Used by
    `plot_curvigram(..., polar=True)`; see its `polar` parameter.

    Returns
    -------
    tuple
        fig, ax, and whether the figure was created here
    """

    ref_color = "0.35"
    sigma_color = "#c0392b"
    text_color = "0.15"

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=(8.5, 6.8))
        ax = fig.add_subplot(projection="polar")
    else:
        if getattr(ax, "name", "") != "polar":
            raise ValueError("With polar=True, `ax` must be a polar axes, "
                             "e.g. plt.subplot(projection='polar').")
        fig = ax.figure

    # Compass orientation: north at the top, azimuths increase clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_deg = np.asarray(theta_deg, dtype=float)
    y = np.asarray(y, dtype=float)
    # close the circle (the full-circle grid omits 360° = 0°)
    th = np.radians(np.append(theta_deg, theta_deg[0] + 360.0))
    yy = np.append(y, y[0])
    circle = np.linspace(0, 2 * np.pi, 361)

    y_min = float(np.nanmin(y)) if y.size else 0.0
    y_max = float(np.nanmax(y)) if y.size else 1.0
    span = max(y_max - y_min, 1.0)
    has_line_labels = draw_reference_lines and bool(_azimuth_line_labels(ref_lines, lunar_lines, ref_color))
    # room for peak labels, plus a band near the rim for the line labels
    r_lo, r_hi = y_min - 0.05 * span, y_max + (0.42 if has_line_labels else 0.32) * span
    ax.set_rorigin(y_min - 0.35 * span)                      # open centre
    ax.set_ylim(r_lo, r_hi)

    if null_band is not None:
        lo, hi = (np.append(np.asarray(b, dtype=float), b[0]) for b in null_band)
        ax.fill_between(th, lo, hi, color="0.6", alpha=0.22, lw=0, zorder=0.5,
                        label="Random datasets: expected to 99th percentile")
    ax.plot(circle, np.zeros_like(circle), color="0.6", lw=0.8, zorder=1)
    # faint fill above the null level, strong fill only where significant
    ax.fill_between(th, 0, yy, where=yy > 0, interpolate=True,
                    alpha=fill_alpha * 0.35, facecolor=fill_color, edgecolor="none", zorder=2)
    if significant is not None and np.any(significant):
        sig = np.append(np.asarray(significant, dtype=bool), bool(significant[0]))
        ax.fill_between(th, 0, yy, where=sig, alpha=min(1.0, fill_alpha * 2),
                        facecolor=fill_color, edgecolor="none", zorder=2.1, label="Significant")
    ax.plot(th, yy, color=line_color, lw=1.6, zorder=3, label=label if label else None)

    if show_three_sigma:
        ax.plot(circle, np.full_like(circle, three_sigma_level), color=sigma_color, lw=1.1,
                ls=(0, (5, 4)), alpha=0.85, zorder=1.5, label=r'3$\sigma$')
    if draw_reference_lines:
        labelled = set()
        for pos in ref_lines:
            is_equinox = bool(np.isclose(pos, 90.0) or np.isclose(pos, 270.0))
            kind = "equinox" if is_equinox else "solstice"
            ax.plot([np.radians(pos)] * 2, [r_lo, r_hi], color=ref_color,
                    ls=(0, (1.5, 2.5)) if is_equinox else "-", lw=1.2 if is_equinox else 1.0,
                    alpha=0.9, zorder=1.5,
                    label=None if kind in labelled else
                          ("Equinoxes (flat horizon)" if is_equinox else "Solstices (flat horizon)"))
            labelled.add(kind)
        for kind, positions in (lunar_lines or {}).items():
            for k, pos in enumerate(positions):
                ax.plot([np.radians(pos)] * 2, [r_lo, r_hi], alpha=0.9, zorder=1.5,
                        label=f"{kind.capitalize()} lunar standstill (flat horizon)" if k == 0 else None,
                        **LUNAR_LINE_STYLES[kind])
        # flat-horizon azimuth written on each solstice and lunar line, near the rim
        r_txt = r_lo + 0.95 * (r_hi - r_lo)
        for pos, col in _azimuth_line_labels(ref_lines, lunar_lines, ref_color):
            angle = (90.0 - pos) % 360.0                   # screen direction of the line
            if 90.0 < angle < 270.0:
                angle -= 180.0                             # keep the text upright
            ax.text(np.radians(pos), r_txt, f"{pos:.1f}°", rotation=angle, rotation_mode="anchor",
                    ha="center", va="center", fontsize=8, color=col, zorder=4.5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    for p in peaks_sig:
        t = np.radians(p["angle"])
        ax.plot(t, p["height"], marker="o", ms=5.5, mfc="white", mec=line_color, mew=1.6, zorder=4)
        ax.annotate(f"{p['angle']:.1f}°", xy=(t, p["height"]),
                    xytext=(t, p["height"] + 0.11 * span), textcoords="data",
                    ha="center", va="center", fontsize=10, color=text_color, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.6, alpha=0.9))

    # Compass labels every 15° (N/E/S/W), tick marks every 5°, light grid,
    # small radial labels
    ticks = np.arange(0, 360, 15)
    names = {0: "N", 90: "E", 180: "S", 270: "W"}
    ax.set_thetagrids(ticks, [names.get(int(t), f"{int(t)}°") for t in ticks])
    for t in np.arange(0, 360, 5):                          # 5° marks at the rim
        tick = 0.035 if t % 15 == 0 else 0.02
        ax.plot([np.radians(t)] * 2, [r_hi - tick * (r_hi - r_lo), r_hi], color="0.45",
                lw=1.0 if t % 15 == 0 else 0.7, zorder=1.6, clip_on=False)
    ax.tick_params(axis="x", labelsize=10, colors=text_color, pad=4)
    ax.tick_params(axis="y", labelsize=8, colors="0.4")
    ax.set_rlabel_position(15)
    ax.grid(color="0.85", lw=0.7)
    ax.spines["polar"].set_color("0.6")
    if "inner" in ax.spines:
        ax.spines["inner"].set_color("0.6")
    ax.set_xlabel("Radius: normalized relative frequency", fontsize=11, color=text_color, labelpad=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", color=text_color, loc="left", pad=22)

    if label or draw_reference_lines or show_three_sigma or lunar_lines or null_band is not None or (significant is not None and np.any(significant)):
        ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.08, 1.0),
                  borderaxespad=0.0, frameon=True, framealpha=0.92,
                  edgecolor="0.85", fancybox=False, borderpad=0.6)

    add_version_watermark(ax)
    return fig, ax, created_fig


def plot_curvigram(
    data: np.ndarray[float],
    latitude: float,
    binwidths: np.ndarray[float],
    curvigram_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    distribution_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    ref_lines: np.ndarray[float] | None = None,
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
    null_sampler: Callable[[int, Generator], np.ndarray] | None = None,
    distribution_kwargs: dict | None = None,
    renormalization: str = "auto",
    show_null_model: bool = False,
    null_model_label: str = "Null model",
    show_null_range: bool = False,
    view: str = "excess",
    show_three_sigma: bool = False,
    xpad_deg: float = 10.0,
    lunar_lines: dict | None = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Plots a renormalized adaptive curvigram of azimuths or declinations.
    Used by `Azimuth.plot_az_adaptive` and `Declination.plot_decl_adaptive`,
    which supply the domain-specific functions and reference lines.

    Parameters
    ----------
    data : np.ndarray[float]
        The azimuths or declinations in degrees
    latitude : float | None
        Latitude of the site; required for azimuths, None for declinations
    binwidths : np.ndarray[float]
        Per-point binwidths in degrees, interpreted as bw = 2*sigma.
        Recommended: twice the measurement error of each point, e.g.
        ``2 * Declination.err_decl(...)``. Any positive values are
        accepted; they are converted internally with
        `HelperFunctions.gaussian_bw_to_epanechnikov_h`.
    curvigram_fn : Callable
        `Azimuth.curvigram_az_adaptive` or `Declination.curvigram_decl_adaptive`
    distribution_fn : Callable
        `Azimuth.distribution_az_adaptive` or `Declination.distribution_decl_adaptive`
    ref_lines : np.ndarray[float]
        Positions (degrees) of the vertical reference lines. Every line
        to draw must be listed explicitly; azimuths are not mirrored.
        Azimuths of 90/270 and a declination of 0 are drawn as dotted
        (equinox) lines, the rest as solid (solstice) lines.
        Default: None (no lines)
    title : str
        Plot title. Default: '' (no title)
    iterations : int
        Number of null-model samples used to renormalize the curve. Default: 100
    label : str | None
        Legend label for the curve. Default: None
    xlim : tuple[float, float]
        Visible x-axis range in degrees. The curve, the threshold and the
        peaks are always computed over the full range, so zooming in does
        not change the results. Default: None, automatic: the full 0-360°
        for azimuths; for declinations the outermost structures ±
        `xpad_deg`, rounded outwards to `xtick_step`, within -90..90°
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
        Draws the `ref_lines`. Default: True
    show_three_sigma : bool
        Draw a dashed line at `three_sigma_level` (only in the 'excess' view
        and the polar plot). It is NOT a significance test: it checks one
        position at a time, so random datasets usually cross it somewhere.
        Significance is shown by the strong fill and the peak markers.
        Default: False
    grid_step : float
        Spacing of the evaluation grid in degrees. Default: 0.5
    wrap_inputs : bool
        Must be True only for azimuths; pass False for declinations.
        Default: True
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
    null_sampler : Callable[[int, Generator], np.ndarray] | None
        Function returning one random dataset of the given size, used as the
        null model for the significance threshold. None draws uniformly
        over [loc, loc + scale]. See
        `Declination.make_null_sampler` for the declination null models.
        Default: None
    distribution_kwargs : dict | None
        Extra keyword arguments passed to `distribution_fn`, e.g. the null
        model settings of `Declination.distribution_decl_adaptive`, so the
        curve and the threshold use the same null model. Default: None
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
    null_model_label : str
        Name of the null model used in the legend and on the right axis
        when `show_null_model=True`. Default: "Null model"
    xpad_deg : float
        Declinations only: padding in degrees around the outermost
        structures for the automatic x-axis range (`xlim=None`). Default: 10.0
    lunar_lines : dict | None
        Positions (degrees) of the lunar standstill lines to draw, as
        {"major": [...], "minor": [...]}. Default: None (none)
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
        structures for both curves; the 3-sigma line is not drawn in this
    view. Significance is the same in both views.
        Not available with `polar=True`. Default: "excess"
    renormalization : str
        How the curve is centred before dividing by the null-model scale.
        'expected' subtracts the null model's expected density at each
        position, so the curve shows the excess over what the null model
        predicts there (scale: typical point-wise null standard deviation).
        'mean' subtracts the curve's own average (the original method; scale:
        mean standard deviation of the null curves). 'auto' uses 'expected'
        when `null_sampler` is given (declination null models
        'orientations' and 'solar') and 'mean' otherwise (azimuths,
        'uniform'), so earlier results are reproduced. 'expected' is needed
        for the 'solar' null model, whose expected curve has strong peaks at
        the solstices. Default: 'auto'

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
            'global_threshold_percentile', 'mean_null_std',
            'null_expectation' (the null model's expected density on the
            grid with `show_null_model=True`, else None), and the peak
            lists 'peaks_all' and 'peaks_significant' (dicts with 'angle',
            'height', 'prominence', 'index')

    Raises
    ------
    ValueError
        If mandatory arguments are None, `data` and `binwidths` differ in
        length, or values are out of range
    """
    
    if data is None or binwidths is  None:
        raise ValueError("`azimuths`, `latitude` and `binwidths` cannot be None.")
        
    if (latitude is not None) and not (-90 <= latitude <= 90):
        raise ValueError("`latitude``should be between [-90,90].")
        
    if len(data) != len(binwidths):
        raise ValueError("`azimuths` and `binwidths` should be of equal lenght")
        
    is_az = curvigram_fn.__name__ == "curvigram_az_adaptive"

    if not is_az and wrap_inputs:
        raise ValueError("`wrap_inputs` can be true only for azimuths.")
    
    if is_az and latitude is None:
        raise ValueError("`latitude` cannot be None when plotting azimuths.")

    if polar and not is_az:
        raise ValueError("`polar=True` is only available for azimuths.")

    if polar and show_null_model:
        raise ValueError("`show_null_model` is not available with `polar=True`.")

    view = str(view).lower()
    if view not in ("excess", "density"):
        raise ValueError("`view` must be 'excess' or 'density'.")
    if polar and view == "density":
        raise ValueError("`view='density'` is not available with `polar=True`.")

    ref_lines = [] if ref_lines is None else ref_lines

    # Visible range: azimuths always 0-360; declinations by default the
    # outermost structures +/- xpad_deg, rounded outwards to the tick spacing
    # and kept within -90..90
    if xlim is None:
        lo_lim, hi_lim = (0.0, 360.0) if is_az else (-90.0, 90.0)
        d = np.asarray(data, dtype=float).ravel()
        d = d[np.isfinite(d)]
        if is_az:
            d = np.mod(d, 360.0)
        if is_az or polar or d.size == 0:
            xlim = (lo_lim, hi_lim)                 # azimuths: always the full 0-360
        else:
            step = float(xtick_step) if xtick_step and xtick_step > 0 else 10.0
            lo = np.floor((d.min() - xpad_deg) / step) * step
            hi = np.ceil((d.max() + xpad_deg) / step) * step
            xlim = (float(max(lo_lim, lo)), float(min(hi_lim, hi)))

    binwidths = gaussian_bw_to_epanechnikov_h(binwidths)

    # The curve, the threshold and the peaks are always computed over the
    # full domain; `xlim` only sets the visible range, so zooming in does
    # not change the results.
    calc_xlim = (0.0, 360.0) if is_az else (-90.0, 90.0)

    if is_az:
        dist_out = distribution_fn(
            data,
            binwidths,
            loc=loc,
            scale=scale,
            iterations=iterations,
            title=title,
            n_uniform=n_uniform,
            rng=rng,
            grid_step=grid_step,
            xlim=calc_xlim,
            wrap_inputs=wrap_inputs,
            normalize=normalize,
            renormalization=renormalization,
            return_null_stats=True,
            **(distribution_kwargs or {}),
        )
    else:
        dist_out = distribution_fn(
            data,
            binwidths,
            loc=loc,
            scale=scale,
            iterations=iterations,
            title=title,
            n_uniform=n_uniform,
            rng=rng,
            grid_step=grid_step,
            xlim=calc_xlim,
            normalize=normalize,
            renormalization=renormalization,
            return_null_stats=True,
            **(distribution_kwargs or {}),
        )

    new_az, normed_density, renorm_density, null_stats = dist_out
    null_expectation = null_stats["expectation"] if show_null_model else None

    # Global threshold (percentile of null maxima), if requested
    thr_global = None
    mean_null_std = None
    if draw_thresholds:
        thr_iters = iterations if null_iterations is None else null_iterations
        thr_global, _null_maxima, mean_null_std = estimate_null_max_threshold_adaptive(
            n_samples=len(np.asarray(data)),
            h_epa=np.asarray(binwidths, dtype=float),
            curvigram_fn=curvigram_fn,
            percentile=null_percentile,
            iterations=thr_iters,
            n_uniform=n_uniform,
            loc=loc, scale=scale,
            grid_step=grid_step,
            xlim=calc_xlim,
            wrap_inputs=wrap_inputs,
            normalize=normalize,
            rng=rng,
            null_sampler=null_sampler,
            renormalization=renormalization,
        )

    if polar:
        x = np.asarray(new_az)[:, 0]
        y = np.asarray(renorm_density)
        thr_used = thr_global if (draw_thresholds and thr_global is not None) else three_sigma_level
        peaks_all, peaks_sig = [], []
        if mark_significant_peaks:
            peaks_all = find_circular_peaks(
                x_deg=x, y=y, min_separation_deg=min_peak_separation_deg, prominence=peak_prominence
            )
            peaks_sig = [p for p in peaks_all if p["height"] >= thr_used]

        fig, ax, created_fig = _draw_curvigram_polar(
            x, y, ax=ax, ref_lines=ref_lines, title=title, label=label,
            line_color=line_color, fill_color=fill_color, fill_alpha=fill_alpha,
            three_sigma_level=three_sigma_level, draw_reference_lines=draw_reference_lines,
            peaks_sig=peaks_sig, significant=y >= thr_used,
            null_band=(null_stats["excess_expected"], null_stats["excess_hi"]) if show_null_range else None,
            show_three_sigma=show_three_sigma, lunar_lines=lunar_lines)

        if created_fig:
            fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
        if show and created_fig:
            plt.show()

        info = {
            "three_sigma_level": three_sigma_level,
            "global_threshold": thr_global,
            "global_threshold_percentile": null_percentile if draw_thresholds else None,
            "mean_null_std": mean_null_std,
            "peaks_all": peaks_all,
            "peaks_significant": peaks_sig,
        }
        return fig, ax, new_az, normed_density, renorm_density, info

    # ------------------------------------------------------------------
    # Styling constants
    # ------------------------------------------------------------------
    ref_color = "0.35"          # reference lines (solstices / equinoxes)
    sigma_color = "#c0392b"     # 3-sigma level
    text_color = "0.15"

    # Prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(11, 4.2))
        created_fig = True
    else:
        fig = ax.figure

    x = np.asarray(new_az)[:, 0]
    y = np.asarray(renorm_density)                      # test statistic (excess units)
    thr_used = thr_global if (draw_thresholds and thr_global is not None) else three_sigma_level
    significant = y >= thr_used                         # passes the significance test

    # What is drawn: the excess curve, or (view='density') the data density
    # and the null model's expected density, both divided by the null's peak
    E = np.asarray(null_stats["expectation"], dtype=float)
    if view == "density":
        unit = float(np.nanmax(E)) if np.nanmax(E) > 0 else 1.0     # null-model peak = 1
        y_disp = np.asarray(normed_density, dtype=float) / unit
        null_disp = E / unit
        band = (null_disp, np.asarray(null_stats["density_hi"]) / unit)          # expected -> p99
    else:
        y_disp = y
        null_disp = None
        band = (np.asarray(null_stats["excess_expected"]), np.asarray(null_stats["excess_hi"]))   # expected -> p99

    # For a full-circle azimuth grid (which omits 360° = 0°), repeat the
    # first point at the end so the drawn curves reach the right edge.
    closing = bool(is_az and x.size > 1 and np.isclose(x[-1] + (x[1] - x[0]) - x[0], 360.0))
    def closed(v):
        v = np.asarray(v)
        return np.append(v, v[0]) if closing else v
    x_plot = np.append(x, x[0] + 360.0) if closing else x
    y_plot = closed(y_disp)

    if show_null_range:
        ax.fill_between(x_plot, closed(band[0]), closed(band[1]), color="0.6", alpha=0.22,
                        lw=0, zorder=0.5, label="Random datasets: expected to 99th percentile")
    if show_null_model and view == "density":
        ax.fill_between(x_plot, 0, closed(null_disp), color="0.6", alpha=0.10, lw=0, zorder=0.6)
        ax.plot(x_plot, closed(null_disp), color="0.45", ls=(0, (4, 3)), lw=1.4, zorder=1.2,
                label=f"{null_model_label} (expected)")

    # Curve: faint fill, strong fill only where the curve is significant
    if view == "excess":
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
        faint = y_plot > 0
    else:
        faint = np.ones(y_plot.shape, dtype=bool)
    ax.fill_between(x_plot, 0, y_plot, where=faint, interpolate=True,
                    alpha=fill_alpha * 0.35, facecolor=fill_color, edgecolor="none", zorder=2)
    sig_plot = closed(significant)
    if sig_plot.any():
        ax.fill_between(x_plot, 0, y_plot, where=sig_plot, alpha=min(1.0, fill_alpha * 2),
                        facecolor=fill_color, edgecolor="none", zorder=2.1, label="Significant")
    ax.plot(x_plot, y_plot, color=line_color, lw=1.6, zorder=3, label=label if label else None)

    # Axes cosmetics
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.4")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.85", lw=0.7, ls="-")
    ax.set_xlabel(r"Azimuth ($^\circ$)" if is_az else r"Declination ($^\circ$)",
                  fontsize=13, color=text_color)
    ax.set_ylabel("Normalized relative frequency" if view == "excess" else
                  "Density (null-model peak = 1)", fontsize=13, color=text_color)
    if title:
        ax.set_title(title, fontsize=14, fontweight="semibold", color=text_color, loc="left")

    visible = (x >= min(xlim)) & (x <= max(xlim))          # scale y to the visible part
    if not visible.any():
        visible = np.ones_like(x, dtype=bool)
    parts = [y_disp[visible]]
    if show_null_range:
        parts += [band[0][visible], band[1][visible]]
    if null_disp is not None and show_null_model:
        parts.append(null_disp[visible])
    y_min = min(float(np.nanmin(p)) for p in parts)
    y_max = max(float(np.nanmax(p)) for p in parts)
    if view == "density":
        y_min = 0.0
    span = max(y_max - y_min, 1.0 if view == "excess" else 0.1)
    headroom = 0.18
    if is_az and draw_reference_lines and _azimuth_line_labels(ref_lines, lunar_lines, "0"):
        headroom = 0.38          # keep peak labels below the azimuth labels at the top
    ax.set_ylim(y_min - (0.05 * span if view == "excess" else 0.0), y_max + headroom * span)

    if show_three_sigma and view == "excess":      # optional; not a significance test
        ax.axhline(y=three_sigma_level, color=sigma_color, lw=1.1, ls=(0, (5, 4)),
                   alpha=0.85, zorder=1.5, label=r'3$\sigma$')

    if draw_reference_lines:
        # Legend labels; each is attached to the first line of its kind only,
        # so the legend shows one entry per line style.
        if is_az:
            equinox_label = "Equinoxes (flat horizon)"
            solstice_label = "Solstices (flat horizon)"
        else:
            equinox_label = r"Equinoxes ($\delta$ = 0$^\circ$)"
            solstice_label = rf"Solstices ($\delta$ = $\pm${obliquity_deg:.2f}$^\circ$)"
        labelled = set()

        # `ref_lines` holds every line to draw (for azimuths: rising and
        # setting), so no mirroring is done here. Equinoxes are dotted,
        # solstices solid; both sit behind the curve.
        for pos in ref_lines:
            if is_az:
                is_equinox = bool(np.isclose(pos, 90.0) or np.isclose(pos, 270.0))
            else:
                is_equinox = bool(np.isclose(pos, 0.0))
            kind = "equinox" if is_equinox else "solstice"
            ax.axvline(x=pos, color=ref_color, ls=(0, (1.5, 2.5)) if is_equinox else "-",
                       lw=1.2 if is_equinox else 1.0, alpha=0.9, zorder=1.5,
                       label=None if kind in labelled else
                             (equinox_label if is_equinox else solstice_label))
            labelled.add(kind)

        # Lunar standstill limits (major: dashed, minor: dash-dot)
        for kind, positions in (lunar_lines or {}).items():
            if not positions:
                continue
            if is_az:
                lunar_label = f"{kind.capitalize()} lunar standstill (flat horizon)"
            else:
                lunar_label = rf"{kind.capitalize()} lunar standstill ($\delta$ = $\pm${abs(positions[0]):.2f}$^\circ$)"
            for k, pos in enumerate(positions):
                ax.axvline(x=pos, alpha=0.9, zorder=1.5, label=lunar_label if k == 0 else None,
                           **LUNAR_LINE_STYLES[kind])

        # Azimuth plots: flat-horizon azimuth written at the top of each
        # solstice and lunar line
        if is_az:
            for pos, col in _azimuth_line_labels(ref_lines, lunar_lines, ref_color):
                if min(xlim) <= pos <= max(xlim):
                    ax.text(pos, 0.985, f"{pos:.1f}°", transform=ax.get_xaxis_transform(),
                            rotation=90, ha="center", va="top", fontsize=8, color=col,
                            zorder=4.5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    # The null-percentile threshold is used for peak significance (fill and
    # markers) and returned in `info`, but it is not drawn as a line.
    legend_x = 1.01
    extra_handles = []
    if show_null_model and view == "excess":
        # Null model's expected curvigram on a second y-axis (density units),
        # drawn behind the data curve
        null_color = "0.45"
        ax_null = ax.twinx()
        e = np.asarray(E, dtype=float)
        e_vis = e[visible]
        ax_null.fill_between(x_plot, 0, closed(e), color="0.6", alpha=0.15, lw=0)
        ax_null.plot(x_plot, closed(e), color=null_color, ls=(0, (4, 3)), lw=1.4)
        ax_null.set_ylim(0, max(float(np.nanmax(e_vis)), 1e-12) * 1.7)
        ax_null.set_ylabel(f"{null_model_label}: expected density", fontsize=11,
                           color=null_color, labelpad=8)
        ax_null.tick_params(axis="y", labelsize=9, colors=null_color, length=3)
        ax_null.spines["right"].set_color("0.7")
        ax_null.spines["top"].set_visible(False)
        ax_null.spines["left"].set_visible(False)
        ax_null.spines["bottom"].set_visible(False)
        ax.set_zorder(ax_null.get_zorder() + 1)      # data curve in front
        ax.patch.set_visible(False)
        from matplotlib.patches import Patch
        extra_handles.append(Patch(facecolor="0.6", alpha=0.35, edgecolor=null_color,
                                   linestyle=(0, (4, 3)),
                                   label=f"{null_model_label}\n(expected, right axis)"))
        legend_x = 1.10                               # room for the right axis

    handles, labels = ax.get_legend_handles_labels()
    handles += extra_handles
    labels += [h.get_label() for h in extra_handles]
    if handles:
        # Legend outside the plot, to the right, so it never hides a peak
        ax.legend(handles, labels, fontsize=10, loc="upper left", bbox_to_anchor=(legend_x, 1.0),
                  borderaxespad=0.0, frameon=True, framealpha=0.92,
                  edgecolor="0.85", fancybox=False, borderpad=0.6)

    add_version_watermark(ax)

    ax.set_xlim(*xlim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.1, xtick_step))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", labelsize=11, colors=text_color, length=4)
    ax.tick_params(axis="both", which="minor", length=2, color="0.5")

    # Peaks (found and tested on the excess curve; drawn on the shown curve)
    peaks_all = []
    peaks_sig = []
    if mark_significant_peaks:
        if is_az:
            peaks_all = find_circular_peaks(
                x_deg=x, y=y, min_separation_deg=min_peak_separation_deg, prominence=peak_prominence
            )
        else:
            peaks_all = find_linear_peaks(
                x_deg=x, y=y, min_separation_deg=min_peak_separation_deg, prominence=peak_prominence
            )           
        peaks_sig = [p for p in peaks_all if p["height"] >= thr_used]
        
        x_span = (xlim[1] - xlim[0]) or 1.0
        for p in peaks_sig:
            if not (min(xlim) <= p["angle"] <= max(xlim)):
                continue        # outside the visible range (still listed in `info`)
            y_pk = float(y_disp[p["index"]])
            ax.plot(p["angle"], y_pk, marker="o", ms=5.5, mfc="white",
                    mec=line_color, mew=1.6, zorder=4)
            # Labels of peaks near the left/right edge are aligned inwards so
            # they do not run into the axis numbers or out of the plot
            frac = (p["angle"] - xlim[0]) / x_span
            ha, dx = ("left", -4) if frac < 0.04 else (("right", 4) if frac > 0.96 else ("center", 0))
            ax.annotate(f"{p['angle']:.1f}°",
                        xy=(p["angle"], y_pk),
                        xytext=(dx, 8), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=10, color=text_color, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.6, alpha=0.9))

    if created_fig:
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show and created_fig:
        plt.show()

    info = {
        "three_sigma_level": three_sigma_level,
        "global_threshold": thr_global,
        "global_threshold_percentile": null_percentile if draw_thresholds else None,
        "mean_null_std": mean_null_std,
        "peaks_all": peaks_all,
        "peaks_significant": peaks_sig,
        "null_expectation": null_expectation,
    }
    
    return fig, ax, new_az, normed_density, renorm_density, info