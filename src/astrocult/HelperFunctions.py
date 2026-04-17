# -*- coding: utf-8 -*-
import numpy as np
from typing import Callable
from numpy.random import default_rng, Generator
import matplotlib.pyplot as plt

def ang_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest signed angular difference a - b in degrees, wrapped to (-180, 180].
    Works with broadcasting.
    
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
    
    d = (a - b + 180.0) % 360.0 - 180.0
    d[d == -180.0] = 180.0
    
    return d


def gaussian_bw_to_epanechnikov_h(bw_gauss_like : np.ndarray[float]) -> np.ndarray[float]:
    """
    Interpret ``bw_gauss_like`` as 2*sigma (i.e., bw = 2σ) and return
    h_epa such that var_epa = var_gauss = σ^2. => h = sqrt(5) * σ*2/2 = sqrt(5)*σ. 
    Hence, we assume that h = sqrt(5) * bw_gauss_like.
    
    Parameters
    ----------
    bw_gauss_like : np.ndarray
        The bin widths as a sequence
    
    Returns
    -------
    np.ndarray
        The transformed values

    Raises
    ------
    ValueError
        If a None value has been provided as an argument
    """
    
    if bw_gauss_like is None:
        raise ValueError("Argument cannot be None")
    
    return (np.sqrt(5.0)) * np.asarray(bw_gauss_like, dtype=float) 
 
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generic adaptive density + null-model renormalization.

    It mirrors functions_v2.behavior but allows plugging either the circular
    or linear curvigram via ``curvigram_fn``.
    
    Parameters
    ----------
    data : Sequence
        The input data
    binwidths : Sequence
        The bin bandwidth
    curvigram_fn : Callable
        The curvigram generator method. Possible values include `Azimuth.curvigram_az_adaptive`
        or `Declination.curvigram_dec_adaptive`
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
    wrap_inputs : bool | None
        Only used for azimuths
    normalize : str
        Possible values: pdf, none, max1. Default=pdf
    loc : float
        Default: 0
    scale : float
        Default: 359
    
    Returns
    -------
    Tupple
        new_data : np.ndarray
            Data grid returned by the function provided to `curvigram_fn` (shape: [N, ...]).
        normed_density : np.ndarray
            Density for the real sample.
        renorm_density : np.ndarray
            Mean-centered real density divided by the average std from the null model.

    Raises
    ------
    ValueError
        If a None value has been provided as an argument
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

    null_stds = np.empty(iterations, dtype=float)
    for i in range(iterations):
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

    mean_null_std = float(np.mean(null_stds))

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
    renorm_density = (normed_density - np.mean(normed_density)) / denom

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
):
    """Return (threshold, maxima, mean_null_std) for renormalized curves.

    Uses the adaptive curvigrams so real and null share identical settings.
    
    Parameters
    ----------
    n_samples : int
    h_epa : np.ndarray
    domain : str
        "az" (circular) or "decl" (linear). Default: az
    curvigram_fn : Callable
        The curvigram generator method. Possible values include `Azimuth.curvigram_az_adaptive`
        or `Declination.curvigram_dec_adaptive`
    percentile : float
        Default: 99
    iterations : int
        Default: 500
    n_uniform : int
        Default: 80_000
    loc : float
        Default: 0
    scale : float
        Default: 359
    grid_step : float
        Default: 0.5
    xlim : Tupple
        Default: (0.0, 360.0)
    wrap_inputs : bool
        Default: true
    normalize : str
        Possible values: pdf, none, max1. Default=pdf
    rng : Generator
        
    Returns
    -------
    Tupple
        

    Raises
    ------
    ValueError
        If a None value has been provided as an argument
    """
    
    if h_epa is None or curvigram_fn is None:
        raise ValueError("`h_epa` and `curvigram_h` cannot be None.")

    rng = default_rng() if rng is None else rng
    n_uniform = max(int(n_uniform), int(n_samples))
    uniform_grid = np.linspace(loc, loc + scale, n_uniform, endpoint=True)

    h_epa = np.asarray(h_epa, dtype=float).ravel()
    if h_epa.size != int(n_samples):
        raise ValueError("`h_epa` must have same length as `n_samples`.")

    #curv = curvigram_az_adaptive if domain == "az" else curvigram_decl_adaptive
    curv = curvigram_fn


    # Pass 1: estimate mean std of null densities
    stds = np.empty(iterations, dtype=float)
    for i in range(iterations):
        null_sample = rng.choice(uniform_grid, size=n_samples, replace=False)
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
    mean_null_std = float(np.mean(stds))
    denom = mean_null_std if mean_null_std > np.finfo(float).eps else 1.0

    # Pass 2: maxima of renormalized null curves
    maxima = np.empty(iterations, dtype=float)
    for i in range(iterations):
        null_sample = rng.choice(uniform_grid, size=n_samples, replace=False)
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
        z = (dens - np.mean(dens)) / denom
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
    """Peak detection on a linear axis. Returns list of dicts with
    keys: value (x), height, prominence, index. Used for declinations.
    
    Parameters
    ----------
    x_deg : np.ndarray,
        Declination angles
    y : np.ndarray,
    min_separation_deg : float
        Default: 2
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


def plot_curvigram(
    data: np.ndarray[float],
    latitude: float,
    binwidths: np.ndarray[float],
    curvigram_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    distribution_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    ref_lines: np.ndarray[float] = [],
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
    """ Plots the curvigram for the provided data and bin widths by using an adaptive approach.
    See also: `self.curvigram_[az|decl]_adaptive` and `self.distribution_[az|decl]_adaptive`.
    
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
    
    if data is None or binwidths is  None:
        raise ValueError("`azimuths`, `latitude` and `binwidths` cannot be None.")
        
    if (latitude != None) and not (-90 <= latitude <= 90):
        raise ValueError("`latitude``should be between [-90,90].")
        
    if len(data) != len(binwidths):
        raise ValueError("`azimuths` and `binwidths` should be of equal lenght")
        
    if curvigram_fn.__name__ == "curvigram_decl_adaptive" and wrap_inputs == True:
        raise ValueError("`wrap_inputs` can be true only for azimuths.")
    
    if curvigram_fn.__name__ == "curvigram_az_adaptive" and latitude == None:
        raise ValueError("`latitude` cannot be None when plotting azimuths.")

    binwidths = gaussian_bw_to_epanechnikov_h(binwidths)

    if curvigram_fn == "curvigram_az_adaptive":
        new_az, normed_density, renorm_density = distribution_fn(
            data,
            binwidths,
            loc=loc,
            scale=scale,
            iterations=iterations,
            title=title,
            n_uniform=n_uniform,
            rng=rng,
            grid_step=grid_step,
            xlim=xlim,
            wrap_inputs=wrap_inputs,
            normalize=normalize,
        )
    else:
        new_az, normed_density, renorm_density = distribution_fn(
            data,
            binwidths,
            loc=loc,
            scale=scale,
            iterations=iterations,
            title=title,
            n_uniform=n_uniform,
            rng=rng,
            grid_step=grid_step,
            xlim=xlim,
            normalize=normalize,
        )

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
            xlim=xlim,
            wrap_inputs=wrap_inputs,
            normalize=normalize,
            rng=rng,
        )

    # Prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        created_fig = True
    else:
        fig = ax.figure

    x = np.asarray(new_az)[:, 0]
    y = np.asarray(renorm_density)

    ax.plot(x, y, color=line_color, lw=1.5, label=label if label else None)
    ax.fill_between(x, 0, y, alpha=fill_alpha, facecolor=fill_color)

    if curvigram_fn.__name__ == "curvigram_az_adaptive":
        ax.set_xlabel(r"Azimuth ($^\circ$)", fontsize=18)
    else:
        ax.set_xlabel(r"Declination ($^\circ$)", fontsize=18)
  
    ax.set_ylabel("Normalized relative frequency", fontsize=18)
    ax.set_title(title, fontsize=16)

    y_min = float(np.nanmin(y)) if y.size else 0.0
    y_max = float(np.nanmax(y)) if y.size else 1.0
    pad = 0.1 * max(y_max - y_min, 1.0)
    ax.set_ylim(y_min, y_max + pad)

    if draw_reference_lines:
        ax.axhline(y=three_sigma_level, color="firebrick", lw=1.0, ls="--", label=r'3$\sigma$')
        if curvigram_fn.__name__ == "curvigram_az_adaptive":
            # Solstices ±ε and symmetric counterparts
            for az in ref_lines:
                ax.axvline(x=az, ymin=0.3, ymax=0.6 if az != 90 or az !=270 else 0.95, color="k", ls=":" if az == 90 or az ==270 else "", lw=1.3)
                ax.axvline(x=(360 - az), ymin=0.3, ymax=0.6 if az != 90 and az !=270 else 0.95, color="k", ls=":" if az == 90 or az ==270 else "", lw=1.3)  
        else:
            # Reference verticals at solstitial declinations ±ε and equatorial 0°
            for dec in ref_lines:
                ax.axvline(x=dec, ymax=0.95 if dec == 0 else 0.6, color="k", ls=":" if dec == 0 else "-", lw=1.3)

    if draw_thresholds and thr_global is not None:
        ax.axhline(y=thr_global, color="purple", lw=1.1, ls="-.", label=f"Null p{null_percentile:.1f}")

    if label or (draw_thresholds and thr_global is not None):
        ax.legend(fontsize=12)

    ax.set_xlim(*xlim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.1, xtick_step))
    ax.tick_params(axis="both", labelsize=18)

    # Peaks
    peaks_all = []
    peaks_sig = []
    if mark_significant_peaks:
        if curvigram_fn.__name__ == "curvigram_az_adaptive":
            peaks_all = find_circular_peaks(
                x_deg=x, y=y, min_separation_deg=min_peak_separation_deg, prominence=peak_prominence
            )
        else:
            peaks_all = find_linear_peaks(
                x_deg=x, y=y, min_separation_deg=min_peak_separation_deg, prominence=peak_prominence
            )           
        thr_used = thr_global if (draw_thresholds and thr_global is not None) else three_sigma_level
        peaks_sig = [p for p in peaks_all if p["height"] >= thr_used]
        
        for p in peaks_sig:
            ax.plot(p["angle"], p["height"], marker="o", ms=5, color="black")
            ax.annotate(f"{p['angle']:.1f}°",
                        xy=(p["angle"], p["height"]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12)

    fig.tight_layout()
    if savepath:
        # slightly smaller labels for saved figure
        if curvigram_fn.__name__ == "curvigram_az_adaptive":
            ax.set_xlabel(r"Azimuth ($^\circ$)", fontsize=14)
        else:
            ax.set_xlabel(r"Declination ($^\circ$)", fontsize=14)
        ax.set_ylabel("Normalized relative frequency", fontsize=14)
        ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.1, 50))
        ax.tick_params(axis="both", labelsize=12)
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