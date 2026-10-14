"""
Test suite for the archaeoastronomy package.

Run from the repository root with:  pytest -v tests/
Tests marked "BUG" assert the correct behaviour and are expected to FAIL until
the corresponding fix is applied.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Find the modules: $ARCHAEO_SRC, else <repo>/src/astrocult, else <repo>/astrocult, else this folder
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_candidates = [os.path.join(_root, "src", "astrocult"), os.path.join(_root, "astrocult"), _here]
sys.path.insert(0, os.environ.get("ARCHAEO_SRC", next(c for c in _candidates if os.path.isfile(os.path.join(c, "Declination.py")))))

import Altitude
import Azimuth
import Clustering
import Declination
import GeoUtils
import HelperFunctions as HF
import JulianHelper


@pytest.fixture
def decl():
    return Declination.Declination()


@pytest.fixture
def az():
    return Azimuth.Azimuth()


@pytest.fixture
def jh():
    return JulianHelper.JulianHelper()


# --------------------------------------------------------------------------
# Declination
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lat, alt, a, expected", [
    (45.0, 0.0, 90.0, 0.0),     # due east on flat horizon -> equator
    (45.0, 0.0, 0.0, 45.0),     # due north on horizon -> 90 - lat
    (45.0, 0.0, 180.0, -45.0),  # due south on horizon
    (0.0, 90.0, 123.0, 0.0),    # zenith at the equator
])
def test_calc_decl_known_values(decl, lat, alt, a, expected):
    assert decl.calc_decl(lat, alt, a) == pytest.approx(expected, abs=1e-9)


def test_calc_decl_rejects_out_of_range(decl):
    with pytest.raises(ValueError):
        decl.calc_decl(91, 0, 0)
    with pytest.raises(ValueError):
        decl.calc_decl(0, 0, float("nan"))


def test_azimuth_is_wrapped(decl):
    # bearings from GeoUtils.get_bearing can be negative (-180..180)
    assert decl.calc_decl(45.2, 1.0, -55.6) == pytest.approx(decl.calc_decl(45.2, 1.0, 304.4))
    assert decl.calc_decl(45.2, 1.0, 361.0) == pytest.approx(decl.calc_decl(45.2, 1.0, 1.0))
    assert decl.err_decl(45.2, -55.6, 1.0) == pytest.approx(decl.err_decl(45.2, 304.4, 1.0))
    assert decl.calc_decl(90.0, 90.0, 0.0) == pytest.approx(90.0)        # no rounding crash


def test_refraction_input_range():
    alt = Altitude.Altitude()
    with pytest.raises(ValueError):
        alt.update_alt_by_atmospheric_refraction(-4.4)                 # formula singular there
    assert alt.update_alt_by_atmospheric_refraction(90.0) == pytest.approx(90.0, abs=1e-3)


@pytest.mark.parametrize("lat, a, alt", [(40, 60, 2), (52, 120, 0.5), (-30, 250, 5)])
def test_err_decl_matches_finite_difference(decl, lat, a, alt):
    e = 1e-5
    d_alt = (decl.calc_decl(lat, alt + e, a) - decl.calc_decl(lat, alt - e, a)) / (2 * e)
    d_az = (decl.calc_decl(lat, alt, a + e) - decl.calc_decl(lat, alt, a - e)) / (2 * e)
    expected = abs(d_alt) * 0.5 + abs(d_az) * 0.5
    assert decl.err_decl(lat, a, alt, 0.5, 0.5) == pytest.approx(expected, rel=1e-4)


def test_azimuth_declination_round_trip(decl, az):
    lat, alt = 41.4, 1.5
    for d in (-23.5, -10.0, 0.0, 15.0, 23.5):
        a_e, a_w = az.calc_az(alt, lat, d)
        assert decl.calc_decl(lat, alt, a_e) == pytest.approx(d, abs=1e-6)
        assert decl.calc_decl(lat, alt, a_w) == pytest.approx(d, abs=1e-6)
        assert 0 < a_e < 180 < a_w < 360


def test_BUG_year_declinations_accept_float_latitude(decl, monkeypatch):
    # Validation uses `not in range(-90, 90)`, which rejects any float.
    # We only check that validation passes; stop before ephemeris loading.
    monkeypatch.setattr(Declination, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        decl.calculate_declinations_for_entire_year(1600, 41.5, 2.17, 0)


def test_curvigram_decl_integrates_to_one(decl):
    x, dens = decl.curvigram_decl_adaptive([-20.0, 0.0, 23.0], [3.0, 3.0, 3.0])
    assert __import__("scipy.integrate").integrate.trapezoid(dens, x[:, 0]) == pytest.approx(1.0, abs=1e-6)


def test_BUG_curvigram_accepts_lowercase_none(decl):
    decl.curvigram_decl_adaptive([0.0], [3.0], normalize="none")


def test_null_sampler_orientations_range(decl):
    rng = np.random.default_rng(0)
    s = decl.make_null_sampler("orientations", latitude=40.0)(100_000, rng)
    assert s.min() >= -50.0 - 1e-9 and s.max() <= 50.0 + 1e-9      # flat horizon: +/-(90 - lat)
    assert s.max() > 49.9 and s.min() < -49.9
    # (latitude, altitude) pairs per site are resampled together
    lats, alts = np.r_[np.full(50, 30.0), np.full(50, 60.0)], np.r_[np.zeros(50), np.full(50, 10.0)]
    s = decl.make_null_sampler("orientations", latitude=lats, null_altitudes=alts)(100_000, rng)
    assert s.max() == pytest.approx(60.0, abs=0.05)


def test_null_sampler_solar(decl):
    rng = np.random.default_rng(0)
    s = decl.make_null_sampler("solar", obliquity_deg=23.44)(100_000, rng)
    assert np.abs(s).max() <= 23.44 + 1e-9
    near_solstice = np.mean(np.abs(np.abs(s) - 23.44) < 2)
    near_equinox = np.mean(np.abs(s) < 2)
    assert near_solstice > 3 * near_equinox                          # pile-up at the solstices
    pool = decl.make_null_sampler("solar", solar_declinations=[1.0, 2.0, np.nan])(500, rng)
    assert set(np.unique(pool)) == {1.0, 2.0}
    assert float(decl.solar_declination_approx(171, 23.44)) == pytest.approx(23.44, abs=0.1)
    assert float(decl.solar_declination_approx(354, 23.44)) == pytest.approx(-23.44, abs=0.1)


def test_null_sampler_errors(decl):
    with pytest.raises(ValueError):
        decl.make_null_sampler("orientations")                       # latitude missing
    with pytest.raises(ValueError):
        decl.make_null_sampler("lunar", latitude=40.0)
    assert decl.make_null_sampler("uniform") is None


def test_decl_plot_null_models_and_zoom(decl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    data = np.r_[rng.normal(0, 1, 15), rng.uniform(-40, 40, 40)]
    bw = np.full(data.size, 2.0)
    with pytest.raises(ValueError):                                  # default needs a latitude
        decl.plot_decl_adaptive(data, None, bw, iterations=5, show=False)
    res = {}
    for xlim in [(-90, 90), (-30, 30)]:
        *_, r, info = decl.plot_decl_adaptive(data, 41.0, bw, iterations=30, null_iterations=60, show=False,
                                              rng=np.random.default_rng(1), xlim=xlim)
        res[xlim] = (r, info["global_threshold"])
        assert info["null_model"] == "orientations"
    np.testing.assert_array_equal(res[(-90, 90)][0], res[(-30, 30)][0])   # zoom does not change results
    assert res[(-90, 90)][1] == res[(-30, 30)][1]
    for model in ("solar", "uniform"):
        *_, info = decl.plot_decl_adaptive(data, 41.0, bw, null_model=model, iterations=10, null_iterations=20,
                                           show=False, rng=np.random.default_rng(1))
        assert info["null_model"] == model
    plt.close("all")


def test_show_null_model(decl, az):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    data = np.r_[rng.normal(0, 1, 15), rng.uniform(-40, 40, 40)]
    bw = np.full(data.size, 2.0)
    for model in ("orientations", "solar", "uniform"):
        kw = dict(null_model=model, iterations=40, null_iterations=40, show=False)
        fig, ax, grid, _, r_on, info = decl.plot_decl_adaptive(data, 41.0, bw, rng=np.random.default_rng(1),
                                                               show_null_model=True, **kw)
        *_, r_off, info_off = decl.plot_decl_adaptive(data, 41.0, bw, rng=np.random.default_rng(1), **kw)
        np.testing.assert_array_equal(r_on, r_off)                     # the flag does not change results
        e = info["null_expectation"]
        assert e is not None and info_off["null_expectation"] is None
        from scipy.integrate import trapezoid
        assert trapezoid(e, grid[:, 0]) == pytest.approx(1.0, abs=0.01)    # a density: area 1
        assert len(fig.axes) >= 2                                      # second y-axis drawn
    with pytest.raises(ValueError):
        az.plot_az_adaptive(np.array([10.0, 20.0, 30.0]), 45.0, np.full(3, 3.0), iterations=5,
                            show=False, polar=True, show_null_model=True)
    plt.close("all")


def test_views_bands_and_three_sigma(decl, az):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    data = np.r_[rng.normal(0, 1, 15), rng.uniform(-40, 40, 40)]
    bw = np.full(data.size, 2.0)
    kw = dict(iterations=30, null_iterations=60, show=False)
    base = decl.plot_decl_adaptive(data, 41.0, bw, rng=np.random.default_rng(1), **kw)
    legend = lambda ax: [t.get_text() for t in ax.get_legend().get_texts()]
    assert not any("sigma" in t for t in legend(base[1]))               # 3-sigma hidden by default
    with3 = decl.plot_decl_adaptive(data, 41.0, bw, rng=np.random.default_rng(1), show_three_sigma=True, **kw)
    assert any("sigma" in t for t in legend(with3[1]))
    dens = decl.plot_decl_adaptive(data, 41.0, bw, rng=np.random.default_rng(1), view="density",
                                   show_null_model=True, show_null_range=True, **kw)
    np.testing.assert_array_equal(base[4], dens[4])                      # views do not change results
    assert base[5]["global_threshold"] == dens[5]["global_threshold"]
    assert len(dens[0].axes) == 1                                        # null drawn on the same axis
    assert any("99th percentile" in t for t in legend(dens[1]))
    with pytest.raises(ValueError):
        decl.plot_decl_adaptive(data, 41.0, bw, view="ratio", **kw)
    with pytest.raises(ValueError):
        az.plot_az_adaptive(np.array([10.0, 20.0, 30.0]), 45.0, np.full(3, 3.0), iterations=5,
                            show=False, polar=True, view="density")
    plt.close("all")


def test_auto_range_lunar_lines_and_band(decl, az):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = np.array([-12.0, -5.0, 0.5, 1.0, 14.0])
    bw = np.full(data.size, 2.0)
    fig, ax, *_ = decl.plot_decl_adaptive(data, 45.0, bw, iterations=20, null_iterations=40, show=False,
                                          show_null_range=True, rng=np.random.default_rng(1))
    assert ax.get_xlim() == pytest.approx((-30.0, 30.0))                 # -12-10 -> -30, 14+10 -> 30
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Major lunar standstill" in t and "28.70" in t for t in texts)
    assert any("Minor lunar standstill" in t for t in texts)
    assert any("99th percentile" in t for t in texts)
    fig, ax, *_ = decl.plot_decl_adaptive(data, 45.0, bw, iterations=20, null_iterations=40, show=False,
                                          draw_lunar_lines=False, xlim=(-90, 90))
    assert ax.get_xlim() == pytest.approx((-90.0, 90.0))
    assert not any("lunar" in t.get_text() for t in ax.get_legend().get_texts())
    # band starts at the null expectation: in the excess view that is ~0 for 'expected' renormalization
    *_, stats = HF.distribution_generic_adaptive(data, HF.gaussian_bw_to_epanechnikov_h(bw),
                                                 curvigram_fn=decl.curvigram_decl_adaptive, iterations=50,
                                                 xlim=(-90, 90), loc=-90, scale=180,
                                                 null_sampler=decl.make_null_sampler("orientations", latitude=45.0),
                                                 rng=np.random.default_rng(2), return_null_stats=True)
    assert np.max(np.abs(stats["excess_expected"])) < 1e-9
    fig, ax, *_ = az.plot_az_adaptive(np.array([118.0, 120.0, 122.0]), 45.0, np.full(3, 3.0), iterations=10,
                                      null_iterations=20, show=False, rng=np.random.default_rng(1))
    assert ax.get_xlim() == pytest.approx((0.0, 360.0))                  # azimuths: never trimmed
    east, west = az.calc_az(0.0, 45.0, 23.56)                             # solstice line labels
    labels = {t.get_text() for t in ax.texts}
    assert f"{east:.1f}°" in labels and f"{west:.1f}°" in labels
    east_major, _ = az.calc_az(0.0, 45.0, 23.56 + 5.145)                 # lunar line labels
    assert f"{east_major:.1f}°" in labels
    fig, ax, *_ = az.plot_az_adaptive(np.array([118.0, 120.0, 122.0]), 45.0, np.full(3, 3.0), iterations=10,
                                      null_iterations=20, show=False, rng=np.random.default_rng(1), polar=True)
    assert len(ax.get_xticklabels()) == 24                                # compass labels every 15 degrees
    assert f"{east:.1f}°" in {t.get_text() for t in ax.texts}
    plt.close("all")


def test_orientation_null_is_calibrated(decl):
    # random orientations analysed with the matching null: ~1% false alarms at the 99th percentile
    n = 60
    h = HF.gaussian_bw_to_epanechnikov_h(np.full(n, 2.0))
    rng = np.random.default_rng(4)
    sampler = decl.make_null_sampler("orientations", latitude=45.0, null_altitudes=rng.uniform(0, 4, 300))
    thr, _, s = HF.estimate_null_max_threshold_adaptive(n, h, curvigram_fn=decl.curvigram_decl_adaptive,
                                                        iterations=800, xlim=(-90, 90), loc=-90, scale=180,
                                                        rng=rng, null_sampler=sampler)
    E = np.mean([decl.curvigram_decl_adaptive(sampler(n, rng), h)[1] for _ in range(800)], axis=0)
    hits = sum(((decl.curvigram_decl_adaptive(sampler(n, rng), h)[1] - E) / s).max() >= thr for _ in range(400))
    assert hits / 400 < 0.04


# --------------------------------------------------------------------------
# Azimuth
# --------------------------------------------------------------------------
def test_err_az_combines_systematic_and_spread(az):
    assert az.err_az([10, 12], 0.5) == pytest.approx(math.sqrt(0.25 + 2.0))


def test_BUG_err_az_single_reading(az):
    # statistics.stdev raises on a single value; should fall back to err_az.
    assert az.err_az([10.0], 0.5) == pytest.approx(0.5)


def test_BUG_calc_az_accepts_ints(az):
    az.calc_az(0, 45, 23)


def test_curvigram_az_wraps_across_north(az):
    x, dens = az.curvigram_az_adaptive([359.0], [5.0])
    grid = x[:, 0]
    assert dens[grid == 1.0][0] > 0      # mass spills past 0°/360°
    assert dens[grid == 357.0][0] > 0


def test_BUG_circular_grid_has_no_duplicate_endpoint(az):
    x, _ = az.curvigram_az_adaptive([0.0], [5.0])
    assert not (x[0, 0] == 0.0 and x[-1, 0] == 360.0)


def test_orientation_map(az):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    lat, lon = [45.60, 45.70, 45.80], [21.10, 21.30, 21.20]
    fig, ax = az.plot_orientation_map(lat, lon, [90.0, 123.0, 300.0], [2.0, 0.0, 5.0],
                                      labels=["A", "B", "C"], show=False)
    wedges = [p for p in ax.patches if isinstance(p, Polygon)]
    assert len(wedges) == 4                                  # fill + outline for the 2 sites with error > 0
    arrows = [c for c in ax.texts if getattr(c, "arrow_patch", None) is not None and c.get_text() == ""]
    assert len(arrows) >= 3
    for i, a in enumerate([90.0, 123.0, 300.0]):             # arrow direction on the map = azimuth
        (x0, y0), (x1, y1) = arrows[i].xyann, arrows[i].xy
        dx = (x1 - x0) * np.cos(np.radians(lat[i]))          # longitude degrees -> distance east
        assert (np.degrees(np.arctan2(dx, y1 - y0)) % 360) == pytest.approx(a, abs=0.5)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert len(set(labels)) == len(labels)                   # no repeated tick labels
    with pytest.raises(ValueError):
        az.plot_orientation_map([45.7, 45.8], [21.2], [90.0, 90.0], show=False)
    plt.close("all")


def test_polar_azimuth_plot_matches_linear(az):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = np.array([1.0, 2.0, 358.0, 121.0, 122.0, 123.0, 200.0])
    bw = np.full(data.size, 3.0)
    fig, ax, _, _, r_polar, info_p = az.plot_az_adaptive(data, 40.0, bw, iterations=10, show=False,
                                                         rng=np.random.default_rng(0), polar=True)
    _, _, _, _, r_lin, info_l = az.plot_az_adaptive(data, 40.0, bw, iterations=10, show=False,
                                                    rng=np.random.default_rng(0))
    assert ax.name == "polar"
    np.testing.assert_allclose(r_polar, r_lin)
    assert [p["angle"] for p in info_p["peaks_all"]] == [p["angle"] for p in info_l["peaks_all"]]
    with pytest.raises(ValueError):                      # ax must be polar
        az.plot_az_adaptive(data, 40.0, bw, iterations=5, show=False, ax=plt.subplots()[1], polar=True)
    plt.close("all")


# --------------------------------------------------------------------------
# Altitude
# --------------------------------------------------------------------------
def test_refraction_at_horizon_about_half_degree():
    r = Altitude.Altitude().update_alt_by_atmospheric_refraction(0.0)
    assert r == pytest.approx(-0.575, abs=0.01)  # Bennett: ~34.5 arcmin at horizon


@pytest.mark.parametrize("h", [0.0, 1.0, 5.0])
def test_BUG_err_alt_matches_refraction_derivative(h):
    alt = Altitude.Altitude()
    f = alt.update_alt_by_atmospheric_refraction
    d = (f(h + 1e-4) - f(h - 1e-4)) / 2e-4
    expected = 0.5 * (abs(d - 1.0) + 1.0)
    assert alt.err_alt(h, 0.5) == pytest.approx(expected, rel=1e-3)


# --------------------------------------------------------------------------
# GeoUtils
# --------------------------------------------------------------------------
def test_bearing_and_distance_basics():
    g = GeoUtils.GeoUtils()
    assert g.get_bearing(0, 0, 0, 1) == pytest.approx(90.0, abs=1e-6)
    assert g.distance_between(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)


def test_horizon_height_round_trip():
    g = GeoUtils.GeoUtils()
    for curvature in (True, False):
        for k in (0.0, 0.13):
            deg = g.horizon_height_degrees(30_000, 1500, curvature=curvature, refraction_k=k)
            assert g.horizon_height_meters(30_000, deg, curvature=curvature, refraction_k=k) == pytest.approx(1500)


def test_horizon_curvature():
    g = GeoUtils.GeoUtils()
    # a point at eye level 10 km away is depressed by d/2R
    assert g.horizon_height_degrees(10_000, 0.0) == pytest.approx(-math.degrees(10_000 / (2 * 6_371_000)), rel=1e-4)
    # the surface drops ~d^2/2R
    assert g.horizon_height_meters(10_000, 0.0) == pytest.approx(10_000 ** 2 / (2 * 6_371_000), rel=1e-4)
    # curvature=False reproduces the flat formula
    assert g.horizon_height_degrees(2000, 120, curvature=False) == pytest.approx(math.degrees(math.atan(120 / 2000)))
    with pytest.raises(ValueError):
        g.horizon_height_degrees(0, 10)


def test_fmap_kernel_widths_independent_of_units():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = Declination.Declination()
    fig = d.plot_fmap([0.0], [0.0], 2.0, -20, 20, -10, 10, grid=401, norm=False,
                      ymarkers=False, show=False, bw_x=10.0)
    Z = fig.axes[0].images[0].get_array().data
    xs, ys = np.linspace(-20, 20, 401), np.linspace(-10, 10, 401)
    py, px = Z.sum(1), Z.sum(0)
    assert np.sqrt((py * ys ** 2).sum() / py.sum()) == pytest.approx(1.0, abs=0.01)   # bw = 2 sigma
    assert np.sqrt((px * xs ** 2).sum() / px.sum()) == pytest.approx(5.0, abs=0.05)
    # same data in years and in centuries -> same picture
    rng = np.random.default_rng(0)
    years, decl = rng.uniform(-3000, 0, 50), rng.normal(0, 15, 50)
    z1 = d.plot_fmap(years, decl, 2.0, -3000, 0, -40, 40, grid=80, show=False).axes[0].images[0].get_array().data
    z2 = d.plot_fmap(years / 100, decl, 2.0, -30, 0, -40, 40, grid=80, show=False).axes[0].images[0].get_array().data
    np.testing.assert_allclose(z1, z2, atol=1e-9)
    plt.close("all")


def test_BUG_heatmap_uses_observer_latitude():
    g = GeoUtils.GeoUtils()
    # Observer at 40°N looks due north to a target at eye level at 40.5°N.
    # On a flat Earth: az=0, alt=0 -> declination = 90 - 40 = 50.
    df = g.compute_error_heatmap([[40.5, 0.0, 0.0]], [["sun", 50.0]], [[40.0], [0.0]],
                                 curvature=False)
    assert df.err_val.iloc[0] == pytest.approx(0.0, abs=0.1)
    # With curvature the 55 km distant target sits ~0.25° below the horizon,
    # which lowers the declination by about that amount
    df = g.compute_error_heatmap([[40.5, 0.0, 0.0]], [["sun", 50.0]], [[40.0], [0.0]])
    assert df.err_val.iloc[0] == pytest.approx(0.2, abs=0.1)


def test_BUG_location_probability_is_a_row_fraction():
    g = GeoUtils.GeoUtils()
    df = pd.DataFrame({"lat": 0.0, "lon": 0.0,
                       "err_val": np.r_[np.zeros(500), np.full(500, 5.0)]})
    assert g.compute_location_probability(df, threshold=2) == pytest.approx(0.5, abs=0.05)


def test_vincenty_published_test_line():
    # Flinders Peak -> Buninyong (Vincenty 1975): azimuth 306 52' 05.37"
    lat1 = -(37 + 57 / 60 + 3.72030 / 3600); lon1 = 144 + 25 / 60 + 29.52440 / 3600
    lat2 = -(37 + 39 / 60 + 10.15610 / 3600); lon2 = 143 + 55 / 60 + 35.38390 / 3600
    az = float(GeoUtils.GeoUtils._inverse_azimuth_wgs84(lat1, lon1, lat2, lon2)) % 360
    assert az == pytest.approx(306 + 52 / 60 + 5.37 / 3600, abs=1e-5)


def test_vectorized_bearings_match_geographiclib():
    geod = pytest.importorskip("geographiclib.geodesic")
    if not hasattr(geod.Geodesic.WGS84, "Inverse") or "stubs" in str(geod.__file__):
        pytest.skip("real geographiclib not available")
    rng = np.random.default_rng(0)
    lat1, lon1 = rng.uniform(-60, 60, 300), rng.uniform(-180, 180, 300)
    lat2, lon2 = lat1 + rng.uniform(-0.5, 0.5, 300), lon1 + rng.uniform(-0.5, 0.5, 300)
    ours = GeoUtils.GeoUtils._inverse_azimuth_wgs84(lat1, lon1, lat2, lon2)
    ref = [geod.Geodesic.WGS84.Inverse(a, b, c, d)["azi1"] for a, b, c, d in zip(lat1, lon1, lat2, lon2)]
    assert np.max(np.abs(HF.ang_diff_deg(ours, np.array(ref)))) < 1e-6


def test_heatmap_cell_on_target_is_nan_and_lengths_checked():
    g = GeoUtils.GeoUtils()
    df = g.compute_error_heatmap([[45.8, 21.3, 100.0]], [["sun", 0.0]], [[45.8, 45.7], [21.3]])
    assert np.isnan(df.err_val.iloc[0]) and not np.isnan(df.err_val.iloc[1])
    with pytest.raises(ValueError):
        g.compute_error_heatmap([[45.8, 21.3, 1.0]], [["sun", 0.0], ["sun", 1.0]], [[45.0], [21.0]])


NOAA_XML = """<maggridresult><model>WMM-2025</model><result>
<declination units="Degree">
7.65734
</declination>
<declination_uncertainty units="Degree">
0.36816
</declination_uncertainty></result></maggridresult>"""


def _fake_urlopen(body=None, exc=None):
    import io
    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()
    def urlopen(url, timeout=None):
        if exc: raise exc
        return Resp(body.encode())
    return urlopen


def test_magnetic_declination_parses_noaa_xml(monkeypatch):
    monkeypatch.setattr(GeoUtils, "urlopen", _fake_urlopen(NOAA_XML))
    assert GeoUtils.GeoUtils().get_magnetic_declination(40, -105.25, 2026, 1, 1) == (7.65734, 0.36816)


def test_magnetic_declination_errors(monkeypatch):
    from urllib.error import URLError
    g = GeoUtils.GeoUtils()
    monkeypatch.setattr(GeoUtils, "urlopen", _fake_urlopen(exc=URLError("offline")))
    with pytest.raises(RuntimeError):
        g.get_magnetic_declination(45.75, 21.23, 2026, 1, 1)
    monkeypatch.setattr(GeoUtils, "urlopen", _fake_urlopen("<html>Error</html>"))
    with pytest.raises(RuntimeError):
        g.get_magnetic_declination(45.75, 21.23, 2026, 1, 1)
    with pytest.raises(ValueError):
        g.get_magnetic_declination(45.75, 21.23, 2023, 1, 1)          # outside WMM years
    with pytest.raises(ValueError):
        g.get_magnetic_declination(45.75, 21.23, 2026, 2, 30)         # invalid date


def test_location_area_matches_spherical_rectangle():
    g = GeoUtils.GeoUtils()
    lats, lons, d = np.arange(45.0, 45.3001, 0.01), np.arange(21.0, 21.5001, 0.01), 0.01
    lat, lon = np.repeat(lats, lons.size), np.tile(lons, lats.size)
    df = pd.DataFrame({"lat": lat, "lon": lon, "err_val": np.where(lat < 45.15 - 1e-9, 1.0, 5.0)})
    below, total = g.compute_location_area(df, 2.0)
    R = 6371.0
    exact = R ** 2 * np.radians(np.ptp(lons) + d) * (np.sin(np.radians(lats.max() + d / 2)) -
                                                  np.sin(np.radians(lats.min() - d / 2)))
    assert total == pytest.approx(exact, rel=1e-3)
    assert 0 < below < total
    df.loc[0, "err_val"] = np.nan                                     # NaN cells are left out
    assert g.compute_location_area(df, 2.0)[1] < total


def test_heatmap_targets_and_combine():
    g = GeoUtils.GeoUtils()
    targets = [[45.615, 21.385, 600.0], [45.785, 21.050, 400.0], [45.705, 21.420, 300.0], [45.610, 21.060, 500.0]]
    bodies = [["sun", -21.0], ["sun", 27.3], ["sun", 0.5], ["moon", -28.0]]
    grid = [np.arange(45.55, 45.85, 0.01) + 0.0003, np.arange(20.95, 21.50, 0.01) + 0.0004]
    for n in (1, 2, 4):                                                  # any number of targets
        df = g.compute_error_heatmap(targets[:n], bodies[:n], grid)
        assert np.isfinite(df.err_val).all()
    rss = g.compute_error_heatmap(targets, bodies, grid, combine="rss").err_val.to_numpy()
    rms = g.compute_error_heatmap(targets, bodies, grid, combine="rms").err_val.to_numpy()
    mx = g.compute_error_heatmap(targets, bodies, grid, combine="max").err_val.to_numpy()
    assert np.max(np.abs(rss - 2 * rms)) <= 0.15                        # rss = rms * sqrt(4)
    assert np.all(mx >= rms - 0.05)
    one = [g.compute_error_heatmap(targets[:1], bodies[:1], grid, combine=c).err_val for c in ("rss", "rms", "max")]
    assert one[0].equals(one[1]) and one[0].equals(one[2])
    with pytest.raises(ValueError):
        g.compute_error_heatmap([], [], grid)
    with pytest.raises(ValueError):
        g.compute_error_heatmap(targets[:1], bodies[:1], grid, combine="mean")


def test_heatmap_visibility_and_distance():
    g = GeoUtils.GeoUtils()
    t, R = [45.70, 21.60, 60.0], 6_371_000.0
    lons = np.arange(21.00, 21.60, 0.0005)
    reach = np.sqrt(2 * R * 1.6) + np.sqrt(2 * R * 61.6)                # both horizon distances
    df = g.compute_error_heatmap([t], [["sun", 0.0]], [[45.70], lons], observer_height_m=1.6)
    d = g._haversine_m(45.70, df.lon.to_numpy(), t[0], t[1])
    assert np.all(np.isinf(df.err_val[d > reach + 50]))
    assert np.all(np.isfinite(df.err_val[d < reach - 50]))
    df = g.compute_error_heatmap([t], [["sun", 0.0]], [[45.70], lons], max_distance_km=20)
    assert np.all(np.isinf(df.err_val[d > 20_050])) and np.all(np.isfinite(df.err_val[d < 19_950]))
    assert df.attrs["max_distance_km"] == 20
    toy = pd.DataFrame({"lat": np.repeat([0.0, 0.01], 2), "lon": np.tile([0.0, 0.01], 2),
                        "err_val": [0.5, np.inf, np.nan, 3.0]})
    assert g.compute_location_probability(toy, 2.0) == pytest.approx(1 / 3)   # inf counts as unsuitable
    below, total = g.compute_location_area(toy, 2.0)
    assert below / total == pytest.approx(1 / 3, abs=0.01)
    with pytest.raises(ValueError):
        g.compute_error_heatmap([t], [["sun", 0.0]], [[45.70], lons], observer_height_m=0)


def test_plot_error_map_legend_and_markers():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = GeoUtils.GeoUtils()
    lats, lons = np.arange(45.6, 45.8001, 0.005), np.arange(21.1, 21.3001, 0.005)
    lat, lon = np.repeat(lats, lons.size), np.tile(lons, lats.size)
    df = pd.DataFrame({"lat": lat, "lon": lon, "err_val": np.hypot(lat - 45.7, lon - 21.2) * 40})
    fig, ax = g.plot_error_map(df, observatory=[45.7, 21.2, 0.0], targetGeoLocations=[[45.65, 21.25, 500.0]],
                               threshold=2.0, show=False)
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "km²" in texts[0] and "mapped" in texts[0]
    assert any("T1 +500 m" in t for t in texts)
    with pytest.raises(ValueError):
        g.plot_error_map(df, threshold=0, show=False)
    plt.close("all")


def test_show_heatmap_best_cells_brightest(monkeypatch, tmp_path):
    calls = {}
    class FakeMap:
        def __init__(self, center=None, zoom=None): calls["center"] = center
        def add_basemap(self, name): pass
        def add_heatmap(self, data, latitude, longitude, value, name, radius): calls["heat"] = (data, value)
        def add_marker(self, *a, **k): pass
        def to_html(self, path): pass
    monkeypatch.setattr(GeoUtils.leafmap, "Map", FakeMap)
    monkeypatch.setattr(GeoUtils.webbrowser, "open_new_tab", lambda url: None)
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"lat": [45.70, 45.71, 45.72, 45.73], "lon": [21.2] * 4,
                       "err_val": [0.05, 0.3, 0.49, 3.0]})
    GeoUtils.GeoUtils().show_heatmap(df, observatory=[45.70, 21.20, 0.05],
                                     targetGeoLocations=[[45.6, 21.4, 600]], filter_threshold=0.5)
    data, value = calls["heat"]
    assert len(data) == 3                                             # 3.0 deg cell filtered out
    assert data.loc[data["err_val"].idxmin(), value] == data[value].max()   # best cell brightest
    assert calls["center"] == pytest.approx((45.715, 21.2))


# --------------------------------------------------------------------------
# JulianHelper
# --------------------------------------------------------------------------
def test_jdn_reference_points(jh):
    jdn = jh._JulianHelper__jdn_from_julian
    assert jdn(-4712, 1, 1) == 0
    assert jdn(1582, 10, 4) == 2299160      # last Julian day before reform


def test_gregorian_from_jdn(jh):
    assert jh._JulianHelper__gregorian_from_jdn(2451545) == (2000, 1, 1)


@pytest.mark.parametrize("y, m, d, off", [(1582, 10, 5, 10), (1700, 3, 1, 11), (2000, 1, 1, 13)])
def test_julian_gregorian_offset(jh, y, m, d, off):
    assert jh.julian_gregorian_offset_days(y, m, d) == off


def test_BUG_offset_on_julian_only_leap_day(jh):
    assert jh.julian_gregorian_offset_days(1900, 2, 29) == 13


def test_BUG_offset_before_1582(jh):
    assert jh.julian_gregorian_offset_days(1000, 3, 21) == 6


def test_leap_years(jh):
    assert jh.is_julian_leap_year(1900) and jh.is_julian_leap_year(-4)
    assert not jh.is_julian_leap_year(1901)


def test_sunrise_equinox_greenwich(jh):
    t = jh.estimate_sunrise_noaa(__import__("datetime").datetime(2024, 3, 20), 0.0, 0.0)
    assert abs((t.hour * 60 + t.minute) - (6 * 60 + 4)) < 10


def test_sunrise_polar_night_returns_none(jh):
    import datetime
    assert jh.estimate_sunrise_noaa(datetime.datetime(2024, 12, 21), 80.0, 0.0) is None


def test_julian_gregorian_round_trips(jh):
    for jdn in range(0, 2_817_152, 997):          # 4712 BCE .. 3000 CE
        assert jh.julian_to_jdn(*jh.jdn_to_julian(jdn)) == jdn
    assert jh.julian_to_gregorian(1582, 10, 4) == (1582, 10, 14)
    assert jh.julian_to_gregorian(1900, 2, 29) == (1900, 3, 13)


def test_offset_before_common_era(jh):
    assert jh.julian_gregorian_offset_days(-500, 6, 21) == -5
    assert jh.julian_gregorian_offset_days(1, 1, 1) == -2


@pytest.mark.parametrize("ymd", [(1901, 2, 29), (2000, 4, 31), (-4713, 1, 1)])
def test_invalid_julian_dates_rejected(jh, ymd):
    with pytest.raises(ValueError):
        jh.julian_to_gregorian(*ymd)


def test_sunrise_east_longitude_sign(jh):
    # Sydney 2024-01-14: sunrise 05:58 AEDT = 18:58 UTC on the previous day
    ut = jh.sunrise_ut_hours_noaa(2024, 1, 14, -33.87, 151.21)
    assert ut == pytest.approx(-(5 + 2 / 60), abs=0.1)


def test_sunrise_datetime_has_utc_date(jh):
    import datetime
    t = jh.estimate_sunrise_noaa(datetime.datetime(2024, 1, 14), -33.87, 151.21)
    assert (t.year, t.month, t.day, t.hour) == (2024, 1, 13, 18)


# --------------------------------------------------------------------------
# HelperFunctions
# --------------------------------------------------------------------------
def test_ang_diff_wraps():
    d = HF.ang_diff_deg(np.array([350.0, 10.0, 0.0]), np.array([10.0, 350.0, 180.0]))
    np.testing.assert_allclose(d, [-20.0, 20.0, 180.0])


def test_BUG_epanechnikov_preserves_variance_for_bw_equal_2sigma():
    sigma = 2.0
    h = HF.gaussian_bw_to_epanechnikov_h(np.array([2 * sigma]))[0]
    assert h / math.sqrt(5) == pytest.approx(sigma)


def test_linear_peaks_found():
    x = np.linspace(-90, 90, 361)
    y = np.exp(-((x - 20) / 3) ** 2) + np.exp(-((x + 30) / 3) ** 2)
    peaks = HF.find_linear_peaks(x, y, prominence=0.5)
    assert [round(p["angle"]) for p in peaks] == [-30, 20]


def test_circular_peak_across_north():
    x = np.arange(0, 360, 0.5)
    y = np.exp(-(HF.ang_diff_deg(x, np.full_like(x, 1.0)) / 3) ** 2)
    peaks = HF.find_circular_peaks(x, y, prominence=0.5)
    assert len(peaks) == 1 and peaks[0]["angle"] == pytest.approx(1.0)


def test_distribution_is_reproducible_with_seed(az):
    data, bw = np.array([90.0, 92.0, 270.0]), np.full(3, 5.0)
    r1 = az.distribution_az_adaptive(data, bw, iterations=5, rng=np.random.default_rng(1))[2]
    r2 = az.distribution_az_adaptive(data, bw, iterations=5, rng=np.random.default_rng(1))[2]
    np.testing.assert_allclose(r1, r2)


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------
def test_BUG_optimal_kmeans_clusters_returns_k():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(np.vstack([rng.normal(c, 0.1, (30, 2)) for c in (0, 5, 10)]),
                     columns=["a", "b"])
    assert Clustering.Clustering().get_optimal_number_of_kmeans_clusters(X) == 3
