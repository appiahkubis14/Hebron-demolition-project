"""Tests for hebron_tools.py (pytest -q test_hebron_tools.py)."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

import hebron_tools as ht


def test_quadkey_bing_example():
    # Bing Maps Tile System documentation: tile (3, 5) at level 3 has quadkey "213"
    assert ht.quadkey(3, 5, 3) == "213"
    # Hebron lies in one or two level-9 tiles, all 9 digits long
    keys = ht.quadkeys_for_bbox((35.04, 31.48, 35.18, 31.60), 9)
    assert 1 <= len(keys) <= 4 and all(len(k) == 9 for k in keys)


def test_search_all_splits_capped_windows():
    days = [date(2020, 1, 1) + timedelta(days=k) for k in range(0, 400, 2)]     # 200 products
    items = [SimpleNamespace(id=f"p{k}", datetime=d) for k, d in enumerate(days)]

    class Q(SimpleNamespace):
        def model_copy(self, update):
            return Q(**{**self.__dict__, **update})

    class Client:
        def search(self, q, providers):
            a, b = date.fromisoformat(q.start_date), date.fromisoformat(q.end_date)
            return [i for i in items if a <= i.datetime <= b][: q.max_results]   # the provider truncates silently

    got = ht.search_all(Client(), Q(max_results=20, start_date=None, end_date=None), ["x"], "2020-01-01", "2021-02-03",
                        window_days=120, log=lambda m: None)
    assert len(got) == 200


def test_lee_filter_keeps_mean_lowers_variance():
    rng = np.random.default_rng(0)
    p = rng.gamma(4.4, 1 / 4.4, (200, 200))        # speckle with ENL 4.4 around 1
    f = ht.lee_filter(p, 5, 4.4)
    assert abs(np.nanmean(f) - 1) < 0.02 and np.nanvar(f) < 0.5 * np.var(p)
    p[50, 50] = np.nan
    assert np.isnan(ht.lee_filter(p, 5)[50, 50])


def make_series(seed=1):
    rng = np.random.default_rng(seed)
    t = np.array([(date(2020, 1, 1) + timedelta(days=6 * k)).toordinal() for k in range(190)], float)   # ~3.1 years
    T = len(t)
    noise = lambda n: rng.normal(0, 1.2, (T, n))
    d0 = date(2021, 9, 10).toordinal()
    x = {}
    x["stable"] = noise(400)
    season = 2.0 * np.sin(2 * np.pi * (t - t[0]) / 365.25)[:, None]
    x["seasonal"] = noise(400) + season
    x["step"] = noise(200) - 4.0 * (t >= d0)[:, None]
    ramp = np.clip((t - d0) / 180.0, 0, 1)[:, None]
    x["gradual"] = noise(200) + 4.0 * ramp
    x["transient"] = noise(200) - 4.0 * ((t >= d0) & (t < d0 + 30))[:, None]
    return t, x, d0


def test_change_test_finds_steps_and_dates_them():
    t, x, d0 = make_series()
    p = ht.TestParams()
    splits = ht.split_dates(t, p)
    r = ht.change_test(x["step"], t, splits, p)
    assert np.median(np.abs(r["t"])) > 6
    assert np.median(r["split"] - d0) <= 6 and np.median(r["split"] - d0) >= 0   # the first acquisition after it
    assert (r["confirm"] == 1).mean() > 0.95
    assert np.median(r["delta"]) < -3


def test_threshold_from_null_controls_false_alarms_and_season_check():
    t, x, d0 = make_series()
    p = ht.TestParams()
    splits = ht.split_dates(t, p)
    thr = np.quantile(ht.null_max_abs_t(x["stable"], t, splits, p, np.random.default_rng(3)), 0.99)
    stable = np.abs(ht.change_test(x["stable"], t, splits, p)["t"])
    assert (stable >= thr).mean() < 0.05
    seasonal = np.abs(ht.change_test(x["seasonal"], t, splits, p)["t"])
    seasonal_off = np.abs(ht.change_test(x["seasonal"], t, splits, ht.TestParams(season_check=False))["t"])
    assert (seasonal >= thr).mean() < (seasonal_off >= thr).mean()     # the season check removes seasonal alarms
    step = np.abs(ht.change_test(x["step"], t, splits, p)["t"])
    assert (step >= thr).mean() > 0.9


def test_transient_not_confirmed_and_gradual_ratio():
    t, x, d0 = make_series()
    p = ht.TestParams()
    splits = ht.split_dates(t, p)
    thr = np.quantile(ht.null_max_abs_t(x["stable"], t, splits, p, np.random.default_rng(3)), 0.99)
    tr = ht.change_test(x["transient"], t, splits, p)
    assert (np.abs(tr["t"]) >= thr).mean() < 0.2                  # a 30-day dip does not persist
    gr = ht.change_test(x["gradual"], t, splits, p)
    hit = np.abs(gr["t"]) >= thr
    assert hit.mean() > 0.5 and np.nanmedian(gr["pre_trend"][hit]) > 0.4     # built up over months
    st = ht.change_test(x["step"], t, splits, p)
    assert np.nanmedian(st["pre_trend"]) < 0.25                                # abrupt


def test_stackstore_roundtrip_and_extension(tmp_path):
    s = ht.StackStore.open(tmp_path, ["2020-01-01", "2020-01-07"], (3, 4))
    s.write("2020-01-07", np.ones((3, 4)), "a.tif")
    s2 = ht.StackStore.open(tmp_path, ["2020-01-01", "2020-01-07", "2020-01-13"], (3, 4))
    mm = s2.memmap()
    assert np.isnan(mm[0]).all() and (mm[1] == 1).all() and np.isnan(mm[2]).all()
    assert s2.filled_dates() == ["2020-01-07"]


def test_footprints_and_clusters():
    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import box

    tr = from_origin(0, 100, 10, 10)                              # 10 x 10 grid of 10 m
    fps = gpd.GeoDataFrame(geometry=[box(0, 70, 30, 100), box(52, 52, 56, 56)], crs=32636)   # 9 px; tiny (no centre)
    fid, pix = ht.footprint_pixels(fps, tr, (10, 10))
    assert (fid == 0).sum() == 9 and (fid == 1).sum() == 1 and pix[fid == 1][0] == 4 * 10 + 5
    score = np.zeros((10, 10)); score[0:3, 0:3] = 9.0
    comb = {"score": score, "split": np.full((10, 10), 737000.0), "confirm": np.ones((10, 10), np.int8),
            "ratio": np.ones((10, 10)), "pre_trend": np.zeros((10, 10))}
    out = ht.footprint_scores(fid, pix, 2, comb, 5.0)
    assert out.loc[0, "score"] == 9.0 and out.loc[0, "frac_above"] == 1.0 and out.loc[1, "score"] == 0.0
    mask = np.zeros((10, 10), bool); mask[6:9, 6:9] = True; mask[0, 9] = True
    lab, feats = ht.change_clusters(mask, tr, 10.0, 500.0, {"score": score})
    assert len(feats) == 1 and feats[0]["properties"]["area_m2"] == 900.0


def test_validation_metrics():
    assert ht.auc([3, 4, 5], [0, 1, 2]) == 1.0
    p, r, th = ht.precision_recall_curve(np.array([0.9, 0.8, 0.1, 0.05]), np.array([1, 0, 1, 0]))
    assert p[0] == 1.0 and r[-1] == 1.0
    s = ht.binary_scores([1, 1, 0, 0], [1, 0, 1, 0])
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["false_positive_rate"] == 0.5


def test_select_seasonal_dates():
    ds = [f"{y}-{m:02d}-{d:02d}" for y in (2022, 2023) for m in range(1, 13) for d in (3, 9, 15, 21, 27)]
    out = ht.select_seasonal_dates(ds, (6, 7, 8, 9, 10), 5)
    assert len(out) == 10 and all(x[5:7] in ("06", "07", "08", "09", "10") and x.endswith("-15") for x in out)
    assert ht.select_seasonal_dates(ds, (6, 7, 8, 9, 10), 5, prefer={"2023-08-03"}).count("2023-08-03") == 1
    assert len(ht.select_seasonal_dates(ds, (6, 7, 8, 9, 10), 3)) == 6
