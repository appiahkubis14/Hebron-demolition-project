"""Building-change detection helpers for the Hebron notebook (v6).

Method, and where each part comes from
--------------------------------------
* **Per-pixel two-sample t-test on Sentinel-1 GRD backscatter** (VV and VH, every track), following the Pixel-Wise
  T-Test (PWTT, arXiv:2405.06323): a reference period of about one year is compared with a short
  inference window; the largest absolute t over polarisations and tracks is the change score; the score of a building
  is the mean over its footprint. PWTT used a fixed pre-war reference; Hebron has no single event date, so here the
  split date slides month by month (reference = the year before the split, inference = the ``post_days`` after it).
* **Persistence**: a detection counts only if the new level still holds in the window after the inference window, as
  LT-CCD requires a detection to recur within the following month (LT-CCD, arXiv:2506.14730).
* **Season check** (optional): the inference window is also compared with the same months one year earlier, as
  LT-CCD matches seasons between its reference and test stacks. A seasonal swing shows no difference there.
* **Speckle**: a Lee filter before the test, as in PWTT (``enl`` 4.4 is ESA's figure for IW GRDH).
* **Multiple testing**: the largest |t| over ~70 split dates and several track/polarisation pairs is far larger than one
  t-value, so the threshold comes from a permutation null (each pixel's dates shuffled, the same search repeated).
* **Footprints**: Microsoft Global ML Building Footprints (QuadKey / Url columns of ``dataset-links.csv``, level-9 quadkeys,
  line-delimited GeoJSON per file), or any footprint file you give.

Nothing here downloads Sentinel-1 data; that stays with pygeofetch in the notebook.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------------------------- archive search


def search_all(client: Any, query: Any, providers: Sequence[str], start: str, end: str, *, window_days: int = 120,
               min_window_days: int = 4, log: Callable[[str], None] = print) -> list[Any]:
    """Every product of ``query`` between ``start`` and ``end``, without hitting the provider's result cap.

    The period is searched in windows of ``window_days``; a window that returns exactly ``query.max_results`` products
    may be truncated, so it is split in two and searched again (down to ``min_window_days``). Products are de-duplicated
    by id. ``query`` is a pygeofetch ``SearchQuery``; only its dates are changed (``model_copy``).
    """
    cap = int(getattr(query, "max_results", 0) or 0)
    seen: dict[str, Any] = {}

    def run(a: date, b: date) -> None:
        q = query.model_copy(update={"start_date": a.isoformat(), "end_date": b.isoformat()})
        items = client.search(q, providers=list(providers)) or []
        if cap and len(items) >= cap and (b - a).days > min_window_days:
            mid = a + (b - a) // 2
            log(f"  {a} .. {b}: {len(items)} results = the cap, splitting")
            run(a, mid)
            run(mid + timedelta(days=1), b)
            return
        if cap and len(items) >= cap:
            log(f"  WARNING {a} .. {b}: still {len(items)} results in {(b - a).days} days; raise max_results")
        for it in items:
            seen.setdefault(str(getattr(it, "id", id(it))), it)

    a, last = date.fromisoformat(start), date.fromisoformat(end)
    while a <= last:
        b = min(a + timedelta(days=window_days - 1), last)
        run(a, b)
        a = b + timedelta(days=1)
    return sorted(seen.values(), key=lambda i: str(getattr(i, "datetime", "")))


# --------------------------------------------------------------------------------------------- speckle


def lee_filter(power: np.ndarray, window: int = 5, enl: float = 4.4) -> np.ndarray:
    """Lee speckle filter on linear power (NaN-aware). ``enl`` = equivalent number of looks of the product
    (ESA: 4.4 for Sentinel-1 IW GRDH). Returns linear power, NaN where the input is NaN."""
    from scipy.ndimage import uniform_filter

    if window <= 1:
        return np.asarray(power, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    ok = np.isfinite(p) & (p > 0)
    w = ok.astype(np.float64)
    pz = np.where(ok, p, 0.0)
    nw = np.maximum(uniform_filter(w, window, mode="nearest"), 1e-12)
    mean = uniform_filter(pz, window, mode="nearest") / nw
    mean_sq = uniform_filter(pz * pz, window, mode="nearest") / nw
    var = np.maximum(mean_sq - mean * mean, 0.0)
    cu2 = 1.0 / enl
    ci2 = var / np.maximum(mean * mean, 1e-30)
    k = np.clip((1.0 - cu2 / np.maximum(ci2, 1e-30)) / (1.0 + cu2), 0.0, 1.0)
    out = mean + k * (pz - mean)
    return np.where(ok, out, np.nan)


# --------------------------------------------------------------------------------------------- on-disk stacks


@dataclass
class StackStore:
    """A (dates, H, W) float32 stack on disk (numpy memmap) plus a JSON record of which dates are filled.

    All planned dates are allocated at creation (NaN), so a later run fills the gaps without rebuilding.
    """

    folder: Path
    dates: list[str]
    shape: tuple[int, int]
    filled: dict[str, str] = field(default_factory=dict)      # date -> source file it was filled from

    @property
    def path(self) -> Path:
        return self.folder / "stack.f32"

    @property
    def meta_path(self) -> Path:
        return self.folder / "stack.json"

    @classmethod
    def open(cls, folder: str | Path, dates: Sequence[str], shape: tuple[int, int]) -> StackStore:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        dates = sorted(set(dates))
        meta = json.loads((folder / "stack.json").read_text()) if (folder / "stack.json").exists() else None
        if meta and meta["dates"] == dates and tuple(meta["shape"]) == tuple(shape) and (folder / "stack.f32").exists():
            return cls(folder, dates, tuple(shape), meta.get("filled", {}))
        old = cls(folder, meta["dates"], tuple(meta["shape"]), meta.get("filled", {})) if meta and (folder / "stack.f32").exists() else None
        store = cls(folder, dates, tuple(shape))
        tmp = folder / "stack.f32.tmp"
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(dates), *shape))
        mm[:] = np.nan
        if old is not None and tuple(old.shape) == tuple(shape):        # keep what an earlier date list already filled
            omm = old.memmap("r")
            for d, src in old.filled.items():
                if d in dates:
                    mm[dates.index(d)] = omm[old.dates.index(d)]
                    store.filled[d] = src
            del omm
        mm.flush()
        del mm
        tmp.replace(store.path)
        store.save()
        return store

    def memmap(self, mode: str = "r") -> np.memmap:
        return np.lib.format.open_memmap(self.path, mode=mode)

    def save(self) -> None:
        self.meta_path.write_text(json.dumps({"dates": self.dates, "shape": list(self.shape), "filled": self.filled}))

    def write(self, d: str, image: np.ndarray, source: str) -> None:
        mm = self.memmap("r+")
        mm[self.dates.index(d)] = np.asarray(image, dtype=np.float32)
        mm.flush()
        del mm
        self.filled[d] = str(source)
        self.save()

    def filled_dates(self) -> list[str]:
        return [d for d in self.dates if d in self.filled]


# --------------------------------------------------------------------------------------------- the change test


@dataclass(frozen=True)
class TestParams:
    ref_days: int = 365            # reference window before the split
    post_days: int = 60            # inference window after the split (PWTT: two months)
    confirm_days: int = 120        # the new level must still hold here (persistence)
    min_ref: int = 15              # dates needed in the reference window
    min_post: int = 3              # ... in the inference window
    min_confirm: int = 3           # ... in the confirmation window (fewer: "not testable", kept but flagged)
    season_check: bool = True      # compare the inference window with the same months a year before
    season_min_dates: int = 3
    season_effect: bool = False    # add each pixel's own season-to-season variation to the test's uncertainty (season_excess_var)


def split_dates(t_days: np.ndarray, p: TestParams, step_days: int | None = None) -> np.ndarray:
    """Candidate split dates (ordinal days) where a full reference window fits before and an inference window after.

    ``step_days=None`` (default): every acquisition date, so a change is dated to the first acquisition that shows it
    (the event lies between that acquisition and the one before). An integer: a regular grid (faster, coarser).
    """
    t = np.sort(np.asarray(t_days, dtype=np.float64))
    if t.size == 0:
        return np.zeros(0)
    first, last = t[0] + p.ref_days, t[-1] - p.post_days
    if last < first:
        return np.zeros(0)
    if step_days is None:
        return t[(t >= first) & (t <= last)]
    return np.arange(first, last + 1e-9, step_days)


def _prefix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ok = np.isfinite(x)
    x0 = np.where(ok, x, 0.0).astype(np.float64)
    zeros = np.zeros((1,) + x.shape[1:])
    cn = np.concatenate([zeros, np.cumsum(ok, axis=0, dtype=np.float64)])
    cs = np.concatenate([zeros, np.cumsum(x0, axis=0)])
    cq = np.concatenate([zeros, np.cumsum(x0 * x0, axis=0)])
    return cn, cs, cq


def _window(cn, cs, cq, i0: int, i1: int):
    n = cn[i1] - cn[i0]
    s = cs[i1] - cs[i0]
    q = cq[i1] - cq[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
        v = (q - s * s / n) / (n - 1)
    return n, m, v


def season_excess_var(x: np.ndarray, t_days: np.ndarray, min_seasons: int = 3) -> np.ndarray:
    """Each pixel's own season-to-season variance beyond what independent dates explain (dB^2), per pixel.

    Seasons are calendar years (the dry-season series has one season a year). From the series ``x`` (T, ...):
    the variance of a season's mean is estimated robustly from the differences between consecutive seasons
    (median |difference| = 0.954 sd, so one real step among them does not inflate it), and what independent dates
    would give (pooled within-season variance / dates per season) is subtracted; negative values become 0.
    With fewer than ``min_seasons`` seasons it returns 0.
    """
    x = np.asarray(x, dtype=np.float64)
    yrs = np.array([date.fromordinal(int(v)).year for v in np.asarray(t_days)])
    seasons = sorted(set(yrs))
    if len(seasons) < min_seasons:
        return np.zeros(x.shape[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.stack([np.nanmean(x[yrs == y], 0) for y in seasons])
        within = np.stack([np.nanvar(x[yrs == y], 0, ddof=1) if (yrs == y).sum() > 1 else np.full(x.shape[1:], np.nan) for y in seasons])
        nper = np.stack([np.isfinite(x[yrs == y]).sum(0) for y in seasons]).astype(float)
        w = np.nanmean(within, 0)
        nbar = np.maximum(np.nanmean(np.where(nper > 0, nper, np.nan), 0), 1)
        sb = (np.nanmedian(np.abs(np.diff(means, axis=0)), 0) / 0.954) ** 2
    out = np.clip(sb - w / nbar, 0, None)
    return np.where(np.isfinite(out), out, 0.0)


def _season_weight(t: np.ndarray, a: int, b: int, c: int) -> float:
    """sum over seasons of (share of inference dates in it - share of reference dates in it)^2: how much of a
    season-to-season difference ends up in the difference of the two windows' means."""
    yrs = np.array([date.fromordinal(int(v)).year for v in t])
    r, q = yrs[a:b], yrs[b:c]
    if r.size == 0 or q.size == 0:
        return 0.0
    ys = set(r) | set(q)
    return float(sum(((q == y).mean() - (r == y).mean()) ** 2 for y in ys))


def change_test(x: np.ndarray, t_days: np.ndarray, splits: np.ndarray, p: TestParams,
                season_var: np.ndarray | float | None = None) -> dict[str, np.ndarray]:
    """Rolling two-sample (Welch) t-test of every pixel's backscatter, the best split kept per pixel.

    ``x`` (T, ...) backscatter in dB (NaN = no data), dates ``t_days`` (T,) sorted ordinal days, ``splits`` (K,) ordinal
    days. For each split s: reference [s - ref_days, s), inference [s, s + post_days), confirmation
    [s + post_days, s + post_days + confirm_days), same season a year before [s - 365, s - 365 + post_days).

    A split is *eligible* when both windows have enough dates, the change holds in the confirmation window (same sign,
    at least half the size) or that window has too few dates to tell (end of the series), and, with ``season_check``,
    the change is not just the season (the inference window also differs from the same months a year before, in the
    same direction, by at least half as much; skipped when that window has too few dates).

    With ``season_effect``, the uncertainty of the difference also includes the pixel's own season-to-season
    variation (``season_excess_var``), weighted by how unevenly the two windows cover the seasons (a split at the
    start of a season compares two whole seasons: weight 2; a split inside a season shares that season: less). A
    building whose level wobbles from one year to the next then needs a larger change to score as high.

    Returns, per pixel, for the eligible split with the largest |t|: ``t`` (signed), ``split`` (ordinal day), ``delta``
    (dB, inference minus reference), ``confirm`` (1 confirmed, -1 not testable, 0 no eligible split), ``ratio``
    (confirmation change / inference change: how well the new level holds), ``season_delta`` (dB), ``n_post`` and
    ``pre_trend``: how far the level had already moved in the ``post_days`` before the split, as a fraction of the
    change (late reference minus early reference, over inference minus early reference). Near 0: an abrupt step
    between two acquisitions; near 1: a change that built up over months.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t_days, dtype=np.float64)
    cn, cs, cq = _prefix(x)
    ss = lambda v: int(np.searchsorted(t, v, side="left"))
    shape = x.shape[1:]
    best = {"t": np.zeros(shape), "split": np.full(shape, np.nan), "delta": np.full(shape, np.nan),
            "confirm": np.zeros(shape, np.int8), "ratio": np.full(shape, np.nan), "season_delta": np.full(shape, np.nan),
            "n_post": np.zeros(shape, np.int16), "pre_trend": np.full(shape, np.nan)}
    absbest = np.zeros(shape)
    sv = None
    if getattr(p, "season_effect", False):                  # given (e.g. one value for the whole area) or per pixel
        sv = season_excess_var(x, t) if season_var is None else np.broadcast_to(np.asarray(season_var, float), shape)
    for s in np.asarray(splits, dtype=np.float64):
        a, b, c = ss(s - p.ref_days), ss(s), ss(s + p.post_days)
        d_ = ss(s + p.post_days + p.confirm_days)
        n1, m1, v1 = _window(cn, cs, cq, a, b)
        n2, m2, v2 = _window(cn, cs, cq, b, c)
        n3, m3, _ = _window(cn, cs, cq, c, d_)
        _, me, _ = _window(cn, cs, cq, a, ss(s - p.ref_days / 2))          # first half of the reference year
        _, ml, _ = _window(cn, cs, cq, ss(s - p.post_days), b)             # the post_days just before the split
        with np.errstate(invalid="ignore", divide="ignore"):
            se2 = v1 / n1 + v2 / n2
            if sv is not None:
                se2 = se2 + _season_weight(t, a, b, c) * sv
            se = np.sqrt(se2)
            tt = (m2 - m1) / se
        delta = m2 - m1
        ok = (n1 >= p.min_ref) & (n2 >= p.min_post) & np.isfinite(tt) & (se > 0)
        testable = n3 >= p.min_confirm
        with np.errstate(invalid="ignore"):
            holds = (np.sign(m3 - m1) == np.sign(delta)) & (np.abs(m3 - m1) >= 0.5 * np.abs(delta))
            ratio = (m3 - m1) / delta
            pre_trend = (ml - me) / (m2 - me)
        conf = np.where(testable, np.where(holds, 1, 0), -1).astype(np.int8)
        ok &= conf != 0
        season = np.full(shape, np.nan)
        if p.season_check:
            n4, m4, _ = _window(cn, cs, cq, ss(s - 365), ss(s - 365 + p.post_days))
            season = np.where(n4 >= p.season_min_dates, m2 - m4, np.nan)
            with np.errstate(invalid="ignore"):
                seasonal = np.isfinite(season) & ~((np.sign(season) == np.sign(delta)) & (np.abs(season) >= 0.5 * np.abs(delta)))
            ok &= ~seasonal
        better = ok & (np.abs(tt) > absbest)
        if better.any():
            absbest = np.where(better, np.abs(tt), absbest)
            best["t"] = np.where(better, tt, best["t"])
            best["split"] = np.where(better, s, best["split"])
            best["delta"] = np.where(better, delta, best["delta"])
            best["confirm"] = np.where(better, conf, best["confirm"]).astype(np.int8)
            best["ratio"] = np.where(better, ratio, best["ratio"])
            best["season_delta"] = np.where(better, season, best["season_delta"])
            best["n_post"] = np.where(better, n2, best["n_post"]).astype(np.int16)
            best["pre_trend"] = np.where(better, pre_trend, best["pre_trend"])
    return best


def season_block_order(t_days: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """``n_perm`` permutations of the dates (n_perm, T) that keep seasons together: the seasons (calendar years) are
    shuffled as blocks and the dates inside each block too, so a pixel's season-to-season variation survives while any
    step at a particular time is destroyed. A season that receives a block of a different length takes its dates in
    turn from that block (repeating or dropping some)."""
    t = np.asarray(t_days)
    yrs = np.array([date.fromordinal(int(v)).year for v in t])
    seasons = sorted(set(yrs))
    idx = [np.flatnonzero(yrs == y) for y in seasons]
    out = np.empty((n_perm, t.size), dtype=int)
    for k in range(n_perm):
        src = rng.permutation(len(seasons))
        for target, s_ in zip(idx, src):
            block = rng.permutation(idx[s_])
            out[k, target] = np.resize(block, target.size)
    return out


def null_max_abs_t(series: np.ndarray, t_days: np.ndarray, splits: np.ndarray, p: TestParams,
                   rng: np.random.Generator) -> np.ndarray:
    """Permutation null of the change score: each column of ``series`` (T, N) has its dates shuffled (any real change
    is destroyed, the values and gaps are kept) and the same best-split search is run. Returns |t| per column (N,).

    Shuffling also removes seasonality and other autocorrelation, so this null is optimistic for seasonal ground; the
    quiet-period test of the notebook measures the false-alarm rate on real, unshuffled data.
    """
    x = np.asarray(series, dtype=np.float64)
    if getattr(p, "season_effect", False):                  # keep each pixel's season-to-season variation (see change_test)
        order = season_block_order(t_days, x.shape[1], rng).T
    else:
        order = np.argsort(rng.random(x.shape), axis=0)
    xs = np.take_along_axis(x, order, axis=0)
    return np.abs(change_test(xs, t_days, splits, p)["t"])


def combine(results: Mapping[str, Mapping[str, np.ndarray]], threshold: float | None = None,
            date_tol_days: float = 60.0) -> dict[str, np.ndarray]:
    """Combine per-(track, polarisation) results as PWTT does: the score is the largest |t|. Also returns which result
    gave it (index into ``list(results)``), its split date, and how many results agree (|t| >= ``threshold`` within
    ``date_tol_days`` of that date)."""
    names = list(results)
    a = np.stack([np.abs(results[n]["t"]) for n in names])
    k = np.argmax(a, axis=0)
    take = lambda key: np.take_along_axis(np.stack([results[n][key] for n in names]), k[None], 0)[0]
    out = {"score": a.max(0), "best": k.astype(np.int8), "split": take("split"), "t": take("t"),
           "confirm": take("confirm"), "ratio": take("ratio"), "delta": take("delta"), "pre_trend": take("pre_trend")}
    if threshold is not None:
        sp = np.stack([results[n]["split"] for n in names])
        agree = (a >= threshold) & (np.abs(sp - out["split"][None]) <= date_tol_days)
        out["agree"] = agree.sum(0).astype(np.int8)
    return out


# --------------------------------------------------------------------------------------------- footprints


def tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Web-Mercator (Bing/OSM) tile containing lon/lat at ``zoom``."""
    n = 2 ** zoom
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def quadkey(x: int, y: int, zoom: int) -> str:
    """Bing Maps quadkey of tile (x, y) at ``zoom``."""
    digits = []
    for i in range(zoom, 0, -1):
        mask = 1 << (i - 1)
        digits.append(str((1 if x & mask else 0) + (2 if y & mask else 0)))
    return "".join(digits)


def quadkeys_for_bbox(bbox: Sequence[float], zoom: int = 9) -> list[str]:
    w, s, e, n = bbox
    x0, y0 = tile_xy(w, n, zoom)
    x1, y1 = tile_xy(e, s, zoom)
    return [quadkey(x, y, zoom) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def download_ms_footprints(bbox: Sequence[float], links_urls: Sequence[str], out_path: str | Path,
                           log: Callable[[str], None] = print) -> Path:
    """Microsoft Global ML Building Footprints for ``bbox`` (lon/lat) as a GeoPackage (EPSG:4326).

    Reads ``dataset-links.csv`` (columns ``QuadKey`` and ``Url``; the first of ``links_urls`` that can be read is used),
    takes every row whose level-9 quadkey touches the box (a quadkey can have rows for several locations), reads each
    file (line-delimited GeoJSON, gzip) and keeps the polygons that intersect the box. Properties kept as they are.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box, shape

    links, err = None, []
    for url in links_urls:
        try:
            links = pd.read_csv(url, dtype=str)
            log(f"dataset-links: {url} ({len(links)} rows)")
            break
        except Exception as exc:                                  # try the next published location
            err.append(f"{url}: {type(exc).__name__}: {exc}")
    if links is None:
        raise RuntimeError("could not read dataset-links.csv:\n" + "\n".join(err))
    cols = {c.lower(): c for c in links.columns}
    if "quadkey" not in cols or "url" not in cols:
        raise RuntimeError(f"dataset-links.csv has columns {list(links.columns)}; expected QuadKey and Url")
    keys = set(quadkeys_for_bbox(bbox, 9))
    rows = links[links[cols["quadkey"]].astype(str).str.zfill(9).isin(keys)]
    if rows.empty:
        raise RuntimeError(f"no footprint files for quadkeys {sorted(keys)}")
    loc = cols.get("location")
    log(f"{len(rows)} footprint file(s) for quadkeys {sorted(keys)}" + (f": {sorted(set(rows[loc]))}" if loc else ""))
    aoi = box(*bbox)
    parts = []
    for url in rows[cols["url"]]:
        df = pd.read_json(url, lines=True)
        geoms = [shape(g) for g in df["geometry"]]
        props = pd.json_normalize(df["properties"]) if "properties" in df else pd.DataFrame(index=df.index)
        g = gpd.GeoDataFrame(props, geometry=geoms, crs=4326)
        parts.append(g[g.intersects(aoi)])
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(out_path, driver="GPKG")
    log(f"{len(out)} footprints written to {out_path}")
    return out_path


def footprint_pixels(fp_grid: Any, transform: Any, shape_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Pixels of every footprint: (footprint index, flat pixel index) pairs. A footprint covers the pixels whose centres
    it contains; one smaller than that gets the pixel containing its centroid (so no building is lost). ``fp_grid``
    is a GeoDataFrame in the grid's CRS, indexed 0..N-1."""
    from rasterio.features import rasterize

    h, w = shape_hw
    ids = rasterize(((g, i + 1) for i, g in enumerate(fp_grid.geometry)), out_shape=shape_hw, transform=transform,
                    fill=0, dtype="int32")
    flat = ids.ravel()
    pix = np.nonzero(flat)[0]
    fid = flat[pix] - 1
    have = np.zeros(len(fp_grid), bool)
    have[fid] = True
    miss = np.nonzero(~have)[0]
    if miss.size:
        c = fp_grid.geometry.iloc[miss].centroid
        inv = ~transform
        cols, rows = inv * (c.x.to_numpy(), c.y.to_numpy())
        r, cc = np.floor(rows).astype(int), np.floor(cols).astype(int)
        inside = (r >= 0) & (r < h) & (cc >= 0) & (cc < w)
        fid = np.concatenate([fid, miss[inside]])
        pix = np.concatenate([pix, (r * w + cc)[inside]])
    return fid, pix


def footprint_scores(fid: np.ndarray, pix: np.ndarray, n: int, comb: Mapping[str, np.ndarray], threshold: float,
                     extra: Mapping[str, np.ndarray] | None = None) -> Any:
    """Per footprint: mean change score over its pixels (PWTT's building score), the share of its pixels above
    ``threshold``, and, over those pixels (or all, if none is above), the median split date, the confirmation status
    and the median change ratio. ``extra`` layers (e.g. VV and VH change in dB) are averaged too."""
    import pandas as pd

    df = pd.DataFrame({"fid": fid, "score": comb["score"].ravel()[pix], "split": comb["split"].ravel()[pix],
                       "confirm": comb["confirm"].ravel()[pix], "ratio": comb["ratio"].ravel()[pix],
                       "pre_trend": comb["pre_trend"].ravel()[pix] if "pre_trend" in comb else np.nan})
    if "agree" in comb:
        df["agree"] = comb["agree"].ravel()[pix]
    for k, v in (extra or {}).items():
        df[k] = np.asarray(v).ravel()[pix]
    df["above"] = df["score"] >= threshold
    g = df.groupby("fid")
    out = pd.DataFrame({"pixels": g.size(), "score": g["score"].mean(), "score_max": g["score"].max(),
                        "frac_above": g["above"].mean()})
    hot = df[df["above"]]
    pick = hot if len(hot) else df
    gp = pick.groupby("fid")
    out["split"] = gp["split"].median()
    out["confirmed_frac"] = gp["confirm"].apply(lambda s: float((s == 1).mean()))
    out["untestable_frac"] = gp["confirm"].apply(lambda s: float((s == -1).mean()))
    out["ratio"] = gp["ratio"].median()
    out["pre_trend"] = gp["pre_trend"].median()
    if "agree" in df:
        out["agree"] = gp["agree"].max()
    for k in (extra or {}):
        out[k] = g[k].mean()
    # footprints with no pixel above the threshold: date etc. from all their pixels
    rest = out["split"].isna() | ~out.index.isin(hot["fid"].unique())
    if rest.any():
        ga = df[df["fid"].isin(out.index[rest])].groupby("fid")
        out.loc[rest, "split"] = ga["split"].median()
        out.loc[rest, "confirmed_frac"] = ga["confirm"].apply(lambda s: float((s == 1).mean()))
        out.loc[rest, "untestable_frac"] = ga["confirm"].apply(lambda s: float((s == -1).mean()))
        out.loc[rest, "ratio"] = ga["ratio"].median()
        out.loc[rest, "pre_trend"] = ga["pre_trend"].median()
    return out.reindex(range(n))


def change_clusters(mask: np.ndarray, transform: Any, cell_m: float, min_area_m2: float,
                    layers: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """8-connected groups of ``mask`` of at least ``min_area_m2``, as polygons (grid CRS) with area and the mean / median
    of each layer. Returns (labels, features)."""
    from rasterio.features import shapes
    from scipy import ndimage
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return lab, []
    sizes = np.bincount(lab.ravel())
    keep = np.nonzero(sizes * cell_m * cell_m >= min_area_m2)[0]
    keep = keep[keep > 0]
    lab = np.where(np.isin(lab, keep), lab, 0)
    feats = []
    for i in keep:
        inside = lab == i
        geom = unary_union([shape(g) for g, v in shapes(inside.astype(np.uint8), mask=inside, transform=transform) if v == 1])
        props: dict[str, Any] = {"cluster": int(i), "area_m2": float(inside.sum() * cell_m * cell_m)}
        for k, v in layers.items():
            vals = np.asarray(v, dtype=np.float64)[inside]
            vals = vals[np.isfinite(vals)]
            props[f"{k}_mean"] = float(vals.mean()) if vals.size else None
            props[f"{k}_median"] = float(np.median(vals)) if vals.size else None
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    return lab, feats


# --------------------------------------------------------------------------------------------- validation


def auc(pos: Iterable[float], neg: Iterable[float]) -> float:
    """Probability that a random positive scores above a random negative (Mann-Whitney U / (n1 n2)), ties counted half."""
    from scipy.stats import rankdata

    pos, neg = np.asarray(list(pos), float), np.asarray(list(neg), float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def precision_recall_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall and thresholds over every distinct score (labels: 1 positive, 0 negative; NaN scores dropped)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    order = np.argsort(-s)
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    prec = tp[last] / np.maximum(tp[last] + fp[last], 1)
    rec = tp[last] / max(int(y.sum()), 1)
    return prec, rec, s[last]


def binary_scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p, t = np.asarray(pred, bool), np.asarray(truth, bool)
    tp, fp, fn, tn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if tp else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan")}


# --------------------------------------------------------------------------------------------- thinning the series


def select_seasonal_dates(dates: Sequence[str], months: Sequence[int], per_year: int,
                          prefer: Iterable[str] = ()) -> list[str]:
    """A thinned series: in every year, ``per_year`` acquisition dates from the months ``months``, one per chosen month,
    the one nearest the 15th (a date in ``prefer``, e.g. already processed, wins within its month).

    With fewer ``per_year`` than ``months``, the chosen months are spread evenly over ``months``. Taking the same months
    every year means each year is compared with the same season the year before, so no season check is needed.
    """
    months = list(months)
    if per_year > len(months):                                   # several dates per month, spread over the month
        k = math.ceil(per_year / len(months))
        targets = [round(1 + (j + 0.5) * 30 / k) for j in range(k)]
        by_m: dict[tuple[int, int], list[date]] = {}
        for d in sorted(set(dates)):
            x = date.fromisoformat(d)
            if x.month in months:
                by_m.setdefault((x.year, x.month), []).append(x)
        out = []
        for ds in by_m.values():
            chosen = []
            for tday in targets:
                rest = [x for x in ds if x not in chosen]
                if rest:
                    chosen.append(min(rest, key=lambda x: abs(x.day - tday)))
            out += [x.isoformat() for x in chosen]
        return sorted(out)
    if per_year == len(months):
        use = months
    else:
        use = [months[int(round(k * (len(months) - 1) / max(per_year - 1, 1)))] for k in range(per_year)]
    prefer = set(prefer)
    by_ym: dict[tuple[int, int], list[date]] = {}
    for d in sorted(set(dates)):
        x = date.fromisoformat(d)
        if x.month in use:
            by_ym.setdefault((x.year, x.month), []).append(x)
    out = []
    for (y, m), ds in sorted(by_ym.items()):
        pref = [x for x in ds if x.isoformat() in prefer]
        pool = pref or ds
        out.append(min(pool, key=lambda x: abs(x.day - 15)).isoformat())
    return out


def footprint_null_scores(series: Mapping[str, np.ndarray], t_days: Mapping[str, np.ndarray],
                          splits: Mapping[str, np.ndarray], pix_group: np.ndarray, p: TestParams,
                          rng: np.random.Generator) -> np.ndarray:
    """Permutation null of the **building** score (mean over a footprint of the combined pixel score).

    ``series[name]`` (T_name, P): the pixels of a sample of footprints, for every track/polarisation ``name``;
    ``pix_group`` (P,): which sampled footprint (0..F-1) each pixel belongs to. For every footprint and every name one
    permutation of the dates is drawn and applied to **all** its pixels together, so the spatial correlation inside a
    footprint (speckle filter, resampling) is kept, and the same search is run. Returns the null building scores (F,).

    A footprint's score averages several correlated pixels, so its null is much narrower than that of a single pixel's
    maximum; using the pixel threshold on building scores would be needlessly strict.
    """
    group = np.asarray(pix_group, dtype=int)
    n_f = int(group.max()) + 1 if group.size else 0
    best = None
    for name, x in series.items():
        x = np.asarray(x, dtype=np.float64)
        if getattr(p, "season_effect", False):              # whole seasons shuffled, as in null_max_abs_t
            perms = season_block_order(t_days[name], n_f, rng)
        else:
            perms = np.argsort(rng.random((n_f, x.shape[0])), axis=1)      # one permutation per footprint
        order = perms[group].T                                              # (T, P)
        xs = np.take_along_axis(x, order, axis=0)
        a = np.abs(change_test(xs, t_days[name], splits[name], p)["t"])
        best = a if best is None else np.maximum(best, a)
    sums = np.bincount(group, weights=best, minlength=n_f)
    counts = np.maximum(np.bincount(group, minlength=n_f), 1)
    return sums / counts


# --------------------------------------------------------------------------------------------- terrain correction

def height_on_grid(dem_path: str | Path, transform: Any, shape_hw: tuple[int, int], crs: Any) -> np.ndarray:
    """DEM heights (m) on a grid (bilinear); raises if the DEM leaves more than 0.1 % of the grid empty."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    out = np.full(shape_hw, np.nan, "float64")
    with rasterio.open(dem_path) as src:
        reproject(rasterio.band(src, 1), out, src_nodata=src.nodata, dst_nodata=np.nan, dst_transform=transform,
                  dst_crs=crs, resampling=Resampling.bilinear)
    out[out < -500] = np.nan
    gap = float(np.mean(~np.isfinite(out)))
    if gap > 0.001:
        raise ValueError(f"the DEM {dem_path} leaves {gap:.1%} of the grid without height: use a DEM covering the box")
    out[~np.isfinite(out)] = np.nanmean(out)
    return out


def terrain_shift_m(height: np.ndarray, k: float, c: float, h_mean: float) -> np.ndarray:
    """Apparent displacement towards the satellite (m) of a point at ``height`` in ellipsoid-georeferenced GRD:
    c + k (h - h_mean), with k = cot(incidence) in theory, both measured by hebron_geolocation_check.ipynb."""
    return c + k * (np.asarray(height, float) - h_mean)


def corrected_sample_lonlat(transform: Any, shape_hw: tuple[int, int], crs: Any, shift_m: np.ndarray,
                            to_sat_en: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat at which the GRD image shows each grid cell: the cell's true position moved ``shift_m`` towards the
    satellite (horizontal unit vector ``to_sat_en`` = east, north)."""
    from pyproj import Transformer

    h, w = shape_hw
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = transform.c + transform.a * cols + transform.b * rows
    y = transform.f + transform.d * cols + transform.e * rows
    x = x + shift_m * to_sat_en[0]
    y = y + shift_m * to_sat_en[1]
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def sample_geotiff(path: str | Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinear sample of a single-band lon/lat GeoTIFF at lon/lat (NaN outside the image or next to no-data)."""
    import rasterio
    from scipy.ndimage import map_coordinates

    with rasterio.open(path) as src:
        if src.crs is not None and not src.crs.is_geographic:
            raise ValueError(f"{path}: expected a lon/lat image, got {src.crs}")
        a = src.read(1).astype("float64")
        if src.nodata is not None and np.isfinite(src.nodata):
            a[a == src.nodata] = np.nan
        col, row = ~src.transform * (lon, lat)
    out = map_coordinates(a, [np.asarray(row) - 0.5, np.asarray(col) - 0.5], order=1, mode="constant", cval=np.nan)
    return out.reshape(np.shape(lon))


def simulated_brightness(height: np.ndarray, cell_m: float, to_sat_en: Sequence[float], incidence_deg: float) -> np.ndarray:
    """cos(local incidence angle) from a DEM on a north-up grid: slopes facing the satellite are bright."""
    th = np.radians(incidence_deg)
    L = np.array([to_sat_en[0] * np.sin(th), to_sat_en[1] * np.sin(th), np.cos(th)])
    gy, gx = np.gradient(np.asarray(height, float), cell_m)
    nx, ny, nz = -gx, gy, np.ones_like(gx)                       # rows run north -> south: d/dnorth = -d/drow
    return (nx * L[0] + ny * L[1] + nz * L[2]) / np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)


def residual_offset_m(db: np.ndarray, sim: np.ndarray, open_ground: np.ndarray, cell_m: float, to_sat_en: Sequence[float],
                      search_m: float = 120.0, step_m: float = 5.0) -> tuple[float, float]:
    """Remaining shift of a (corrected) backscatter image along the line of sight, from its correlation with the
    DEM simulation over open ground: the d that maximises corr(image(p), sim(p - d s)). Returns (d in m, best
    correlation). Both images are band-passed (speckle smoothed, regional trend removed) first."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    def bp(a):
        a = np.where(np.isfinite(a), a, np.nanmean(a))
        return gaussian_filter(a, 1.5) - gaussian_filter(a, 25)

    m = np.asarray(open_ground, bool).copy()
    m[:40], m[-40:], m[:, :40], m[:, -40:] = False, False, False, False
    a = bp(db)[m]
    a = (a - a.mean()) / a.std()
    simb = bp(sim)
    rr, cc = np.nonzero(m)
    best = (0.0, -np.inf)
    for d in np.arange(-search_m, search_m + step_m / 2, step_m):
        s = map_coordinates(simb, [rr + d * to_sat_en[1] / cell_m, cc - d * to_sat_en[0] / cell_m], order=1, mode="nearest")
        r = float(np.mean(a * (s - s.mean()) / s.std()))
        if r > best[1]:
            best = (float(d), r)
    return best


# --------------------------------------------------------------------------------------------- co-registration

def offset_lonlat(transform: Any, shape_hw: tuple[int, int], crs: Any, de_m: np.ndarray | float,
                  dn_m: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat of every grid cell's centre moved by (de_m east, dn_m north) metres: where to sample an image that
    shows the cell displaced by that much. ``corrected_sample_lonlat`` is the special case of a shift towards the
    satellite."""
    from pyproj import Transformer

    h, w = shape_hw
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = transform.c + transform.a * cols + transform.b * rows + de_m
    y = transform.f + transform.d * cols + transform.e * rows + dn_m
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def _bandpass(a: np.ndarray, fine: float = 1.0, coarse: float = 12.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    a = np.where(np.isfinite(a), a, np.nanmean(a) if np.isfinite(a).any() else 0.0)
    return gaussian_filter(a, fine) - gaussian_filter(a, coarse)


def tile_offsets(img: np.ndarray, ref: np.ndarray, tile_px: int = 128, max_shift_px: int = 12,
                 min_peak: float = 0.25) -> dict[str, np.ndarray]:
    """How far ``img`` is displaced from ``ref``, tile by tile (both on the same grid, e.g. dB).

    For each tile the displacement d = (rows, cols) maximises the normalised cross-correlation of the band-passed
    images, so that img(p + d) ~ ref(p); sub-pixel by a parabola through the peak. Returns per tile its centre
    (``row``, ``col``), ``dr``, ``dc`` (pixels), the ``peak`` correlation and ``ok``: peak >= ``min_peak``, not on
    the edge of the search window, and fewer than 20 % no-data pixels in the tile.
    """
    a_all, b_all = _bandpass(img), _bandpass(ref)
    bad = ~(np.isfinite(img) & np.isfinite(ref))
    h, w = img.shape
    n = tile_px
    taper = np.outer(np.hanning(n), np.hanning(n))
    out = {k: [] for k in ("row", "col", "dr", "dc", "peak", "ok")}
    for r0 in range(0, h - n + 1, n):
        for c0 in range(0, w - n + 1, n):
            a = a_all[r0:r0 + n, c0:c0 + n]
            b = b_all[r0:r0 + n, c0:c0 + n]
            a = (a - a.mean()) * taper
            b = (b - b.mean()) * taper
            den = math.sqrt(float((a * a).sum() * (b * b).sum())) or 1.0
            cc = np.real(np.fft.ifft2(np.fft.fft2(a, s=(2 * n, 2 * n)) * np.conj(np.fft.fft2(b, s=(2 * n, 2 * n))))) / den
            cc = np.fft.fftshift(cc)                                     # zero shift at (n, n)
            m = max_shift_px
            win = cc[n - m:n + m + 1, n - m:n + m + 1]
            i, j = np.unravel_index(int(np.argmax(win)), win.shape)
            peak = float(win[i, j])

            def sub(v_m, v_0, v_p):
                d = v_m - 2 * v_0 + v_p
                return 0.0 if d >= 0 else 0.5 * (v_m - v_p) / d

            fr = sub(win[i - 1, j], win[i, j], win[i + 1, j]) if 0 < i < 2 * m else 0.0
            fc = sub(win[i, j - 1], win[i, j], win[i, j + 1]) if 0 < j < 2 * m else 0.0
            edge = i in (0, 2 * m) or j in (0, 2 * m)
            out["row"].append(r0 + n / 2)
            out["col"].append(c0 + n / 2)
            out["dr"].append(i - m + fr)
            out["dc"].append(j - m + fc)
            out["peak"].append(peak)
            out["ok"].append(peak >= min_peak and not edge and bad[r0:r0 + n, c0:c0 + n].mean() < 0.2)
    return {k: np.asarray(v) for k, v in out.items()}


def fit_affine(row: np.ndarray, col: np.ndarray, d: np.ndarray, *, k: float = 3.0, n_iter: int = 5) -> dict[str, Any]:
    """Robust plane d ~ a0 + a1 row + a2 col (a shift plus a linear stretch/shear), fitted by least squares with
    outliers beyond ``k`` robust standard deviations removed, repeated ``n_iter`` times. Returns ``coef`` (3,),
    ``scatter`` (robust standard deviation of the residuals of the tiles kept) and ``used`` (boolean mask)."""
    row, col, d = (np.asarray(v, float) for v in (row, col, d))
    A = np.column_stack([np.ones_like(row), row, col])
    used = np.ones(row.size, bool)
    coef = np.zeros(3)
    for _ in range(n_iter):
        if used.sum() < 3:
            break
        coef, *_ = np.linalg.lstsq(A[used], d[used], rcond=None)
        res = d - A @ coef
        s = 1.4826 * np.median(np.abs(res[used] - np.median(res[used]))) + 1e-9
        new = np.abs(res) <= k * max(s, 0.05)
        if (new == used).all():
            break
        used = new
    res = d - A @ coef
    scatter = float(1.4826 * np.median(np.abs(res[used]))) if used.any() else float("nan")
    return {"coef": coef, "scatter": scatter, "used": used}


def coregistration(img: np.ndarray, ref: np.ndarray, *, tile_px: int = 128, max_shift_px: int = 12, min_peak: float = 0.25,
                   min_tiles: int = 8) -> dict[str, Any]:
    """Shift + linear stretch of ``img`` relative to ``ref`` from tile offsets (``tile_offsets`` + ``fit_affine``).

    Returns ``dr``, ``dc`` (full-grid arrays, pixels: img(p + d(p)) ~ ref(p)), the tiles, the fitted coefficients,
    the scatter of the tiles around the fit (pixels), how many tiles were used, and ``ok`` (at least ``min_tiles``
    good tiles). Use the offsets to resample the source file once (``offset_lonlat``), not the gridded image again.
    """
    t = tile_offsets(img, ref, tile_px, max_shift_px, min_peak)
    good = t["ok"]
    h, w = img.shape
    rr, cc = np.mgrid[0:h, 0:w].astype(float)
    res = {"tiles": t, "n_tiles": int(good.sum()), "ok": bool(good.sum() >= min_tiles)}
    if not res["ok"]:
        res.update(dr=np.zeros((h, w)), dc=np.zeros((h, w)), coef_r=np.zeros(3), coef_c=np.zeros(3), scatter_px=float("nan"))
        return res
    fr = fit_affine(t["row"][good], t["col"][good], t["dr"][good])
    fc = fit_affine(t["row"][good], t["col"][good], t["dc"][good])
    res.update(coef_r=fr["coef"], coef_c=fc["coef"], scatter_px=float(np.hypot(fr["scatter"], fc["scatter"])),
               n_tiles=int((fr["used"] & fc["used"]).sum()),
               dr=fr["coef"][0] + fr["coef"][1] * rr + fr["coef"][2] * cc,
               dc=fc["coef"][0] + fc["coef"][1] * rr + fc["coef"][2] * cc)
    return res


# --------------------------------------------------------------------------------------- neighbourhood level

def local_level(img: np.ndarray, window_px: int, block_px: int = 10) -> np.ndarray:
    """The typical value around every pixel: the median over about ``window_px`` x ``window_px`` pixels.

    Computed in two robust steps so it is fast on a full scene: the median of every ``block_px`` x ``block_px`` block,
    then the median of the blocks within the window, interpolated back to every pixel. One building (a few pixels)
    barely moves it; a change shared by the whole neighbourhood moves it fully. NaN pixels are ignored.
    """
    from scipy.ndimage import generic_filter, zoom

    h, w = img.shape
    b = block_px
    hb, wb = -(-h // b), -(-w // b)
    pad = np.full((hb * b, wb * b), np.nan, dtype="float64")
    pad[:h, :w] = img
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        blocks = np.nanmedian(pad.reshape(hb, b, wb, b).transpose(0, 2, 1, 3).reshape(hb, wb, b * b), axis=2)
        k = max(1, int(round(window_px / b)))
        k += (k + 1) % 2                                        # odd window, in blocks
        med = generic_filter(blocks, np.nanmedian, size=k, mode="nearest")
    med = np.where(np.isfinite(med), med, np.nanmedian(med) if np.isfinite(med).any() else 0.0)
    up = zoom(med, (hb * b / hb, wb * b / wb), order=1, mode="nearest", grid_mode=True)
    return up[:h, :w].astype(img.dtype)


def step_test(x: np.ndarray, t_days: np.ndarray, p: TestParams | None = None, min_before: int = 10, min_after: int = 10,
              season_var: np.ndarray | float | None = None, season_starts_only: bool = False) -> dict[str, np.ndarray]:
    """A lasting step anywhere in the series: for every acquisition s with at least ``min_before`` dates before it and
    ``min_after`` from it on, all dates before s are compared with all dates from s on (Welch, plus each series' own
    season-to-season variation weighted by how the two sides cover the seasons, as in ``change_test``). A demolition
    lasts, so every later season is evidence; using them all, rather than one season after, divides the year-to-year
    wobble by the number of seasons on each side. ``season_starts_only``: test only the first date of each season
    (a change during a season is then dated to the next season's start). Returns per series the largest |t|: ``t`` (signed), ``split``
    (ordinal day), ``delta`` (dB, after minus before)."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t_days, dtype=np.float64)
    cn, cs, cq = _prefix(x)
    shape = x.shape[1:]
    sv = season_excess_var(x, t) if season_var is None else np.broadcast_to(np.asarray(season_var, float), shape)
    best = {"t": np.zeros(shape), "split": np.full(shape, np.nan), "delta": np.full(shape, np.nan)}
    absbest = np.zeros(shape)
    T = t.size
    yrs = np.array([date.fromordinal(int(v)).year for v in t])
    for b in range(min_before, T - min_after + 1):
        if b > 0 and t[b] == t[b - 1]:
            continue
        if season_starts_only and yrs[b] == yrs[b - 1]:       # only between seasons: fewer tries, a lower bar
            continue
        n1, m1, v1 = _window(cn, cs, cq, 0, b)
        n2, m2, v2 = _window(cn, cs, cq, b, T)
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(v1 / n1 + v2 / n2 + _season_weight(t, 0, b, T) * sv)
            tt = (m2 - m1) / se
        ok = (n1 >= min_before) & (n2 >= min_after) & np.isfinite(tt)
        better = ok & (np.abs(tt) > absbest)
        if better.any():
            absbest = np.where(better, np.abs(tt), absbest)
            best["t"] = np.where(better, tt, best["t"])
            best["split"] = np.where(better, t[b], best["split"])
            best["delta"] = np.where(better, m2 - m1, best["delta"])
    return best