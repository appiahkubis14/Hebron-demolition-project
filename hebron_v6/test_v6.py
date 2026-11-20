"""Run hebron_demolition_vs_construction_v6.ipynb on a synthetic town, without any download.

    python test_v6.py [TEST_DIR]

Writes, in the notebook's own file layout: sigma0 COGs (linear power, EPSG:4326) of two tracks x VV/VH from 2022 to mid
2025, a footprint file, building-level labels, a labelled-area polygon and the GRD inventory. The town: ~200 buildings
on bare ground, a seasonal field, per-date moisture offsets and speckle (ENL 4.4). Events: 10 demolitions (2023-2024;
VV falls; VH falls for 6, rises for 4) and 4 construction sites off the footprints (backscatter builds up over 150 days).
No events in 2025-01-01 .. 2025-04-30 (the quiet period). Every cell except those tagged "acquisition" is executed,
with pygeofetch functions that need the upgraded package stubbed. The executed notebook is saved to TEST_DIR.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import nbformat
import numpy as np
import rasterio
from nbclient import NotebookClient
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

HERE = Path(__file__).resolve().parent
BBOX = (35.090, 31.520, 35.110, 31.538)
import os
MODE = os.environ.get("MODE", "dry_season")       # "dry_season" (5 dates a year, the notebook's default) or "all"
FIRST = date(2021, 1, 3) if MODE == "dry_season" else date(2022, 1, 3)
TRACKS = {"D094": (("descending", 94), FIRST), "A087": (("ascending", 87), FIRST + timedelta(days=6))}
TRACKS = dict(list(TRACKS.items())[: int(os.environ.get("NTRACKS", "1"))])
PER_YEAR = int(os.environ.get("PER_YEAR", "10"))
END = date(2025, 10, 31) if MODE == "dry_season" else date(2025, 6, 30)
QUIET = ("2025-01-01", "2025-10-31") if MODE == "dry_season" else ("2025-01-01", "2025-04-30")
EV0, EV_DAYS = (date(2022, 3, 1), 900) if MODE == "dry_season" else (date(2023, 3, 1), 600)
CON0 = date(2022, 9, 1) if MODE == "dry_season" else date(2023, 5, 1)


def town(rng):
    to_utm = Transformer.from_crs(4326, 32636, always_xy=True)
    x0, y0 = to_utm.transform(BBOX[0], BBOX[1])
    x1, y1 = to_utm.transform(BBOX[2], BBOX[3])
    blds = []
    for gx in np.arange(x0 + 150, x1 - 150, 45):
        for gy in np.arange(y0 + 150, y1 - 450, 45):
            w, h = rng.uniform(9, 24), rng.uniform(9, 24)
            cx, cy = gx + rng.uniform(-8, 8), gy + rng.uniform(-8, 8)
            blds.append(box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    fp = gpd.GeoDataFrame({"bid": range(len(blds))}, geometry=blds, crs=32636)
    big = fp[fp.area >= 200].index.to_numpy()
    dem_ids = rng.choice(big, 10, replace=False)
    days = [EV0 + timedelta(days=int(d)) for d in rng.integers(0, EV_DAYS, 10)]
    vh_up = set(dem_ids[:4])
    con = []
    for k in range(4):                                        # open ground north of the town
        cx, cy = x0 + 300 + 350 * k, y1 - 250
        con.append((box(cx - 18, cy - 18, cx + 18, cy + 18), CON0 + timedelta(days=150 * k)))
    field = box(x1 - 500, y0 + 100, x1 - 150, y0 + 450)
    return fp, list(zip(dem_ids, days)), vh_up, con, field


def write_data(test_dir: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fp, dems, vh_up, cons, field = town(rng)
    fp.to_crs(4326).to_file(test_dir / "footprints.gpkg", driver="GPKG")
    # grid of the sigma0 files: EPSG:4326, ~10 m, the box plus a margin
    res_lon, res_lat = 0.0001, 0.00009
    m = 0.004
    tr = from_origin(BBOX[0] - m, BBOX[3] + m, res_lon, res_lat)
    w = int((BBOX[2] - BBOX[0] + 2 * m) / res_lon)
    h = int((BBOX[3] - BBOX[1] + 2 * m) / res_lat)
    fp4 = fp.to_crs(4326)
    ids = rasterize(((g, i + 1) for i, g in enumerate(fp4.geometry)), out_shape=(h, w), transform=tr, fill=0, dtype="int32") - 1
    to4 = lambda g: gpd.GeoSeries([g], crs=32636).to_crs(4326).iloc[0]
    con_masks = [rasterize([(to4(g), 1)], out_shape=(h, w), transform=tr, fill=0, dtype="uint8").astype(bool) for g, _ in cons]
    field_m = rasterize([(to4(field), 1)], out_shape=(h, w), transform=tr, fill=0, dtype="uint8").astype(bool)
    dem_day = {int(b): d for b, d in dems}
    inventory = {}
    sig = test_dir / "data" / "grd" / "sigma0"
    for key, (track, first) in TRACKS.items():
        dates = []
        d = first
        while d <= END:
            dates.append(d)
            d += timedelta(days=12)
        if MODE == "dry_season":
            sys.path.insert(0, str(HERE))
            import hebron_tools as ht
            keep = set(ht.select_seasonal_dates([x.isoformat() for x in dates], (6, 7, 8, 9, 10), PER_YEAR))
            dates = [x for x in dates if x.isoformat() in keep]
        inventory[key] = {"track": list(track), "dates": [x.isoformat() for x in dates]}
        geo = 1.0 if key == "D094" else 0.8                   # buildings look a little different from the other side
        for pol in ("VV", "VH"):
            folder = sig if (key == "D094" and pol == "VH") else sig / key / pol.lower()
            folder.mkdir(parents=True, exist_ok=True)
            ground, building = (-12.0, -2.0 * geo) if pol == "VV" else (-19.0, -10.0 * geo)
            for d in dates:
                out = folder / f"S1_{pol.lower()}_sigma0_{d.isoformat()}.tif"
                if out.exists():
                    continue
                img = np.full((h, w), ground)
                img[ids >= 0] = building
                for b, dd in dem_day.items():
                    if d >= dd:                                 # demolished: rubble
                        img[ids == b] = (ground + 3.0) if pol == "VV" else ((-7.0) if b in vh_up else (ground + 2.0))
                for cm, (g, d0) in zip(con_masks, cons):
                    frac = np.clip((d - d0).days / 150.0, 0, 1)
                    img[cm] = ground + frac * (building - ground)
                doy = d.timetuple().tm_yday
                img[field_m] += 3.0 * np.sin(2 * np.pi * doy / 365.25)
                img += rng.normal(0, 0.8)                       # moisture: the whole scene at once
                lin = 10 ** (img / 10) * rng.gamma(4.4, 1 / 4.4, (h, w))
                with rasterio.open(out, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs="EPSG:4326",
                                   transform=tr, nodata=np.nan) as dst:
                    dst.write(lin.astype("float32"), 1)
    res = test_dir / "data" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "grd_inventory.json").write_text(json.dumps(inventory))
    # labels: every event, plus 40 unchanged buildings
    rows = []
    for b, d in dems:
        rows.append({"class": "demolition", "date": d.isoformat(), "geometry": fp.geometry.iloc[b].centroid})
    for g, d0 in cons:
        rows.append({"class": "construction", "date": (d0 + timedelta(days=75)).isoformat(), "geometry": g.centroid})
    calm = [i for i in range(len(fp)) if i not in dem_day]
    for i in rng.choice(calm, 40, replace=False):
        rows.append({"class": "none", "date": None, "geometry": fp.geometry.iloc[i].centroid})
    gpd.GeoDataFrame(rows, crs=32636).to_crs(4326).to_file(test_dir / "labels.gpkg", driver="GPKG")
    gpd.GeoDataFrame(geometry=[box(*BBOX)], crs=4326).to_file(test_dir / "label_area.gpkg", driver="GPKG")
    print(f"synthetic town: {len(fp)} buildings, {len(dems)} demolitions, {len(cons)} construction sites")


STUBS = r'''
import sys, types, inspect, functools
import pygeofetch.sar, pygeofetch.sar.extraction as _ex
from pygeofetch.insar import InterferogramGenerator as _IG
sys.modules.setdefault("pygeofetch.insar.tops", types.ModuleType("pygeofetch.insar.tops"))
_ex._verify_product = lambda p, require_crs=False: (True, "")
pygeofetch.sar.GRDExtractor.extract_sigma0 = lambda self, *a, **k: None
pygeofetch.sar.GRDExtractor.sigma0_filename = lambda self, day, **k: f"S1_{self._pol}_sigma0_{day}.tif"
if "streaming_cleanup" not in inspect.signature(_IG.process_pairs).parameters:
    _orig = _IG.process_pairs
    def _pp(self, specs, out, resume=True, streaming_cleanup=False, **kw):
        return _orig(self, specs, out, resume=resume, **kw)
    _IG.process_pairs = _pp
print("TEST STUBS installed")
'''


def main(test_dir: Path) -> None:
    test_dir.mkdir(parents=True, exist_ok=True)
    write_data(test_dir)
    nb = nbformat.read(HERE / "hebron_demolition_vs_construction_v6.ipynb", as_version=4)
    cells = [c for c in nb.cells if "acquisition" not in c.metadata.get("tags", [])]
    codes = [i for i, c in enumerate(cells) if c.cell_type == "code"]
    cells.insert(codes[0], nbformat.v4.new_code_cell(STUBS))
    codes = [i for i, c in enumerate(cells) if c.cell_type == "code"]
    settings_i, paths_i = codes[2], codes[3]
    assert "BBOX = (" in cells[settings_i].source and "DATA = Path(" in cells[paths_i].source
    cells[paths_i].source = cells[paths_i].source.replace(
        'DATA = Path("/run/media/samuel-appiah-kubi/New Volume/hebronv5")', f'DATA = Path({str(test_dir / "data")!r})')
    cells.insert(settings_i + 1, nbformat.v4.new_code_cell(f'''
BBOX = {BBOX}
START, END = "{FIRST.isoformat()}", "{END.isoformat()}"
SAMPLING = "{MODE}"
DATES_PER_YEAR = {PER_YEAR}
if SAMPLING == "all":
    TEST = ht.TestParams(ref_days=365, post_days=60, confirm_days=120, min_ref=15, min_post=3, min_confirm=3, season_check=True)
    DATE_TOL_DAYS = 60
FOOTPRINTS_PATH = Path({str(test_dir / "footprints.gpkg")!r})
BUILDING_REF_PATH = Path({str(test_dir / "labels.gpkg")!r})
LABEL_AREA_PATH = Path({str(test_dir / "label_area.gpkg")!r})
QUIET_PERIOD = {QUIET}
RUN_COHERENCE = False
N_NULL = 5000
print("TEST SETTINGS", BBOX)
'''))
    nb.cells = cells
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, test_dir / "hebron_v6_test.ipynb")
        print("executed notebook:", test_dir / "hebron_v6_test.ipynb")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/hebron_v6_test"))
