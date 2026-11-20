# ruff: noqa: E501
"""Write hebron_demolition_vs_construction_v6.ipynb (python build_v6.py).

Cells that search or download (Copernicus, OpenTopography, Microsoft footprints are NOT among them: that cell skips itself
when FOOTPRINTS_PATH exists) carry the tag "acquisition"; test_v6.py runs everything else on synthetic data.
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip("\n")))


def code(s, tags=()):
    c = new_code_cell(s.strip("\n"))
    if tags:
        c.metadata["tags"] = list(tags)
    cells.append(c)


ACQ = ("acquisition",)

# =============================================================================================== 0 intro
md(r"""
# Hebron: building demolitions and construction from Sentinel-1 (v6)

**Research question.** Can Sentinel-1 detect building demolitions and new construction in Hebron, date them, and tell them apart?

**What this version produces.** A *candidate* map: buildings (footprints) where the backscatter changed lastingly, with a date, a score and a type, plus new built-up clusters off existing footprints. Every candidate is meant to be confirmed on dated very-high-resolution imagery before it is reported as a demolition. Published Sentinel-1 damage methods describe their output the same way (a screening tool, not a verified assessment), and section 9 measures how good the screening is.

### What changed from v5, and why

| v5 | v6 | Reason |
|---|---|---|
| GRD search capped at 100 results: the run covered only 2026-05-01 – 2026-10-28 | the whole period, searched in windows that split when they hit the cap, then thinned to 10 dry-season dates per year (two per month, June–October) | the test needs a reference year before every date; every acquisition of every track would be ~1,450 GB |
| VH only | **VV and VH** from the same download; one track by default, more when the volume allows | PWTT tests both polarisations and both pass directions; in a Ukraine study VV carried more destruction information than VH, both together best |
| 30 m cells, step between 3-date means | **10 m pixels**, Lee filter, two-sample t-test per pixel of a reference period against the period after each acquisition date; the new level must hold afterwards. Default series: 10 dry-season dates per year (last season = reference, this season = test, next season = confirmation); with every acquisition: a year of reference against 60 days, 120 days to confirm, season check | PWTT (one-year reference, two-month inference window, t-test per pixel, 10 m); LT-CCD (detection must recur; seasons matched). The thinned default keeps the downloads to about 120 GB for one track |
| fixed thresholds (3 dB, z 3) | threshold from a permutation null of the same statistic | the score is a maximum over ~100 dates and 4–6 track/polarisation pairs: a single-test threshold would flag noise everywhere |
| type from the sign of the VH step | type from **what stood there before**: change on an existing footprint = demolition candidate; change off footprints = construction candidate | backscatter after destruction can rise, fall or stay level (Ukraine study); no study we found validates the sign rule |
| cells, 3-cell blobs | **per-building scores** (mean over the footprint, as PWTT) | a house (100–200 m²) is below the 20 × 22 m resolution of GRDH; scoring the footprint uses every pixel it touches |
| coherence drop as a required layer | coherence as **supporting evidence** where SLC pairs exist | coherence covers only the SLC months; requiring it hid every event outside them |
| hit rate at locality points | **quiet-period false-alarm rate** and **building-level** precision / recall / AUC against your own labels | OCHA coordinates are mostly at locality level; LT-CCD measured false positives over a period without events |

### Sources of the method
* *Open Access Battle Damage Detection via Pixel-Wise T-Test on Sentinel-1 Imagery*, arXiv:2405.06323, 2024 (PWTT: building-level AUC 0.81 Gaza, 0.88 Ukraine; F1 falls with class imbalance; 26 % of footprints smaller than one pixel; false positives cluster next to real damage, median 31 m).
* *An Open-Source Tool for Mapping War Destruction at Scale in Ukraine using Sentinel-1 Time Series*, arXiv:2406.02506, 2024 (VV more informative than VH; backscatter after destruction can increase, decrease or stay stable; buildings ≤ 50 m² excluded; a screening tool).
* *Active InSAR monitoring of building damage in Gaza*, arXiv:2506.14730, 2025 (LT-CCD: matched-season pre-event reference, z < −2, recurrence within a month; FPR 1.2 % measured on a counterfactual period; neighbourhood-scale at 40 m).
* ESA, Sentinel-1 Level-1 GRD: IW GRDH resolution 20 × 22 m, pixel spacing 10 m, ENL 4.4.
* Microsoft Global ML Building Footprints (imagery 2014–2021 for most areas, later updates; Middle East precision 95.7 %, recall 85.4 % in their evaluation).

### What it cannot do (state it with the results)
* **Single small structures** (sheds, tents, walls, water tanks) are far below the resolution; expect them to be missed.
* **Rare events.** Demolitions are a tiny share of Hebron's buildings, so precision, not recall, will be the weak number (PWTT's F1 dropped from 0.76 on balanced samples to 0.64 in Gaza overall).
* **The "existed before" rule** is only as good as the footprints and their date: a building built after the footprint imagery and later demolished is labelled construction; a building demolished before it is invisible. Check the footprint date for Hebron (section 5).
* **Neighbours.** A demolition changes backscatter in neighbouring pixels too, so adjacent buildings can be flagged.

**Sections.** 1 setup · 2 complete GRD search · 3 calibrated σ⁰, VV and VH, all tracks · 4 10 m stacks on disk · 5 settlement mask and footprints · 6 change test and threshold · 7 candidates and type · 8 coherence support (optional, SLC) · 9 validation · 10 export and maps.
""")

# =============================================================================================== 1 setup
md("## 1. Setup and configuration")
code(r'''
import gc
import getpass
import inspect
import json
import os
import warnings
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform as warp_xy, transform_bounds
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, distance_transform_edt, label, map_coordinates
from shapely.geometry import box, shape

from pygeofetch import PyGeoFetch
from pygeofetch.core.orbits import fetch_orbit_file
from pygeofetch.insar import InterferogramGenerator
from pygeofetch.insar.interferogram import InterferogramResult
from pygeofetch.insar.workflow import InSARProject
from pygeofetch.models.download_task import DownloadOptions
from pygeofetch.models.search_query import BoundingBox, SearchQuery
from pygeofetch.sar import GRDExtractor
from pygeofetch.sar.extraction import _verify_product      # the same check extract_sigma0 runs before it deletes anything

import hebron_tools as ht                                   # hebron_tools.py sits next to this notebook

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")

assert hasattr(GRDExtractor, "extract_sigma0"), "Install the upgraded sar/extraction.py (it adds GRDExtractor.extract_sigma0)."
assert "streaming_cleanup" in inspect.signature(InterferogramGenerator.process_pairs).parameters, \
    "Install the upgraded insar/interferogram.py (it adds process_pairs(streaming_cleanup=...))."
try:
    import pygeofetch.insar.tops  # noqa: F401  (TOPS deramping: without it Sentinel-1 coherence collapses)
except ImportError:
    raise ImportError("This pygeofetch has no insar/tops.py: install the TOPS-fixed version.")
''')
code(r'''
# ── Run size ────────────────────────────────────────────────────────────────
PILOT = False              # True: a short pipeline check (newest PILOT_N scenes per track); the test needs years, not weeks
PILOT_N = 30
MAX_DOWNLOAD_GB = 200      # refuse to start the GRD downloads above this (raise it consciously: see section 2)
STREAMING_CLEANUP = True   # delete each raw download only after its products are saved AND verified
CHUNK_ROWS = 500           # SLC rows processed at a time (section 8)

# ── Study area and periods ──────────────────────────────────────────────────
BBOX = (35.04, 31.48, 35.18, 31.60)       # Hebron: (min_lon, min_lat, max_lon, max_lat)
AOI_PATH = None                            # optional GeoJSON/GPKG/SHP: its bounds replace BBOX
START, END = "2020-01-01", "2026-10-31"    # GRD series. The first year is reference only: detections start a year after START
GRID_M = 30                                # coherence grid (m), section 8
GRID10_M = 10                              # backscatter grid (m): the GRDH pixel spacing

# ── GRD: tracks, polarisations, search ─────────────────────────────────────
GRD_TRACKS = [("descending", 94)]   # start with one track (v5's, its 2026 VH files are reused); None = every track covering the area
MIN_TRACK_DATES = 5        # with GRD_TRACKS = None: tracks with fewer (thinned) dates are left out
GRD_POLS = ("VV", "VH")
GRD_SEARCH_MAX = 500       # results per search window; a window that hits it is split and searched again
GRD_SEARCH_WINDOW_DAYS = 120
LEGACY_VH_TRACK = ("descending", 94)       # v5 wrote this track's VH files straight into grd/sigma0: reused, not downloaded again

# ── Thinning: how many acquisitions per year are downloaded ─────────────────
SAMPLING = "dry_season"    # "dry_season": DATES_PER_YEAR dates per year, in SEASON_MONTHS, the same months every year; "all": every acquisition
DATES_PER_YEAR = 10       # two per month of SEASON_MONTHS; 5 = one per month (half the download, see section 2)
SEASON_MONTHS = (6, 7, 8, 9, 10)   # Hebron's dry months (rain falls mainly November to April): soil moisture and vegetation change least

# ── Change test (section 6) ────────────────────────────────────────────────
LEE_WINDOW = 5             # Lee speckle filter window (pixels) before the test; 0 = off. ENL 4.4 (ESA, IW GRDH)
if SAMPLING == "dry_season":    # one season per year: last season is the reference, this season the test, next season the confirmation
    TEST = ht.TestParams(ref_days=365, post_days=150, confirm_days=365, min_ref=4, min_post=2, min_confirm=2, season_check=False)
else:                           # every acquisition: PWTT-like windows (a year of reference, two months of test)
    TEST = ht.TestParams(ref_days=365, post_days=60, confirm_days=120, min_ref=15, min_post=3, min_confirm=3, season_check=True)
T_THRESHOLD = None         # None = from the permutation null (NULL_QUANTILE); or a fixed |t|
NULL_QUANTILE = 0.999      # pixel threshold (maps, construction clusters): share of null pixel scores below it
N_NULL = 20_000            # pixels sampled for the pixel null
T_THRESHOLD_BUILDING = None   # None = from the building null (NULL_QUANTILE_BUILDING); or a fixed mean |t|
NULL_QUANTILE_BUILDING = 0.995   # building threshold: share of null building scores below it (0.995 = 5 per 1,000 unchanged buildings)
N_NULL_BUILDINGS = 4000    # footprints sampled for the building null
BLOCK_ROWS = 32            # rows of the 10 m grid tested at a time (memory: ~BLOCK_ROWS x width x dates x 30 bytes)
DATE_TOL_DAYS = 150 if SAMPLING == "dry_season" else 60   # results of different tracks/polarisations agree when this close

# ── Buildings and type (sections 5 and 7) ──────────────────────────────────
FOOTPRINTS_PATH = None     # your footprint file (any vector format); None = Microsoft Global ML Building Footprints, downloaded once
MS_LINKS_URLS = ["https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv",
                 "https://bfppub.blob.core.windows.net/%24web/2026-08-13/dataset-links.csv"]
MIN_FOOTPRINT_M2 = 50      # smaller footprints dropped (under half a 10 m pixel; as in the Ukraine study)
MIN_CONFIRMED_FRAC = 0.5   # share of a building's changed pixels whose new level held (persistence)
CON_MIN_AREA_M2 = 400      # construction clusters off footprints: at least 4 pixels of 10 m
CONSTRUCTION_ZONE_M = 300  # construction searched within this distance of existing footprints (and the settlement mask)
SPILLOVER_M = 30           # a construction cluster this close to a demolition candidate of the same date is the same event
WSF_PATH = DW_PATH = None  # optional settlement masks (WSF value > 0, Dynamic World class 6)

# ── Coherence support (section 8, optional) ────────────────────────────────
RUN_COHERENCE = True
SLC_START, SLC_END = "2026-06-01", "2026-10-31"
MAX_SLC, SLC_DATES = 16, None
TRACK = LEGACY_VH_TRACK    # SLC track (the coherence chain of v5)
COH_DROP, COH_STABLE, TOL_DAYS = 0.30, 0.40, 24

# ── Validation (section 9) ─────────────────────────────────────────────────
QUIET_PERIOD = None        # ("YYYY-MM-DD", "YYYY-MM-DD"): a period with NO known demolitions in QUIET_AREA_PATH (or the whole AOI)
QUIET_AREA_PATH = None
BUILDING_REF_PATH = None   # your building-level labels: points or polygons, column "class" (demolition / construction / none), optional "date"
LABEL_AREA_PATH = None     # polygons where EVERY change was mapped (needed for precision); None = recall and AUC only
REF_MATCH_M = 20           # a candidate matches a label within this distance
OCHA_PATH = PEACENOW_PATH = ARIJ_PATH = None    # locality-level references (section 9c)
REF_RADIUS_M = 150
SEED = 0

if PILOT:
    print(f"PILOT: newest {PILOT_N} GRD scenes per track. With fewer than about a year and a half of dates the test "
          "has no reference year: use it to check the chain only.")
''')
code(r'''
DATA = Path("/run/media/samuel-appiah-kubi/New Volume/hebronv5")
RAW_GRD, SIGMA0_DIR, DEM_DIR, IFG_DIR, RESULTS = (DATA / s for s in ("grd/raw", "grd/sigma0", "dem", "ifg", "results"))
STACK_DIR, CHANGE_DIR, FOOT_DIR = DATA / "grd/stacks10", RESULTS / "change", DATA / "footprints"
for p in (RAW_GRD, SIGMA0_DIR, DEM_DIR, IFG_DIR, RESULTS, DATA / "slc", STACK_DIR, CHANGE_DIR, FOOT_DIR):
    p.mkdir(parents=True, exist_ok=True)
if AOI_PATH:
    BBOX = tuple(float(v) for v in gpd.read_file(AOI_PATH).to_crs(4326).total_bounds)
aoi, aoi_poly = BoundingBox.from_string(",".join(map(str, BBOX))), box(*BBOX)

# Two grids in UTM 36N (Hebron lies at 35.1 deg E): 10 m for backscatter, 30 m for coherence.
GRID_CRS = CRS.from_epsg(32636)
left, bottom, right, top = transform_bounds("EPSG:4326", GRID_CRS, *BBOX)
W, H = int((right - left) // GRID_M), int((top - bottom) // GRID_M)
GRID_T = from_origin(left, top, GRID_M, GRID_M)
EXTENT = (left, left + W * GRID_M, top - H * GRID_M, top)
W10, H10 = int((right - left) // GRID10_M), int((top - bottom) // GRID10_M)
GRID10_T = from_origin(left, top, GRID10_M, GRID10_M)
EXTENT10 = (left, left + W10 * GRID10_M, top - H10 * GRID10_M, top)
INVENTORY = RESULTS / "grd_inventory.json"      # dates per track (section 2), so later sections work in a fresh kernel
print(f"backscatter grid {H10} x {W10} at {GRID10_M} m; coherence grid {H} x {W} at {GRID_M} m")
''')
code(r'''
client = PyGeoFetch()
# client.add_credentials("copernicus", username=os.environ.get("CDSE_USER") or input("Copernicus e-mail: "),
#                        password=os.environ.get("CDSE_PASS") or getpass.getpass("Copernicus password: "))
# client.add_credentials("opentopography", api_key=os.environ.get("OPENTOPO_KEY") or getpass.getpass("OpenTopography API key: "))
''', ACQ)
code(r'''
def name_of(item):
    return item.properties.get("name") or item.id

def track_of(item):
    return (str(item.pass_direction).lower(), int(item.relative_orbit))

def track_key(track):
    """('descending', 94) -> 'D094' (used in folder and layer names)."""
    return f"{track[0][0].upper()}{int(track[1]):03d}"

def covers_aoi(item):
    return item.geometry is None or shape(item.geometry).buffer(0).contains(aoi_poly)

def one_per_date(items, track, only=None):
    """One scene per acquisition date on `track`, oldest first (only = optional list of dates)."""
    out = {}
    for item in sorted(items, key=lambda i: i.datetime):
        if track_of(item) == tuple(track) and (not only or str(item.datetime)[:10] in only):
            out.setdefault(str(item.datetime)[:10], item)
    return list(out.values())

def newest_or_first(items, n):
    """Pilot: the newest n scenes; full run: the first n. None = all."""
    return items if not n else (items[-n:] if PILOT else items[:n])

def size_gb(items):
    return sum((i.assets["download"].size_bytes or 0) for i in items if "download" in i.assets) / 1e9


GRD_KW = dict(lut="sigmaNought", remove_noise=True, out_db=False, aoi_bbox=BBOX)
_NAME_KW = {k: GRD_KW[k] for k in ("lut", "out_db", "aoi_bbox")}
EXTRACTOR = {pol: GRDExtractor(pol) for pol in GRD_POLS}


def sigma0_dir(track, pol):
    if tuple(track) == tuple(LEGACY_VH_TRACK) and pol == "VH":
        return SIGMA0_DIR                                       # v5 layout, reused
    d = SIGMA0_DIR / track_key(track) / pol.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d


def sigma0_path(track, pol, day):
    return sigma0_dir(track, pol) / EXTRACTOR[pol].sigma0_filename(day, **_NAME_KW)
''')

# =============================================================================================== 2 search
md(r"""
## 2. Complete GRD search, then a thinned series

v5 asked for at most 100 results and got the 100 newest products, so its series started on 2026-05-01. Here the period is searched in windows of `GRD_SEARCH_WINDOW_DAYS`; a window that returns `GRD_SEARCH_MAX` results may be truncated and is split in two and searched again (`ht.search_all`).

**Thinning (`SAMPLING = "dry_season"`).** Every acquisition of every track from 2020 would be about 1,450 GB of downloads. Instead, `DATES_PER_YEAR` dates per year are kept in `SEASON_MONTHS` (10: two per month, near the 8th and the 23rd; 5: one per month, nearest the 15th; dates already processed win), on the tracks in `GRD_TRACKS`. For one track from 2020, 10 per year is about 70 scenes (~120 GB), 5 per year about 35 (~60 GB). Taking the same dry months every year means each year is compared with the same season a year earlier, so moisture and vegetation differ as little as possible between the two. The table shows both volumes per track.

**What thinning costs**, stated with the results:
* **Dating** is to the season: a change is placed between the last acquisition that does not show it and the first that does, which can be up to about eight months apart (October to June).
* **Statistics**: about 10 reference and up to 10 test dates per pixel instead of ~30 and ~5 (PWTT). The threshold comes from the permutation null of the same thinned series, and sections 9a–9b measure the result.
* **On a synthetic town** (1 track, 1,120 buildings, 10 demolitions, 4 construction sites; `test_v6.py`): every acquisition found 10/10 demolitions dated within 5 days (median); 10 dry-season dates per year found 10/10, dated within 49 days; 5 per year found only 5/10, dated within 112 days. Precision was 0.42–0.50 in all three (about 5 false candidates per 1,000 buildings at the default building threshold). That is why the default is 10, not 5. These are synthetic numbers; only section 9 measures Hebron.
* **One track** sees the town from one side only; buildings in radar shadow or layover from that side are not tested. Add a second track (an ascending one) when the volume allows: PWTT combines both pass directions.

Downloads are streamed: one raw scene (~1.7 GB) is on disk at a time; what stays is ~20 MB per date (VV + VH, cropped).
""")
code(r'''
query = SearchQuery(bbox=aoi, start_date=START, end_date=END, product_type="GRD", polarisation="VH", max_results=GRD_SEARCH_MAX)
found = ht.search_all(client, query, ["copernicus"], START, END, window_days=GRD_SEARCH_WINDOW_DAYS)
grd = [i for i in found if "_GRD" in name_of(i) and "COG" not in name_of(i) and covers_aoi(i)]
assert grd, "No Sentinel-1 GRD scene covers the AOI in this period."

def thinned(items, track):
    """The dates kept for the test: all, or DATES_PER_YEAR per year in SEASON_MONTHS (dates already processed win)."""
    if SAMPLING != "dry_season":
        return items
    dates = [str(i.datetime)[:10] for i in items]
    done = {d for d in dates if any(sigma0_path(track, pol, d).exists() for pol in GRD_POLS)}
    keep = set(ht.select_seasonal_dates(dates, SEASON_MONTHS, DATES_PER_YEAR, prefer=done))
    return [i for i in items if str(i.datetime)[:10] in keep]


rows = []
for tr, n in Counter(track_of(i) for i in grd).most_common():
    its = one_per_date(grd, tr)
    th = thinned(its, tr)
    rows.append({"track": tr, "key": track_key(tr), "dates": len(its), "first": str(its[0].datetime)[:10],
                 "last": str(its[-1].datetime)[:10], "satellites": dict(Counter(name_of(i)[:3] for i in its)),
                 "GB, every date": round(size_gb(its), 1), "dates kept": len(th), "GB, dates kept": round(size_gb(th), 1)})
track_table = pd.DataFrame(rows).set_index("key")
display(track_table)

tracks = [tuple(t) for t in GRD_TRACKS] if GRD_TRACKS else [r["track"] for r in rows if r["dates kept"] >= MIN_TRACK_DATES]
grd_items = {track_key(t): newest_or_first(thinned(one_per_date(grd, t), t), PILOT_N if PILOT else None) for t in tracks}
inventory = {track_key(t): {"track": list(t), "dates": [str(i.datetime)[:10] for i in grd_items[track_key(t)]]} for t in tracks}
INVENTORY.write_text(json.dumps(inventory, indent=1))
todo_items = [i for k, v in grd_items.items() for i in v
              if not all(sigma0_path(tuple(inventory[k]["track"]), pol, str(i.datetime)[:10]).exists() for pol in GRD_POLS)]
total = size_gb(todo_items)
print(f"tracks kept: {[track_key(t) for t in tracks]}; {sum(len(v) for v in grd_items.values())} dates, of which "
      f"{len(todo_items)} still to download: about {total:.0f} GB, one scene at a time (~{total * 1e3 / 3 / 3600:.0f} h at 3 MB/s). "
      f"Disk: ~1.7 GB at a time for the raw scene, then ~20 MB per date")
assert total <= MAX_DOWNLOAD_GB or PILOT, (f"{total:.0f} GB > MAX_DOWNLOAD_GB = {MAX_DOWNLOAD_GB}: raise it, or set GRD_TRACKS "
                                           "to fewer tracks, or START later")
''', ACQ)

# =============================================================================================== 3 sigma0
md(r"""
## 3. Calibrated σ⁰, VV and VH, one scene at a time

As in v5: σ⁰ = (DN² − N) / A² with the calibration table A and the thermal noise N removed (`extract_sigma0(lut="sigmaNought", remove_noise=True)`), linear power, cropped to the area, saved as a Cloud-Optimized GeoTIFF and verified. **New:** both polarisations are taken from the same download (dual-polarisation SDV products), and each track and polarisation has its own folder (`grd/sigma0/<track>/<pol>`), so files of different tracks on the same day cannot collide. Only the last polarisation of a scene runs with `streaming_cleanup`, so the raw file is deleted after both are verified. v5's VH files of track D094 (in `grd/sigma0`) are reused.
""")
code(r'''
def process_grd_scene(item, track):
    """Download one GRD scene once, calibrate every polarisation still missing, verify, then delete the raw file
    (STREAMING_CLEANUP, on the last polarisation only). Returns {pol: COG path} for the polarisations that exist."""
    day = str(item.datetime)[:10]
    have, todo = {}, []
    for pol in GRD_POLS:
        p = sigma0_path(track, pol, day)
        if p.exists() and _verify_product(p, require_crs=False)[0]:
            have[pol] = p
        else:
            todo.append(pol)
    if not todo:
        return have                                               # resume: nothing is downloaded again
    try:
        result = client.download([item], destination=RAW_GRD, options=DownloadOptions(parallel=1, resume=True))[0]
        if not result.success or result.output_path is None or not Path(result.output_path).exists():
            print(day, "DOWNLOAD FAILED:", result.error_type, result.error)
            return have
        for k, pol in enumerate(todo):
            last = k == len(todo) - 1
            cog = EXTRACTOR[pol].extract_sigma0(result, sigma0_dir(track, pol), label=day,
                                               streaming_cleanup=STREAMING_CLEANUP and last, **GRD_KW)
            if cog is None:
                print(day, pol, "CALIBRATION FAILED: see the log above; the raw file was kept")
            else:
                have[pol] = Path(cog)
    except Exception as exc:                                       # one bad scene must not stop a 6-year series
        print(day, f"FAILED: {type(exc).__name__}: {exc}")
    finally:
        gc.collect()
    return have


for key, items in grd_items.items():
    track = tuple(inventory[key]["track"])
    done = Counter()
    for n, item in enumerate(items, 1):
        got = process_grd_scene(item, track)
        done.update(got)
        if n % 10 == 0 or n == len(items):
            print(f"{key}: {n}/{len(items)} scenes; {dict(done)}; raw files left: {len(list(RAW_GRD.glob('*.zip')))}")
''', ACQ)

# =============================================================================================== 4 stacks
md(r"""
## 4. 10 m stacks on disk

Each σ⁰ file is resampled to the 10 m UTM grid (bilinear on linear power), Lee-filtered (`LEE_WINDOW`), converted to dB, and the date's median over the area is removed (rain and soil moisture shift the whole scene at once; a building change is local). One stack per track and polarisation is kept on disk as a memory-mapped array (`grd/stacks10/<track>_<pol>/`), so a six-year series never has to fit in memory. Missing dates stay empty (NaN); re-running fills only what is new.
""")
code(r'''
inventory = json.loads(INVENTORY.read_text())
COMBOS = [(key, pol) for key in inventory for pol in GRD_POLS]
combo_name = lambda key, pol: f"{key}_{pol}"


def cog_to_grid10(path):
    """sigma0 COG (linear power) -> 10 m grid: Lee filter, dB, scene median removed."""
    out = np.full((H10, W10), np.nan, "float32")
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, src_nodata=np.nan, dst_nodata=np.nan,
                  dst_transform=GRID10_T, dst_crs=GRID_CRS, resampling=Resampling.bilinear)
    out[~(out > 0)] = np.nan                                      # after noise subtraction a pixel can be <= 0
    out = ht.lee_filter(out, LEE_WINDOW) if LEE_WINDOW else out
    with np.errstate(invalid="ignore", divide="ignore"):
        db = 10 * np.log10(out)
    return (db - np.nanmedian(db)).astype("float32")


STORES = {}
for key, pol in COMBOS:
    track = tuple(inventory[key]["track"])
    store = ht.StackStore.open(STACK_DIR / combo_name(key, pol), inventory[key]["dates"], (H10, W10))
    added = 0
    for d in store.dates:
        p = sigma0_path(track, pol, d)
        if d not in store.filled and p.exists():
            store.write(d, cog_to_grid10(p), str(p))
            added += 1
    STORES[(key, pol)] = store
    print(f"{combo_name(key, pol)}: {len(store.filled_dates())}/{len(store.dates)} dates filled ({added} new)")
COMBOS = [c for c in COMBOS if len(STORES[c].filled_dates()) >= TEST.min_ref + TEST.min_post]
assert COMBOS, "no track/polarisation has enough dates for the test"
''')

# =============================================================================================== 5 masks, footprints
md(r"""
## 5. Settlement mask and building footprints

**Footprints** come from `FOOTPRINTS_PATH` or, if it is not set, from Microsoft Global ML Building Footprints (downloaded once for the area into `footprints/ms_buildings.gpkg`). Footprints under `MIN_FOOTPRINT_M2` are dropped. Each footprint covers the 10 m pixels whose centres fall inside it; a footprint smaller than that gets the pixel containing its centroid, so no building is lost.

**Check the footprint date.** The type rule (section 7) treats a footprint as "a building that stood before the change". Microsoft's imagery for most areas is from 2014–2021 (some regions updated later); the cell prints any date fields the file carries. Changes before that date cannot be typed by this rule, and buildings built after it are missing (their later demolition would be typed as construction). A footprint file with a known date (e.g. OSM history, or your own digitising from an image of a known date) removes that uncertainty.

**Settlement mask** (optional, WSF or Dynamic World, as in v5) limits the construction search together with `CONSTRUCTION_ZONE_M` around footprints.
""")
code(r'''
if FOOTPRINTS_PATH is None:
    FOOTPRINTS_PATH = FOOT_DIR / "ms_buildings.gpkg"
    if not FOOTPRINTS_PATH.exists():
        ht.download_ms_footprints(BBOX, MS_LINKS_URLS, FOOTPRINTS_PATH)
fp = gpd.read_file(FOOTPRINTS_PATH)
fp = fp.set_crs(4326) if fp.crs is None else fp
date_cols = [c for c in fp.columns if any(k in c.lower() for k in ("date", "year", "capture"))]
print(f"{len(fp)} footprints in {Path(FOOTPRINTS_PATH).name}; date fields: {date_cols or 'none (check the source for the imagery date)'}")
for c in date_cols:
    print(f"  {c}: {fp[c].astype(str).value_counts().head(5).to_dict()}")
fp = fp.to_crs(GRID_CRS)
fp = fp[fp.intersects(box(left, bottom, right, top))]
fp["area_m2"] = fp.geometry.area
n0 = len(fp)
fp = fp[fp["area_m2"] >= MIN_FOOTPRINT_M2].reset_index(drop=True)
print(f"{len(fp)} footprints in the area ({n0 - len(fp)} under {MIN_FOOTPRINT_M2} m² dropped); "
      f"median {fp['area_m2'].median():.0f} m², {100 * (fp['area_m2'] < GRID10_M ** 2).mean():.0f}% smaller than one 10 m pixel")
FID, FPIX = ht.footprint_pixels(fp, GRID10_T, (H10, W10))
ON_FOOTPRINT = np.zeros(H10 * W10, bool); ON_FOOTPRINT[FPIX] = True; ON_FOOTPRINT = ON_FOOTPRINT.reshape(H10, W10)


def on_grid(path, transform, shape_hw, resampling=Resampling.mode):
    out = np.zeros(shape_hw, "float32")
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=transform, dst_crs=GRID_CRS, resampling=resampling)
    return out


def settlement_mask(transform, shape_hw):
    m = np.zeros(shape_hw, bool)
    if WSF_PATH:
        m |= on_grid(WSF_PATH, transform, shape_hw) > 0
    if DW_PATH:
        m |= on_grid(DW_PATH, transform, shape_hw) == 6
    return m


settle10 = settlement_mask(GRID10_T, (H10, W10))
near_fp = distance_transform_edt(~ON_FOOTPRINT) * GRID10_M <= CONSTRUCTION_ZONE_M
CON_ZONE = near_fp | settle10
print(f"footprint pixels {ON_FOOTPRINT.mean():.0%} of the area; construction search zone {CON_ZONE.mean():.0%}"
      + ("" if settle10.any() else " (no WSF/DW mask: footprints only)"))
''')

# =============================================================================================== 6 change test
md(r"""
## 6. The change test and its threshold

**Per pixel, per track and polarisation.** For every acquisition date *s* with a full year before it: the backscatter of the year before *s* (reference) is compared with the `post_days` after it (inference) by a two-sample t-test (Welch). With the dry-season series (`SAMPLING = "dry_season"`) the reference is the previous season (and the earlier dates of this one), the inference window the rest of this season (150 days) and the confirmation the next season (365 days); the season check is off because every year has the same months. A split counts only if

1. both windows have enough dates (`min_ref`, `min_post`);
2. the new level still holds in the `confirm_days` after the inference window (same direction, at least half the change), as LT-CCD requires a detection to recur; near the end of the series this cannot be tested, and such detections are kept but flagged *unconfirmed*;
3. with `season_check`, the inference window also differs from the same months one year earlier (same direction, at least half the change). A seasonal swing returns every year, so it shows no difference there.

The split with the largest |t| is kept: its date is the first acquisition that shows the change (the event lies between it and the previous acquisition of that track).

**Combining.** As in PWTT, the change score of a pixel is the largest |t| over the tracks and polarisations; `agree` counts how many of them exceed the threshold within `DATE_TOL_DAYS` of that date.

**Threshold.** The score is a maximum over ~100 split dates and several track/polarisation pairs, so even unchanged pixels reach |t| values far above single-test values. The threshold is therefore taken from a **permutation null**: for `N_NULL` pixels, the dates of each series are shuffled (any real change destroyed, values and gaps kept), the same search is run, and the `NULL_QUANTILE` quantile of the null scores is the threshold. Shuffling also removes seasonality, so this null is optimistic for seasonal ground: the false-alarm rate on real data is measured in section 9a.

**Change timing shape** (`pre_trend`): how far the level had already moved in the `post_days` before the split, as a fraction of the change. Near 0: an abrupt step between two acquisitions; near 1: a change that built up over months. It is reported as an attribute and tested in section 9; it is **not** used to assign the type.
""")
code(r'''
PARAMS_TAG = json.dumps({"test": TEST.__dict__, "lee": LEE_WINDOW, "n_null": N_NULL, "seed": SEED}, sort_keys=True)
rng = np.random.default_rng(SEED)
null_pix = np.sort(rng.choice(H10 * W10, size=min(N_NULL, H10 * W10), replace=False))      # same pixels for every combo
LAYERS = ("t", "split", "delta", "confirm", "ratio", "season_delta", "pre_trend")


def run_combo(key, pol):
    """Change test of one track/polarisation, in row blocks; results cached in CHANGE_DIR/<combo>/."""
    out_dir = CHANGE_DIR / combo_name(key, pol)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = STORES[(key, pol)]
    tag = PARAMS_TAG + json.dumps(store.filled_dates())
    if (out_dir / "params.json").exists() and (out_dir / "params.json").read_text() == tag and \
            all((out_dir / f"{k}.npy").exists() for k in LAYERS + ("null",)):
        return {k: np.load(out_dir / f"{k}.npy") for k in LAYERS}, np.load(out_dir / "null.npy")
    filled = store.filled_dates()
    idx = np.array([store.dates.index(d) for d in filled])
    t_days = np.array([date.fromisoformat(d).toordinal() for d in filled], float)
    splits = ht.split_dates(t_days, TEST)
    mm = store.memmap("r")
    res = {k: np.full((H10, W10), np.nan, "float32") for k in LAYERS}
    for r0 in range(0, H10, BLOCK_ROWS):
        r1 = min(H10, r0 + BLOCK_ROWS)
        block = np.asarray(mm[idx, r0:r1, :], dtype="float64")
        rb = ht.change_test(block, t_days, splits, TEST)
        for k in LAYERS:
            res[k][r0:r1] = rb[k]
    series = np.stack([np.asarray(mm[i]).ravel()[null_pix] for i in idx]).astype("float64")   # (T, N_NULL), one date at a time
    null = ht.null_max_abs_t(series, t_days, splits, TEST, np.random.default_rng(SEED + 1))
    del mm
    for k in LAYERS:
        np.save(out_dir / f"{k}.npy", res[k])
    np.save(out_dir / "null.npy", null)
    (out_dir / "params.json").write_text(tag)
    return res, null


RESULTS_BY_COMBO, NULLS = {}, {}
for key, pol in COMBOS:
    t0 = pd.Timestamp.now()
    RESULTS_BY_COMBO[combo_name(key, pol)], NULLS[combo_name(key, pol)] = run_combo(key, pol)
    print(f"{combo_name(key, pol)}: {(pd.Timestamp.now() - t0).total_seconds() / 60:.1f} min")

null_score = np.nanmax(np.stack(list(NULLS.values())), axis=0)          # the combined statistic under the null
THRESH = float(T_THRESHOLD) if T_THRESHOLD else float(np.nanquantile(null_score, NULL_QUANTILE))
COMB = ht.combine(RESULTS_BY_COMBO, THRESH, DATE_TOL_DAYS)
COMB_NAMES = list(RESULTS_BY_COMBO)
valid10 = np.isfinite(COMB["split"])
print(f"threshold |t| = {THRESH:.2f} ({'fixed' if T_THRESHOLD else f'{NULL_QUANTILE} quantile of the permutation null'}); "
      f"{(COMB['score'][valid10] >= THRESH).mean():.2%} of tested pixels above it")
''')
code(r'''
year = lambda ordinal: np.where(np.isfinite(ordinal), 1 + (np.nan_to_num(ordinal) - date(1, 1, 1).toordinal()) / 365.2425, np.nan)
hot = (COMB["score"] >= THRESH) & (COMB["confirm"] != 0)
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True, sharex=True, sharey=True)
im = axes[0].imshow(np.where(valid10, COMB["score"], np.nan), extent=EXTENT10, cmap="magma", vmin=0, vmax=max(THRESH * 2, 1))
axes[0].contour(np.arange(W10) * GRID10_M + left, top - np.arange(H10) * GRID10_M, hot.astype(float), [0.5], colors="c", linewidths=0.5)
axes[0].set_title(f"change score max|t| (threshold {THRESH:.1f}, cyan)"); fig.colorbar(im, ax=axes[0], shrink=0.8)
im = axes[1].imshow(np.where(hot, year(COMB["split"]), np.nan), extent=EXTENT10, cmap="viridis", interpolation="nearest")
axes[1].set_title("date of the change (year), pixels above the threshold"); fig.colorbar(im, ax=axes[1], shrink=0.8)
im = axes[2].imshow(np.where(hot, COMB["agree"], np.nan), extent=EXTENT10, cmap="Blues", vmin=0, vmax=len(COMB_NAMES), interpolation="nearest")
axes[2].set_title(f"how many of {len(COMB_NAMES)} track/polarisation results agree"); fig.colorbar(im, ax=axes[2], shrink=0.8)
plt.show()
print("which result gave the score (share of pixels above the threshold):",
      {COMB_NAMES[k]: round(float((COMB["best"][hot] == k).mean()), 2) for k in range(len(COMB_NAMES))})
''')

# =============================================================================================== 7 candidates
md(r"""
## 7. Candidates and their type

| Candidate | Rule | Why this rule |
|---|---|---|
| **Demolition candidate** | a footprint whose mean change score ≥ the **building threshold**, and at least `MIN_CONFIRMED_FRAC` of its changed pixels held their new level | demolition needs a building that stood before; PWTT scores a building by the mean over its footprint |

**Building threshold.** A building's score averages several pixels that are correlated (speckle filter, resampling), so its null distribution is narrower than that of one pixel's maximum. It has its own permutation null: for `N_NULL_BUILDINGS` random footprints, the dates are shuffled (one permutation per footprint, applied to all its pixels together), the same search is run, and the footprint mean is taken; the `NULL_QUANTILE_BUILDING` quantile is the threshold. 0.995 means about 5 of every 1,000 unchanged buildings exceed it under the null; the quiet-period test (9a) measures the real rate.
| **Construction candidate** | a group of ≥ `CON_MIN_AREA_M2` changed pixels off every footprint (and one pixel away from them), within `CONSTRUCTION_ZONE_M` of buildings or in the settlement mask, and **not** within `SPILLOVER_M` of a demolition candidate of the same date (± `DATE_TOL_DAYS`) | construction appears where no building stood; a demolition also changes the pixels around the footprint (PWTT: false positives cluster next to real damage), which would otherwise appear as "construction" |

Attributes on every candidate: date (first acquisition showing the change), score, `agree`, VV and VH change (dB), `pre_trend` (abrupt or gradual), confirmed / unconfirmed, and (section 8) coherence support. A change on an existing footprint can also be an extension or a new storey: the type rule cannot tell that from demolition, which is one more reason to confirm each candidate on imagery. **The sign of the backscatter change is reported but not used**: published results show it can go either way after destruction.
""")
code(r'''
delta_by_pol = {}
for pol in GRD_POLS:
    names = [n for n in COMB_NAMES if n.endswith(pol)]
    if names:                                   # the change of the result with the largest |t| of that polarisation
        a = np.stack([np.abs(RESULTS_BY_COMBO[n]["t"]) for n in names])
        k = np.argmax(np.nan_to_num(a), axis=0)
        delta_by_pol[f"d{pol.lower()}_db"] = np.take_along_axis(np.stack([RESULTS_BY_COMBO[n]["delta"] for n in names]), k[None], 0)[0]

# Building threshold from a building-level permutation null: the dates of a sample of footprints are shuffled (one
# permutation per footprint, applied to all its pixels), the same search is run, and the mean over the footprint taken.
rng_b = np.random.default_rng(SEED + 2)
sample_f = np.sort(rng_b.choice(len(fp), size=min(N_NULL_BUILDINGS, len(fp)), replace=False))
sel = np.isin(FID, sample_f)
s_pix, s_grp = FPIX[sel], np.searchsorted(sample_f, FID[sel])
series_b, t_b, sp_b = {}, {}, {}
for key, pol in COMBOS:
    store = STORES[(key, pol)]
    filled = store.filled_dates()
    mm = store.memmap("r")
    series_b[combo_name(key, pol)] = np.stack([np.asarray(mm[store.dates.index(d)]).ravel()[s_pix] for d in filled]).astype("float64")
    del mm
    t_b[combo_name(key, pol)] = np.array([date.fromisoformat(d).toordinal() for d in filled], float)
    sp_b[combo_name(key, pol)] = ht.split_dates(t_b[combo_name(key, pol)], TEST)
null_b = ht.footprint_null_scores(series_b, t_b, sp_b, s_grp, TEST, rng_b)
THRESH_B = float(T_THRESHOLD_BUILDING) if T_THRESHOLD_BUILDING else float(np.nanquantile(null_b, NULL_QUANTILE_BUILDING))
print(f"building threshold: mean |t| = {THRESH_B:.2f} ({'fixed' if T_THRESHOLD_BUILDING else f'{NULL_QUANTILE_BUILDING} quantile of '
      f'the building null, {len(sample_f)} footprints'}); pixel threshold {THRESH:.2f}")

fs = ht.footprint_scores(FID, FPIX, len(fp), COMB, THRESH, extra=delta_by_pol)
bld = fp[["area_m2", "geometry"]].join(fs)
bld["date"] = [date.fromordinal(int(v)).isoformat() if np.isfinite(v) else None for v in bld["split"]]
bld["changed"] = bld["score"] >= THRESH_B
bld["status"] = np.where(bld["untestable_frac"] > 0.5, "unconfirmed (end of series)",
                         np.where(bld["confirmed_frac"] >= MIN_CONFIRMED_FRAC, "confirmed", "not held"))
DEM = bld[bld["changed"] & (bld["status"] != "not held")].copy()
DEM["type"] = "demolition candidate"
print(f"{len(bld)} buildings scored; {int(bld['changed'].sum())} above the threshold; "
      f"{len(DEM)} demolition candidates ({int((DEM['status'] == 'confirmed').sum())} confirmed)")

off_fp = ~binary_dilation(ON_FOOTPRINT, iterations=1)
con_mask = (COMB["score"] >= THRESH) & (COMB["confirm"] != 0) & off_fp & CON_ZONE
layers = {"score": COMB["score"], "split": COMB["split"], "pre_trend": COMB["pre_trend"], "agree": COMB["agree"], **delta_by_pol}
CON_LAB, con_feats = ht.change_clusters(con_mask, GRID10_T, GRID10_M, CON_MIN_AREA_M2, layers)
CON = gpd.GeoDataFrame.from_features(con_feats, crs=GRID_CRS) if con_feats else gpd.GeoDataFrame(geometry=[], crs=GRID_CRS)
if len(CON):
    CON["date"] = [date.fromordinal(int(round(v))).isoformat() for v in CON["split_median"]]
    CON["type"] = "construction candidate"
if len(CON) and len(DEM):        # spillover: a demolition also changes the pixels next to the footprint
    dd = np.array([date.fromisoformat(d).toordinal() if d else np.nan for d in DEM["date"]])
    spill = []
    for g, sp in zip(CON.geometry, CON["split_median"]):
        near = (DEM.geometry.distance(g) <= SPILLOVER_M).to_numpy() & (np.abs(dd - sp) <= DATE_TOL_DAYS)
        spill.append(bool(near.any()))
    spill = np.array(spill)
    print(f"{int(spill.sum())} off-footprint clusters next to a demolition candidate of the same date: treated as its spillover")
    CON = CON[~spill].reset_index(drop=True)
print(f"{len(CON)} construction candidates ({CON['area_m2'].sum() / 1e4 if len(CON) else 0:.1f} ha)")

per_year = pd.DataFrame({
    "demolition candidates": pd.Series([d[:4] for d in DEM["date"].dropna()]).value_counts(),
    "construction candidates": pd.Series([d[:4] for d in CON["date"]] if len(CON) else [], dtype=str).value_counts(),
}).fillna(0).astype(int).sort_index()
display(per_year)
if len(DEM):
    display(DEM[["area_m2", "score", "agree", "date", "status", "pre_trend", *delta_by_pol]].describe().round(2))
''')

# =============================================================================================== 8 coherence
md(r"""
## 8. Coherence support (optional, SLC)

The SLC chain of v5 (same-satellite 12-day pairs, with the fixes made since: DEM margin 0.1°, the "is this pair done" check that reads the saved results, and the workaround for pygeofetch's `download_and_extract` date lookup). Coherence is **supporting evidence** here: for every candidate whose date falls inside the coherence period, `coh_support` says whether the coherence dropped by at least `COH_DROP` at that place within `TOL_DAYS` of the candidate's date (in cells normally coherent, ≥ `COH_STABLE`). Outside the period it is empty. Set `RUN_COHERENCE = False` to skip the section.
""")
code(r'''
def fetch_orbits(items):
    """{date: orbit file}. Precise orbits appear about 3 weeks after acquisition; newer scenes get restituted orbits."""
    out = {}
    for item in items:
        for kind in ("precise", "restituted"):
            try:
                path = fetch_orbit_file(product_name=name_of(item), output_dir=str(DATA / "slc/orbits"), orbit_type=kind)
            except Exception:
                path = None
            if path:
                out[str(item.datetime)[:10]] = path
                break
    return out


def plan_pairs(items, orbit_files, max_days=13):
    """Consecutive acquisitions of the same satellite, at most max_days apart, both with an orbit file."""
    by_sat = {}
    for item in items:
        d = str(item.datetime)[:10]
        if d in orbit_files:
            by_sat.setdefault(name_of(item)[:3], []).append(d)
    return [(a, b) for ds in by_sat.values() for a, b in zip(sorted(ds), sorted(ds)[1:])
            if (date.fromisoformat(b) - date.fromisoformat(a)).days <= max_days]


if RUN_COHERENCE:
    project = InSARProject(name="hebron", aoi=aoi, output_dir=DATA / "slc", polarisation="VV", providers=["copernicus"],
                           streaming_cleanup=STREAMING_CLEANUP)
    slc_found = project.search(start_date=SLC_START, end_date=SLC_END, max_results=500, show_map=False)
    slc_items = one_per_date([i for i in slc_found if covers_aoi(i)], TRACK, SLC_DATES)
    if not SLC_DATES:      # one satellite only: its scenes make the pairs
        sat = Counter(name_of(i)[:3] for i in slc_items).most_common(1)[0][0]
        slc_items = [i for i in slc_items if name_of(i)[:3] == sat]
    slc_items = newest_or_first(slc_items, MAX_SLC)
    slc_by_date = {str(i.datetime)[:10]: i for i in slc_items}
    orbits = fetch_orbits(slc_items)
    pairs = plan_pairs(slc_items, orbits)
    print(f"{len(slc_items)} SLC acquisitions on track {TRACK}; orbit files for {len(orbits)}; {len(pairs)} pairs")

    # DEM for the whole radar crop: box plus 35 % of its span on each side, at least 0.1 deg (0.05 left 1.3 % uncovered)
    import math
    need_dem = (math.floor((BBOX[0] - max(0.35 * (BBOX[2] - BBOX[0]), 0.10)) * 1e4) / 1e4,
                             math.floor((BBOX[1] - max(0.35 * (BBOX[3] - BBOX[1]), 0.10)) * 1e4) / 1e4,
                             math.ceil((BBOX[2] + max(0.35 * (BBOX[2] - BBOX[0]), 0.10)) * 1e4) / 1e4,
                             math.ceil((BBOX[3] + max(0.35 * (BBOX[3] - BBOX[1]), 0.10)) * 1e4) / 1e4)
    dem_bbox = BoundingBox(min_lon=need_dem[0], min_lat=need_dem[1], max_lon=need_dem[2], max_lat=need_dem[3])
    dem_items = client.search(SearchQuery(bbox=dem_bbox, product_type="DEM"), providers=["opentopography"])
    dem_item = (next((i for i in dem_items if "COP30" in i.id.upper()), None)
                or next((i for i in dem_items if "SRTMGL1" in i.id.upper()), dem_items[0]))
    DEM_PATH = client.download([dem_item], destination=DEM_DIR)[0].output_path
    (RESULTS / "dem_path.txt").write_text(str(DEM_PATH))
    print("DEM:", DEM_PATH, "bbox:", need_dem)
''', ACQ)
code(r'''
if RUN_COHERENCE:
    gen = InterferogramGenerator(coherence_window=None, esd_enabled=True, use_gpu=False, use_real_burst_processing=True,
                                 remove_flat_earth_phase=True, memory_guard="warn")
    MIN_DEM_COVERAGE = 0.999

    def dem_coverage(a, b):
        """DEM coverage stored with a saved pair (None if the pair has no complete save)."""
        try:
            return json.loads((Path(IFG_DIR) / f"{a}_{b}" / "result.json").read_text())["metadata"]["geometry"].get("dem_coverage_fraction")
        except Exception:
            return None

    def saved_ok(a, b):
        cov = dem_coverage(a, b)
        return cov is not None and cov >= MIN_DEM_COVERAGE

    def scene_paths(d):
        return [project.extracted_slcs[d], project.download_results[d].output_path] if d in project.extracted_slcs else []

    def have(d):
        return d in project.extracted_slcs and Path(project.extracted_slcs[d]).exists()

    def process_slc_pairs(todo, redo=()):
        """One pair at a time: download what it lacks, form it, verify, delete scenes no later pair needs."""
        records = []
        for i, (a, b) in enumerate(todo):
            missing = [d for d in (a, b) if not have(d)]
            if missing:
                # pygeofetch builds its date lookup from `selected` but loops over every download of the session:
                # let it see only this call's downloads, then merge them back
                earlier = project.download_results
                project.download_results = {}
                try:
                    project.download_and_extract(selected=[slc_by_date[d] for d in missing], max_scenes=len(missing), show_map=False)
                finally:
                    project.download_results = {**earlier, **project.download_results}
            if not (have(a) and have(b)):
                print(f"{a} -> {b}: a scene could not be downloaded or extracted; pair skipped")
                continue
            spec = dict(reference=project.extracted_slcs[a], secondary=project.extracted_slcs[b],
                        reference_date=a, secondary_date=b,
                        reference_safe_zip=project.download_results[a].output_path,
                        secondary_safe_zip=project.download_results[b].output_path,
                        reference_orbit_file=orbits[a], secondary_orbit_file=orbits[b])
            still_needed = [p for pair in todo[i + 1:] for d in pair for p in scene_paths(d)]
            records += gen.process_pairs([spec], IFG_DIR, resume=(a, b) not in redo, streaming_cleanup=STREAMING_CLEANUP,
                                         keep_inputs=still_needed, dem=DEM_PATH, looks_azimuth=2, looks_range=8,
                                         apply_goldstein_filter=False, aoi_bbox=aoi, chunk_rows=CHUNK_ROWS)
            cov = dem_coverage(a, b)
            if cov is not None and cov < MIN_DEM_COVERAGE:
                print(f"  WARNING {a} -> {b}: DEM covers only {100 * cov:.1f}% of the interferogram")
            gc.collect()
        return records

    redo = {tuple(p) for p in pairs if dem_coverage(*p) is not None and not saved_ok(*p)}
    todo = [p for p in pairs if not saved_ok(*p)]
    print(f"{len(todo)} of {len(pairs)} pairs to process" + (f"; recomputed for DEM coverage: {sorted(redo)}" if redo else ""))
    records = process_slc_pairs(todo, redo) if todo else []
''', ACQ)
code(r'''
COH = None
saved_pairs = sorted(tuple(p.name.split("_")) for p in Path(IFG_DIR).iterdir()
                     if p.is_dir() and InterferogramResult.saved_fingerprint(p) is not None) if RUN_COHERENCE else []
if RUN_COHERENCE and len(saved_pairs) >= 2:
    DEM_PATH = Path((RESULTS / "dem_path.txt").read_text().strip()) if (RESULTS / "dem_path.txt").exists() else globals().get("DEM_PATH")
    if DEM_PATH is None:
        raise RuntimeError("no DEM path: run the SLC acquisition cells of this section first")
    with rasterio.open(DEM_PATH) as src:
        dem, dem_T = src.read(1).astype("float64"), src.transform
    dem = np.where(dem > -1000, dem, np.median(dem[dem > -1000]))
    gx, gy = np.meshgrid(left + GRID_M * (np.arange(W) + 0.5), top - GRID_M * (np.arange(H) + 0.5))

    def dem_at(lon, lat):
        c, r = ~dem_T * (lon, lat)
        return map_coordinates(dem, [r - 0.5, c - 0.5], order=1, mode="nearest")

    def coherence_on_grid(folder):
        """Coherence of one saved pair on the 30 m grid, every radar cell placed at its terrain height."""
        res = InterferogramResult.load(folder)
        g = res.geometry
        lon0, lat0, dlon, dlat = (g.dense(k, dtype=np.float64) for k in ("lon", "lat", "dlon_dh", "dlat_dh"))
        h = np.zeros_like(lon0)
        for _ in range(3):
            h = dem_at(lon0 + dlon * h, lat0 + dlat * h)
        x, y = warp_xy("EPSG:4326", GRID_CRS, (lon0 + dlon * h).ravel().tolist(), (lat0 + dlat * h).ravel().tolist())
        coh = np.asarray(res.coherence, "float64").ravel()
        ok = np.isfinite(coh) & (coh >= 0)
        out = griddata((np.asarray(x)[ok], np.asarray(y)[ok]), coh[ok], (gx, gy), method="linear").astype("float32")
        del res, g
        gc.collect()
        return out

    stack = np.stack([coherence_on_grid(IFG_DIR / f"{a}_{b}") for a, b in saved_pairs])
    m = np.nanmedian(stack, axis=(1, 2))
    stack = stack - m[:, None, None] + np.median(m)
    mid = np.array([(date.fromisoformat(a).toordinal() + date.fromisoformat(b).toordinal()) / 2 for a, b in saved_pairs])
    med = np.nanmedian(stack, 0) if len(stack) >= 3 else np.nanmax(stack, 0)
    low = np.where(np.isfinite(stack), stack, np.inf).argmin(0)
    drop = med - np.take_along_axis(stack, low[None], 0)[0]
    COH = dict(drop=drop, drop_day=mid[low], testable=(med >= COH_STABLE) & np.isfinite(drop), first=mid.min() - 6, last=mid.max() + 6)
    COH["anomalous"] = COH["testable"] & (COH["drop"] >= COH_DROP)
    print(f"coherence: {len(saved_pairs)} pairs ({saved_pairs[0][0]} to {saved_pairs[-1][1]}); "
          f"{COH['testable'].mean():.0%} of cells coherent enough to test")


def coherence_support(gdf):
    """True / False where the candidate's date lies in the coherence period, else None."""
    if COH is None or not len(gdf):
        return [None] * len(gdf)
    c = gdf.geometry.centroid
    cols = ((c.x.to_numpy() - left) // GRID_M).astype(int)
    rows = ((top - c.y.to_numpy()) // GRID_M).astype(int)
    an = binary_dilation(COH["anomalous"], iterations=1)
    out = []
    for r, cc, d in zip(rows, cols, gdf["date"]):
        if d is None or not (0 <= r < H and 0 <= cc < W):
            out.append(None); continue
        day = date.fromisoformat(d).toordinal()
        if not (COH["first"] <= day <= COH["last"]) or not COH["testable"][r, cc]:
            out.append(None); continue
        out.append(bool(an[r, cc] and abs(COH["drop_day"][r, cc] - day) <= TOL_DAYS + 12))
    return out


DEM["coh_support"] = coherence_support(DEM)
if len(CON):
    CON["coh_support"] = coherence_support(CON)
print("coherence support of demolition candidates:", DEM["coh_support"].value_counts(dropna=False).to_dict())
''')

# =============================================================================================== 9 validation
md(r"""
## 9. Validation

**9a False alarms on real data (quiet period).** Choose a period, and optionally an area (`QUIET_AREA_PATH`), in which you have good reason to believe **no** demolition or construction happened (from OCHA records and imagery). Every candidate dated inside it is a false alarm. This is LT-CCD's counterfactual test; it measures what the permutation null cannot (seasonality, moisture, vehicles, layover changes on real data). Reported per 1,000 buildings per year.

**9b Building-level labels.** `BUILDING_REF_PATH`: your own labels from dated very-high-resolution imagery, points or polygons with a `class` column (`demolition`, `construction` or `none`) and an optional `date`. For each label, the score is the largest change score within `REF_MATCH_M`. Reported:
* **AUC** of the score, changed labels against `none` labels (independent of the threshold);
* **recall** at the threshold: share of labelled changes with a candidate of the **same** type within `REF_MATCH_M`; and the type confusion (which type the candidate got);
* **precision**, only inside `LABEL_AREA_PATH` (where every change was mapped): share of candidates there that match a label;
* **date error**: candidate date minus label date.

**9c Locality-level references** (OCHA, Peace Now, ARIJ, as in v5): candidates within `REF_RADIUS_M` of each point, against random control points ≥ 300 m away. These coordinates are often at locality level, so this is a coarse check only.
""")
code(r'''
def union(g):
    return g.union_all() if hasattr(g, "union_all") else g.unary_union


def year_span(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days / 365.25


quiet_table = None
if QUIET_PERIOD:
    q0, q1 = QUIET_PERIOD
    area = union(gpd.read_file(QUIET_AREA_PATH).to_crs(GRID_CRS)) if QUIET_AREA_PATH else box(left, bottom, right, top)
    in_area = bld.geometry.intersects(area)
    dq = DEM[DEM.geometry.intersects(area) & DEM["date"].between(q0, q1)]
    cq = CON[CON.geometry.intersects(area) & CON["date"].between(q0, q1)] if len(CON) else CON
    yrs = year_span(q0, q1)
    quiet_table = pd.DataFrame({
        "buildings in area": [int(in_area.sum())], "years": [round(yrs, 2)],
        "demolition false alarms": [len(dq)], "per 1,000 buildings per year": [1e3 * len(dq) / max(in_area.sum(), 1) / yrs],
        "construction false alarms": [len(cq)], "construction false-alarm area (ha / yr)": [cq["area_m2"].sum() / 1e4 / yrs if len(cq) else 0.0]})
    display(quiet_table.round(3))
else:
    print("QUIET_PERIOD not set: the false-alarm rate on real data is not measured (the permutation null alone is optimistic)")
''')
code(r'''
label_table = conf_table = pr = None
if BUILDING_REF_PATH:
    lab = gpd.read_file(BUILDING_REF_PATH)
    lab = (lab.set_crs(4326) if lab.crs is None else lab).to_crs(GRID_CRS)
    lab["class"] = lab["class"].str.lower().str.strip()
    dcol = next((c for c in lab.columns if c.lower() == "date"), None)
    lab["label_day"] = pd.to_datetime(lab[dcol], errors="coerce").map(lambda t: t.toordinal() if pd.notna(t) else np.nan) if dcol else np.nan
    pts = lab.geometry.representative_point()
    xy = np.c_[pts.x, pts.y]
    # score: the largest change score within REF_MATCH_M of the label
    rr = ((top - xy[:, 1]) / GRID10_M).astype(int); cc = ((xy[:, 0] - left) / GRID10_M).astype(int)
    win = int(np.ceil(REF_MATCH_M / GRID10_M))
    sc = np.full(len(lab), np.nan)
    for i, (r, c) in enumerate(zip(rr, cc)):
        if 0 <= r < H10 and 0 <= c < W10:
            sl = COMB["score"][max(r - win, 0):r + win + 1, max(c - win, 0):c + win + 1]
            sc[i] = np.nanmax(sl) if np.isfinite(sl).any() else np.nan
    lab["score"] = sc
    changed = lab["class"].isin(["demolition", "construction"])
    auc_all = ht.auc(lab.loc[changed, "score"], lab.loc[lab["class"] == "none", "score"])
    # candidates of each type, as points (footprint or cluster centroids)
    cand = pd.concat([DEM.assign(type="demolition")[["type", "date", "geometry"]],
                      CON.assign(type="construction")[["type", "date", "geometry"]] if len(CON) else None], ignore_index=True)
    cand = gpd.GeoDataFrame(cand, crs=GRID_CRS)
    rows, conf = [], Counter()
    for cls in ("demolition", "construction"):
        L = lab[lab["class"] == cls]
        if not len(L):
            continue
        hit_same, err = 0, []
        for i in L.index:
            p = pts.loc[i]
            near = cand[cand.geometry.distance(p) <= REF_MATCH_M] if len(cand) else cand
            conf[(cls, near["type"].iloc[0] if len(near) else "none")] += 1
            same = near[near["type"] == cls]
            if len(same):
                hit_same += 1
                if np.isfinite(lab.at[i, "label_day"]) and same["date"].notna().any():
                    err.append(min(date.fromisoformat(d).toordinal() for d in same["date"].dropna()) - lab.at[i, "label_day"])
        rows.append({"class": cls, "labels": len(L), "recall (same type)": hit_same / len(L),
                     "median date error (d)": float(np.median(err)) if err else np.nan,
                     "within 30 d": float(np.mean(np.abs(err) <= 30)) if err else np.nan})
    label_table = pd.DataFrame(rows)
    conf_table = pd.Series(conf).unstack(fill_value=0) if conf else None
    if LABEL_AREA_PATH and len(cand):
        area = union(gpd.read_file(LABEL_AREA_PATH).to_crs(GRID_CRS))
        ca = cand[cand.geometry.intersects(area)]
        la = lab[lab.geometry.intersects(area) & changed]
        matched = [any((la["class"] == t) & (la.geometry.distance(g) <= REF_MATCH_M)) for g, t in zip(ca.geometry, ca["type"])]
        label_table["precision (labelled area)"] = [np.mean([m for m, t in zip(matched, ca["type"]) if t == c]) if (ca["type"] == c).any() else np.nan
                                                    for c in label_table["class"]]
    print(f"{len(lab)} labels: {lab['class'].value_counts().to_dict()}; AUC changed vs none: {auc_all:.3f}")
    display(label_table.round(3))
    if conf_table is not None:
        print("type confusion (rows: label, columns: nearest candidate within REF_MATCH_M):"); display(conf_table)
    if (lab["class"] == "none").any():
        y = np.r_[np.ones(int(changed.sum())), np.zeros(int((lab["class"] == "none").sum()))]
        s = np.r_[lab.loc[changed, "score"], lab.loc[lab["class"] == "none", "score"]]
        prec, rec, thr = ht.precision_recall_curve(s, y)
        pr = pd.DataFrame({"threshold": thr, "precision": prec, "recall": rec})
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        ax.plot(rec, prec, "-o", ms=2); k = np.argmin(np.abs(thr - THRESH))
        ax.plot(rec[k], prec[k], "rs", label=f"threshold {THRESH:.1f}")
        ax.set_xlabel("recall"); ax.set_ylabel("precision (labels only: not the map's precision)"); ax.legend(); ax.set_title(f"labels, AUC {auc_all:.2f}")
        plt.show()
else:
    print("BUILDING_REF_PATH not set: building-level validation skipped (see the reference-data checklist)")
''')
code(r'''
def read_points(path):
    """CSV (lon/lat columns) or any vector file -> GeoDataFrame in the grid CRS with an event_day column, inside BBOX."""
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)
        col = lambda names: next(c for c in df.columns if c.lower() in names)
        g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[col(("lon", "long", "longitude", "x"))],
                                                              df[col(("lat", "latitude", "y"))]), crs=4326)
    else:
        g = gpd.read_file(path).to_crs(4326)
        g["geometry"] = g.geometry.representative_point()
    g = g.cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]]
    dcol = next((c for c in g.columns if "date" in c.lower()), None)
    g["event_day"] = [t.toordinal() if pd.notna(t) else np.nan for t in pd.to_datetime(g[dcol], errors="coerce")] if dcol else np.nan
    return g.to_crs(GRID_CRS)


loc_table = None
refs = [(cls, p) for cls, ps in {"demolition": [OCHA_PATH], "construction": [PEACENOW_PATH, ARIJ_PATH]}.items() for p in ps if p]
if refs:
    rng_c = np.random.default_rng(SEED)
    rows = []
    for cls, p in refs:
        g = read_points(p)
        cand = DEM if cls == "demolition" else CON
        hit = [bool(len(cand)) and bool((cand.geometry.distance(pt) <= REF_RADIUS_M).any()) for pt in g.geometry]
        # controls: random points in the area at least 300 m from every reference point
        allp = np.c_[g.geometry.x, g.geometry.y]
        ctrl = []
        while len(ctrl) < 5 * len(g) and len(allp):
            x, y = rng_c.uniform(left, right), rng_c.uniform(bottom, top)
            if np.min(np.hypot(allp[:, 0] - x, allp[:, 1] - y)) >= 300:
                ctrl.append((x, y))
        from shapely.geometry import Point
        chit = [bool(len(cand)) and bool((cand.geometry.distance(Point(x, y)) <= REF_RADIUS_M).any()) for x, y in ctrl]
        rows.append({"class": cls, "source": Path(p).stem, "points": len(g), "hit rate": np.mean(hit) if hit else np.nan,
                     "control hit rate": np.mean(chit) if chit else np.nan})
    loc_table = pd.DataFrame(rows)
    display(loc_table.round(3))
else:
    print("no locality-level reference files set (OCHA_PATH, PEACENOW_PATH, ARIJ_PATH)")
''')

# =============================================================================================== 10 export
md(r"""
## 10. Export and maps

`results/hebron_candidates_v6.gpkg` with layers `demolition_candidates` (footprints) and `construction_candidates` (clusters), every attribute included; GeoTIFFs on the 10 m grid: `change_score`, `change_date_year`, `change_agree` and each track/polarisation's signed t; CSV tables; the map. Footprint data: Microsoft Global ML Building Footprints (ODbL), unless you set your own.
""")
code(r'''
def save_tif10(arr, path, dtype="float32", nodata=np.nan):
    with rasterio.open(path, "w", driver="GTiff", dtype=dtype, count=1, height=H10, width=W10, crs=GRID_CRS,
                       transform=GRID10_T, nodata=nodata, compress="deflate") as dst:
        dst.write(np.asarray(arr, dtype=dtype), 1)
    return path


out = [save_tif10(np.where(valid10, COMB["score"], np.nan), RESULTS / "change_score.tif"),
       save_tif10(np.where(COMB["score"] >= THRESH, year(COMB["split"]), np.nan), RESULTS / "change_date_year.tif"),
       save_tif10(np.where(valid10, COMB["agree"], 0), RESULTS / "change_agree.tif", "uint8", 0)]
for n, r in RESULTS_BY_COMBO.items():
    out.append(save_tif10(r["t"], RESULTS / f"t_{n}.tif"))
gpkg = RESULTS / "hebron_candidates_v6.gpkg"
keep = ["area_m2", "score", "score_max", "frac_above", "agree", "date", "status", "pre_trend", "ratio", *delta_by_pol, "coh_support", "type", "geometry"]
DEM[[c for c in keep if c in DEM]].to_crs(4326).to_file(gpkg, layer="demolition_candidates", driver="GPKG")
if len(CON):
    CON.to_crs(4326).to_file(gpkg, layer="construction_candidates", driver="GPKG")
per_year.to_csv(RESULTS / "candidates_per_year.csv")
for name, t in (("validation_quiet_period", quiet_table), ("validation_labels", label_table), ("validation_localities", loc_table), ("validation_pr_curve", pr)):
    if t is not None:
        t.to_csv(RESULTS / f"{name}.csv", index=False)
(RESULTS / "run_v6.json").write_text(json.dumps({"threshold": THRESH, "null_quantile": NULL_QUANTILE, "building_threshold": THRESH_B,
                                                 "building_null_quantile": NULL_QUANTILE_BUILDING, "sampling": SAMPLING, "combos": COMB_NAMES,
                                                 "test": TEST.__dict__, "lee_window": LEE_WINDOW, "footprints": str(FOOTPRINTS_PATH),
                                                 "demolition_candidates": len(DEM), "construction_candidates": len(CON)}, indent=1))
print(f"written: {gpkg.name} and {len(out)} GeoTIFFs in {RESULTS}")
''')
code(r'''
VERM, BLUE = "#D55E00", "#0072B2"          # colour-blind-safe pair: demolition / construction candidates
fig, ax = plt.subplots(figsize=(11, 10.5), constrained_layout=True)
ax.imshow(np.where(valid10, COMB["score"], np.nan), extent=EXTENT10, cmap="Greys", vmin=0, vmax=THRESH * 1.5, alpha=0.6)
fp.boundary.plot(ax=ax, color="0.6", linewidth=0.2)
if len(DEM):
    DEM.plot(ax=ax, color=VERM, edgecolor=VERM, linewidth=0.8)
if len(CON):
    CON.plot(ax=ax, facecolor="none", edgecolor=BLUE, linewidth=1.2)
    ax.plot([], [], color=BLUE, label=f"construction candidates ({len(CON)})")
ax.plot([], [], "s", color=VERM, label=f"demolition candidates ({len(DEM)})")
ax.set_xlim(EXTENT10[0], EXTENT10[1]); ax.set_ylim(EXTENT10[2], EXTENT10[3])
ax.set_xlabel("UTM 36N easting (m)"); ax.set_ylabel("UTM 36N northing (m)")
ax.legend(loc="lower left", framealpha=0.9)
ax.set_title(f"Hebron: change candidates {START} to {END} (score threshold {THRESH:.1f}); grey = change score, "
             "outlines = building footprints", fontsize=10)
fig.savefig(RESULTS / "hebron_candidates_v6.png", dpi=250)
plt.show()
''')
md(r"""
**Reading the result.** Treat every candidate as a place to look, not a finding. Before reporting numbers: (1) check the footprint date against your period (section 5); (2) run the quiet-period test (9a) and report its false-alarm rate; (3) label a sample of candidates **and** of random buildings on dated imagery (9b), and report AUC, recall, precision in the labelled area and the type confusion. Only the type confusion tells you whether the "existed before" rule separates demolition from construction in Hebron; `pre_trend` and the VV/VH signs are there to test whether timing shape or polarisation adds to it.
""")

nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}})
out = Path(__file__).with_name("hebron_demolition_vs_construction_v6.ipynb")
nbformat.validate(nb)
nbformat.write(nb, out)
print("wrote", out, len(cells), "cells")
