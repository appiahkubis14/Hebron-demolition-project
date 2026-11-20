# Hebron: Sentinel-1 demolition vs. construction detection

Can Sentinel-1 radar distinguish building demolitions from settlement construction in Hebron?

This project answers that question with two independent radar signals, validated against reference data, using pygeofetch for everything from data search to the final map. All work is in one notebook: `hebron_demolition_vs_construction.ipynb`.

## Contents

- [Overview](#overview)
- [Data](#data)
- [Method](#method)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Quick test with 3 images](#quick-test-with-3-images-pilot-mode)
- [Parameters](#parameters)
- [Vegetated and low-coherence areas](#vegetated-and-low-coherence-areas)
- [Outputs](#outputs)
- [Validation](#validation)
- [Notebook structure](#notebook-structure)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Status of testing](#status-of-testing)

## Overview

| | |
|---|---|
| **Question** | Can Sentinel-1 tell demolitions from construction in Hebron? |
| **Signal 1** | VH backscatter time series from Sentinel-1 GRD: threshold detection of a persistent step |
| **Signal 2** | InSAR coherence from Sentinel-1 SLC: detection of anomalous coherence drops |
| **Mask** | Built-up area from World Settlement Footprint (WSF) and Dynamic World |
| **Reference** | OCHA demolitions (demolition) and Peace Now / ARIJ outposts (construction) |
| **Products** | Change/demolition map and anomalous coherence drop map, as PNG and GeoTIFF |

The type of change (demolition or construction) comes from the **sign of the backscatter step**. A coherence drop confirms that something changed on that date but says nothing about what. The notebook tests whether the sign separates the two classes; it does not assume it.

## Data

| Data | Source | How it is obtained |
|---|---|---|
| Sentinel-1 GRD (VH) | Copernicus Data Space | `client.search` and `client.download` (pygeofetch) |
| Sentinel-1 SLC (VV) | Copernicus Data Space | `InSARProject` (pygeofetch) |
| Orbit files | ESA (precise, restituted) | `fetch_orbit_file` (pygeofetch) |
| DEM | OpenTopography (Copernicus DEM 30 m or SRTM 30 m) | `client.search` / `client.download` (pygeofetch) |
| AOI | Default bounding box, or your own file | `BBOX` or `AOI_PATH` |
| Settlement extent | WSF, Dynamic World | **You supply** (`WSF_PATH`, `DW_PATH`) |
| Demolitions | OCHA | **You supply** (`OCHA_PATH`) |
| Construction | Peace Now, ARIJ | **You supply** (`PEACENOW_PATH`, `ARIJ_PATH`) |

pygeofetch has no loaders for WSF, Dynamic World, OCHA, Peace Now or ARIJ, so these are read from local files.

**File formats**

- WSF: raster, value above 0 means settlement.
- Dynamic World: label raster, value 6 means built.
- Reference points: CSV with longitude and latitude columns (`lon`/`long`/`longitude`/`x`, `lat`/`latitude`/`y`) or any vector file readable by GeoPandas (polygons are reduced to a representative point). If a column name contains `date`, it is used as the event date.
- AOI: GeoJSON, GeoPackage or Shapefile. Its bounds replace `BBOX`.

## Method

### 1. Track selection
Backscatter is only comparable between dates seen with the same geometry, so one relative orbit and pass direction is used for the whole study. Only footprints that contain the whole AOI are kept, with one scene per date.

### 2. Backscatter stack
For each GRD scene the VH band is extracted and georeferenced (`GRDExtractor`), averaged onto a 30 m UTM 36N grid as the root-mean-square of the digital numbers (so power is averaged), converted to dB, and cached. The downloaded zip is then deleted. A per-date median is subtracted to remove acquisition-to-acquisition offsets.

Values are **relative backscatter** (DN² in dB), not calibrated σ⁰. pygeofetch's `SARProcessor.calibrate` uses a constant calibration factor of 1 rather than the product's calibration table. On a single track the factor is the same for every date at a given pixel, so it cancels in change detection.

### 3. Method 1: backscatter threshold detection
For every cell and every candidate date, the mean of the next `K_WIN` dates is compared with the mean of the previous `K_WIN` dates. The largest step is kept. A step is flagged when:

1. it is at least `MIN_ABS_DB` in size,
2. its robust z-score (against the scatter of steps over the whole settlement) is at least `Z_MIN`,
3. the new level **persists**: the mean of all later dates still differs from the mean of all earlier dates in the same direction, by at least half the step, and
4. it belongs to a cluster of at least `MIN_PIX` cells.

Flagged steps are split by sign into *demolition-like* (sign equal to `DEMOLITION_SIGN`) and *construction-like* (opposite sign).

### 4. Coherence
Only pairs of the **same satellite one repeat cycle apart** (at most 13 days apart) are formed. Pairs across satellites have different burst timing and decorrelate even on stable ground. Interferograms are produced by `InterferogramGenerator.process_pairs`.

Each pair's coherence is placed on the analysis grid using the exact radar geometry and the DEM height of every cell. On Hebron's relief an ellipsoid-only position would be off by about a kilometre. Pair-wide offsets (atmosphere, residual coregistration) are removed by matching each pair's settlement median.

### 5. Method 2: coherence change detection
A cell is **anomalous** when it is normally coherent (median over pairs at least `COH_STABLE`) and one pair falls at least `COH_DROP` below that median. The pair that carries the drop dates it.

### 6. Combined classes
A VH step is marked as confirmed when a coherence-anomalous cell lies within one cell of it and the two dates agree within `TOL_DAYS`.

| Code | Class |
|---|---|
| 0 | No change |
| 1 | Demolition-like (VH) |
| 2 | Construction-like (VH) |
| 3 | Demolition-like with coherence drop |
| 4 | Construction-like with coherence drop |

## Requirements

- Python 3.10 or newer and Jupyter
- **pygeofetch with the TOPS deramping fix** (the package contains `pygeofetch/insar/tops.py`). Without it, Sentinel-1 coherence is unreliable. The notebook warns if the module is missing.
- Python packages: `numpy`, `scipy`, `pandas`, `geopandas`, `shapely`, `rasterio` (with GDAL 3.3 or newer for RMS resampling), `matplotlib`
- Accounts:
  - Copernicus Data Space (e-mail and password)
  - OpenTopography API key
- Disk and time:
  - GRD: about 1 GB per scene, processed one at a time and deleted afterwards
  - SLC: about 4 GB per scene, kept (16 scenes by default)
  - Interferogram processing takes several minutes per pair

## Quick start

1. Install the TOPS-fixed pygeofetch and the packages above.
2. Open `hebron_demolition_vs_construction.ipynb`.
3. In section 0, set the paths to your reference files and review the periods:

   ```python
   WSF_PATH = "data/ref/wsf_hebron.tif"
   DW_PATH = "data/ref/dynamic_world_built.tif"
   OCHA_PATH = "data/ref/ocha_demolitions.csv"
   PEACENOW_PATH = "data/ref/peacenow_outposts.geojson"
   ARIJ_PATH = None
   SLC_START, SLC_END = "2026-06-01", "2026-10-31"   # months that contain your OCHA events
   ```

4. Provide credentials, either as environment variables or when prompted:


5. Run the cells in order. Re-running resumes from the cache, so an interrupted run does not repeat finished downloads or interferograms.

Section 8 needs the reference files; without them it stops with a message and the maps in section 9 still work.

## Quick test with 3 images (pilot mode)

`PILOT = True` is the default. It limits the run to a pipeline check:

- the search window shrinks to the 45 days before `END`,
- 3 GRD scenes (about 3 GB) and 3 SLC scenes of one satellite (about 12 GB) are used,
- `K_WIN` is set to 1, so backscatter compares single dates, and the 3 SLC scenes give 2 coherence pairs.

Results are noisy and nothing is validated; this only shows that search, download, processing and maps work end to end. To bracket a known event instead, set explicit dates (they must be acquisitions of the selected track; the SLC dates must be one satellite, 12 days apart):

```python
GRD_DATES = ["2026-06-05", "2026-06-17", "2026-06-29"]
SLC_DATES = ["2026-06-05", "2026-06-17", "2026-06-29"]
```

Set `PILOT = False` for the full run. A guard (`MAX_DOWNLOAD_GB`, default 30) stops any section that would download more than that. With two coherence pairs the normal coherence is the better of the two; with a single pair no drop can be computed.

## Parameters

All parameters are in the section 0 cell.

| Parameter | Default | Meaning |
|---|---|---|
| `PILOT`, `PILOT_N` | `True`, 3 | Pilot mode and its number of scenes |
| `MAX_DOWNLOAD_GB` | 30 | Stop before downloading more than this per section |
| `BBOX` | `(35.04, 31.48, 35.18, 31.60)` | Study area (min lon, min lat, max lon, max lat) |
| `AOI_PATH` | `None` | AOI file; its bounds replace `BBOX` |
| `START`, `END` | 2025-01-01, 2026-10-31 | GRD backscatter period |
| `SLC_START`, `SLC_END` | 2026-06-01, 2026-10-31 | SLC coherence period |
| `MAX_GRD`, `MAX_SLC` | `None`, 16 | Caps on the number of scenes |
| `GRD_DATES`, `SLC_DATES` | `None` | Explicit lists of dates to use (quick test) |
| `TRACK` | `None` | `(pass, relative orbit)`, for example `("descending", 21)`; `None` picks the track with most acquisitions |
| `GRID_M` | 30 | Analysis cell in metres, matched to the multilooked SLC cell |
| `K_WIN` | 3 | Dates averaged either side of a candidate break |
| `MIN_ABS_DB` | 1.5 | Smallest backscatter step reported (dB) |
| `Z_MIN` | 3.0 | Minimum robust z-score of a step |
| `MIN_PIX` | 2 | Smallest changed cluster (cells) |
| `DEMOLITION_SIGN` | -1 | Sign of the VH step called demolition-like; a hypothesis tested in section 8 |
| `COH_STABLE` | 0.40 | Median coherence a cell needs to be testable |
| `COH_DROP` | 0.25 | Coherence drop counted as anomalous |
| `TOL_DAYS` | 24 | Allowed gap between backscatter step and coherence drop, and between detection and recorded event |
| `REF_RADIUS_M` | 150 | A reference point counts as detected if a flagged cell lies within this distance |

The default thresholds are starting points. They have not been tuned on real Hebron data; tune `Z_MIN` and `MIN_ABS_DB` on a calibration area before reporting results.

## Vegetated and low-coherence areas

Coherence can only show a drop where it was high to begin with. Vegetated and bare ground usually sits near the estimator's noise floor (about 0.2), so a change there cannot be seen by the coherence method. The project handles this as follows.

**How it is handled now**

| Issue | Handling |
|---|---|
| Cells that are never coherent | Excluded from the coherence test. A cell is tested only if its normal coherence is at least `COH_STABLE` (0.40). |
| How much can be tested | Section 7 prints the share of settlement cells that are coherent enough to test. Report this share with the coherence map; the map says nothing about the rest. |
| Vegetation in the analysis | The settlement mask (WSF, Dynamic World) keeps most vegetated ground out of both methods. |
| Backscatter on vegetated ground | It needs no coherence, but VH changes with season. The persistence test rejects short disturbances. It does not reject a slow trend lasting many months. |
| Weak coherence in individual pairs | Pairs are same-satellite and 12 days apart, and each pair's median coherence is matched to the others before the drop is measured. |

**What is not handled yet**

- **Construction on vegetated or bare land.** There, coherence starts low and rises after buildings appear. The signal is a coherence *gain*, and the notebook only looks for drops, so this case is missed. A gain detector needs about 4 or 5 pairs and cannot run in pilot mode.
- **The settlement mask can hide construction.** WSF and Dynamic World describe land that was built up when they were made. New outposts are often on land that was not, so the mask can remove them. Demolition is not affected, because the buildings already exist. Until this is changed, the construction results are biased towards construction inside existing settlements.

**Recommended changes (not yet in the notebook)**

1. Apply the settlement mask to the demolition analysis only. For construction, use no mask or a mask widened by a few hundred metres.
2. Add a coherence-gain detector for construction: low coherence in the early pairs, high in the later pairs.
3. Report, for each method, the share of the AOI it can test.

Until these are done, state in any results that construction detection covers only existing settlement areas.

## Outputs

Written to `data/hebron/results/`:

| File | Content |
|---|---|
| `change_classes.tif` | Class codes 0 to 4 (uint8) |
| `vh_step_db.tif` | Signed VH step in dB where flagged |
| `coherence_drop.tif` | Coherence drop where anomalous |
| `hebron_maps.png` | Left: change/demolition map. Right: anomalous coherence drop map |

All rasters use EPSG:32636 (UTM 36N) at `GRID_M` resolution. In the PNG the changed cells are enlarged for display only; the GeoTIFFs are not.

Other folders under `data/hebron/`:

| Folder | Content |
|---|---|
| `grd/cache_<pass>_<orbit>/` | Cached VH arrays, one per date |
| `slc/` | Downloaded SLC, extracted bursts, orbit files |
| `ifg/` | One folder per pair with interferogram, coherence and geometry |
| `dem/` | DEM |

## Validation

Section 8 compares the detections with the reference points. For each reference point the strongest signal within `REF_RADIUS_M` is taken and compared with up to 500 random settlement cells at least 300 m from any reference (the *control* class).

The notebook prints:

- hit rates per class: share of points with a demolition-like step, a construction-like step, and a coherence anomaly nearby,
- AUC values (0.5 means no separation, 1.0 perfect):

  | Comparison | Meaning |
  |---|---|
  | Change vs control, VH step | Does the backscatter method see change at all? |
  | Change vs control, coherence drop | Does the coherence method see change at all? |
  | Demolition vs construction, VH sign | **The research question**: does the sign separate the two? |
  | Demolition vs construction, coherence drop | Does coherence separate the two? |

- timing: for demolitions with a recorded date, the median difference between the detected break and the recorded date, and the share within `TOL_DAYS`.

If the demolition-vs-construction AUC for the VH sign is below 0.5, the sign assumed in `DEMOLITION_SIGN` is reversed. Do not flip it and then report the same points as validation; use a calibration subset or independent points.

## Notebook structure

| Section | Content |
|---|---|
| 0 | Parameters, grid, credentials |
| 1 | Search GRD, choose track |
| 2 | VH backscatter stack |
| 3 | Settlement mask |
| 4 | Method 1: backscatter threshold detection |
| 5 | SLC, orbits, DEM |
| 6 | Coherence of same-satellite pairs, geocoding |
| 7 | Method 2: coherence change detection, combined classes |
| 8 | Validation against reference data |
| 9 | Maps and export |

## Limitations

- **Geocoding of GRD.** GRD is geocoded with the product's tie points and is not terrain-corrected, so on Hebron's relief a feature can sit tens of metres from its true position. `REF_RADIUS_M` absorbs this. Only one relative orbit is used, so the same displacement applies to every date.
- **Resolution.** The analysis cell is 30 m. A single house is smaller than one cell, so the method detects clusters and larger buildings, not individual small structures.
- **Relative backscatter.** DN² in dB with a per-date median removed, not calibrated σ⁰.
- **Vegetated ground.** Coherence cannot show change where it is already low; see [Vegetated and low-coherence areas](#vegetated-and-low-coherence-areas).
- **Coherence.** Same-satellite 12-day pairs only. The estimator has a noise floor of about 0.2, so cells that are not normally coherent cannot show a drop and are excluded by `COH_STABLE`. The number of pairs is limited by the SLC period and `MAX_SLC`.
- **Type of change.** It comes from the sign of the VH step. Whether demolition is a decrease or an increase in Hebron's dense, hilly urban fabric is tested in section 8, not assumed.
- **Reference data.** OCHA locations may be recorded at locality level rather than per building, and Peace Now and ARIJ outposts are a partial list of construction. Hit rates are lower bounds of detection, not accuracy.
- **Temporal coverage.** A change that lasts fewer than `K_WIN` dates, and anything in the last `K_WIN` dates, cannot be detected.
- **No comparison with other processors.** Results have not been compared with SNAP, ISCE or HyP3 products.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Warning about `insar/tops.py` | Install the TOPS-fixed pygeofetch before section 5. |
| Search returns no GRD scenes | The catalogue may label the product type differently. The notebook already retries without the filter; check the dates and AOI. |
| Download fails with an authentication error | Check the Copernicus password at the website and re-enter it. |
| `No settlement mask found` | `WSF_PATH` and `DW_PATH` are unset or do not overlap the AOI. The whole AOI is used, with more false alarms. |
| Few or no usable pairs | Check the printed orbit-file count. Precise orbits appear about three weeks after acquisition; restituted orbits are used as a fallback. Pairs across satellites are excluded on purpose. |
| `no radar geometry saved` | Orbit files were missing for that pair. Fix the orbit step and delete the pair's folder in `ifg/` to recompute it. |
| Many false alarms | Provide the settlement mask, raise `Z_MIN` or `MIN_ABS_DB`, or raise `MIN_PIX`. |
| Few detections | Lower `Z_MIN`, check that at least `2 * K_WIN` dates exist, and check the track choice. |
| Run interrupted | Run the cells again. Cached VH arrays and finished interferograms are reused. |

## Status of testing

- The numerical parts (backscatter step detection, coherence anomaly logic, validation and maps) were tested on synthetic data with planted changes. The break date was recovered, a two-date transient was rejected, and no false alarms appeared elsewhere with the default settings. The terrain-height geocoding was checked against a known 1.1 km shift.
- The search, download, orbit and interferogram calls follow the working Mexico City notebook, and their signatures were checked against the pygeofetch source.
- The notebook has **not yet been run on real Sentinel-1 scenes**, and the default thresholds have not been tuned on real data.

## Author

Samuel Appiah Kubi