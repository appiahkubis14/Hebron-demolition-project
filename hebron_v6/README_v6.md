# Hebron demolition vs construction, v6

| File | What it is |
|---|---|
| `hebron_demolition_vs_construction_v6.ipynb` | The notebook (built by `build_v6.py`) |
| `hebron_tools.py` | The method: complete archive search, Lee filter, on-disk stacks, the rolling t-test with persistence and season check, permutation threshold, Microsoft footprints, building scores, construction clusters, validation metrics. Must sit next to the notebook |
| `test_hebron_tools.py` | Unit tests (`pytest -q test_hebron_tools.py`): steps found and dated to the first acquisition, false alarms near the null quantile, seasonal swings removed by the season check, 30-day dips rejected by persistence, gradual vs abrupt |
| `test_v6.py` | Runs the whole notebook (without downloads) on a synthetic town: `python test_v6.py /tmp/hebron_v6_test` |
| `example_run_synthetic.ipynb` | That run, with every output |

## Before the real run

1. Copy `hebron_tools.py` next to the notebook. Same pygeofetch as v5 (upgraded `sar/extraction.py`, `insar/interferogram.py`, `insar/workflow.py`, `insar/tops.py`).
2. Section 2 prints, per track, the volume for every acquisition and for the thinned series. Default: track D094, 10 dry-season dates per year, ~120 GB for 2020–2026 (each scene downloaded once, deleted after VV and VH are verified). `MAX_DOWNLOAD_GB` is 200.
3. Disk on the drive: one raw scene (~1.7 GB) at a time while downloading; then ~20 MB per date (σ⁰, VV + VH) and the 10 m stacks (`grd/stacks10`, ~0.5 GB per polarisation for 70 dates).
4. Footprints: leave `FOOTPRINTS_PATH = None` to download Microsoft's once, or point it at a footprint file whose date you know (better for the type rule).
5. For results you can report: set `QUIET_PERIOD` (section 9a) and make building-level labels (`BUILDING_REF_PATH`, ideally with `LABEL_AREA_PATH`, section 9b).

## What the synthetic runs show (and what they do not)

Synthetic town, one track (VV + VH), 1,120 buildings, 10 demolitions and 4 construction sites, speckle, moisture, a seasonal field (`test_v6.py`; `MODE`, `PER_YEAR`, `NTRACKS` select the series):

| Series | Demolitions found (right type) | Median date error | Precision, demolitions | Construction found | False alarms, quiet period |
|---|---|---|---|---|---|
| every acquisition (12-day) | 10/10 | 5 d | 0.42 | 4/4 | 4 (11 per 1,000 buildings per year) |
| **10 dry-season dates / year (default)** | **10/10** | **49 d** | **0.48** | **4/4** | **0** |
| 5 dry-season dates / year | 5/10 | 112 d | 0.50 | 3/4 | 0 |

Volume for one track 2020–2026: every acquisition ~480 GB; 10 per year ~120 GB; 5 per year ~60 GB. Downloads are streamed (one ~1.7 GB scene on disk at a time); what stays is ~20 MB per date plus the 10 m stacks (well under 1 GB per polarisation for 70 dates).

This checks that the code does what it is designed to do and shows the cost of thinning. It says nothing about Hebron: the synthetic buildings are larger and brighter than many real ones, and real scenes have layover, vehicles and partial changes. Only sections 9a and 9b on real data measure the method there.
