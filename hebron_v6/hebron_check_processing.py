"""Which satellite and which ESA processing made each date? (Hebron geolocation check, after section 6.)

Run inside the notebook, after section 6, with:   %run -i hebron_check_processing.py
It only reads: the raw GRD products in grd/raw (their manifest.safe) and the sigma0 files' headers.
"""
import re, zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from IPython.display import display


def product_info(p):
    """Satellite, ESA processor (IPF) version and processing time of one raw GRD product (zip or .SAFE folder)."""
    m = re.search(r"(S1[A-D])_IW_GRD\w*?_(\d{8})T", p.name)
    if not m:
        return None
    rec = {"date": f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:]}", "satellite": m.group(1), "product": p.name,
           "cog_product": "_COG" in p.name.upper(), "ipf": None, "processed": None}
    try:
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                name = next((n for n in z.namelist() if n.endswith("manifest.safe")), None)
                text = z.read(name).decode("utf-8", "replace") if name else ""
        else:
            mf = p / "manifest.safe"
            text = mf.read_text("utf-8", "replace") if mf.exists() else ""
        v = re.search(r'name="Sentinel-1 IPF"\s+version="([^"]+)"', text) or re.search(r'IPF[^>]*version="([^"]+)"', text)
        t = re.search(r'<safe:processing[^>]*name="[^"]*"[^>]*start="([^"]+)"', text) or re.search(r'<safe:processing[^>]*start="([^"]+)"', text)
        rec["ipf"], rec["processed"] = (v.group(1) if v else None), (t.group(1)[:10] if t else None)
    except Exception as exc:
        rec["ipf"] = f"unreadable ({type(exc).__name__})"
    return rec


raw = [product_info(p) for p in sorted((DATA / "grd" / "raw").rglob("S1*")) if p.suffix.lower() == ".zip" or p.suffix.upper() == ".SAFE"]
PROD = pd.DataFrame([r for r in raw if r]).drop_duplicates("date") if any(raw) else pd.DataFrame(columns=["date", "satellite", "ipf", "processed", "cog_product", "product"])
print(f"{len(PROD)} raw GRD products found in {DATA / 'grd' / 'raw'}"
      + ("" if len(PROD) else ": the raw files are not kept, so satellite and processor version are unknown here"))

# the sigma0 files themselves: size, pixel, extent and when they were written, per date
rows = []
for pol in FILES:
    for d, path in FILES[pol].items():
        with rasterio.open(path) as src:
            rows.append({"pol": pol.upper(), "date": d, "width": src.width, "height": src.height,
                         "pixel_deg": round(src.res[0], 8), "west": round(src.bounds.left, 4), "north": round(src.bounds.top, 4),
                         "written": pd.Timestamp(path.stat().st_mtime, unit="s").date()})
FILE_INFO = pd.DataFrame(rows)
tab = FILE_INFO.merge(RES.assign(date=RES["date"].dt.strftime("%Y-%m-%d"))[["pol", "date", "residual_m", "correlation"]],
                      on=["pol", "date"], how="left").merge(PROD, on="date", how="left")
tab["year"] = tab["date"].str[:4]
cols = [c for c in ("satellite", "ipf", "processed", "cog_product", "width", "height", "pixel_deg", "west", "north", "written") if c in tab]
print("per year and polarisation: the distinct values of each property, with the median shift and match")
summary = tab.groupby(["year", "pol"]).agg(
    **{c: (c, lambda s: ", ".join(sorted({str(v) for v in s.dropna()}))[:60]) for c in cols},
    shift_m=("residual_m", "median"), match=("correlation", "median"))
with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 60):
    display(summary)
