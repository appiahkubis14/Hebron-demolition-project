# 9d: OCHA demolition records against the demolition candidates, place and time
import time
import urllib.parse
import urllib.request
from shapely.geometry import Point

ocha_table = None
OCHA_DIR = RESULTS / "ocha_check"
ocha_files = OCHA_FILES or [DATA / "references" / "demo-repository.xlsx"]
files = [f for f in ocha_files if Path(f).exists()]
if not files:
    print(f"no OCHA file found ({[str(f) for f in ocha_files]}): put demo-repository.xlsx there, or set OCHA_FILES")
else:
    OCHA_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in files:
        t = pd.read_excel(f) if str(f).lower().endswith((".xlsx", ".xls")) else pd.read_csv(f)
        frames.append(t.rename(columns={"Dateofincident": "date", "community_name": "community", "TotalNumberofStructures": "structures",
                                        "NumberofDisplacedPeople": "displaced", "GeneralLocation": "area"}))
    OCHA = pd.concat(frames, ignore_index=True)
    OCHA["date"] = pd.to_datetime(OCHA["date"])
    OCHA = OCHA[(OCHA["Governorate"] == OCHA_GOVERNORATE) & (OCHA["date"] >= OCHA_START)].drop_duplicates(["date", "community", "structures"])
    OCHA["community"] = OCHA["community"].astype(str).str.strip()
    print(f"OCHA: {len(OCHA)} incidents, {OCHA['structures'].sum()} structures, {OCHA['date'].min():%Y-%m-%d} .. "
          f"{OCHA['date'].max():%Y-%m-%d}, {OCHA['community'].nunique()} communities")

    # 1. locate the communities (OpenStreetMap, once; cached)
    cache_f = RESULTS / "ocha_communities.csv"
    cache = pd.read_csv(cache_f) if cache_f.exists() else pd.DataFrame(columns=["community", "lon", "lat", "osm_name", "how"])
    todo = [c for c in OCHA["community"].unique() if c not in set(cache["community"]) and c not in OCHA_MANUAL_COORDS]
    VIEW = (34.85, 31.30, 35.35, 31.72)

    def nominatim(q):
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
            {"q": q, "format": "json", "limit": 1, "viewbox": ",".join(map(str, VIEW)), "bounded": 1})
        req = urllib.request.Request(url, headers={"User-Agent": "hebron-change-research/1.0 (research use)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read().decode())
        time.sleep(1.1)                                        # Nominatim's usage policy: at most one request a second
        return out[0] if out else None

    found = []
    for c in todo:
        hit = None
        for q in (c, c.replace("Khirbet ", "Khirbat "), c.replace("Al ", "Al-").replace("Ar ", "Ar-").replace("As ", "As-")):
            try:
                hit = nominatim(q)
            except Exception as exc:
                print(f"lookup failed for {c!r}: {exc}")
                break
            if hit:
                break
        found.append({"community": c, "lon": float(hit["lon"]) if hit else np.nan, "lat": float(hit["lat"]) if hit else np.nan,
                      "osm_name": hit["display_name"][:80] if hit else "", "how": "OpenStreetMap" if hit else "not found"})
    if found:
        cache = pd.concat([cache.dropna(axis=1, how="all"), pd.DataFrame(found)], ignore_index=True)
        cache.to_csv(cache_f, index=False)
    COMM = cache.set_index("community")
    for c, (lon, lat) in OCHA_MANUAL_COORDS.items():
        COMM.loc[c, ["lon", "lat", "osm_name", "how"]] = [lon, lat, "", "by hand"]
    COMM = COMM.loc[COMM.index.isin(OCHA["community"].unique())].copy()
    COMM["inside"] = COMM["lon"].between(BBOX[0], BBOX[2]) & COMM["lat"].between(BBOX[1], BBOX[3])
    COMM = COMM.join(OCHA.groupby("community").agg(incidents=("date", "size"), structures=("structures", "sum")))
    print(f"communities located: {int(COMM['lon'].notna().sum())} of {len(COMM)}; inside the study area: {int(COMM['inside'].sum())}; "
          f"not found: {sorted(COMM.index[COMM['lon'].isna()])}")
    display(COMM[COMM["inside"]].sort_values("structures", ascending=False)[["how", "lon", "lat", "incidents", "structures"]])

    # 2. the incidents inside the area, and the whole-city ones, for the check by eye
    INC = OCHA.merge(COMM[["lon", "lat", "inside"]], left_on="community", right_index=True, how="left")
    INC["skipped"] = INC["community"].isin(OCHA_SKIP)
    inside = INC[INC["inside"].fillna(False).astype(bool) & ~INC["skipped"]].sort_values("date").reset_index(drop=True)
    inside.insert(0, "id", [f"O{i + 1:03d}" for i in range(len(inside))])
    city = INC[INC["skipped"]].sort_values("date")
    marks = "".join(f"<Placemark><name>{r.id} {r.community}</name><description>{r.date:%Y-%m-%d}: {r.structures} structure(s)"
                    f"</description><Point><coordinates>{r.lon:.6f},{r.lat:.6f},0</coordinates></Point></Placemark>" for r in inside.itertuples())
    (OCHA_DIR / "ocha_incidents_inside.kml").write_text('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
                                                        f"<Document><name>OCHA incidents in the study area</name>{marks}</Document></kml>", encoding="utf-8")
    city.assign(date=city["date"].dt.strftime("%Y-%m-%d"))[["date", "community", "structures", "displaced"]].to_csv(OCHA_DIR / "ocha_incidents_whole_city.csv", index=False)

    # 3. place and time: candidates near a community dated where the test would date that incident
    acq = np.array(sorted({date.fromisoformat(d).toordinal() for c in COMBOS for d in STORES[c].filled_dates()}), float)
    yrs_acq = np.array([date.fromordinal(int(v)).year for v in acq])
    STEP_DATES = BUILDING_TEST == "step" and STEP_SEASON_STARTS_ONLY
    if STEP_DATES:
        # the step test dates a change to a season's first image: an incident during a season can come out at that
        # season's start or the next one's; one between seasons at the next season's start. Tested starts: those with a
        # full season of dates before and after (the step test's minimum)
        starts = np.array(sorted({acq[yrs_acq == y].min() for y in set(yrs_acq)}))
        ends = {y: acq[yrs_acq == y].max() for y in set(yrs_acq)}
        testable = starts[[(acq < s_).sum() >= 10 and (acq >= s_).sum() >= 10 for s_ in starts]]

        def expected(t):
            y = date.fromordinal(int(t)).year
            in_season = y in ends and acq[yrs_acq == y].min() <= t <= ends[y]
            nxt = starts[starts > t]
            out = ([acq[yrs_acq == y].min()] if in_season else []) + ([nxt.min()] if nxt.size else [])
            return np.array([v for v in out if v in testable])
        tol = 0
    else:
        testable = np.unique(np.concatenate([ht.split_dates(np.array(sorted(date.fromisoformat(d).toordinal()
                                                                                for d in STORES[c].filled_dates()), float), TEST) for c in COMBOS]))

        def expected(t):
            f = acq[acq > t]
            return np.array([f.min()]) if f.size and f.min() in testable else np.array([])
        tol = OCHA_DATE_TOL_DAYS
    cand = DEM[DEM["date"].notna()].copy()
    ct = np.array([date.fromisoformat(d).toordinal() for d in cand["date"]], float)
    cx, cy = cand.geometry.centroid.x.to_numpy(), cand.geometry.centroid.y.to_numpy()
    pts = gpd.GeoSeries([Point(xy) for xy in zip(inside["lon"], inside["lat"])], crs=4326).to_crs(GRID_CRS)
    near = [np.flatnonzero(np.hypot(cx - q.x, cy - q.y) <= OCHA_RADIUS_M) for q in pts]
    t_inc = np.array([d.toordinal() for d in inside["date"].dt.date], float)
    exp = [expected(t) for t in t_inc]
    ok = np.array([e.size > 0 for e in exp])
    hits = lambda nb, e: int(sum((np.abs(ct[nb][:, None] - e[None, :]) <= tol).any(1))) if e.size and len(nb) else 0
    h_obs = np.array([hits(nb, e) for nb, e in zip(near, exp)])
    inside["expected_candidate_dates"] = [", ".join(date.fromordinal(int(v)).isoformat() for v in e) for e in exp]
    inside["testable"], inside["candidates_near_in_time"], inside["candidates_near_any_time"] = ok, h_obs, [len(nb) for nb in near]
    inside.assign(date=inside["date"].dt.strftime("%Y-%m-%d")).to_csv(OCHA_DIR / "ocha_incidents_inside.csv", index=False)
    print(f"{len(inside)} incidents inside the area ({int(ok.sum())} testable: the date the test would give them is one it tests); "
          f"{len(city)} recorded only as a whole city or region (listed for the check by eye)")
    if ok.sum():
        # chance: the same incidents on random days (uniform over the testable period), dated the same way
        rng_o = np.random.default_rng(SEED)
        lo, hi = (testable.min() - 300, testable.max()) if STEP_DATES else (testable.min() - 1, testable.max())
        near_ok = [near[i] for i in np.flatnonzero(ok)]
        draws = []
        for _ in range(OCHA_N_DRAWS):
            row = []
            for nb in near_ok:
                e = np.array([])
                while not e.size:
                    e = expected(rng_o.uniform(lo, hi))
                row.append(hits(nb, e))
            draws.append(row)
        draws = np.array(draws)
        obs_c, obs_i = int(h_obs[ok].sum()), int((h_obs[ok] > 0).sum())
        ch_c, ch_i = float(draws.sum(1).mean()), float((draws > 0).sum(1).mean())
        ocha_table = pd.DataFrame([
            {"measure": "candidates near an incident at the right time", "observed": obs_c, "by chance (mean)": round(ch_c, 1),
             "enrichment": round(obs_c / ch_c, 2) if ch_c else np.nan, "p (chance >= observed)": float(np.mean(draws.sum(1) >= obs_c))},
            {"measure": "incidents with at least one such candidate", "observed": obs_i, "by chance (mean)": round(ch_i, 1),
             "enrichment": round(obs_i / ch_i, 2) if ch_i else np.nan, "p (chance >= observed)": float(np.mean((draws > 0).sum(1) >= obs_i))}])
        display(ocha_table)
        display(inside.loc[ok, ["id", "date", "community", "structures", "expected_candidate_dates", "candidates_near_in_time", "candidates_near_any_time"]])
        rows_m = []
        for i in np.flatnonzero(ok & (h_obs > 0)):
            sel_m = near[i][(np.abs(ct[near[i]][:, None] - exp[i][None, :]) <= tol).any(1)]
            for j in sel_m:
                rows_m.append({"incident": inside.loc[i, "id"], "community": inside.loc[i, "community"],
                               "ocha_date": inside.loc[i, "date"].strftime("%Y-%m-%d"), "candidate_date": cand.iloc[j]["date"],
                               "score": round(float(cand.iloc[j]["score"]), 2), "geometry": cand.geometry.iloc[j]})
        if rows_m:
            M = gpd.GeoDataFrame(rows_m, crs=GRID_CRS)
            cen = M.geometry.centroid.to_crs(4326)
            M.to_crs(4326).to_file(OCHA_DIR / "ocha_matched_candidates.gpkg", driver="GPKG")
            M.drop(columns="geometry").assign(lon=cen.x.round(6), lat=cen.y.round(6)).to_csv(OCHA_DIR / "ocha_matched_candidates.csv", index=False)
            print(f"{len(M)} candidates matched an incident: {OCHA_DIR / 'ocha_matched_candidates.csv'} (confirm them on imagery)")
    print("written to", OCHA_DIR)