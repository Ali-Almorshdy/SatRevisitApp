# app.py
"""
Streamlit front-end for Satellite Pass & GA optimizer.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, time as dt_time
from typing import List, Any
import traceback
from passes import compute_passes
try:
    from gaoptm import run_ga
    GA_AVAILABLE = True
except Exception:
    run_ga = None
    GA_AVAILABLE = False

# Styling: pro black theme
st.set_page_config(page_title="Satellite Revisit Optimizer", layout="wide", page_icon="🚀")
st.markdown(
    """
    <style>
      body, .stApp, .block-container { background-color: #000000; color: #e6eef6; }
      section[data-testid="stSidebar"] { background-color: #000000; color: #e6eef6; }
      .stButton>button, .stDownloadButton>button {
          background: linear-gradient(180deg,#141414,#0d0d0d);
          color: #fff;
          border-radius: 10px;
          padding: 8px 12px;
          border: 1px solid #222;
          font-weight: 600;
      }
      .metric-card { background: linear-gradient(180deg,#0f1620,#081016); padding:12px; border-radius:12px; border:1px solid #172028; }
      .small { font-size:0.9rem; color:#cbd6df; }
    </style>
    """, unsafe_allow_html=True
)

# ---------------- session init ----------------
def ensure_state():
    if "locations" not in st.session_state:
        st.session_state.locations = []   # list of dicts {name,lat,lon,key}
    if "processed_drawn_keys" not in st.session_state:
        st.session_state.processed_drawn_keys = set()
    if "ignored_drawn_keys" not in st.session_state:
        st.session_state.ignored_drawn_keys = set()
    if "manual_sats" not in st.session_state:
        st.session_state.manual_sats = [] # list of dicts {a,e,TA,RA,incl,w}
    if "last_results" not in st.session_state:
        st.session_state.last_results = None
    if "ga_running" not in st.session_state:
        st.session_state.ga_running = False
    if "ga_best" not in st.session_state:
        st.session_state.ga_best = None
    if "ga_history" not in st.session_state:
        st.session_state.ga_history = []
ensure_state()

# ---------------- helpers ----------------
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

def get_key(lat: float, lon: float, prec: int = 6) -> str:
    return f"{round(float(lat), prec)}_{round(float(lon), prec)}"

def add_location(lat: float, lon: float, name: str | None = None) -> bool:
    key = get_key(lat, lon)
    if key in st.session_state.ignored_drawn_keys:
        return False
    if any(loc.get("key") == key for loc in st.session_state.locations):
        st.session_state.processed_drawn_keys.add(key)
        return False
    if name is None:
        name = f"location{len(st.session_state.locations) + 1}"
    st.session_state.locations.append({"name": name, "lat": float(lat), "lon": float(lon), "key": key})
    st.session_state.processed_drawn_keys.add(key)
    return True

def renumber_locations():
    new = []
    for i, loc in enumerate(st.session_state.locations, start=1):
        lat = float(loc["lat"]); lon = float(loc["lon"])
        new.append({"name": f"location{i}", "lat": lat, "lon": lon, "key": get_key(lat, lon)})
    st.session_state.locations = new

def manual_coes_to_array(manual_sats: List[dict]) -> np.ndarray:
    if not manual_sats:
        return np.empty((0,6), dtype=float)
    arr = []
    for s in manual_sats:
        arr.append([float(s["a"]), float(s["e"]), float(s["TA"]), float(s["RA"]), float(s["incl"]), float(s["w"])])
    return np.array(arr, dtype=float)

def compute_passes_safe(locs_np, coes, epoch, dt, duration_days, gap_seconds):
    try:
        res = compute_passes(locs_np, coes, epoch=epoch, dt=float(dt), duration_days=float(duration_days), gap_seconds=float(gap_seconds))
        return res, None
    except Exception as e:
        return None, traceback.format_exc()

# ---------------- header ----------------
col1, col2 = st.columns([4,1])
with col1:
    st.title("🚀 Satellite Pass & GA Optimizer")
with col2:
    if st.button("🧹 Clear All"):
        st.session_state.locations = []
        st.session_state.manual_sats = []
        st.session_state.last_results = None
        st.session_state.ga_best = None
        st.session_state.ga_history = []
        st.session_state.ga_running = False
        safe_rerun()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", ["Manual", "GA Optimization"])

    st.markdown("---")
    st.subheader("Epoch (UTC)")
    date_in = st.date_input("Date", value=datetime(2025,9,2).date())
    time_in = st.time_input("Time (UTC)", value=dt_time(9,0,0))
    epoch = datetime.combine(date_in, time_in)

    st.subheader("Propagation")
    dt = st.number_input("Final dt (s)", value=10.0, min_value=1.0, step=1.0)
    duration_days = st.number_input("Final duration (days)", value=1.0, min_value=0.001, step=0.5)
    gap_seconds = st.number_input("gap_seconds (s) for sep()", value=20.0, min_value=1.0, step=1.0)

    st.markdown("---")
    st.subheader("Stored Locations (sidebar)")
    with st.form("add_loc_form", clear_on_submit=True):
        lat_in = st.number_input("Latitude", -90.0, 90.0, 30.0, format="%.6f")
        lon_in = st.number_input("Longitude", -180.0, 180.0, 30.0, format="%.6f")
        add_loc = st.form_submit_button("➕ Add Location")
        if add_loc:
            ok = add_location(lat_in, lon_in)
            if ok:
                st.success("Location added.")
                safe_rerun()
            else:
                st.warning("Location exists or is ignored.")

    if st.session_state.locations:
        for idx, loc in enumerate(st.session_state.locations):
            c1, c2 = st.columns([3,1])
            c1.write(f"**{loc['name']}**")
            c1.write(f"{loc['lat']:.6f}, {loc['lon']:.6f}")
            if c2.button("❌", key=f"del_loc_{idx}"):
                removed = st.session_state.locations.pop(idx)
                st.session_state.ignored_drawn_keys.add(removed.get("key", get_key(removed["lat"], removed["lon"])))
                renumber_locations()
                safe_rerun()
    else:
        st.info("No stored locations. Add via map or the form above.")

    if st.button("🗑️ Clear All Locations"):
        st.session_state.locations = []
        st.session_state.processed_drawn_keys = set()
        st.session_state.ignored_drawn_keys = set()
        safe_rerun()

    st.markdown("---")
    if mode == "Manual":
        st.subheader("🛰 Manual Satellite (UI order)")
        with st.form("add_sat_form", clear_on_submit=True):
            a = st.number_input("a [km]", value=7000.0, step=1.0)
            e = st.number_input("e", value=0.01, min_value=0.0, max_value=1.0, step=0.001)
            TA = st.number_input("TA [deg]", value=0.0, step=0.1)
            RA = st.number_input("RA (RAAN) [deg]", value=0.0, step=0.1)
            incl = st.number_input("incl [deg]", value=98.0, step=0.1)
            w = st.number_input("w (arg) [deg]", value=0.0, step=0.1)
            add_sat = st.form_submit_button("➕ Add Satellite")
            if add_sat:
                st.session_state.manual_sats.append({"a":float(a),"e":float(e),"TA":float(TA),"RA":float(RA),"incl":float(incl),"w":float(w)})
                st.success("Manual satellite added.")
                safe_rerun()
    else:
        st.subheader("🤖 GA Options")
        if not GA_AVAILABLE:
            st.warning("gaoptm.run_ga not found — put gaoptm.py with run_ga(...) to enable GA.")
        n_sats = st.number_input("Number of satellites", min_value=1, value=3, step=1)
        pop_size = st.number_input("Population size", min_value=4, value=20, step=1)
        generations = st.number_input("Generations", min_value=1, value=30, step=1)
        mutation_rate = st.number_input("Mutation rate (per gene)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        crossover_rate = st.number_input("Crossover rate", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        ga_dt = st.number_input("GA dt (s) — coarser speeds GA", value=60.0, min_value=1.0, step=1.0)
        ga_duration_days = st.number_input("GA duration (days)", value=1.0, min_value=0.001, step=0.5)
        st.markdown("**SMA (a) bounds for GA**")
        a_min = st.number_input("SMA min (km)", value=6800.0, step=1.0)
        a_max = st.number_input("SMA max (km)", value=7500.0, step=1.0)
        st.markdown("Inclination limits:")
        auto_incl = st.checkbox("Auto-set incl min from stored locations' max absolute latitude", value=True)
        if auto_incl and st.session_state.locations:
            maxlat = max(abs(loc["lat"]) for loc in st.session_state.locations)
            incl_min_auto = float(np.clip(maxlat, 0.0, 90.0))
            st.write(f"Auto incl_min = {incl_min_auto:.2f}° (max abs lat)")
            incl_min = float(incl_min_auto)
            incl_max = st.number_input("Incl max (deg)", value=98.0, step=0.1)
        else:
            incl_min = st.number_input("Incl min (deg)", value=0.0, step=0.1)
            incl_max = st.number_input("Incl max (deg)", value=98.0, step=0.1)

        seed = st.number_input("Random seed (0 = random)", value=0, step=1)
        if seed == 0:
            seed = None

# ---------------- Map (marker-only) ----------------
st.subheader("Map — add ground points (click / draw marker)")

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

if st.session_state.locations:
    avg_lat = np.mean([c["lat"] for c in st.session_state.locations])
    avg_lon = np.mean([c["lon"] for c in st.session_state.locations])
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=2, control_scale=True, tiles=None,min_zoom=2)
else:
    m = folium.Map(location=[20, 2], zoom_start=2, control_scale=True, tiles=None)

folium.TileLayer(tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                 attr="&copy; OpenStreetMap contributors", control=False, no_wrap=True, max_zoom=19, min_zoom=2).add_to(m)

# add markers
for loc in st.session_state.locations:
    folium.Marker([loc["lat"], loc["lon"]], popup=loc["name"], draggable=False).add_to(m)

# Draw marker only
draw = Draw(draw_options={"polyline": False, "polygon": False, "rectangle": False, "circle": False, "circlemarker": False, "marker": True},
            edit_options={"edit": False, "remove": False})
draw.add_to(m)

map_result = st_folium(m, width=920, height=450, returned_objects=["all_drawings", "last_clicked"])

# process map_result
if map_result:
    added = False
    last = map_result.get("last_clicked")
    if last and isinstance(last, dict):
        lat = last.get("lat"); lon = last.get("lng")
        if lat is not None and lon is not None:
            key = get_key(lat, lon)
            if key not in st.session_state.processed_drawn_keys and key not in st.session_state.ignored_drawn_keys:
                add_location(lat, lon); added = True

    raw = map_result.get("all_drawings")
    if raw:
        features = []
        if isinstance(raw, dict):
            features = raw.get("features", []) or []
        elif isinstance(raw, list):
            features = raw
        for feat in features:
            if not isinstance(feat, dict): continue
            geom = feat.get("geometry", {})
            if not geom or geom.get("type") != "Point": continue
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                lon = float(coords[0]); lat = float(coords[1])
                key = get_key(lat, lon)
                if key in st.session_state.ignored_drawn_keys: continue
                if key in st.session_state.processed_drawn_keys: continue
                ok = add_location(lat, lon)
                if ok: added = True
    if added:
        safe_rerun()

# ---------------- Manual sats display ----------------
st.markdown("---")
st.subheader("Manual Satellites (UI order: a,e,TA,RA,incl,w)")
if st.session_state.manual_sats:
    rm_idx = None
    for i, s in enumerate(st.session_state.manual_sats):
        c1, c2 = st.columns([8,1])
        with c1:
            st.write(f"Sat{i+1}: a={s['a']:.1f} km, e={s['e']:.4f}, TA={s['TA']:.2f}°, RA={s['RA']:.2f}°, incl={s['incl']:.2f}°, w={s['w']:.2f}°")
        with c2:
            if st.button("❌", key=f"del_manual_{i}"):
                rm_idx = i
    if rm_idx is not None:
        st.session_state.manual_sats.pop(rm_idx)
        safe_rerun()
else:
    st.info("No manual satellites yet.")

# ---------------- Controls: Compute & GA ----------------
st.markdown("---")
colA, colB, colC = st.columns([1,1,2])
with colA:
    compute_manual_btn = False
    if mode == "Manual":
        compute_manual_btn = st.button("🚀 Compute Passes & Revisits (Manual)")
with colB:
    ga_start = False; ga_stop = False
    if mode != "Manual":
        ga_start = st.button("▶️ Run GA")
        ga_stop = st.button("⏹️ Stop GA")
with colC:
    export_all = st.button("📤 Export last results (CSV)")

# ---------------- Manual compute ----------------
if compute_manual_btn:
    if compute_passes is None:
        st.error("compute_passes not found. Put it in functions.py / passes.py / psses.py.")
    elif len(st.session_state.manual_sats) == 0:
        st.warning("Add at least one manual satellite.")
    elif len(st.session_state.locations) == 0:
        st.warning("Add at least one location.")
    else:
        locs_np = np.array([[c["lat"], c["lon"]] for c in st.session_state.locations], dtype=float)
        coes_ui = manual_coes_to_array(st.session_state.manual_sats)  # UI order
        with st.spinner("Computing passes..."):
            res, err = compute_passes_safe(locs_np, coes_ui, epoch, dt, duration_days, gap_seconds)
            if err:
                st.error(f"compute_passes failed:\n{err}")
            else:
                st.session_state.last_results = res
                st.session_state.last_params = {"mode":"manual","epoch":epoch,"dt":dt,"duration_days":duration_days}
                st.success("Computation finished.")

# ---------------- GA run/stop ----------------
if mode != "Manual":
    if ga_start:
        if not GA_AVAILABLE:
            st.error("GA engine not available. Put gaoptm.py with run_ga(...) in the project.")
        elif compute_passes is None:
            st.error("compute_passes not found. GA requires compute_passes.")
        elif len(st.session_state.locations) == 0:
            st.warning("Add at least one location.")
        else:
            # set GA running flag and placeholders
            st.session_state.ga_running = True
            st.session_state.ga_history = []
            st.session_state.ga_best = None

            # progress placeholders
            prog_txt = st.empty()
            prog_bar = st.progress(0)
            gen_table_spot = st.empty()

            def progress_cb(gen, best_score, best_coes, history):
                # called from within GA loop (synchronous)
                try:
                    pct = int(100 * (gen / max(1, int(generations))))
                    prog_bar.progress(min(max(pct, 0), 100))
                    gen_text = f"Generation {gen}/{generations} — best avg revisit = {best_score:.2f} s ({best_score/3600.0:.4f} hr)"
                    prog_txt.markdown(f"**{gen_text}**")
                    # show best_coes table (UI order)
                    try:
                        df = pd.DataFrame(best_coes, columns=["a","e","TA","RA","incl","w"])
                        gen_table_spot.dataframe(df)
                    except Exception:
                        pass
                except Exception:
                    pass

            # prepare GA bounds incl from UI settings
            sma_b = (float(a_min), float(a_max))
            e_b = (0.0, 0.05)  # keep small eccentricity default; user can adjust code if needed
            incl_b = (float(incl_min), float(incl_max))
            ra_b = (0.0, 360.0)
            ta_b = (0.0, 360.0)
            w_b = (0.0, 360.0)

            # call run_ga (synchronous). Provide stop_flag and progress_callback.
            try:
                with st.spinner("Running GA — this may take time. Progress will update per generation."):
                    out_best_coes, out_best_val, out_history = run_ga(
                        np.array([[c["lat"], c["lon"]] for c in st.session_state.locations], dtype=float),
                        num_sats=int(n_sats),
                        pop_size=int(pop_size),
                        generations=int(generations),
                        mutation_rate=float(mutation_rate),
                        crossover_rate=float(crossover_rate),
                        duration_days=float(ga_duration_days),
                        dt=float(ga_dt),
                        seed=seed,
                        sma_bounds=sma_b,
                        e_bounds=e_b,
                        incl_bounds=incl_b,
                        ra_bounds=ra_b,
                        ta_bounds=ta_b,
                        w_bounds=w_b,
                        epoch=epoch,
                        gap_seconds=float(gap_seconds),
                        stop_flag=lambda: not st.session_state.ga_running,
                        progress_callback=progress_cb
                    )
                # done running
                st.session_state.ga_running = False
                st.session_state.ga_history = out_history if out_history is not None else []
                # ensure as numpy array
                best_coes_arr = np.asarray(out_best_coes, dtype=float)
                if best_coes_arr.ndim == 1:
                    # reshape to num_sats x 6 if possible
                    if best_coes_arr.size == int(n_sats) * 6:
                        best_coes_arr = best_coes_arr.reshape((int(n_sats), 6))
                    else:
                        # fallback reshape to (-1,6)
                        best_coes_arr = best_coes_arr.reshape((-1,6))
                st.session_state.ga_best = {"coes_ui": best_coes_arr, "val": float(out_best_val)}
                st.success(f"GA finished. Best avg revisit = {float(out_best_val):.2f} s ({float(out_best_val)/3600.0:.4f} hr)")

                # Final compute of passes using final dt/duration_days from top sidebar
                try:
                    locs_np = np.array([[c["lat"], c["lon"]] for c in st.session_state.locations], dtype=float)
                    with st.spinner("Computing passes for GA best solution (final dt/duration)..."):
                        res, err = compute_passes_safe(locs_np, best_coes_arr, epoch, dt, duration_days, gap_seconds)
                        if err:
                            st.warning(f"compute_passes on GA best failed:\n{err}")
                        else:
                            st.session_state.last_results = res
                            st.session_state.last_params = {"mode":"GA_final","epoch":epoch,"dt":dt,"duration_days":duration_days}
                            st.success("Computed passes for GA best solution.")
                except Exception as e:
                    st.warning(f"Error computing passes for GA best: {e}")

            except Exception as e:
                st.session_state.ga_running = False
                st.error(f"GA run failed: {e}\n{traceback.format_exc()}")

    if ga_stop:
        st.session_state.ga_running = False
        st.success("Stop requested. GA will stop between generations (if gaoptm polls stop_flag).")

# ---------------- Export CSV all passes ----------------
if export_all and st.session_state.last_results is not None:
    per_pair = st.session_state.last_results.get("per_pair", {})
    rows = []
    for (k, g), info in per_pair.items():
        df = info.get("passes")
        if df is None or df.empty: continue
        df2 = df.copy(); df2["sat_id"] = int(k)+1; df2["loc_id"] = int(g)+1
        rows.append(df2)
    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        st.download_button("Download all passes (CSV)", all_df.to_csv(index=False).encode("utf-8"), file_name="all_passes.csv", mime="text/csv")
    else:
        st.info("No pass data to export.")

# ---------------- Show GA best ----------------
st.markdown("---")
st.header("GA Result (if available)")
if st.session_state.ga_best is None:
    st.info("No GA result yet.")
else:
    best_arr = st.session_state.ga_best["coes_ui"]
    best_val = st.session_state.ga_best["val"]
    df_best = pd.DataFrame(best_arr, columns=["a","e","TA","RA","incl","w"])
    st.dataframe(df_best)
    st.metric("GA best avg revisit (s)", f"{best_val:.2f}", delta=f"{best_val/3600.0:.3f} hr")
    if st.button("➕ Add GA best to Manual satellites"):
        for row in best_arr:
            st.session_state.manual_sats.append({"a":float(row[0]),"e":float(row[1]),"TA":float(row[2]),"RA":float(row[3]),"incl":float(row[4]),"w":float(row[5])})
        st.success("Added GA-best satellites to Manual list.")
        safe_rerun()
    if st.session_state.ga_history:
        hist = np.array(st.session_state.ga_history, dtype=float)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(np.arange(len(hist)), hist/3600.0, marker='o', linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best avg revisit (hr)")
        ax.grid(alpha=0.2)
        st.pyplot(fig)

# ---------------- Pass results display ----------------
st.markdown("---")
st.header("Pass Results & Per-location Details")

if st.session_state.last_results is None:
    st.info("No pass computation results yet.")
else:
    results = st.session_state.last_results
    per_pair = results.get("per_pair", {})
    gs = results.get("global_stats", {})
    maxRev = gs.get("maxRev")
    meanRev = gs.get("meanRev")

    st.subheader("Global Revisit Stats")
    c1, c2 = st.columns(2)
    with c1:
        if maxRev is not None:
            st.markdown(f'<div class="metric-card"><h3>Max Revisit</h3><p class="small">{maxRev/3600:.2f} hr ({maxRev:.1f} s)</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-card"><h3>Max Revisit</h3><p class="small">N/A</p></div>', unsafe_allow_html=True)
    with c2:
        if meanRev is not None:
            st.markdown(f'<div class="metric-card"><h3>Mean Revisit</h3><p class="small">{meanRev/3600:.2f} hr ({meanRev:.1f} s)</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-card"><h3>Mean Revisit</h3><p class="small">N/A</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Per-location pass details")
    per_pair_keys = list(per_pair.keys()) if per_pair else []
    sat_idxs = set([k for (k,g) in per_pair_keys]) if per_pair_keys else set()
    N = (max(sat_idxs)+1) if sat_idxs else 0

    for g_idx, loc in enumerate(st.session_state.locations):
        st.markdown(f"### 📍 {loc['name']} — Lat {loc['lat']:.6f}, Lon {loc['lon']:.6f}")
        options = ["All satellites"] + [f"Satellite {i+1}" for i in range(N)]
        sel = st.selectbox(f"Choose satellite view for {loc['name']}", options, key=f"select_{g_idx}")

        if sel == "All satellites":
            rows = []
            timeline = []
            for (k,g), info in per_pair.items():
                if g != g_idx: continue
                df = info.get("passes")
                if df is not None and not df.empty:
                    df2 = df.copy(); df2["sat_id"] = int(k)+1
                    rows.append(df2)
                revisits = info.get("revisits")
                if revisits is not None and len(revisits) > 0 and info.get("passes") is not None:
                    for idx_row, stt in enumerate(info["passes"]["Start"]):
                        dur_s = info["passes"]["Duration [s]"].iat[idx_row]
                        timeline.append({"sat_id": int(k)+1, "Start": stt, "Duration_s": dur_s})
            if rows:
                agg_df = pd.concat(rows, ignore_index=True).sort_values("Start")
                st.dataframe(agg_df)
                st.download_button(f"Download passes for {loc['name']}", agg_df.to_csv(index=False).encode("utf-8"), file_name=f"passes_{loc['name']}.csv", mime="text/csv")
            else:
                st.info("No passes for this location.")
            if timeline:
                tdf = pd.DataFrame(timeline)
                fig, ax = plt.subplots(figsize=(8,2))
                ax.scatter(pd.to_datetime(tdf["Start"]), tdf["sat_id"], s=np.clip(tdf["Duration_s"]/10, 6, 100))
                ax.set_xlabel("Pass Start Time")
                ax.set_ylabel("Satellite ID")
                ax.grid(alpha=0.2)
                st.pyplot(fig)
        else:
            sat_idx = options.index(sel) - 1
            pair = per_pair.get((sat_idx, g_idx))
            if pair is None:
                st.info(f"No passes for Satellite {sat_idx+1} at this location.")
            else:
                df = pair.get("passes")
                if df is None or df.empty:
                    st.info(f"No passes for Satellite {sat_idx+1} at this location.")
                else:
                    st.dataframe(df)
                    st.download_button(f"Download passes (sat{sat_idx+1}_{loc['name']})", df.to_csv(index=False).encode("utf-8"), file_name=f"passes_sat{sat_idx+1}_{loc['name']}.csv", mime="text/csv")
                    revisits = pair.get("revisits", [])
                    if revisits is not None and len(revisits) > 0:
                        revisits = np.array(revisits, dtype=float)
                        st.write("Revisit stats:")
                        st.write(f"  Mean revisit = {revisits.mean():.2f} s ({revisits.mean()/3600:.2f} hr)")
                        st.write(f"  Max revisit = {revisits.max():.2f} s ({revisits.max()/3600:.2f} hr)")
                        fig, ax = plt.subplots(figsize=(6,3))
                        ax.hist(revisits, bins=20)
                        ax.set_xlabel("Revisit (s)")
                        ax.set_ylabel("Count")
                        st.pyplot(fig)
                    else:
                        st.info("No revisit intervals available.")

# footer
st.markdown("---")
