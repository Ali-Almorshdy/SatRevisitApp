# put this in functions.py (or a new module) so you can "from functions import compute_passes"
from typing import Any, Dict, List, Tuple
from functions import gmst, sep, dEOM, COE2RV   # your existing helpers
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.integrate import solve_ivp
from astropy.time import Time

def compute_passes(
    locations: np.ndarray,
    coes: np.ndarray,
    epoch: datetime | None = None,
    dt: float = 10.0,
    duration_days: float = 1.0,
    gap_seconds: float = 20.0,
) -> Dict[str, Any]:
    """
    Compute passes and revisit times for a list of ground locations and satellites (COEs).

    Parameters
    ----------
    locations : array-like (M,2)
        Each row = [lat_deg, lon_deg].
    coes : array-like (N,6)
        Each row = [a, e, i, RAAN, w, TA] (degrees where applicable, km for a).
    epoch : datetime (UTC) or None
        Reference epoch (start). If None, defaults to 2025-09-02 09:00:00 UTC.
    dt : float
        Propagation timestep in seconds.
    duration_days : float
        Number of days to analyze (propagation length).
    gap_seconds : float
        Gap threshold (seconds) passed to `sep` to separate contiguous pass intervals.

    Returns
    -------
    dict with keys:
      - "PassDetails": M x N list-of-lists where entry [m][n] is a pandas.DataFrame with columns
                       ["Start","End","Duration [s]"] or None if no passes.
      - "per_pair": dict keyed by (sat_index, loc_index) 0-based -> {"passes": DataFrame, "revisits": np.array, "min_rev": float, "max_rev": float}
      - "allRevisitTimes": np.ndarray of all revisit intervals (s) across everything
      - "global_stats": {"maxRev": float|None, "meanRev": float|None}
      - "tdt": numpy array of datetime objects for all time samples
      - "tf": last datetime
    """
    if epoch is None:
        epoch = datetime(2025, 9, 2, 9, 0, 0)

    locs = np.asarray(locations, dtype=float)
    coes = np.asarray(coes, dtype=float)

    if locs.ndim != 2 or locs.shape[1] != 2:
        raise ValueError("locations must be shape (M,2) with [lat, lon] per row.")
    if coes.ndim != 2 or coes.shape[1] != 6:
        raise ValueError("coes must be shape (N,6): [a,e,i,RAAN,w,TA].")

    M = locs.shape[0]
    N = coes.shape[0]

    # time grid
    times = np.arange(0.0, duration_days * 24.0 * 3600.0 + dt, dt)  # seconds from epoch
    t0 = Time(epoch).jd
    t0_array = t0 + times / 86400.0
    tdt = Time(t0_array, format="jd").to_datetime()
    tf = tdt[-1]

    # gmst expects (jd - 2400000.5) in your code pattern; keep same
    theta = gmst(t0_array - 2400000.5)  # assumed to return radians array (len = len(times))

    Re = 6378.0  # km

    PassDetails: List[List[Any]] = [[None for _ in range(N)] for _ in range(M)]
    per_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}
    allRevisitTimes: List[float] = []

    # propagate per satellite
    for k in range(N):
        r0, v0 = COE2RV(*coes[k])      # user-provided function
        X0 = np.hstack((r0, v0))

        sol = solve_ivp(lambda t, y: dEOM(t, y), [times[0], times[-1]], X0,
                        t_eval=times, rtol=1e-8, atol=1e-8)
        if not sol.success:
            raise RuntimeError(f"Propagation failed for satellite {k}: {sol.message}")
        State = sol.y.T    # shape (T,6)
        r_sat = State[:, :3]    # (T,3)
        state_norm = np.linalg.norm(r_sat, axis=1)   # (T,)

        # half-cone angle delt (clip to [-1,1] to avoid domain errors)
        ratio = Re / state_norm
        ratio = np.clip(ratio, -1.0, 1.0)
        delt = np.degrees(np.arccos(ratio))   # (T,)

        # loop locations
        for g in range(M):
            lat = float(locs[g, 0])
            lon = float(locs[g, 1])

            # ground point ECI at each sample (theta in radians)
            thi = np.deg2rad(lon) + np.asarray(theta)   # (T,)
            coslat = np.cos(np.deg2rad(lat))
            sinlat = np.sin(np.deg2rad(lat))

            RR_x = Re * np.cos(thi) * coslat
            RR_y = Re * np.sin(thi) * coslat
            RR_z = Re * sinlat * np.ones_like(thi)
            RR = np.vstack((RR_x, RR_y, RR_z)).T   # (T,3)

            RR_norm = np.linalg.norm(RR, axis=1)
            dotVals = np.sum(RR * r_sat, axis=1)
            denom = RR_norm * state_norm

            with np.errstate(divide='ignore', invalid='ignore'):
                cos_angle = dotVals / denom
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            value = np.degrees(np.arccos(cos_angle))   # (T,)

            mask = value <= delt
            if not np.any(mask):
                # no visibility at any sample
                continue

            # times when visible (as datetimes). Reverse to match your original behavior
            visible_times = np.array(tdt)[mask][::-1]
            if visible_times.size == 0:
                continue

            # use your sep() to group contiguous times into intervals (expects datetimes)
            sep_result = sep(visible_times, gap_seconds)
            # sep must return intervals at index 1 like in your MATLAB code: (_, intervals, _)
            if sep_result is None:
                continue
            # try to unpack defensively
            try:
                _, intervals, _ = sep_result
            except Exception:
                # assume sep returned only intervals
                intervals = sep_result

            if intervals is None or len(intervals) == 0:
                continue

            # normalize intervals to numpy array of datetime objects with shape (K,2)
            intervals = np.asarray(intervals)
            if intervals.ndim == 1 and intervals.dtype == object:
                # e.g. list of (start,end) tuples -> stack
                intervals = np.vstack(intervals)

            starts = intervals[:, 0]
            ends = intervals[:, 1]

            # durations in seconds
            durations = np.array([(e - s).total_seconds() for s, e in zip(starts, ends)], dtype=float)

            # store DataFrame
            df = pd.DataFrame({"Start": starts, "End": ends, "Duration [s]": durations})
            PassDetails[g][k] = df

            # revisit times: from start of first pass to epoch, between passes, and from last pass end to tf
            revisits = []
            revisits.append((starts[0] - tdt[0]).total_seconds())
            for idx in range(1, len(starts)):
                revisits.append((starts[idx] - ends[idx - 1]).total_seconds())
            revisits.append((tf - ends[-1]).total_seconds())
            revisits = np.array(revisits, dtype=float)

            allRevisitTimes.extend(revisits.tolist())

            per_pair[(k, g)] = {
                "passes": df,
                "revisits": revisits,
                "min_rev": float(np.min(revisits)),
                "max_rev": float(np.max(revisits))
            }

    if len(allRevisitTimes) > 0:
        allRevisitTimes = np.array(allRevisitTimes, dtype=float)
        maxRev = float(np.max(allRevisitTimes))
        meanRev = float(np.mean(allRevisitTimes))
    else:
        allRevisitTimes = np.array([], dtype=float)
        maxRev = None
        meanRev = None

    return {
        "PassDetails": PassDetails,
        "per_pair": per_pair,
        "allRevisitTimes": allRevisitTimes,
        "global_stats": {"maxRev": maxRev, "meanRev": meanRev},
        "tdt": tdt,
        "tf": tf
    }
