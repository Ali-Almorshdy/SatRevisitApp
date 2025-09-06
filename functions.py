import numpy as np
from datetime import datetime, timedelta

# ----------------------------
# GMST
# ----------------------------
def gmst(mjd_ut1):
    Secs = 86400.0
    MJD_J2000 = 51544.5

    mjd_0 = np.floor(mjd_ut1)
    UT1   = Secs * (mjd_ut1 - mjd_0)
    T_0   = (mjd_0 - MJD_J2000) / 36525.0
    T     = (mjd_ut1 - MJD_J2000) / 36525.0

    gmst_sec = (24110.54841 +
                8640184.812866 * T_0 +
                1.002737909350795 * UT1 +
                (0.093104 - 6.2e-6 * T) * T * T)

    # fractional part
    gmstime = 2 * np.pi * ((gmst_sec / Secs) % 1.0)
    return gmstime


# ----------------------------
# Two-body dynamics
# ----------------------------
def dEOM(t, y, mu=398600.44189):
    rv = y[0:3]
    r = np.linalg.norm(rv)
    rdot = y[3:6]
    rddot = -mu * rv / r**3
    return np.concatenate((rdot, rddot))


# ----------------------------
# Split into passes
# ----------------------------
def sep(oo, gap_sec=20):
    """
    Split datetime list into passes separated by >= gap_sec seconds.
    """
    if len(oo) == 0:
        return [], np.empty((0, 2), dtype="datetime64[ns]"), np.array([])

    oo = sorted(oo)
    gaps = np.diff([dt.timestamp() for dt in oo])

    new_pass_idx = np.where(gaps >= gap_sec)[0]

    starts = np.concatenate(([0], new_pass_idx + 1))
    ends   = np.concatenate((new_pass_idx, [len(oo) - 1]))

    passes = [oo[s:e+1] for s, e in zip(starts, ends)]
    intervals = np.array([[oo[s], oo[e]] for s, e in zip(starts, ends)])
    durations = np.array([(oo[e] - oo[s]).total_seconds() for s, e in zip(starts, ends)])

    return passes, intervals, durations


# ----------------------------
# COE → RV
# ----------------------------
def COE2RV(a, e, TA, RA, incl, w, mu=398600.0):
    Pp = a * (1 - e*e)
    h = np.sqrt(Pp * mu)

    TA_rad = np.radians(TA)
    RA_rad = np.radians(RA)
    incl_rad = np.radians(incl)
    w_rad = np.radians(w)

    rp = (h**2/mu) * (1/(1 + e*np.cos(TA_rad))) * \
         (np.cos(TA_rad) * np.array([1, 0, 0]) +
          np.sin(TA_rad) * np.array([0, 1, 0]))
    vp = (mu/h) * (-np.sin(TA_rad) * np.array([1, 0, 0]) +
                   (e + np.cos(TA_rad)) * np.array([0, 1, 0]))

    # Rotation matrices
    R3_W = np.array([[ np.cos(RA_rad),  np.sin(RA_rad), 0],
                     [-np.sin(RA_rad),  np.cos(RA_rad), 0],
                     [0,                0,              1]])

    R1_i = np.array([[1, 0, 0],
                     [0, np.cos(incl_rad),  np.sin(incl_rad)],
                     [0,-np.sin(incl_rad),  np.cos(incl_rad)]])

    R3_w = np.array([[ np.cos(w_rad),  np.sin(w_rad), 0],
                     [-np.sin(w_rad),  np.cos(w_rad), 0],
                     [0,               0,             1]])

    Q_pX = (R3_w @ R1_i @ R3_W).T

    r = Q_pX @ rp
    v = Q_pX @ vp

    return r, v
