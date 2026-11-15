"""Persistence length l_p and bend modulus K of smectic filaments from the thermal fluctuations of their centerlines.

    python persistence_length.py data/centerlines results.csv

Input: a folder of .npz centerline files (or a single file) written by track_filaments.py. Each file holds, for every
tracked frame, the points x_um, y_um along the filament and the local width_um (npts points per frame), plus the
frame rate fps and the temperature T_C.
Output: one line per filament on screen and results.csv with the columns described in README.txt. "accepted" says
whether the filament passed every criterion, "reason" names the first criterion it failed.

Method (Gittes, Mickey, Nettleton and Howard, J. Cell Biol. 120, 923 (1993)):
  1. The tangent angle theta(s) of the centerline is expanded in cosine modes,
         theta(s) = sqrt(2 / L) * sum_n a_n cos(n pi s / L),        s = arc length, L = contour length.
  2. In thermal equilibrium every mode holds kB T / 2 of bending energy, so over the movie
         Var(a_n) = L^2 / (l_p pi^2 n^2)                             (thermal_variance).
     Independent localization errors sigma of the centerline points add
         (4 sigma^2 / L) [1 + (N - 1) sin^2(n pi / 2N)]              (noise_variance, N = number of contour segments),
     which grows with n. The lowest modes relax slowly and can be undersampled or carry a slow change of shape; the
     highest modes are dominated by noise.
  3. A window of consecutive modes that follow the n^-2 law is selected and tested (scan, widest, step 1 of analyze).
  4. l_p and sigma are fitted together to Var(a_n) from the first window mode up to mode 12 (fit_lp_sigma). The 95 % CI
     is the spread of that fit over 300 bootstrap resamples of the frames.
  5. K = 8 kB T l_p / (pi R_F^2), with R_F half the median width of the filament.
Lengths are in um unless a name says otherwise."""
import sys, glob, os, re
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import curve_fit, least_squares

KB = 1.380649e-23                   # Boltzmann constant (J / K)

# Fourier analysis
DS, TRIM = 0.25, 1.0                # contour resampling step and trim at each end (um)
NMODES, NMAX = 20, 12               # modes computed; highest mode scanned and fitted

# A window of modes is thermal if, for its modes: the log-log slope of Var(a_n) against n is -2 +/- SLOPE_TOL with
# R2 >= MIN_R2; n^2 Var(a_n), which is constant for thermal modes, has a robust coefficient of variation <= MAX_CV;
# and no two amplitudes are correlated by more than MAX_CORR (thermal modes are independent).
SLOPE_TOL, MIN_R2, MAX_CV, MAX_CORR = 0.45, 0.90, 0.30, 0.55        # thermal-window criteria

# Window fit: R2 of the slope -2 line through the window, smallest l_p (um) taken as a filament result, and largest
# half-width of a 95 % CI relative to the value (applied to the window fit and to the reported fit).
MIN_FIT_R2, MIN_LP_UM, MAX_REL_CI = 0.8, 500.0, 0.25                # window fit; CI half-width / value

# Slow modes: mode 1 or 2 may lie at most this factor above the plateau of n^2 Var(a_n); the movie must last at least
# this many relaxation times of the lowest mode used.
MAX_LOW_MODE_EXCESS, MIN_TAU_SAMPLED = 1.6, 5

# Largest localization noise (um) and filament radius (um) accepted.
MAX_SIGMA_UM, MAX_R_UM = 0.25, 2.3

# Shape stability over the movie, see shape_stability: spread of the bow of the filament, largest move of its centroid
# in 25 frames (um), relative drift of its length.
MAX_SAG_SPREAD, MAX_JUMP_UM, MAX_LENGTH_DRIFT = 0.05, 10.0, 0.6
SAME_FILAMENT_UM = 5.0              # two traces of one movie closer than this are the same filament


def load(path):
    """Centerline file -> list of x arrays and list of y arrays (one per frame), widths, frame rate, temperature.
    The file stores all frames end to end; npts gives the number of points of each frame."""
    d = np.load(path)
    cut = np.cumsum(d["npts"])[:-1]
    return np.split(d["x_um"].astype(float), cut), np.split(d["y_um"].astype(float), cut), d["width_um"], float(d["fps"]), float(d["T_C"])


def arclength(x, y):
    """Arc length from the first point to each point of a centerline."""
    return np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]


def mode_amplitudes(xs, ys):
    """Cosine-mode amplitudes a_n(t) of the tangent angle (Gittes et al. 1993) on a common contour length L -> a, L, N.

    The tracked length varies a little from frame to frame, and the mode amplitudes must all refer to the same L. L is
    the 5th percentile of the frame lengths after trimming TRIM at each end, so 95 % of the frames are long enough;
    the shorter ones are left out. Every kept frame is resampled every DS from s = TRIM to TRIM + L.
    a has one row per kept frame and one column per mode n = 1..NMODES; N is the number of contour segments."""
    s = [arclength(x, y) for x, y in zip(xs, ys)]
    L = np.percentile([q[-1] - 2 * TRIM for q in s], 5)
    sc = np.arange(0, L + DS / 2, DS)
    keep = [i for i, q in enumerate(s) if q[-1] >= TRIM + L]
    X = np.array([np.interp(sc + TRIM, s[i], xs[i]) for i in keep])
    Y = np.array([np.interp(sc + TRIM, s[i], ys[i]) for i in keep])
    # tangent angle along the contour, made continuous along s (no 2 pi jumps)
    theta = np.unwrap(np.arctan2(np.gradient(Y, sc, axis=1), np.gradient(X, sc, axis=1)), axis=1)
    # a_n = sqrt(2 / L) * integral of theta(s) cos(n pi s / L) ds. A constant angle integrates to zero for n >= 1,
    # so the orientation of the filament and its rotation during the movie do not enter.
    w = np.full(len(sc), DS)
    w[[0, -1]] = DS / 2             # trapezoid rule
    a = np.sqrt(2 / L) * (theta * w) @ np.cos(np.outer(sc, np.arange(1, NMODES + 1)) * np.pi / L)
    return a, L, len(sc) - 1


def scan(var, corr, first=1):
    """Every window of 3-7 consecutive modes within modes first..12 -> (first, last, slope, passes thermal criteria, score).

    var[n - 1] is the variance of a_n over the frames and corr the correlation matrix of the amplitudes. The score
    (lower is better) ranks the windows when none passes all criteria; it is only needed for the bootstrap resamples."""
    out = []
    for a in range(first, NMAX - 1):
        for b in range(a + 2, min(a + 6, NMAX) + 1):
            n = np.arange(a, b + 1)
            v = var[n - 1]
            x, y = np.log10(n), np.log10(v)
            slope, icpt = np.polyfit(x, y, 1)
            r2 = 1 - np.sum((y - icpt - slope * x) ** 2) / np.sum((y - y.mean()) ** 2)
            c = n ** 2 * v          # flat for thermal modes
            cv = 1.4826 * np.median(np.abs(c - np.median(c))) / np.median(c)        # robust: 1.4826 x MAD / median
            cmax = np.abs(corr[np.ix_(n - 1, n - 1)][~np.eye(len(n), dtype=bool)]).max()    # largest off-diagonal |r|
            passes = abs(slope + 2) <= SLOPE_TOL and r2 >= MIN_R2 and cv <= MAX_CV and cmax <= MAX_CORR
            score = 3 * abs(slope + 2) + 1.5 * max(0, 1 - r2) + 2 * cv + cmax + 0.15 * (a - first)
            out.append((a, b, slope, passes, score))
    return out


def widest(windows):
    """Widest passing window (lowest first mode on ties); if none passes, the window with the lowest score."""
    ok = [w for w in windows if w[3]]
    return min(ok, key=lambda w: (w[0] - w[1], w[0])) if ok else min(windows, key=lambda w: (w[4], w[0] - w[1], w[0]))


def plateau_lp(var, win, L):
    """l_p from the median of n^2 Var(a_n) over the window, and R2 of that slope -2 line in log space.

    This window fit ignores noise. It is used to test the window (R2, smallest l_p, bootstrap CI) and as the starting
    value of the reported fit, not as the result."""
    n = np.arange(win[0], win[1] + 1)
    v = var[n - 1]
    lp = L ** 2 / (np.pi ** 2 * np.median(n ** 2 * v))
    y = np.log10(v)
    res = y - np.log10(L ** 2 / (np.pi ** 2 * lp * n ** 2))
    return lp, 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)


def thermal_variance(n, lp, L):
    """Variance of mode n of a filament of persistence length lp in thermal equilibrium."""
    return L ** 2 / (np.pi ** 2 * lp * n ** 2)


def noise_variance(n, sigma, L, N):
    """Variance added to mode n by independent localization errors sigma at the N + 1 contour points (Gittes et al. 1993)."""
    return 4 * sigma ** 2 / L * (1 + (N - 1) * np.sin(n * np.pi / (2 * N)) ** 2)


def screening_sigma(var, L, N):
    """Noise level (um) from a least-squares fit of thermal + noise variance to modes 1-12.

    Used only for the acceptance criterion sigma <= MAX_SIGMA_UM. The fit is in linear space over all twelve modes,
    whatever the window, so the same test applies to every filament. Starting value of sigma: 0.1 pixel."""
    n = np.arange(1, NMAX + 1.0)
    v = var[:NMAX]
    model = lambda n, lp, sigma: thermal_variance(n, lp, L) + noise_variance(n, sigma, L, N)
    p0 = [np.median(L ** 2 / (np.pi ** 2 * n[:3] ** 2 * v[:3])), 0.017115]
    try:
        return curve_fit(model, n, v, p0=p0, bounds=([1e-6, 0], [1e15, 10]), maxfev=50000)[0][1]
    except Exception:
        return 0.0


def fit_lp_sigma(var, first, L, N, p0):
    """Reported fit: ln Var(a_n) = ln[thermal + noise variance] on modes first..12 -> l_p, sigma (um).

    Fitting in log space weights every mode equally although the variances span orders of magnitude. Including the
    noise term lets the modes above the window, where noise is no longer negligible, constrain sigma, so l_p is not
    biased low by noise. p0 is the starting (l_p, sigma); l_p is kept within 1 um - 10 m and sigma within 0.1 nm - 1 um."""
    n = np.arange(first, NMAX + 1.0)
    y = np.log(var[first - 1:NMAX])
    res = lambda p: np.log(thermal_variance(n, np.exp(p[0]), L) + noise_variance(n, np.exp(p[1]), L, N)) - y
    return np.exp(least_squares(res, np.log(p0), bounds=(np.log([1, 1e-4]), np.log([1e7, 1]))).x)


def relaxation_time(a, dt):
    """Exponential fit to the autocorrelation of a mode amplitude.

    a is the amplitude in successive frames, dt the frame interval; returns the relaxation time in the units of dt.
    The autocorrelation is computed up to a lag of one third of the record. If it has already decayed below 0.2 after
    one frame, the mode is faster than the frame rate (0.5 dt is returned). If it is still above 0.5 at the largest lag,
    the time follows from that last value. Otherwise a line is fitted to ln(autocorrelation) where it is above 0.15."""
    x = a - a.mean()
    m = max(4, len(x) // 3)
    acf = np.correlate(x, x, "full")[len(x) - 1:len(x) - 1 + m] / (len(x) - np.arange(m)) / x.var()
    if acf[1] < 0.2:
        return 0.5 * dt
    if acf[-1] > 0.5:
        return m * dt / -np.log(acf[-1])
    use = np.where(acf > 0.15)[0]
    if len(use) < 3:
        return dt
    slope = np.polyfit(use[1:] * dt, np.log(acf[use[1:]]), 1)[0]
    return -1 / slope if slope < 0 else np.nan


def shape_stability(xs, ys):
    """Spread of the chord deviation / length, largest 25-frame move of the smoothed centroid (um), relative drift of
    the smoothed length, and the mean length (um).

    These detect traces that are not one filament fluctuating about a fixed shape:
    - the bow of a frame is the largest distance of the centerline from the straight line through its two ends,
      divided by its length; its spread (95th - 5th percentile over the frames) is large when the filament bends or
      straightens during the movie;
    - a jump of the centroid (51-frame running median, displacement over 25 frames) means the trace moved to another
      object or the filament was carried away;
    - a drift of the length (101-frame running median, maximum - minimum over the median length) means the filament
      grew or shrank, or the trace lost part of it."""
    length = pd.Series([arclength(x, y)[-1] for x, y in zip(xs, ys)])
    sag = []
    for x, y, l in zip(xs, ys, length):
        cx, cy = x[-1] - x[0], y[-1] - y[0]
        sag.append(np.abs((x - x[0]) * cy - (y - y[0]) * cx).max() / np.hypot(cx, cy) / l)
    centroid = pd.DataFrame({"x": [x.mean() for x in xs], "y": [y.mean() for y in ys]})
    move = centroid.rolling(51, center=True, min_periods=10).median().diff(25)
    smooth = length.rolling(101, center=True, min_periods=20).median()
    return (np.percentile(sag, 95) - np.percentile(sag, 5), np.hypot(move.x, move.y).max(),
            (smooth.max() - smooth.min()) / length.median(), length.mean())


def analyze(path, n_boot=300):
    """Full analysis of one centerline file -> dictionary with one entry per column of results.csv.

    Step 1 selects and tests the window of thermal modes; a filament that fails here is returned without l_p.
    Step 2 computes the reported l_p, sigma and K. Step 3 applies the remaining acceptance criteria; a filament that
    fails there keeps its values in the table but is marked as not accepted."""
    xs, ys, width, fps, T = load(path)
    a, L, N = mode_amplitudes(xs, ys)
    var, corr = a.var(axis=0, ddof=1), np.corrcoef(a.T)         # over the frames: variance of each mode, correlations
    R = float(np.nanmedian(width)) / 2                          # R_F: half the median width over all points and frames
    sag, jump, drift, length = shape_stability(xs, ys)
    out = dict(filament=os.path.basename(path)[:-4], accepted=False, reason="", T_C=T, frames=len(xs), length_um=length,
               L_um=L, R_um=R, x_um=np.mean([x.mean() for x in xs]), y_um=np.mean([y.mean() for y in ys]))

    # 1. window of thermal modes
    win = widest(scan(var, corr))
    if not win[3]:
        return {**out, "reason": "no thermal window"}

    # Sampling rule. The variance of a mode is only meaningful if the movie covers many of its relaxation times.
    # Bending modes relax as tau_n = tau_1 / n^4, with tau_1 measured on mode 1, so the movie (frames / fps seconds)
    # covers at least MIN_TAU_SAMPLED relaxation times of every mode n >= n_min = (MIN_TAU_SAMPLED tau_1 / movie)^(1/4).
    tau1 = relaxation_time(a[:, 0], 1 / fps)
    n_min = max(1, int(np.ceil((MIN_TAU_SAMPLED * tau1 * fps / len(xs)) ** 0.25))) if np.isfinite(tau1) else 1
    sampled = win[0] >= n_min       # every window mode recorded for at least 5 of its relaxation times tau1 / n^4
    if not sampled:                 # else start at the first well-sampled mode, on the window with slope closest to -2
        ok = [w for w in scan(var, corr) if w[3] and w[0] >= n_min]
        if not ok:
            return {**out, "reason": "slow modes undersampled"}
        win = min(ok, key=lambda w: (abs(w[2] + 2), w[0] - w[1]))

    # Bootstrap: the frames are drawn again with replacement n_boot times. The fixed seed makes the confidence
    # intervals identical from run to run.
    rng = np.random.default_rng(12345)
    boot = []                       # bootstrap: variance and correlation of the modes for resampled frames
    for _ in range(n_boot):
        b = a[rng.integers(0, len(a), len(a))]
        boot.append((b.var(axis=0, ddof=1), np.corrcoef(b.T)))

    # Tests of the window fit. The loop runs again only if the window is moved up by the low-mode test at its end.
    first = 1
    while True:
        lp_w, r2 = plateau_lp(var, win, L)
        if r2 < MIN_FIT_R2:
            return {**out, "reason": "window fit R2 below 0.8"}
        if lp_w < MIN_LP_UM:
            return {**out, "reason": "l_p below 0.5 mm"}
        # 95 % CI of the window fit. The window is selected again in every resample, so the CI includes the
        # uncertainty of the window itself; a window set by the sampling rule is kept fixed.
        lo, hi = np.percentile([plateau_lp(v, widest(scan(v, c, first)) if sampled else win, L)[0] for v, c in boot], [2.5, 97.5])
        if (hi - lo) / (2 * lp_w) > MAX_REL_CI:
            return {**out, "reason": "window fit CI above 25 %"}
        # Low-mode test. A slow change of shape adds variance to modes 1 and 2 and would lower l_p. If a window
        # starts at mode 1 or 2 and one of them lies more than MAX_LOW_MODE_EXCESS above the plateau, the search is
        # repeated from the next mode up.
        n = np.arange(win[0], win[1] + 1)
        c = n ** 2 * var[n - 1]
        if win[0] > 2 or c[n <= 2].max() <= MAX_LOW_MODE_EXCESS * np.median(c):
            break
        first = win[0] + 1          # modes 1-2 above the plateau: raise the lowest mode by one
        win = widest(scan(var, corr, first))
        if not sampled or not win[3]:
            return {**out, "reason": "excess variance in modes 1-2"}

    # 2. reported fit from the first window mode to mode 12, its 95 % CI, and K = 8 kB T l_p / (pi R^2)
    lp, sigma = fit_lp_sigma(var, win[0], L, N, (lp_w, 0.05))
    lo, hi = np.percentile([fit_lp_sigma(v, win[0], L, N, (lp, sigma))[0] for v, _ in boot], [2.5, 97.5])
    K = 8 * KB * (T + 273.15) / (np.pi * R ** 2) * 1e18             # pN per um of l_p
    out.update(window=f"{win[0]}-{win[1]}", lp_um=lp, lp_lo=lo, lp_hi=hi, sigma_nm=1e3 * sigma, K_pN=K * lp, K_lo=K * lo, K_hi=K * hi)

    # 3. remaining acceptance criteria
    if screening_sigma(var, L, N) > MAX_SIGMA_UM:
        out["reason"] = "noise above 250 nm"
    elif R > MAX_R_UM:
        out["reason"] = "R_F above 2.3 um"
    elif (hi - lo) / (2 * lp) > MAX_REL_CI:
        out["reason"] = "CI above 25 %"
    elif sag > MAX_SAG_SPREAD or jump > MAX_JUMP_UM or drift > MAX_LENGTH_DRIFT:
        out["reason"] = "shape not stable"
    out["accepted"] = out["reason"] == ""
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results.csv"
    files = glob.glob(os.path.join(src, "*.npz")) if os.path.isdir(src) else [src]
    files.sort(key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(f))])    # f2 before f10
    with Pool() as pool:            # one filament per processor core
        rows = pool.map(analyze, files)
    # The detector can return the same filament twice in one movie. Two accepted traces of the same movie whose mean
    # positions are closer than SAME_FILAMENT_UM are one filament.
    movie = lambda r: r["filament"].split("__")[0]
    kept = []                       # a filament traced twice counts once: keep the trace whose window starts lowest
    for r in sorted((r for r in rows if r["accepted"]), key=lambda r: (movie(r), int(r["window"].split("-")[0]))):
        same = [q["filament"] for q in kept
                if movie(q) == movie(r) and np.hypot(q["x_um"] - r["x_um"], q["y_um"] - r["y_um"]) < SAME_FILAMENT_UM]
        if same:
            r.update(accepted=False, reason=f"same filament as {same[0]}")
        else:
            kept.append(r)
    for r in rows:
        result = f"l_p {r['lp_um'] / 1e3:6.2f} mm  K {r['K_pN']:5.1f} pN  modes {r['window']}" if r["accepted"] else f"rejected: {r['reason']}"
        print(f"{r['filament']:34s} {result}")
    pd.DataFrame(rows).to_csv(dst, index=False)
