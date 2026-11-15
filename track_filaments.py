"""Find isolated filaments in a polarized-optical-microscopy movie and track their centerlines.

    python track_filaments.py data/videos/63p2C_2.avi 63.2 edge tracked

Arguments: movie, temperature (C), centre estimator (edge: midpoint of the two half-maximum edges; centroid: intensity
centroid between them), output folder. Writes <movie>__f<i>.npz (read by persistence_length.py) and <movie>__f<i>.png
(first, middle and last frame with the trace, for visual inspection) for every candidate filament.

The filaments are bright on a dark background. The script works in two stages:
  1. Detection (find_filaments, detect). Five frames spread over the movie are searched for straight, elongated
     objects that touch nothing else. An object found in at least three of them is a candidate, and the rectangle that
     contains it in all of them, plus a margin, is its region of interest.
  2. Tracking (track, envelope, trace), one candidate per processor core. In every frame the filament is segmented
     inside its region, intensity profiles are cut across it at every pixel along its length, and the two edges of
     each profile are located at half maximum with sub-pixel resolution. The centre of each profile gives one point
     of the centerline and the distance between the edges gives the local width.
Nothing is smoothed or filtered in time: every frame is traced on its own.

Centerline file: frame (index of each tracked frame), npts (number of points in each), x_um, y_um and width_um (all
frames end to end, in um, origin at the top-left corner of the image), fps (from the movie file), T_C.

For another microscope set UM below. The other lengths are in um, or in pixels where the name or comment says so."""
import sys, os
from multiprocessing import Pool
import cv2, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, distance_transform_edt
from skimage import measure, morphology

UM = 0.17115                                    # um per pixel

# Segmentation: a pixel belongs to an object if it is brighter than the local background by K_SIGMA times the
# background noise. Objects smaller than MIN_AREA are dropped and gaps up to about CLOSING pixels are bridged.
K_SIGMA, MIN_AREA, CLOSING = 6.0, 200, 5        # threshold (x background noise), smallest object (px), closing radius (px)

# Detection. A candidate is 15-250 um long, at least 4 times longer than wide, nearly straight (end-to-end distance
# >= 0.93 of its contour length) and not buckled (largest deviation from a straight line <= 0.20 of its length).
# detection: 5 probe frames, object seen in at least 3 of them
MIN_LEN_UM, MAX_LEN_UM, MIN_ELONGATION, MIN_STRAIGHT, MAX_BUCKLE = 15.0, 250.0, 4.0, 0.93, 0.20
# Its tips are at least 4 um and its body at least 1.5 um from any other object. PAD (px) is the margin around the
# filament in its region of interest, and the distance it must keep from the image border. An object within MATCH_PX
# of a candidate of the previous probe frame is the same filament. At most MAX_FILAMENTS candidates are tracked.
TIP_GAP_UM, BODY_GAP_UM, PAD, MATCH_PX, MAX_FILAMENTS = 4.0, 1.5, 40, 250.0, 20

# Tracking. HALF_WIDTH and DV: half length and sampling step of the transverse profiles (px). EDGE_LEVEL: edges are
# where the profile crosses this fraction of its height above the background. END_TRIM_UM: cut from each end, where
# the rounded tip has no two edges. A frame needs MIN_SLICES valid profiles and a filament MIN_FRAMES traced frames.
# tracking: transverse profiles of half width 28 px sampled every 0.25 px, edges at half maximum, 2 um trimmed at each end
HALF_WIDTH, DV, EDGE_LEVEL, END_TRIM_UM, MIN_SLICES, MIN_FRAMES = 28.0, 0.25, 0.5, 2.0, 40, 200


def flatten(img, blur):
    """Maximum over the colour channels minus a blurred copy; robust noise of the background.

    The colour of a filament between crossed polarizers depends on its thickness, so the brightest channel is used.
    Subtracting a copy blurred over `blur` pixels removes uneven illumination. The noise is 1.4826 x the median
    absolute deviation of the darkest 70 % of the pixels, which are background."""
    f = img.astype(np.float32).max(axis=2)
    d = f - gaussian_filter(f, blur)
    low = d[d < np.percentile(d, 70)]
    return d, 1.4826 * np.median(np.abs(low - np.median(low))) + 1e-9


def objects(d, noise):
    """Binary image of everything brighter than the threshold (flattened image d, its background noise)."""
    mask = morphology.remove_small_objects(d > K_SIGMA * noise, MIN_AREA)
    return morphology.binary_closing(mask, morphology.disk(CLOSING))


def axes(mask):
    """Centroid, long axis and centred pixel coordinates of a mask.

    The long axis is the direction of largest variance of the pixel positions. Coordinates are (row, column)."""
    rc = np.column_stack(np.nonzero(mask)).astype(float)
    c = rc.mean(axis=0)
    val, vec = np.linalg.eigh(np.cov((rc - c).T))
    t = vec[:, np.argmax(val)]
    return c, t / np.linalg.norm(t), rc - c


def detect(frame):
    """Straight, elongated objects whose tips and body are clear of every other object, in one frame."""
    labels = measure.label(objects(*flatten(frame, 61)))
    h, w = labels.shape
    found = []
    for p in measure.regionprops(labels):
        # size and aspect ratio, from the ellipse with the same second moments as the object
        if p.major_axis_length / max(p.minor_axis_length, 1) < MIN_ELONGATION or not MIN_LEN_UM <= p.major_axis_length * UM <= MAX_LEN_UM:
            continue
        # midline: position across the long axis (v) as a function of position along it (u), in bins of about 4 px
        comp = labels == p.label
        _, t, q = axes(comp)
        u, v = q @ t, q @ np.array([-t[1], t[0]])                # along and across the long axis
        nb = max(12, int(np.ptp(u) / 4))                         # midline in bins along the axis
        b = np.clip(np.digitize(u, np.linspace(u.min(), u.max(), nb + 1)) - 1, 0, nb - 1)
        full = [i for i in range(nb) if (b == i).sum() >= 3]
        uu, vv = np.array([u[b == i].mean() for i in full]), np.array([np.median(v[b == i]) for i in full])
        if len(uu) < 6:
            continue
        # contour length of a cubic through the midline, then straightness and buckling
        g = np.linspace(uu[0], uu[-1], 200)
        ds = np.sqrt(1 + np.polyval(np.polyder(np.polyfit(uu, vv, min(3, len(uu) - 2))), g) ** 2)
        length = np.sum((ds[1:] + ds[:-1]) / 2 * np.diff(g))     # contour length (px)
        bend = np.abs(vv - np.polyval(np.polyfit(uu, vv, 1), uu)).max()
        if (uu[-1] - uu[0]) / length < MIN_STRAIGHT or bend / length > MAX_BUCKLE:
            continue
        # room for the margin inside the image
        r0, c0, r1, c1 = p.bbox
        if r0 - PAD < 0 or c0 - PAD < 0 or r1 + PAD > h or c1 + PAD > w:
            continue
        # isolation: distance from every pixel of the object, and from its two tips, to the nearest other object
        others = (labels > 0) & ~comp
        if others.any():
            gap = distance_transform_edt(~others) * UM
            tips = np.column_stack(np.nonzero(comp))[[np.argmin(u), np.argmax(u)]]
            if gap[comp].min() < BODY_GAP_UM or gap[tips[:, 0], tips[:, 1]].min() < TIP_GAP_UM:
                continue
        found.append(dict(centroid=np.array(p.centroid), bbox=p.bbox, length=length * UM))
    return found


def find_filaments(video):
    """Regions of interest (x, y, w, h) of the candidate filaments, longest first.

    Requiring an object in three of five frames spread over the movie keeps the filaments that stay isolated and in
    the field of view for most of the movie."""
    cap = cv2.VideoCapture(video)
    n = 0
    while cap.grab():                                            # frames that can be read (a truncated file holds
        n += 1                                                   # fewer than its header says)
    tracks = []                                                  # one list of detections per candidate
    for k in np.unique(np.linspace(0, max(0, n - 1), 5).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, frame = cap.read()
        if not ok:
            continue
        found = detect(frame)
        used = set()
        for tr in tracks:                                        # link each track to the nearest unused object
            dist = [np.linalg.norm(c["centroid"] - tr[-1]["centroid"]) if i not in used else np.inf for i, c in enumerate(found)]
            if dist and min(dist) <= MATCH_PX:
                tr.append(found[int(np.argmin(dist))])
                used.add(int(np.argmin(dist)))
        tracks += [[c] for i, c in enumerate(found) if i not in used]       # the others start new tracks
    cap.release()
    tracks = sorted([tr for tr in tracks if len(tr) >= 3], key=lambda tr: -np.mean([c["length"] for c in tr]))[:MAX_FILAMENTS]
    rois = []
    for tr in tracks:                                            # rectangle around all detections of the track, plus PAD
        r0, c0, r1, c1 = (f([c["bbox"][i] for c in tr]) for i, f in enumerate((min, min, max, max)))
        rois.append((int(c0 - PAD), int(r0 - PAD), int(c1 - c0 + 2 * PAD), int(r1 - r0 + 2 * PAD)))
    return rois


def envelope(d, noise, prior):
    """Silhouette of the filament: the largest elongated object, preferring the one found in the previous frame.

    d is the flattened region of interest and prior the (dilated) silhouette of the previous traced frame. Objects are
    ranked by area x aspect ratio; overlap with prior adds to the rank, so the trace stays on the same filament when
    something else drifts into the region. Returns None if the region holds no object with aspect ratio >= 2."""
    labels = measure.label(morphology.remove_small_holes(objects(d, noise), area_threshold=5000))
    best, score = None, -np.inf
    for p in measure.regionprops(labels):
        e = p.major_axis_length / max(p.minor_axis_length, 1.0)
        s = p.area * e
        if prior is not None:
            s += 5.0 * np.sum((labels == p.label) & prior) * p.major_axis_length
        if e >= 2 and s > score:
            best, score = p.label, s
    return None if best is None else labels == best


def trace(d, mask, center):
    """Sub-pixel centerline (row, col) and width (px) from transverse intensity profiles cut every pixel along the filament.

    d is the flattened region of interest, mask the silhouette from envelope. Returns None if the frame cannot be
    traced (fewer than MIN_SLICES valid profiles, or a trace shorter than 10 um)."""
    # slices: one every pixel along the long axis of the silhouette (u), each sampled across it (v)
    c, t, q = axes(mask)
    t = -t if t[0] < 0 else t                                    # same direction in every frame
    nrm = np.array([-t[1], t[0]])
    trim = END_TRIM_UM / UM
    u = np.arange((q @ t).min() + trim, (q @ t).max() - trim, 1.0)
    v = np.arange(-HALF_WIDTH, HALF_WIDTH + DV / 2, DV)
    if len(u) < MIN_SLICES:
        return None
    base = c + u[:, None] * t                                    # foot point and direction of every slice
    normal = np.tile(nrm, (len(u), 1))
    rows = np.arange(len(u))
    # Three passes. The first cuts the slices perpendicular to the straight long axis. After the first and the second,
    # a 5th-order polynomial through the centres gives the curved axis, and the slices are moved onto it and turned
    # perpendicular to it, so the last pass measures the filament at right angles to its own contour.
    for it in range(3):
        coords = [(base[:, :1] + v * normal[:, :1]).ravel(), (base[:, 1:] + v * normal[:, 1:]).ravel()]
        prof = map_coordinates(d, coords, order=1, mode="nearest").reshape(len(u), len(v))      # bilinear interpolation
        sm = gaussian_filter1d(prof, 1.0 / DV, axis=1)           # profiles smoothed over 1 px
        floor = np.median(np.c_[sm[:, :8], sm[:, -8:]], axis=1)  # background: the outer 2 px on both sides
        peak = sm.max(axis=1)
        level = floor + EDGE_LEVEL * (peak - floor)              # half maximum
        above = sm >= level[:, None]
        i0 = np.argmax(above, axis=1)                            # outermost crossings of the half-maximum level
        i1 = len(v) - 1 - np.argmax(above[:, ::-1], axis=1)
        # a slice is valid if both edges lie inside the profile and the filament stands out from the background
        good = above.any(axis=1) & (i0 > 0) & (i1 < len(v) - 1) & (peak - floor > 5)
        # position of the crossing, by linear interpolation between the sample outside and the sample inside
        cross = lambda out, inn: v[out] + (level - sm[rows, out]) * (v[inn] - v[out]) / (sm[rows, inn] - sm[rows, out] + 1e-12)
        left = np.where(good, cross(np.clip(i0 - 1, 0, None), i0), np.nan)
        right = np.where(good, cross(np.clip(i1 + 1, None, len(v) - 1), i1), np.nan)
        if center == "edge":                                     # midpoint of the two edges
            mid = (left + right) / 2
        else:                                                    # centroid of the intensity between the two edges
            wgt = np.clip(prof, 0, None) * ((v >= left[:, None]) & (v <= right[:, None]))
            with np.errstate(invalid="ignore"):
                mid = (wgt * v).sum(axis=1) / wgt.sum(axis=1)
        ok = np.isfinite(mid)
        if ok.sum() < MIN_SLICES:
            return None
        pts = base + np.where(ok, mid, 0)[:, None] * normal
        if it < 2:                                               # re-lay the slices perpendicular to the local tangent
            co = np.polyfit(u[ok], mid[ok], 5)
            base = base + np.polyval(co, u)[:, None] * normal
            tan = t + np.polyval(np.polyder(co), u)[:, None] * nrm
            tan /= np.linalg.norm(tan, axis=1, keepdims=True)
            normal = np.c_[-tan[:, 1], tan[:, 0]]
    pts = pts[ok]
    if np.hypot(*np.diff(pts, axis=0).T).sum() * UM < 10:
        return None
    return pts, (right - left)[ok]                               # width: distance between the two edges


def track(video, roi, center, T, out):
    """Trace one candidate (region of interest roi) through the whole movie and write out.npz and out.png.

    Frames in which the filament cannot be traced are skipped; their indices are missing from "frame" in the file.
    Returns a line of text for the screen."""
    x, y, w, h = roi
    cap = cv2.VideoCapture(video)
    prior = None
    frames, pts, wid = [], [], []
    for k in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok, frame = cap.read()
        if not ok:
            break
        d, noise = flatten(frame[y:y + h, x:x + w], 41)
        mask = envelope(d, noise, prior)
        tr = trace(d, mask, center) if mask is not None else None
        if tr is None:
            continue
        prior = morphology.binary_dilation(mask, morphology.disk(8))
        frames.append(k)
        pts.append((tr[0] + [y, x]) * UM)                        # (row, col) in the full frame, um
        wid.append(tr[1] * UM)
    name = os.path.basename(out)
    if len(frames) < MIN_FRAMES:
        return f"{name}: tracked in {len(frames)} frames only, skipped"
    xy = np.concatenate(pts)
    np.savez_compressed(out + ".npz", frame=np.array(frames, np.int32), npts=np.array([len(p) for p in pts], np.int32),
                        x_um=xy[:, 1].astype(np.float32), y_um=xy[:, 0].astype(np.float32),
                        width_um=np.concatenate(wid).astype(np.float32), fps=cap.get(cv2.CAP_PROP_FPS), T_C=T)
    # picture for the visual inspection: first, middle and last traced frame with the centerline in red
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    x0, y0 = max(x - 130, 0), max(y - 130, 0)                    # 130 px of surroundings around the region
    for a, i in zip(ax, (0, len(frames) // 2, len(frames) - 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        img = cap.read()[1][y0:y + h + 130, x0:x + w + 130, ::-1]
        a.imshow(np.clip(img * (200.0 / img.max()), 0, 255).astype(np.uint8))
        a.plot(pts[i][:, 1] / UM - x0, pts[i][:, 0] / UM - y0, "r-", lw=0.8)
        a.set_title(f"frame {frames[i]}")
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=110)
    plt.close(fig)
    cap.release()
    length = np.median([np.hypot(*np.diff(p, axis=0).T).sum() for p in pts])
    return f"{name}: {len(frames)} frames, length {length:.1f} um, R_F {np.median(np.concatenate(wid)) / 2:.2f} um"


if __name__ == "__main__":
    video, T, center, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    os.makedirs(out, exist_ok=True)
    rois = find_filaments(video)
    print(len(rois), "candidate filaments", flush=True)
    stem = os.path.join(out, os.path.splitext(os.path.basename(video))[0])
    with Pool() as pool:                                         # one candidate per processor core
        for line in pool.starmap(track, [(video, roi, center, T, f"{stem}__f{i}") for i, roi in enumerate(rois)]):
            print(line)
