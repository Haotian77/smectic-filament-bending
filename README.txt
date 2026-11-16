Bending rigidity of smectic filaments: analysis code and data


track_filaments.py      movie -> centerline of every isolated filament (.npz) and a picture for inspection (.png)
persistence_length.py   centerlines -> persistence length l_p, 95 % CI, bend modulus K, accepted or rejected
figures.py              accepted filaments -> K per sample, exponents, l_p versus R_F figure
data/filaments.csv      the 89 accepted filament traces behind the manuscript figures
data/centerlines/       their tracked centerlines
data/videos/            three of the 56 movies they come from (20x, 0.17115 um per pixel, 10 frames per second):
                        63p2C_2.avi           pristine 12OCB, 63.2 C
                        5CB2pct_62C_2.avi     12OCB + 5CB, x_add = 6.67 %, 62.0 C
                        Chol1pct_62p9C_1.avi  12OCB + cholesterol, x_add = 3.33 %, 62.9 C
                        The movies are too large for the repository (0.6 to 0.7 GB each). Download them from
                        https://github.com/Haotian77/smectic-filament-bending/releases and put them in data/videos/.

License: code MIT (LICENSE), data CC BY 4.0 (LICENSE-DATA.txt).

Python 3.11 with numpy 2.2, scipy 1.16, pandas 2.3, matplotlib 3.10, opencv-python 4.12, scikit-image 0.25.

Manuscript values from the centerlines (about one minute):
    python persistence_length.py data/centerlines results.csv
    python figures.py data/filaments.csv
The first command recomputes l_p, K and the mode window of every filament; they equal those in data/filaments.csv,
and the confidence intervals agree within about 1 % (bootstrap). The second gives the numbers of the manuscript
from the table (pristine 12OCB: K = 48.9 +/- 1.6 pN, exponent 1.91). Add results.csv as a second argument to use
the recomputed values instead.

From a movie (5 to 20 minutes per movie on ten cores):
    python track_filaments.py data/videos/63p2C_2.avi 63.2 edge tracked
    python track_filaments.py data/videos/5CB2pct_62C_2.avi 62.0 edge tracked
    python track_filaments.py data/videos/Chol1pct_62p9C_1.avi 62.9 centroid tracked
    python persistence_length.py tracked tracked.csv
The arguments of track_filaments.py are the movie, its temperature in C, the centre estimator and the output
folder. The estimator used for each filament is in the column "centre" of data/filaments.csv. The centerlines of
the accepted filaments are identical to those in data/centerlines.

The last acceptance step is visual: a filament that passes persistence_length.py is kept only if its picture
shows a whole filament with two free ends. Traces that are segments of a longer filament, arcs of a loop or
have an attached end are excluded. In the sample movies this excludes 63p2C_2__f9.

How it works, in the order of the scripts (details and all thresholds are explained at the top of each script):
  1. track_filaments.py finds the straight, isolated filaments of a movie and, in every frame, locates the two
     edges of each filament with sub-pixel resolution along its length. The midline between the edges is the
     centerline, their distance the width (R_F is half the median width).
  2. persistence_length.py expands the tangent angle of each centerline in cosine modes. For a filament in thermal
     equilibrium the variance of mode n over the movie is L^2 / (l_p pi^2 n^2). The script selects the modes that
     follow this n^-2 law, fits l_p together with the localization noise, and gives K = 8 kB T l_p / (pi R_F^2).
     A filament is accepted only if it passes every criterion; the column "reason" of the output names the first
     one a rejected filament failed.
  3. figures.py combines the accepted filaments of each sample.

Your own movies: set the pixel size UM at the top of track_filaments.py. The frame rate is read from the movie file.
The filaments must be bright on a dark background, free at both ends and clear of other objects. The movie has to
last several relaxation times of the slowest bending mode used; persistence_length.py tests this and rejects or
shortens the mode window otherwise (5-minute movies were used here). Look at every .png before using a result.
Centerlines from another tracker can be analysed if they are saved in the same format (see the top of
track_filaments.py).

Names: <movie>__f<i> is the i-th candidate of a movie, longest first. In movie names, 5CB2pct means 2 wt % of
the whole mixture, i.e. x_add = 2 / 0.30 = 6.67 % of the liquid crystal. Suffixes: __c re-tracked with the
centroid estimator, __w<a>-<b> frames a to b only, __r candidate of a movie whose file is truncated.

Columns: T_C temperature; frames tracked frames; length_um mean tracked length; L_um analysed length; R_um radius;
window first-last mode of the thermal window; lp_um, lp_lo, lp_hi persistence length and its 95 % CI (um);
sigma_nm fitted noise level; K_pN, K_lo, K_hi bend modulus (pN); x_um, y_um mean position in the movie;
same_filament_as marks a second trace of a filament already in the table.
