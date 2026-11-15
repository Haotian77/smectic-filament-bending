"""Bend modulus K per sample and persistence length versus radius, from a table of accepted filaments.

    python figures.py data/filaments.csv [results.csv]

data/filaments.csv is the table behind the manuscript figures. With results.csv (from persistence_length.py), its
R_F, l_p, CI and K replace the tabulated ones. Writes K_per_sample.csv and lp_vs_radius.pdf / .png.

A sample is one dopant at one loading x_add (pristine 12OCB is dopant "none"). For every sample the script gives two
estimates of K. K_mean is the mean of the K values of its filaments, each from K = 8 kB T l_p / (pi R_F^2). K_fit
comes from fitting l_p = pi K R_F^2 / (8 kB T) to all filaments of the sample at once, with K the only free
parameter. The exponent alpha of l_p ~ R_F^alpha is fitted separately to test the R_F^2 law.

Columns of K_per_sample.csv: n filaments; T_C median temperature; K_mean and K_ci95 (half-width of its 95 % CI, pN);
p_vs_pristine (Welch t test of the filament K values against pristine 12OCB); K_fit and K_fit_se (standard error, pN);
alpha with its 95 % CI alpha_lo, alpha_hi (given for samples of at least four filaments)."""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

KB = 1.380649e-23                                                                              # Boltzmann constant (J / K)
FAMILY = {"none": ("pristine 12OCB", "#2a78d6", "o"), "5CB": ("5CB", "#eb6834", "s"),         # name, colour, marker
          "6OCB": ("6OCB", "#1baf7a", "^"), "cholesterol": ("cholesterol", "#eda100", "v")}
SHADES = {"none": ["#2a78d6"], "5CB": ["#fdc1ac", "#fd8456", "#d95821", "#9a3501"],            # one per loading
          "6OCB": ["#66e7ae", "#2db983", "#0c7c55"], "cholesterol": ["#eda100", "#a86f00", "#634000"]}
DODGE = {"none": 0, "5CB": -0.35, "6OCB": 0.35, "cholesterol": 0}                              # x offset in panel d

t = pd.read_csv(sys.argv[1])
if len(sys.argv) > 2:               # recomputed values: keep the filaments accepted in results.csv and take its numbers
    r = pd.read_csv(sys.argv[2])
    r = r[r.accepted].set_index("filament")
    t = t[t.filament.isin(r.index)].copy()
    for c in ("R_um", "lp_um", "lp_lo", "lp_hi", "K_pN"):
        t[c] = r.loc[t.filament, c].values
# Fits are done on ln l_p. A 95 % CI from lp_lo to lp_hi corresponds to a standard deviation of ln l_p of
# ln(lp_hi / lp_lo) / 3.92 (the CI spans 2 x 1.96 standard deviations). It is kept above 0.02 so that one filament
# with a very narrow CI cannot dominate a fit.
t["w"] = 1 / np.maximum(np.log(t.lp_hi / t.lp_lo) / 3.92, 0.02) ** 2      # weight: inverse variance of ln l_p from its 95 % CI
t["y"] = np.log(t.lp_um / t.R_um ** 2)                                    # constant within a sample if l_p ~ R_F^2


def K_fit(T, y, w):
    """K (pN) from the weighted fit of l_p = pi K R_F^2 / (8 kB T).

    With the exponent fixed at 2 the fit reduces to the weighted mean of y = ln(l_p / R_F^2), and
    K = 8 kB T / pi * exp(mean). T is the median temperature of the sample; 1e18 converts J / um to pN."""
    return 8 * KB * (273.15 + np.median(T)) / np.pi * np.exp(np.average(y, weights=w)) * 1e18


def exponent(R, lp, w):
    """alpha of the weighted fit l_p ~ R_F^alpha."""
    return np.polyfit(np.log(R), np.log(lp), 1, w=np.sqrt(w))[0]        # polyfit takes 1 / standard deviation as weight


# K per sample: mean of the filaments with its 95 % CI, Welch test against pristine, fit of the R_F^2 law, free exponent
samples = [(f, x, d.reset_index(drop=True)) for f in FAMILY for x, d in t[t.dopant == f].groupby("x_add_pct")]
K0 = t.K_pN[t.dopant == "none"]
rng_K, rng_a = np.random.default_rng(0), np.random.default_rng(0)      # fixed seeds: same numbers in every run
rows = []
for f, x, d in samples:
    n = len(d)
    T, y, w, R, lp = (d[c].to_numpy() for c in ("T_C", "y", "w", "R_um", "lp_um"))
    # Uncertainty of the fits: the n filaments of the sample are drawn again with replacement 4000 times and the fit
    # is repeated. The standard deviation of K_fit over the draws is its standard error; the 2.5th and 97.5th
    # percentiles of alpha are its 95 % CI.
    boot_K = [K_fit(T[i], y[i], w[i]) for i in (rng_K.integers(0, n, n) for _ in range(4000))]     # resample filaments
    boot_a = [exponent(R[i], lp[i], w[i]) for i in (rng_a.integers(0, n, n) for _ in range(4000)) if len(np.unique(R[i])) > 1] if n >= 4 else [np.nan]
    rows.append(dict(dopant=f, x_add_pct=x, n=n, T_C=np.median(T), K_mean=d.K_pN.mean(), K_ci95=stats.t.ppf(0.975, n - 1) * d.K_pN.sem(),
                     p_vs_pristine=stats.ttest_ind(d.K_pN, K0, equal_var=False).pvalue if f != "none" else np.nan,
                     K_fit=K_fit(T, y, w), K_fit_se=np.std(boot_K, ddof=1),
                     alpha=exponent(R, lp, w), alpha_lo=np.percentile(boot_a, 2.5), alpha_hi=np.percentile(boot_a, 97.5)))
tab = pd.DataFrame(rows)
tab.to_csv("K_per_sample.csv", index=False)
print(tab.round(3).to_string(index=False))

# common exponent: weighted fit of ln l_p against ln R_F with one prefactor per sample
# Design matrix D: first column ln R_F (its coefficient is the exponent), then one 0/1 column per sample (its
# coefficient is the prefactor of that sample). The standard error is that of weighted linear least squares.
sample = t.dopant + t.x_add_pct.astype(str)
D = np.column_stack([np.log(t.R_um)] + [sample == s for s in sample.unique()]).astype(float)
w = t.w.to_numpy()
b = np.linalg.lstsq(D * np.sqrt(w)[:, None], np.log(t.lp_um) * np.sqrt(w), rcond=None)[0]
res = np.log(t.lp_um) - D @ b
se = np.sqrt((res ** 2 * w).sum() / (len(t) - D.shape[1]) * np.linalg.inv((D * w[:, None]).T @ D)[0, 0])
print(f"common exponent, one prefactor per sample: {b[0]:.2f} +/- {se:.2f}")

# a-c) l_p versus R_F for each dopant with the pristine filaments, lines: R_F^2 law with K_fit; d) K versus loading
fig, ax = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
for a, fam in zip(ax, ("5CB", "6OCB", "cholesterol")):
    for f, x, d in [s for s in samples if s[0] in ("none", fam)]:
        m = tab[tab.dopant == f].reset_index(drop=True)
        k = m.index[m.x_add_pct == x][0]                                    # position of this loading within its dopant
        a.errorbar(d.R_um, d.lp_um / 1e3, yerr=[(d.lp_um - d.lp_lo) / 1e3, (d.lp_hi - d.lp_um) / 1e3], fmt=FAMILY[f][2], color=SHADES[f][k],
                   mec="w", mew=0.5, ms=6, elinewidth=0.7, label=FAMILY[f][0] if f == "none" else f"{x:g} %")
        R = np.array([d.R_um.min() * 0.85, d.R_um.max() * 1.15])
        # l_p = pi K R_F^2 / (8 kB T); 1e-21 converts pN um^2 / J to mm
        a.plot(R, np.pi * m.K_fit[k] * R ** 2 / (8 * KB * (273.15 + d.T_C.median())) * 1e-21, "-", color=SHADES[f][k], lw=1)
    a.set(xscale="log", yscale="log", xlim=(0.55, 2.6), ylim=(0.25, 40), xlabel=r"$R_\mathrm{F}$ (µm)", ylabel=r"$l_\mathrm{p}$ (mm)")
    a.set_xticks([0.6, 1, 1.5, 2.5], ["0.6", "1", "1.5", "2.5"])
    a.set_xticks([], minor=True)
    a.set_yticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
    a.legend(title=f"{fam}, " + r"$x_\mathrm{add}$", fontsize=8, title_fontsize=8, loc="lower right", frameon=False)
for f, (name, colour, marker) in FAMILY.items():                           # panel d: K_mean with its 95 % CI
    m = tab[tab.dopant == f]
    ax[3].errorbar(m.x_add_pct + DODGE[f], m.K_mean, yerr=m.K_ci95, fmt=marker, color=colour, mec="w", mew=0.5, ms=8, capsize=3, label=name)
    if f != "none":
        ax[3].plot([0, *(m.x_add_pct + DODGE[f])], [tab.K_mean[0], *m.K_mean], "-", color=colour, lw=1)
ax[3].axhline(tab.K_mean[0], color=FAMILY["none"][1], ls=":", lw=1)        # dotted line: pristine 12OCB
ax[3].legend(fontsize=8, frameon=False, loc="lower right")
ax[3].set(xlabel=r"$x_\mathrm{add}$ (%)", ylabel=r"$K$ (pN)", xlim=(-1.5, 19), ylim=(0, 85))
for a, letter in zip(ax, "abcd"):
    a.set_title(f"{letter})", loc="left", fontweight="bold")
fig.savefig("lp_vs_radius.pdf")
fig.savefig("lp_vs_radius.png", dpi=200)
