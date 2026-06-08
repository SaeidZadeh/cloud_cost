#!/usr/bin/env python3
###############################################################################
#  cloud_cost.py   (COMPLETE, runnable)
#
#  Python port of "Cloud Price.R" + the revision experiments.
#  Reproduces every figure in the paper with the same matplotlib style,
#  prints Table tab:ga and the Wilcoxon p-values, and writes scalability.png.
#
#  Run with:   python3 cloud_cost.py
#  Requires:   numpy, scipy, matplotlib   (pip install numpy scipy matplotlib)
#
#  Figures written to ./figures/ :
#     Job1allsettings.png/.pdf      Scenario-1 scatter (per-use) w/ markers+legend
#     BoundrySolution.png/.pdf      boundary solutions, alpha in [0,50]
#     Job1allsettings_pertime.png   per-time scatter
#     distorigin1/disttime1/distprice1.png   generalized, per-use   (3 panels)
#     distorigin2/disttime2/distprice2.png   generalized, per-time  (3 panels)
#     scalability.png               NEW runtime figure (replaces old fig:scal img)
#
#  Console output:
#     Scenario-1 optimal allocations; Table tab:ga numbers; B3 p-values; timings.
#
#  Notes / fixes carried over from the R revision:
#     * per-time Min-Price selector corrected
#     * per-time cost = sum(time * price) over all 7 tasks (no dropped terms)
#     * all normalizations computed before use
#     * Scenario-1 brute force fully vectorized (10^7 assignments)
###############################################################################

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
np.random.seed(1)                      # reproducibility
OUT = "C:/Users/saeid/Desktop/Leiria Work/cloud cost/figures"
os.makedirs(OUT, exist_ok=True)
def P(name): return os.path.join(OUT, name)

# global matplotlib style to match the paper figures
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 13,
    "axes.linewidth": 1.0,
    "legend.framealpha": 0.9,
})

# =============================================================================
# SHARED DATA
# =============================================================================
# Average task times (seconds): rows = 10 "node types", cols = 7 tasks.
TimeI   = [44.5, 22.25, 11.125, 5.5625, 2.78125, 1.390625,
           0.927083333, 0.6953125, 0.463541667, 0.34765625]
TimeII  = [447.5, 223.75, 111.875, 55.9375, 27.96875, 13.984375,
           9.322916667, 6.9921875, 4.661458333, 3.49609375]
TimeIII = [7.20E-02, 3.60E-02, 1.80E-02, 9.00E-03, 4.50E-03, 2.25E-03,
           1.50E-03, 1.13E-03, 7.50E-04, 5.63E-04]
TimeIV  = [2.00E-02, 1.00E-02, 5.00E-03, 2.50E-03, 1.25E-03, 6.25E-04,
           4.17E-04, 3.13E-04, 2.08E-04, 1.56E-04]
TimeVc  = [6.61E-03, 3.31E-03, 1.65E-03, 8.26E-04, 4.13E-04, 2.07E-04,
           1.38E-04, 1.03E-04, 6.89E-05, 5.16E-05]
TimeVI  = [2.10E-02, 1.05E-02, 5.25E-03, 2.63E-03, 1.31E-03, 6.56E-04,
           4.38E-04, 3.28E-04, 2.19E-04, 1.64E-04]
TimeVII = [1.09E-01, 5.45E-02, 2.73E-02, 1.36E-02, 6.81E-03, 3.41E-03,
           2.27E-03, 1.70E-03, 1.14E-03, 8.52E-04]
# TimeM[type, task]  (10 x 7)
TimeM = np.array([TimeI, TimeII, TimeIII, TimeIV, TimeVc, TimeVI, TimeVII]).T

Pricea = np.array([0.054, 0.052, 0.208, 0.63, 1.26, 2.52, 3.78, 3.456, 7.56, 10.08])
Priceg = np.array([0.045, 0.09, 0.2208, 0.6048, 1.2096, 2.4192, 3.6288, 2.88, 7.2576, 9.6768])


def job_time_columns(Tcols):
    """Two execution flows from the task graph: {1,2,3,6,7} and {1,2,4,5,6,7}.
    Tcols has shape (..., 7); returns the max of the two flow times (the makespan)."""
    t1 = Tcols[..., 0] + Tcols[..., 1] + Tcols[..., 2] + Tcols[..., 5] + Tcols[..., 6]
    t2 = Tcols[..., 0] + Tcols[..., 1] + Tcols[..., 3] + Tcols[..., 4] + Tcols[..., 5] + Tcols[..., 6]
    return np.maximum(t1, t2)


def all_combos(n_types):
    """All n_types^7 assignments as an (n_types^7, 7) int array of type indices."""
    grids = np.meshgrid(*[np.arange(n_types)] * 7, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


# =============================================================================
# PART A1 : Scenario 1 brute force (per-use) + scatter
# =============================================================================
print("Scenario 1: enumerating 10^7 assignments (vectorized)...")
G = all_combos(10)                                   # 10^7 x 7
Tcols = TimeM[G, np.arange(7)]                       # time of each task's chosen type
tt = job_time_columns(Tcols)                         # makespan per assignment
pa = Pricea[G].sum(axis=1)                           # Amazon total price
pg = Priceg[G].sum(axis=1)                           # Google total price
del Tcols

tt1 = (tt - tt.min()) / (tt.max() - tt.min())
pa1 = (pa - pa.min()) / (pa.max() - pa.min())
pg1 = (pg - pg.min()) / (pg.max() - pg.min())


def our_idx(alpha):
    return int(np.argmin(np.sqrt((alpha * tt1) ** 2 + pa1 ** 2)))

b_eq = our_idx(1.0)     # equal importance
b_pr = our_idx(0.5)     # price 2x more important
b_tm = our_idx(2.0)     # time  2x more important

print("\n--- Scenario 1 (Amazon per-use) optimal allocations ---")
for nm, b in [("equal", b_eq), ("price2x", b_pr), ("time2x", b_tm)]:
    print(f"{nm:8s} types={(G[b] + 1).tolist()}  time={tt[b]:.2f}s  price={pa[b]:.2f}")

# near-optimal reference sets (thresholds you tuned in the paper figure)
thr_eq, thr_pr, thr_tm = 0.06, 0.043, 0.074
L_eq = np.where(np.sqrt(tt1 ** 2 + pa1 ** 2) <= thr_eq)[0]
L_pr = np.where(np.sqrt((0.5 * tt1) ** 2 + pa1 ** 2) <= thr_pr)[0]
L_tm = np.where(np.sqrt((2.0 * tt1) ** 2 + pa1 ** 2) <= thr_tm)[0]


def plot_job1(fname):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    # all assignments: tiny black dots, restricted view [0,0.1]
    ax.scatter(pa1, tt1, s=7, c="black", marker=".", linewidths=0, label="All Assignments")
    ax.scatter(pa1[L_eq], tt1[L_eq], s=70, c="red",
               marker="x", linewidths=1.6, label="Near Optimal Assignments\n(Equal importance)")
    ax.scatter(pa1[b_eq], tt1[b_eq], s=180, c="blue", marker="P",
               label="Optimal Assignment\n(Equal importance)")
    ax.scatter(pa1[L_pr], tt1[L_pr], s=70, c="orange",
               marker="x", linewidths=1.6, label="Near Optimal Assignments\n(Price more importance)")
    ax.scatter(pa1[b_pr], tt1[b_pr], s=180, c="blue", marker="o",
               label="Optimal Assignment\n(Price more importance)")
    ax.scatter(pa1[L_tm], tt1[L_tm], s=70, c="green",
               marker="x", linewidths=1.6, label="Near Optimal Assignments\n(Time more importance)")
    ax.scatter(pa1[b_tm], tt1[b_tm], s=180, c="blue", marker="D",
               label="Optimal Assignment\n(Time more importance)")
    ax.set_xlim(0.0, 0.1)
    ax.set_ylim(0.0, 0.09)
    ax.set_xlabel("normalized total price of using cloud")
    ax.set_ylabel("normalized total time of using cloud")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=10, frameon=True, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

plot_job1(P("Job1allsettings.png"))
plot_job1(P("Job1allsettings.pdf"))

# =============================================================================
# PART A2 : Boundary solution, alpha in [0, 50] step 0.01
# =============================================================================
alphas = np.arange(0, 50.0001, 0.01)
# The min-distance assignment for ANY alpha must lie on the lower-left Pareto
# frontier of (price, time): both objectives are minimized, so a dominated
# point can never beat the point that dominates it for any positive weight.
# We reduce 10^7 -> the frontier (a handful of points), then sweep alpha on it.
_order = np.argsort(pa1)                       # by price ascending
_ts_sorted = tt1[_order]
_cummin = np.minimum.accumulate(_ts_sorted)    # running min of time
_front = _order[_ts_sorted <= _cummin]         # Pareto-optimal indices
_fp, _ft = pa1[_front], tt1[_front]
_D = np.sqrt((alphas[:, None] * _ft[None, :]) ** 2 + _fp[None, :] ** 2)
_sel = np.argmin(_D, axis=1)
bp = _fp[_sel]
btt = _ft[_sel]
boundary = np.unique(np.stack([bp, btt], axis=1), axis=0)


def plot_boundary(fname):
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.scatter(boundary[:, 0], boundary[:, 1], s=90, facecolors="none",
               edgecolors="black", marker="D", linewidths=1.4, label="Boundary Solution")
    ax.scatter(pa1[b_eq], tt1[b_eq], s=110, c="red", marker="x", linewidths=2,
               label="Solution for Equally Important Time and Price")
    ax.set_xlim(-0.01, 0.21)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("normalized total price of using cloud")
    ax.set_ylabel("normalized total time of using cloud")
    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

plot_boundary(P("BoundrySolution.png"))
plot_boundary(P("BoundrySolution.pdf"))

# =============================================================================
# PART A3 : per-time (price-per-second) small brute force (3^7)
# =============================================================================
TimeS = np.array([
    [0.445, 4.475, 7.2e-4, 2.0e-4, 6.61e-5, 2.1e-4, 1.09e-3],
    [0.153, 1.538, 4.1e-4, 7.74e-5, 1.94e-5, 1.3e-4, 4.01e-3],
    [0.047, 0.470, 1.5e-4, 3.46e-5, 9.96e-6, 4.75e-5, 2.7e-4],
])                                                   # 3 types x 7 tasks
PriceS = np.array([0.019, 0.029, 0.048])
GS = all_combos(3)                                   # 3^7 x 7
TcolS = TimeS[GS, np.arange(7)]
PcolS = PriceS[GS]
ttS = job_time_columns(TcolS)
pS = (TcolS * PcolS).sum(axis=1)                     # per-time cost = sum(time*price)
ttS1 = (ttS - ttS.min()) / (ttS.max() - ttS.min())
pS1 = (pS - pS.min()) / (pS.max() - pS.min())
bS = int(np.argmin(np.sqrt(ttS1 ** 2 + pS1 ** 2)))

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.scatter(pS1, ttS1, s=45, facecolors="none", edgecolors="black",
           marker="D", linewidths=0.8, alpha=0.6, label="all allocations")
ax.scatter(pS1[bS], ttS1[bS], s=200, c="blue", marker="P", label="ours (equal)")
ax.set_xlabel("normalized total price of using cloud")
ax.set_ylabel("normalized total time of using cloud")
ax.legend(loc="upper right", fontsize=12)
fig.tight_layout()
fig.savefig(P("Job1allsettings_pertime.png"))
plt.close(fig)
del TcolS, PcolS

# =============================================================================
# PART A4/A5 : generalized random architectures
# =============================================================================
def gen_arch(n):
    Time = np.zeros((n, 7))
    Time[0] = np.abs(np.random.normal(0, 2, 7))
    Price = np.zeros(n)
    Price[0] = abs(np.random.normal(0, 1))
    for j in range(1, n):
        Time[j] = abs(np.random.normal(0, 2)) * Time[0]
        Price[j] = abs(np.random.normal(0, 1)) * Price[0]
    Time = Time / Time[0, 0]
    Price = Price / Price[0]
    return Time, Price


def eval_assign(idx, Time, Price, mode):
    Tb = Time[idx, np.arange(7)]
    Pb = Price[idx]
    ti = job_time_columns(Tb)
    co = Pb.sum() if mode == "use" else (Pb * Tb).sum()
    return np.array([ti, co])


def select_methods(Time, Price, mode):
    hvs = np.empty(7, dtype=int)
    mt = np.empty(7, dtype=int)
    mp = np.empty(7, dtype=int)
    for k in range(7):
        if mode == "use":
            hvs[k] = np.argmax(1 / np.sqrt(Time[:, k] ** 2 + Price ** 2))
            mp[k] = np.argmin(Price)                      # cheapest
        else:
            hvs[k] = np.argmax(1 / np.sqrt(Time[:, k] ** 2 * (1 + Price ** 2)))
            mp[k] = np.argmin(Time[:, k] ** 2 * (Price ** 2))   # FIX vs original
        mt[k] = np.argmin(Time[:, k])                     # fastest
    return hvs, mt, mp


def run_generalized(mode="use", m=200, ns=range(2, 12)):
    ns = list(ns)
    keys = ["Our", "OurT", "OurP", "Ttime", "TtimeT", "TtimeP", "Pprice", "PpriceT", "PpriceP"]
    res = {k: np.zeros(len(ns)) for k in keys}
    res.update({k + "sd": np.zeros(len(ns)) for k in keys})
    for ni, n in enumerate(ns):
        vo = np.zeros((2, m)); voT = np.zeros((2, m)); voP = np.zeros((2, m))
        for i in range(m):
            Time, Price = gen_arch(n)
            h, t_, p_ = select_methods(Time, Price, mode)
            vo[:, i] = eval_assign(h, Time, Price, mode)
            voT[:, i] = eval_assign(t_, Time, Price, mode)
            voP[:, i] = eval_assign(p_, Time, Price, mode)
        d = lambda v: np.sqrt(v[0] ** 2 + v[1] ** 2)
        res["Our"][ni] = d(vo).mean();    res["Oursd"][ni] = d(vo).std()
        res["OurT"][ni] = vo[0].mean();   res["OurTsd"][ni] = vo[0].std()
        res["OurP"][ni] = vo[1].mean();   res["OurPsd"][ni] = vo[1].std()
        res["Ttime"][ni] = d(voT).mean(); res["Ttimesd"][ni] = d(voT).std()
        res["TtimeT"][ni] = voT[0].mean(); res["TtimeTsd"][ni] = voT[0].std()
        res["TtimeP"][ni] = voT[1].mean(); res["TtimePsd"][ni] = voT[1].std()
        res["Pprice"][ni] = d(voP).mean(); res["Ppricesd"][ni] = d(voP).std()
        res["PpriceT"][ni] = voP[0].mean(); res["PpriceTsd"][ni] = voP[0].std()
        res["PpriceP"][ni] = voP[1].mean(); res["PpricePsd"][ni] = voP[1].std()
    res["ns"] = np.array(ns)
    return res


def plot_triplet(fname, ns, our, oursd, tt_, ttsd, pp, ppsd, ylab, legloc):
    """Matches the paper: filled circles + lines + capped error bars (0.1*sd)."""
    f = 0.1
    fig, ax = plt.subplots(figsize=(5.3, 3.64))
    for y, ys, c in [(our, oursd, "black"), (tt_, ttsd, "red"), (pp, ppsd, "blue")]:
        ax.errorbar(ns, y, yerr=f * ys, fmt="o-", color=c, mfc=c, mec=c,
                    ms=9, lw=1.6, capsize=4, elinewidth=1.4, markeredgewidth=1.4)
    ax.set_xlabel("Types of cloud servers")
    ax.set_ylabel(ylab)
    ax.legend(["Ours", "Min-Time", "Min-Price"], loc=legloc, fontsize=12)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


print("\nrunning generalized architectures (per-use)...")
rU = run_generalized("use")
ns = rU["ns"]
plot_triplet(P("distorigin1.png"), ns, rU["Our"], rU["Oursd"], rU["Ttime"], rU["Ttimesd"],
             rU["Pprice"], rU["Ppricesd"], "Normalized distance to the origin", "upper right")
plot_triplet(P("disttime1.png"), ns, rU["OurT"], rU["OurTsd"], rU["TtimeT"], rU["TtimeTsd"],
             rU["PpriceT"], rU["PpriceTsd"], "Normalized time of using cloud", "upper right")
plot_triplet(P("distprice1.png"), ns, rU["OurP"], rU["OurPsd"], rU["TtimeP"], rU["TtimePsd"],
             rU["PpriceP"], rU["PpricePsd"], "Normalized price of using cloud", "lower left")

print("running generalized architectures (per-time)...")
rT = run_generalized("time")
plot_triplet(P("distorigin2.png"), ns, rT["Our"], rT["Oursd"], rT["Ttime"], rT["Ttimesd"],
             rT["Pprice"], rT["Ppricesd"], "Normalized distance to the origin", "upper right")
plot_triplet(P("disttime2.png"), ns, rT["OurT"], rT["OurTsd"], rT["TtimeT"], rT["TtimeTsd"],
             rT["PpriceT"], rT["PpriceTsd"], "Normalized time of using cloud", "upper right")
plot_triplet(P("distprice2.png"), ns, rT["OurP"], rT["OurPsd"], rT["TtimeP"], rT["TtimePsd"],
             rT["PpriceP"], rT["PpricePsd"], "Normalized price of using cloud", "upper right")

# =============================================================================
# PART B1 : NSGA-II baseline + hypervolume + timing  -> Table tab:ga
# =============================================================================
def nondominated_sort(F):
    """Fast non-dominated sorting for 2 objectives (minimization).
    Repeatedly peel off the lower-left staircase frontier; O(n log n) per front,
    far faster than the O(n^2) pairwise method at our population sizes."""
    n = len(F)
    remaining = list(range(n))
    fronts = []
    while remaining:
        idx = np.array(remaining)
        sub = F[idx]
        order = np.lexsort((sub[:, 1], sub[:, 0]))   # obj0 asc, tie obj1 asc
        s = sub[order]
        best = np.inf
        keep = np.zeros(len(s), bool)
        for i in range(len(s)):
            if s[i, 1] < best:          # strictly improves obj1 => non-dominated
                keep[i] = True
                best = s[i, 1]
        front = idx[order[keep]].tolist()
        fronts.append(front)
        fset = set(front)
        remaining = [r for r in remaining if r not in fset]
    return fronts


def crowding(F):
    n = len(F)
    d = np.zeros(n)
    for o in range(2):
        order = np.argsort(F[:, o])
        d[order[0]] = d[order[-1]] = np.inf
        rng = F[:, o].max() - F[:, o].min()
        if rng == 0:
            rng = 1
        for i in range(1, n - 1):
            d[order[i]] += (F[order[i + 1], o] - F[order[i - 1], o]) / rng
    return d


def nsga2(Time, Price, mode, pop=40, gens=60):
    n = Time.shape[0]
    Pp = [np.random.randint(0, n, 7) for _ in range(pop)]
    evalP = lambda L: np.array([eval_assign(g, Time, Price, mode) for g in L])
    for _ in range(gens):
        kids = []
        for _ in range(pop):
            a = Pp[np.random.randint(pop)]
            b = Pp[np.random.randint(pop)]
            mask = np.random.random(7) < 0.5
            child = np.where(mask, a, b).copy()
            if np.random.random() < 0.3:
                child[np.random.randint(7)] = np.random.randint(n)
            kids.append(child)
        R = Pp + kids
        F = evalP(R)
        fronts = nondominated_sort(F)
        newP = []
        fi = 0
        while fi < len(fronts) and len(newP) + len(fronts[fi]) <= pop:
            newP.extend(fronts[fi])
            fi += 1
        if len(newP) < pop and fi < len(fronts):
            fr = fronts[fi]
            cd = crowding(F[fr])
            order = np.argsort(-cd)
            newP.extend([fr[i] for i in order[:pop - len(newP)]])
        Pp = [R[i] for i in newP]
    F = evalP(Pp)
    t = (F[:, 0] - F[:, 0].min()) / (np.ptp(F[:, 0]) + 1e-9)
    p = (F[:, 1] - F[:, 1].min()) / (np.ptp(F[:, 1]) + 1e-9)
    return F[np.argmin(np.sqrt(t ** 2 + p ** 2))]


def hv_point(pt, ref):
    return max(0, ref[0] - pt[0]) * max(0, ref[1] - pt[1])


print("\n=== B1: HVS vs NSGA-II (per-use, 200 architectures of 11 types) ===")
m, n = 200, 11
o_hvs = np.zeros((m, 2)); o_ga = np.zeros((m, 2))
o_mt = np.zeros((m, 2)); o_mp = np.zeros((m, 2))
t_hvs = t_ga = 0.0
for i in range(m):
    Time, Price = gen_arch(n)
    t0 = time.perf_counter()
    h, t_, p_ = select_methods(Time, Price, "use")
    o_hvs[i] = eval_assign(h, Time, Price, "use")
    t_hvs += time.perf_counter() - t0
    o_mt[i] = eval_assign(t_, Time, Price, "use")
    o_mp[i] = eval_assign(p_, Time, Price, "use")
    t0 = time.perf_counter()
    o_ga[i] = nsga2(Time, Price, "use")
    t_ga += time.perf_counter() - t0

d = lambda O: np.sqrt((O ** 2).sum(axis=1))
d_hvs, d_ga, d_mt, d_mp = d(o_hvs), d(o_ga), d(o_mt), d(o_mp)
ref = 1.1 * np.array([max(o_hvs[:, 0].max(), o_ga[:, 0].max(), o_mt[:, 0].max(), o_mp[:, 0].max()),
                      max(o_hvs[:, 1].max(), o_ga[:, 1].max(), o_mt[:, 1].max(), o_mp[:, 1].max())])
H = lambda O: np.mean([hv_point(o, ref) for o in O])

print("\n--- Table tab:ga (paste these numbers) ---")
print(f"{'HVS':9s}  dist={d_hvs.mean():.3f} +/- {d_hvs.std():.3f}   HV={H(o_hvs):.3f}   time={1000*t_hvs/m:.3f} ms")
print(f"{'NSGA-II':9s}  dist={d_ga.mean():.3f} +/- {d_ga.std():.3f}   HV={H(o_ga):.3f}   time={1000*t_ga/m:.3f} ms")
print(f"{'Min-Time':9s}  dist={d_mt.mean():.3f} +/- {d_mt.std():.3f}   HV={H(o_mt):.3f}")
print(f"{'Min-Price':9s}  dist={d_mp.mean():.3f} +/- {d_mp.std():.3f}   HV={H(o_mp):.3f}")
print(f"medians  HVS={np.median(d_hvs):.3f}  NSGA={np.median(d_ga):.3f}  "
      f"MinT={np.median(d_mt):.3f}  MinP={np.median(d_mp):.3f}")

# =============================================================================
# PART B3 : Wilcoxon signed-rank tests (paired)
# =============================================================================
print("\n=== B3: Wilcoxon signed-rank tests on distance-to-origin ===")
print("HVS vs NSGA-II :  p =", wilcoxon(d_hvs, d_ga).pvalue)
print("HVS vs Min-Time:  p =", wilcoxon(d_hvs, d_mt).pvalue)
print("HVS vs Min-Price: p =", wilcoxon(d_hvs, d_mp).pvalue)

# =============================================================================
# PART B2 : scalability runtime vs |T| and |V|  -> scalability.png
# =============================================================================
print("\n=== B2: scalability (runtime of HVS, ms) ===")
Ts = list(np.power(2,range(1,20)))
Vs = [10, 100, 500]
M = np.zeros((len(Ts), len(Vs)))
for ci, nv in enumerate(Vs):
    for ri, mt_ in enumerate(Ts):
        Time = np.abs(np.random.normal(1, 0.5, (nv, mt_))) + 0.01
        Price = np.abs(np.random.normal(1, 0.5, nv)) + 0.01
        t0 = time.perf_counter()
        _ = np.argmax(1 / np.sqrt(Time ** 2 + Price[:, None] ** 2), axis=0)
        M[ri, ci] = 1000 * (time.perf_counter() - t0) + 1e-3
        print(f"  |T|={mt_:5d} |V|={nv:4d} -> {M[ri, ci]:8.3f} ms")

fig, ax = plt.subplots(figsize=(5.3, 3.64))
mk = ["o", "s", "^"]
cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for ci, nv in enumerate(Vs):
    ax.plot(Ts, M[:, ci], marker=mk[ci], color=cols[ci], lw=1.8, ms=8, label=f"|V| = {nv}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of tasks |T|")
ax.set_ylabel("Runtime (ms)")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="upper left", fontsize=12)
fig.tight_layout()
fig.savefig(P("scalability.png"))
plt.close(fig)

print(f"\nDone. All figures are in ./{OUT}/")
print("Paste the tab:ga numbers and the B3 p-values into the paper as instructed.")
