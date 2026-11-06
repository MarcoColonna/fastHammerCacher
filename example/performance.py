"""Iterative reweighting with HammerCacher vs FastHammerCacher: timing and agreement.

Mimics what a fit does: ask the cacher for the histogram at many different
FF (and optionally WC) points, one after the other.

    python example/performance.py                         # default file, FFs only
    python example/performance.py --float-wc V_qLlL       # also vary Re(C_VL)
    python example/performance.py --config cfg.json --mode B02DstMuNu   # redist-style json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fastHammerCacher.HammerCacher import HammerCacher           # noqa: E402
from fastHammerCacher.fastHammerCacher import FastHammerCacher   # noqa: E402

# B0 -> D* mu nu, BGL (10 eigenvector variations), 3D histogram mmiss2 x q2 x E_mu
DEFAULT = {
    "fileNames": ["/ceph/users/mcolonna/Muonic_Side/VTX_reco/hammer/DstMu_BGLVar_newnu.dat"],
    "histoname": "mmiss2_q2_el",
    "ffscheme": {"Process": "BtoD*", "SchemeVar": "BGLVar", "name": "SchemeBGL"},
    "wcscheme": "BtoCMuNu",
    "formfactors": {k: 0.0 for k in ["delta_a0", "delta_a1", "delta_a2", "delta_b0", "delta_b1",
                                     "delta_b2", "delta_c1", "delta_c2", "delta_d0", "delta_d1"]},
    "wilsoncoefficients": {"SM": [1.0, 0.0], "S_qLlL": [0.0, 0.0], "S_qRlL": [0.0, 0.0],
                           "T_qLlL": [0.0, 0.0], "V_qLlL": [0.0, 0.0], "V_qRlL": [0.0, 0.0]},
    "scalefactor": 1.0,
}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--config", help="redist-style json with one entry per mode")
ap.add_argument("--mode", help="entry of --config to use")
ap.add_argument("--n-iter", type=int, default=20, help="number of reweighting points")
ap.add_argument("--sigma", type=float, default=1.0, help="spread of the random FF variations")
ap.add_argument("--float-wc", nargs="*", default=[], help="WCs whose real part is varied, e.g. V_qLlL")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

c = json.load(open(args.config))[args.mode] if args.config else DEFAULT
cacher_args = (c["fileNames"][0], c["histoname"], c["ffscheme"], c["wcscheme"])
ffs0 = dict(c["formfactors"])
wcs0 = {k: complex(*v) for k, v in c["wilsoncoefficients"].items()}

# the points of the "iterative reweighting"
rng = np.random.default_rng(args.seed)
points = []
for _ in range(args.n_iter):
    ffs = {k: v + rng.normal(0, args.sigma) for k, v in ffs0.items()}
    wcs = dict(wcs0)
    for k in args.float_wc:
        wcs[k] = wcs0[k] + rng.normal(0, 0.3)
    points.append((wcs, ffs))


def run(name, make):
    t0 = time.perf_counter()
    hc = make()
    t_init = time.perf_counter() - t0
    calls0 = hc.n_hammer_calls
    histos, times = [], []
    for wcs, ffs in points:
        t0 = time.perf_counter()
        histos.append(hc.getHisto(dict(wcs), dict(ffs)))
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    print(f"{name:18s} init {t_init:7.1f} s ({calls0} Hammer calls) | per point {1e3 * times.mean():9.3f} ms "
          f"| {len(points)} points {times.sum():7.2f} s ({hc.n_hammer_calls - calls0} Hammer calls)")
    return np.array(histos), t_init, times


print(f"file   : {cacher_args[0]}\nhisto  : {cacher_args[1]}, scheme {c['ffscheme']}\n"
      f"FFs    : {len(ffs0)}, floating WCs: {args.float_wc or 'none'}, points: {args.n_iter}\n")
h_van, init_van, t_van = run("HammerCacher", lambda: HammerCacher(*cacher_args, dict(ffs0), dict(wcs0), c["scalefactor"]))
h_fast, init_fast, t_fast = run("FastHammerCacher", lambda: FastHammerCacher(
    *cacher_args, dict(ffs0), dict(wcs0), c["scalefactor"], float_wcs=args.float_wc))

# agreement (relative to the largest bin, so empty bins do not blow up the ratio)
diff = np.abs(h_fast - h_van).max(axis=1) / h_van.max(axis=1)
print(f"\nmax |fast - vanilla| / max bin over all points: {diff.max():.1e}")

speedup = t_van.mean() / t_fast.mean()
breakeven = init_fast / (t_van.mean() - t_fast.mean())
print(f"speed-up per reweighting point: {speedup:,.0f}x")
print(f"setup pays off after ~{breakeven:.0f} points (a fit with MINOS easily needs 1e3-1e5)")
