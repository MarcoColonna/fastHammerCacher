# fastHammerCacher

A minimal, standalone example of how to make Hammer reweighting about 200× faster
when you ask for the same histogram at many form-factor (FF) / Wilson-coefficient (WC) points,
as a fit does. The only dependencies are the Hammer python bindings and numpy (no redist, pyhf or ROOT).

```
fastHammerCacher/
├── README.md
├── example/performance.py              # timing + agreement test
└── fastHammerCacher/
    ├── HammerCacher.py                 # vanilla cacher (standalone copy of redist's HammerCacher)
    └── fastHammerCacher.py             # FastHammerCacher, same interface
```

## The two cachers

**`HammerCacher`** loads a Hammer `.dat` file (run header plus histograms) and returns the
histogram for the given FFs and WCs. Whenever these change, it calls `Hammer.get_histogram`,
which takes about 0.1–1 s for a 3D histogram with O(10^5) events.

**`FastHammerCacher`** has the same constructor and methods. The trick is that Hammer's
amplitude is **linear** in the FF eigenvector variations `d` (linear `...Var` schemes such as `BGLVar`)
and in the WCs. Each histogram bin is a sum of |amplitude|², so it is **exactly quadratic**:

```
h_bin(d) = c + b·d + d·Q·d
```

With n FF parameters there are `1 + 2n + n(n-1)/2` unknowns per bin. They are fixed once
from that many Hammer calls (finite differences are exact for a quadratic). After that, every
histogram is a numpy `einsum`, which takes milliseconds instead of a second.
Floating WCs (for example Re C_VL) work the same way: `c, b, Q` are themselves quadratic in each WC,
so the expansion is done twice. A WC that changes without being declared is added automatically.

Because the polynomial is exact, this is **not an approximation**. If the amplitude is not linear in your FF
parameters (CLN with ρ² as a parameter, for example), it does not hold, and the agreement check in
`performance.py` will show that.

## Usage

```python
import sys; sys.path.insert(0, "/path/to/fastHammerCacher")
from fastHammerCacher.HammerCacher import HammerCacher
from fastHammerCacher.fastHammerCacher import FastHammerCacher

args = ("DstMu_BGLVar.dat", "mmiss2_q2_el",
        {"Process": "BtoD*", "SchemeVar": "BGLVar", "name": "SchemeBGL"}, "BtoCMuNu")
ffs = {"delta_a0": 0., "delta_a1": 0., ...}
wcs = {"SM": 1., "V_qLlL": 0., ...}

hc = FastHammerCacher(*args, ffs, wcs, 1.0, float_wcs=("V_qLlL",))   # or HammerCacher(...)
h = hc.getHisto(wcs_new, ffs_new)          # normalized to scaleFactor
h = hc.getHistoNoScale(wcs_new, ffs_new)   # raw bin contents
hc.save("coeffs.npz"); hc.load("coeffs.npz")   # optional: reuse the expansion
```

## Example

```bash
python example/performance.py                     # FFs only
python example/performance.py --float-wc V_qLlL   # FFs + Re(C_VL)
python example/performance.py --config cfg.json --mode B02DstMuNu   # redist-style json
```

Result on B0→D*μν, BGL (10 FF parameters), 5848-bin histogram (`DstMu_BGLVar_newnu.dat`):

| | setup | per point | Hammer calls in the loop |
|---|---|---|---|
| HammerCacher | 4.8 s | 957 ms | 1 per point |
| FastHammerCacher (FFs) | 65 s (66 calls) | 4.6 ms | 0 |
| FastHammerCacher (FFs + Re C_VL) | 189 s (198 calls) | 6.8 ms | 0 |

The maximum difference is |fast − vanilla| / max bin ≈ 1e-14 (machine precision).
The setup pays for itself after about 70 points (FFs only) or about 180 points (FFs + Re C_VL),
and a fit with MINOS needs 10³–10⁵.
