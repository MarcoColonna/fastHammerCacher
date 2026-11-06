"""FastHammerCacher: same interface as HammerCacher, without calling Hammer at every step.

Idea
----
Hammer's amplitude is linear in the FF eigenvector variations d (for the linear
"...Var" schemes, e.g. BGLVar) and in the Wilson coefficients. The histogram is
|amplitude|^2 summed over events, so every bin is *exactly* a quadratic polynomial

    h(d) = c + b.d + d.Q.d          (c: scalar, b: vector, Q: matrix, per bin)

With n FF parameters this has 1 + 2n + n(n-1)/2 unknowns per bin. We get them
once from that many Hammer calls (finite differences are exact for a quadratic),
and afterwards every histogram is a numpy einsum: microseconds instead of ~1 s.

Floating WCs (e.g. Re(V_qLlL)) are handled the same way: c, b, Q are themselves
quadratic in each WC direction t, so the coefficients are expanded twice
(quadratic in t, of quadratic in d). A WC that changes without being declared in
`float_wcs` is added automatically (costs one re-expansion).

This is exact, not an approximation, as long as the amplitude is linear in the
FF parameters. Check it on your scheme with example/performance.py.
"""
import itertools
import time

import numpy as np

from .HammerCacher import HammerCacher


def quad_fit(f, n):
    """Exact coefficients (c, b, Q) of f(x) = c + b.x + x.Q.x, x in R^n, f array-valued.
    Uses f at 0, +-e_i and e_i+e_j: 1 + 2n + n(n-1)/2 evaluations."""
    f0 = np.asarray(f(np.zeros(n)), dtype=float)
    b = np.zeros(f0.shape + (n,))
    Q = np.zeros(f0.shape + (n, n))
    e = np.eye(n)
    for i in range(n):
        fp, fm = f(e[i]), f(-e[i])
        b[..., i] = (fp - fm) / 2
        Q[..., i, i] = (fp + fm) / 2 - f0
    for i, j in itertools.combinations(range(n), 2):
        fij = f(e[i] + e[j])
        Q[..., i, j] = Q[..., j, i] = (fij - f0 - b[..., i] - b[..., j] - Q[..., i, i] - Q[..., j, j]) / 2
    return f0, b, Q


def quad_eval(c, b, Q, x):
    """c + b.x + x.Q.x, broadcast over the leading (bin) axes."""
    return c + b @ x + np.einsum("...ij,i,j->...", Q, x, x)


def n_calls(n):
    return 1 + 2 * n + n * (n - 1) // 2


class FastHammerCacher(HammerCacher):
    """Drop-in replacement for HammerCacher.

    Same constructor arguments, plus
    float_wcs : WC names whose real part will vary, e.g. ("V_qLlL",).
                Optional: anything else that changes is picked up automatically.
    The WCs/FFs given to the constructor are the expansion point.
    """

    def __init__(self, *args, float_wcs=(), verbose=False, **kwargs):
        super().__init__(*args, verbose=verbose, **kwargs)
        self._verbose = verbose
        self._ff_names = list(self._FFs)
        self._ff_state = {k: float(v) for k, v in self._FFs.items()}
        self._wc_base = {k: complex(v) for k, v in self._wcs.items()}
        self._dirs = [(k, "re") for k in float_wcs]   # WC expansion directions
        self._coeffs = None
        self.setup_time = 0.0
        self._build()

    # --- expansion -------------------------------------------------------
    def _hammer(self, t, d):
        """Exact Hammer histogram at WC offsets t (along self._dirs) and FFs d."""
        wcs = dict(self._wc_base)
        for (k, part), tv in zip(self._dirs, t):
            wcs[k] = wcs.get(k, 0j) + (tv if part == "re" else 1j * tv)
        return super().getHistoNoScale(wcs, dict(zip(self._ff_names, map(float, d))))

    def _build(self):
        n, m = len(self._ff_names), len(self._dirs)
        t0 = time.perf_counter()

        def packed(t):   # FF coefficients at WC point t, packed per bin as [c, b, Q]
            c, b, Q = quad_fit(lambda d: self._hammer(t, d), n)
            return np.concatenate([c[:, None], b, Q.reshape(len(c), n * n)], axis=1)

        self._coeffs = quad_fit(packed, m)
        self.setup_time += time.perf_counter() - t0
        if self._verbose:
            print(f"FastHammerCacher: {n} FF pars, WC dirs {self._dirs}, "
                  f"{n_calls(m) * n_calls(n)} Hammer calls, {time.perf_counter() - t0:.1f} s")

    def _wc_offsets(self, wcs):
        """WC offsets from the expansion point; adds new directions if needed."""
        wcs = {k: complex(v) for k, v in wcs.items()}
        new = []
        for k, v in wcs.items():
            dv = v - self._wc_base.get(k, 0j)
            for part, x in (("re", dv.real), ("im", dv.imag)):
                if x != 0 and (k, part) not in self._dirs:
                    new.append((k, part))
        if new:
            self._dirs += new
            self._build()
        return np.array([getattr(wcs.get(k, 0j) - self._wc_base.get(k, 0j), "real" if p == "re" else "imag")
                         for k, p in self._dirs])

    # --- same interface as HammerCacher ------------------------------------
    def getHistoNoScale(self, wcs, FFs):
        for k, v in FFs.items():
            if k in self._ff_state:
                self._ff_state[k] = float(v)
        t = self._wc_offsets(wcs)
        d = np.array([self._ff_state[k] for k in self._ff_names])
        n = len(d)
        coef = quad_eval(*self._coeffs, t)             # FF coefficients at this WC point
        c, b, Q = coef[:, 0], coef[:, 1:n + 1], coef[:, n + 1:].reshape(-1, n, n)
        res = quad_eval(c, b, Q, d)
        self._normFactor = res.sum()
        return res

    # --- optional: save / load the coefficients ----------------------------
    def save(self, path):
        np.savez(path, *self._coeffs, ff_names=self._ff_names, dirs=np.array(self._dirs, dtype=str))

    def load(self, path):
        z = np.load(path)
        assert list(z["ff_names"]) == self._ff_names
        self._dirs = [tuple(x) for x in z["dirs"]]
        self._coeffs = (z["arr_0"], z["arr_1"], z["arr_2"])
