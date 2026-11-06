"""Vanilla HammerCacher: standalone copy of redist.modifier_hammer.HammerCacher.

Only depends on the Hammer python bindings. Every time the FFs or WCs change,
Hammer recomputes the full histogram (one `get_histogram` call, ~0.1-1 s).
"""
import numpy as np
from hammer import Hammer, IOBuffer, RecordType


class HammerCacher:
    """Reads one Hammer histogram from a .dat file and returns it for given WCs and FFs.

    fileName           : Hammer .dat file (run header + histograms)
    histoName          : histogram name stored in the file, e.g. "mmiss2_q2_el"
    FFscheme           : {"Process": "BtoD*", "SchemeVar": "BGLVar", "name": "SchemeBGL"}
    WilsonSet          : WC set name, e.g. "BtoCMuNu"
    FormFactors        : {FF eigenvector name: value}, e.g. {"delta_a0": 0., ...}
    WilsonCoefficients : {WC name: complex value}, e.g. {"SM": 1., "V_qLlL": 0., ...}
    scaleFactor        : the normalized histogram sums to this
    """

    def __init__(self, fileName, histoName, FFscheme, WilsonSet, FormFactors, WilsonCoefficients,
                 scaleFactor=1.0, verbose=False):
        self._histoName = histoName
        self._FFScheme = FFscheme
        self._WilsonSet = WilsonSet
        self._scaleFactor = scaleFactor
        self._wcs = dict(WilsonCoefficients)
        self._FFs = dict(FormFactors)
        self.n_hammer_calls = 0

        self._ham = Hammer()
        self._ham.set_units("GeV")
        buf = IOBuffer(RecordType.UNDEFINED)
        if verbose:
            print(f"fileName = {fileName}\nhistoName = {histoName}")
        with open(fileName, "rb", buffering=0) as fin:
            if buf.load(fin) and self._ham.load_run_header(buf):
                self._ham.init_run()
                if buf.load(fin):
                    while buf.kind in (RecordType.HISTOGRAM, RecordType.HISTOGRAM_DEFINITION):
                        if buf.kind == RecordType.HISTOGRAM_DEFINITION:
                            self._ham.load_histogram_definition(buf)
                        else:
                            self._ham.load_histogram(buf)
                        if not buf.load(fin):
                            break

        self._ham.set_ff_eigenvectors(FFscheme["Process"], FFscheme["SchemeVar"], self._FFs)
        self._ham.set_wilson_coefficients(WilsonSet, self._wcs)
        self._shape = list(self._ham.get_histogram_shape(histoName))
        self._nobs = int(np.prod(self._shape))
        self._update()

    def _update(self):
        """The expensive part: Hammer reweights the histogram for the current FFs/WCs."""
        histo = self._ham.get_histogram(self._histoName, self._FFScheme["name"])
        self._values = np.array([h.sum_wi for h in histo])
        self._normFactor = self._values.sum()
        self.n_hammer_calls += 1

    @staticmethod
    def _check_cache(cache, new):
        """Update `cache` with `new`; True if nothing changed."""
        cached = True
        for k, v in new.items():
            if k not in cache or cache[k] != v:
                cache[k] = v
                cached = False
        return cached

    def getHistoNoScale(self, wcs, FFs):
        """Raw (unnormalized) bin contents for the given WCs and FFs."""
        update = False
        if not self._check_cache(self._FFs, FFs):
            p, v = self._FFScheme["Process"], self._FFScheme["SchemeVar"]
            self._ham.reset_ff_eigenvectors(p, v)
            self._ham.set_ff_eigenvectors(p, v, self._FFs)
            update = True
        if not self._check_cache(self._wcs, wcs):
            self._ham.reset_wilson_coefficients(self._WilsonSet)
            self._ham.set_wilson_coefficients(self._WilsonSet, self._wcs)
            update = True
        if update:
            self._update()
        return self._values.copy()

    def getHisto(self, wcs, FFs):
        """Bin contents normalized to scaleFactor (as redist's MultiHammerCacher.getHisto)."""
        res = self.getHistoNoScale(wcs, FFs)
        return res * self._scaleFactor / res.sum()

    def getHistoTotal(self):
        return self._normFactor
