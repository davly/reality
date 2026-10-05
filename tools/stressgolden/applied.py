#!/usr/bin/env python3
"""Generate testdata/stress/precision_applied.json: precision claims of the
applied-formula packages (physics, fluids, em, acoustics, color).

Every case pairs one function with the precision its own docstring promises
and an input where that promise is hardest to keep: cancellation, near-unit
logarithms, ill-conditioned branches, overflow and underflow of intermediate
products, branch points, and long accumulations. A typical input accompanies
the hard ones, so "fails everywhere" can be told apart from "fails at the
edge". Reference values are the documented formula evaluated exactly at the
binary64 value of every input: python Fractions where the formula is
rational, mpmath at 60 digits otherwise. The reference is then rounded to the
nearest float64. The library's own output is never used as a reference.

This is a development-time generator, not an implementation: Go code never
imports it, and the library keeps zero dependencies. Requirements: Python 3.8+
and mpmath (version recorded in the output). Run from the repository root:

    python tools/stressgolden/applied.py

Tolerance rules (recorded per case in `claim`; `tol` states the CLAIM, never
the error the code achieves):
  - "exact (single operation)": the IEEE operation is correctly rounded, so
    the result must equal the nearest float64 of the true value (tol 0).
  - "exact (n operations)": n roundings, each at most 2^-53 relative, bound
    n * 2^-53 * (1 + n * 2^-53) relative for non-cancelling data. A case that
    breaks it is cancellation, overflow or underflow, not ordinary rounding.
    Each literal decimal constant counts as one rounding; a product of
    constants that Go folds exactly (math.Pi * math.Pi) counts as one.
  - "exact (summation)": read as "within a few ulps of the exact sum", taken
    as 4 ulps = 4 * 2^-52 relative.
  - "limited by float64 X (~15 significant digits)": an approximate claim of
    ~1e-15 relative, enforced with 10x slack: 1e-14 relative. "~12
    significant digits": 1e-11. "typically 1e-10": 1e-10 absolute.
  - A claim "limited by the uncertainty in G (or eps0)" is about the constant,
    which no arithmetic test can check; the arithmetic is compared with the
    formula evaluated at the library's documented decimal value of the
    constant.

Case ids are applied/<function>-<class>-<name>, with class one of: typ
(typical), hard (hard input), extreme (an intermediate product overflows or
underflows although the result is representable), cond (ill-conditioned:
any float64 evaluation of the formula loses this much).

How the hard inputs were chosen, so that a miss is not an accident of one
build: a case that misses is only worth keeping if it misses by a wide margin
and in the same way on the default amd64 build and on GOAMD64=v3 (which fuses
multiply-adds). Where the miss is a rounding error that an exponential
amplifies, the input is the one at which float64 rounds the exponent worst
(worst_rounded_partner); where it is a branch decided by rounding noise (the
CIEDE2000 hue difference of exactly 180 degrees), the pairs were found by
scanning and are listed here. The CIEDE2000 reference is checked against the
34 published test pairs of Sharma, Wu and Dalal on every run.
"""

import json
import math
import os
from fractions import Fraction as Fr

import mpmath as mp

mp.mp.dps = 60
OUT = os.path.join("testdata", "stress", "precision_applied.json")
U = Fr(1, 2 ** 53)  # unit roundoff of binary64 with round to nearest
ULP = 2.0 ** -52

CASES = []
SEEN = set()
ORACLES = {}  # function name -> the exact reference, kept so other tools can reuse it


# ---------------------------------------------------------------------------
# exact arithmetic helpers
# ---------------------------------------------------------------------------

def ex(x):
    """Exact mpf of a float, int, Fraction or decimal string."""
    if isinstance(x, mp.mpf):
        return x
    if isinstance(x, str):
        x = Fr(x)
    if isinstance(x, Fr):
        return mp.mpf(x.numerator) / mp.mpf(x.denominator)
    return mp.mpf(x)


def fl(x):
    """Nearest float64 of an exact value (Fraction or mpf)."""
    if isinstance(x, Fr):
        return float(x)
    return float(x)


def dec(s):
    """Exact rational of a decimal literal such as '0.161'."""
    return Fr(s)


def tol_ops(n):
    """Relative bound of n roundings (round to nearest)."""
    t = Fr(n) * U * (1 + Fr(n) * U)
    return float(t)


def add(cid, func, args, want, tol, kind, claim):
    cid = "applied/" + cid
    assert cid not in SEEN, cid
    SEEN.add(cid)
    assert math.isfinite(want), (cid, want)
    CASES.append({"id": cid, "func": func, "args": args, "want": want, "tol": tol, "tol_kind": kind, "claim": claim})


class Rule:
    """A claim's tolerance reading."""

    def __init__(self, tol, kind, reading):
        self.tol, self.kind, self.reading = tol, kind, reading


def EXACT1():
    return Rule(0.0, "rel", "a single correctly rounded operation: exact match with the nearest float64")


def OPS(n, what):
    t = tol_ops(n)
    return Rule(t, "rel", f"{n} roundings ({what}): relative {t:.3g} = {n}*2^-53*(1+{n}*2^-53)")


def FEW_ULPS(k=4):
    return Rule(k * ULP, "rel", f"'exact' read as within {k} ulps of the exact result: relative {k}*2^-52")


def D15():
    return Rule(1e-14, "rel", "'~15 significant digits' = ~1e-15 relative with the generator's 10x slack: relative 1e-14")


def D12():
    return Rule(1e-11, "rel", "'~12 significant digits' = ~1e-12 relative with the generator's 10x slack: relative 1e-11")


def run(prefix, func, src, quote, rule, oracle, items, skip=()):
    """Add one case per (name, args) item; oracle(**args) gives the true value.
    `skip` names items left out for this function: cases whose outcome sits
    within 2.5x of the tolerance on the side where a compiler that fuses a
    multiply-add differently (another Go version, arm64) could flip it."""
    ORACLES[func] = oracle
    for name, args in items:
        if name in skip:
            continue
        want = oracle(**args)
        add(f"{prefix}-{name}", func, args, want, rule.tol, rule.kind,
            f"{src} {func.split('.', 1)[1]}: '{quote}' -> {rule.reading}")


# ---------------------------------------------------------------------------
# acoustics
# ---------------------------------------------------------------------------

AC = "acoustics/acoustics.go"


def a_weight_exact(f):
    """The documented analytic approximation, constants as written, exactly."""
    f2 = ex(f) ** 2
    num = ex(Fr(12194) ** 2) * f2 * f2
    d1 = f2 + ex(dec("20.6") ** 2)
    d2 = f2 + ex(Fr(12194) ** 2)
    d3 = mp.sqrt((f2 + ex(dec("107.7") ** 2)) * (f2 + ex(dec("737.9") ** 2)))
    return 20 * mp.log10(num / (d1 * d2 * d3)) + 2


def a_weight_root():
    """The float64 nearest the zero of the A-weighting value near 1 kHz."""
    root = mp.findroot(lambda x: a_weight_exact(x), mp.mpf(999.8))
    return float(root)


def acoustics():
    # SoundSpeed: c = sqrt(gamma R T / M)
    def o(gamma, R, T, M):
        return fl(mp.sqrt(ex(gamma) * ex(R) * ex(T) / ex(M)))
    run("soundspeed", "acoustics.SoundSpeed", AC, "Precision: limited by float64 sqrt (~15 significant digits)", D15(), o, [
        ("typ-air-20C", dict(gamma=1.4, R=8.314462618, T=293.15, M=0.02896)),
        ("typ-helium", dict(gamma=5.0 / 3.0, R=8.314462618, T=1000.0, M=0.0040026)),
        ("hard-cryogenic", dict(gamma=1.4, R=8.314462618, T=1e-3, M=0.02896)),
        ("hard-plasma", dict(gamma=1.4, R=8.314462618, T=1e9, M=0.02896)),
    ])

    # SoundIntensity: I = P / (4 pi r^2)
    def o(P, r):
        return fl(ex(P) / (4 * mp.pi * ex(r) * ex(r)))
    run("soundintensity", "acoustics.SoundIntensity", AC, "Precision: limited by float64 representation of π (~15 significant digits)", D15(), o, [
        ("typ-1w-1m", dict(P=1.0, r=1.0)),
        ("typ-speaker", dict(P=0.01, r=3.7)),
        ("hard-near-field", dict(P=1e5, r=1e-3)),
        ("hard-far-field", dict(P=1e3, r=1e10)),
        ("extreme-r2-overflow", dict(P=1e300, r=1e160)),
    ])

    # Decibel scales
    def spl(p, pRef):
        return fl(20 * mp.log10(ex(p) / ex(pRef)))

    def il(I, IRef):
        return fl(10 * mp.log10(ex(I) / ex(IRef)))
    def near(ref, delta):
        """The float p near ref*(1+delta) at which float64 rounds the quotient p/ref worst."""
        return worst_rounded_partner(ref, ref * (1 + delta), lambda r, p: p / r, lambda r, p: Fr(p) / Fr(r))
    items_spl = [
        ("typ-1pa", dict(p=1.0, pRef=20e-6)),
        ("typ-quiet", dict(p=3e-5, pRef=20e-6)),
        # pRef = 1 makes p/pRef exact, so these three control cases keep the logarithm itself on trial
        ("hard-near-0db-exact-ratio-above", dict(p=1.001, pRef=1.0)),
        ("hard-near-0db-exact-ratio-1e-10", dict(p=1.0000000001, pRef=1.0)),
        ("hard-near-0db-exact-ratio-below", dict(p=0.9999999999, pRef=1.0)),
        ("hard-adjacent-floats", dict(p=1.0000000000000002, pRef=1.0)),
        # the standard 20 micropascal reference: the quotient rounds, and the level just above 0 dB keeps few digits
        ("hard-near-0db-ref20upa-1e-3", dict(p=near(20e-6, 1e-3), pRef=20e-6)),
        ("hard-near-0db-ref20upa-1e-6", dict(p=near(20e-6, 1e-6), pRef=20e-6)),
        ("hard-near-0db-ref20upa-1e-10", dict(p=near(20e-6, 1e-10), pRef=20e-6)),
        ("hard-near-0db-ref20upa-below-1e-8", dict(p=near(20e-6, -1e-8), pRef=20e-6)),
        ("hard-equal", dict(p=20e-6, pRef=20e-6)),
        ("hard-huge-ratio", dict(p=1e5, pRef=1e-300)),
        ("extreme-ratio-overflow", dict(p=1e200, pRef=1e-200)),
        ("extreme-ratio-underflow", dict(p=1e-200, pRef=1e200)),
    ]
    run("dbspl", "acoustics.DecibelSPL", AC, "Precision: limited by float64 log10 (~15 significant digits)", D15(), spl, items_spl)
    items_il = [
        ("typ-1e-6", dict(I=1e-6, IRef=1e-12)),
        ("typ-quiet", dict(I=3e-12, IRef=1e-12)),
        ("hard-near-0db-exact-ratio-above", dict(I=1.001, IRef=1.0)),
        ("hard-near-0db-exact-ratio-1e-10", dict(I=1.0000000001, IRef=1.0)),
        ("hard-near-0db-exact-ratio-below", dict(I=0.9999999999, IRef=1.0)),
        ("hard-adjacent-floats", dict(I=1.0000000000000002, IRef=1.0)),
        ("hard-near-0db-ref1e-12-1e-3", dict(I=near(1e-12, 1e-3), IRef=1e-12)),
        ("hard-near-0db-ref1e-12-1e-6", dict(I=near(1e-12, 1e-6), IRef=1e-12)),
        ("hard-near-0db-ref1e-12-1e-10", dict(I=near(1e-12, 1e-10), IRef=1e-12)),
        ("hard-near-0db-ref1e-12-below-1e-8", dict(I=near(1e-12, -1e-8), IRef=1e-12)),
        ("hard-equal", dict(I=1e-12, IRef=1e-12)),
        ("extreme-ratio-overflow", dict(I=1e200, IRef=1e-200)),
        ("extreme-ratio-underflow", dict(I=1e-200, IRef=1e200)),
    ]
    run("dbintensity", "acoustics.DecibelFromIntensity", AC, "Precision: limited by float64 log10 (~15 significant digits)", D15(), il, items_il)

    # SabineRT60: 0.161 V / A (the literal 0.161 is one rounding)
    def o(V, A):
        return fl(dec("0.161") * Fr(V) / Fr(A))
    run("sabine", "acoustics.SabineRT60", AC, "Precision: exact (multiplication and division)", OPS(3, "the literal 0.161, the product, the quotient"), o, [
        ("typ-classroom", dict(V=200.0, A=20.0)),
        ("typ-cathedral", dict(V=1e5, A=1.3e3)),
        ("hard-tiny-room", dict(V=1e-3, A=7.3)),
        ("hard-large-absorption", dict(V=1234.5678, A=0.3333333333333333)),
    ])

    # DopplerShift: f0 (c + vr) / (c + vs); c + v is exact for v near -c (Sterbenz)
    def o(f0, vs, vr, c):
        return fl(Fr(f0) * (Fr(c) + Fr(vr)) / (Fr(c) + Fr(vs)))
    run("doppler", "acoustics.DopplerShift", AC, "Precision: exact (arithmetic only)", OPS(4, "c+vr, the product, c+vs, the quotient"), o, [
        ("typ-receding-source", dict(f0=440.0, vs=30.0, vr=0.0, c=343.0)),
        ("typ-approaching-source", dict(f0=440.0, vs=-30.0, vr=0.0, c=343.0)),
        ("typ-moving-receiver", dict(f0=1000.0, vs=0.0, vr=12.5, c=343.0)),
        ("hard-near-sonic-approach", dict(f0=440.0, vs=-342.9999999, vr=0.0, c=343.0)),
        ("hard-near-sonic-1e-13", dict(f0=440.0, vs=-342.9999999999997, vr=3.3, c=343.0)),
        ("hard-receiver-cancels", dict(f0=440.0, vs=5.0, vr=-342.99999999999994, c=343.0)),
    ])

    # ResonantFrequency: n c / (2 L)
    def o(L, n, c):
        return fl(Fr(n) * Fr(c) / (2 * Fr(L)))
    run("resonantfreq", "acoustics.ResonantFrequency", AC, "Precision: exact (arithmetic only)", OPS(2, "n*c and the quotient"), o, [
        ("typ-organ-pipe", dict(L=0.5, n=1, c=343.0)),
        ("typ-harmonic-7", dict(L=0.25, n=7, c=343.0)),
        ("hard-1000th-harmonic", dict(L=2.0, n=1000, c=343.0)),
        ("hard-short-pipe", dict(L=1e-3, n=3, c=343.0)),
    ])

    # WaveLength: c / f
    def o(f, c):
        return fl(Fr(c) / Fr(f))
    run("wavelength", "acoustics.WaveLength", AC, "Precision: exact (single division)", EXACT1(), o, [
        ("typ-440hz", dict(f=440.0, c=343.0)),
        ("typ-20khz", dict(f=20000.0, c=343.0)),
        ("hard-radio", dict(f=1e-3, c=3e8)),
        ("hard-odd-quotient", dict(f=3.0, c=1.0)),
        ("hard-subnormal-result", dict(f=3e10, c=1e-300)),
    ])

    # AWeighting: the documented analytic approximation, constants as written
    def o(f):
        return fl(a_weight_exact(f))
    run("aweight", "acoustics.AWeighting", AC,
        "Precision: limited by float64 log and sqrt (~12 significant digits at extreme frequencies)", D12(), o, [
            ("typ-100hz", dict(f=100.0)),
            ("typ-10khz", dict(f=10000.0)),
            ("typ-20hz", dict(f=20.0)),
            ("typ-20khz", dict(f=20000.0)),
            ("hard-6khz", dict(f=6300.0)),
            ("hard-near-zero-1khz", dict(f=a_weight_root() + 1e-6)),
            ("hard-infrasound", dict(f=1e-3)),
            ("hard-1ghz", dict(f=1e9)),
            ("hard-1e-70hz", dict(f=1e-70)),
            ("extreme-f4-underflow", dict(f=1e-80)),
            ("extreme-f4-overflow", dict(f=1e80)),
        ])


# ---------------------------------------------------------------------------
# em
# ---------------------------------------------------------------------------

EM = "em/em.go"
EPS0 = dec("8.8541878128e-12")  # the library's documented CODATA 2018 value


def coulomb_k():
    return 1 / (4 * mp.pi * ex(EPS0))


def em():
    arith = ("the uncertainty in eps0 is a property of the constant; the arithmetic is compared with k = 1/(4 pi eps0) at the "
             "documented decimal eps0 = 8.8541878128e-12, relative 1e-14 ('~15 significant digits' with 10x slack)")
    rule = Rule(1e-14, "rel", arith)

    def o(q1, q2, r):
        return fl(coulomb_k() * ex(q1) * ex(q2) / (ex(r) * ex(r)))
    run("coulomb", "em.CoulombForce", EM, "Precision: limited by uncertainty in ε₀ and float64 arithmetic", rule, o, [
        ("typ-two-electrons", dict(q1=-1.602176634e-19, q2=-1.602176634e-19, r=1e-10)),
        ("typ-microcoulombs", dict(q1=1e-6, q2=1e-6, r=1.0)),
        ("typ-attractive", dict(q1=3e-9, q2=-5e-9, r=0.02)),
        ("hard-nuclear-distance", dict(q1=79 * 1.602176634e-19, q2=2 * 1.602176634e-19, r=1e-14)),
        ("hard-astronomical-distance", dict(q1=1e-3, q2=1e-3, r=1e12)),
        ("extreme-tiny-charges-underflow", dict(q1=1e-170, q2=1e-170, r=1e-100)),
        ("extreme-huge-charges-overflow", dict(q1=1e160, q2=1e160, r=1e150)),
    ])

    def o(q, r):
        return fl(coulomb_k() * ex(q) / (ex(r) * ex(r)))
    run("efield", "em.ElectricField", EM, "Precision: limited by uncertainty in ε₀ and float64 arithmetic", rule, o, [
        ("typ-proton-bohr-radius", dict(q=1.602176634e-19, r=5.29177210903e-11)),
        ("typ-negative-charge", dict(q=-2e-6, r=0.3)),
        ("hard-far-field", dict(q=1e-9, r=1e9)),
        ("extreme-tiny-charge-underflow", dict(q=1e-310, r=1e-5)),
    ])

    def o(V, R):
        return fl(Fr(V) / Fr(R))
    run("ohm", "em.OhmsLaw", EM, "Precision: exact (single division)", EXACT1(), o, [
        ("typ-12v-220", dict(V=12.0, R=220.0)),
        ("typ-3v3-4k7", dict(V=3.3, R=4.7e3)),
        ("hard-micro-volts-mega-ohms", dict(V=1e-6, R=1e6)),
        ("hard-thirds", dict(V=1.0, R=3.0)),
        ("hard-subnormal-result", dict(V=1e-300, R=3e10)),
    ])

    def o(V, I):
        return fl(Fr(V) * Fr(I))
    run("power", "em.PowerElectric", EM, "Precision: exact (single multiplication)", EXACT1(), o, [
        ("typ-mains", dict(V=230.0, I=13.0)),
        ("typ-logic", dict(V=3.3, I=0.7e-3)),
        ("hard-power-line", dict(V=7.65e5, I=1.9e3)),
        ("hard-nano", dict(V=1e-9, I=3.3e-9)),
        ("hard-subnormal-result", dict(V=1e-160, I=3.3e-150)),
    ])

    def series(**kw):
        vals = kw["r"] if "r" in kw else None
        if vals is not None:
            return fl(sum((Fr(x) for x in vals), Fr(0)))
        return fl(Fr(kw["value"]) * kw["count"])
    ser = "Precision: exact (summation)"
    run("series", "em.ResistorsInSeries", EM, ser, FEW_ULPS(4), series, [
        ("typ-three", dict(r=[100.0, 220.0, 330.0])),
        ("typ-tenths", dict(r=[0.1, 0.2, 0.3])),
        ("typ-e24-ladder", dict(r=[10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0])),
        ("hard-30-tenths", dict(count=30, value=0.1)),
        ("hard-100-tenths", dict(count=100, value=0.1)),
        ("hard-1e3-tenths", dict(count=1000, value=0.1)),
        ("hard-1e5-tenths", dict(count=100000, value=0.1)),
        ("hard-1e6-ohm-values", dict(count=1000000, value=1.1)),
        ("hard-wide-spread", dict(r=[1e8, 1.0, 1e-8, 3.3, 7.7, 1e-3])),
    ])

    def parallel(**kw):
        vals = kw["r"] if "r" in kw else [kw["value"]] * kw["count"]
        return fl(1 / sum((1 / Fr(x) for x in vals), Fr(0)))

    ORACLES["em.ResistorsInParallel"] = parallel

    def parallel_rule(n):
        t = tol_ops(n + 1)
        return Rule(t, "rel", f"accumulation of n={n} terms: n+1 roundings on any path, relative {t:.3g} = (n+1)*2^-53*(1+(n+1)*2^-53)")
    par = "Precision: limited by float64 accumulation"
    for name, kw, n in [
        ("typ-three", dict(r=[100.0, 220.0, 330.0]), 3),
        ("typ-equal-ten", dict(count=10, value=470.0), 10),
        ("hard-1e3-equal", dict(count=1000, value=1e3), 1000),
        ("hard-1e5-equal", dict(count=100000, value=1e3), 100000),
        ("hard-wide-spread", dict(r=[1e-6, 1e6, 3.3, 1e12, 7.7e-3]), 5),
    ]:
        rule_n = parallel_rule(n)
        add(f"parallel-{name}", "em.ResistorsInParallel", kw, parallel(**kw), rule_n.tol, rule_n.kind,
            f"{EM} ResistorsInParallel: '{par}' -> {rule_n.reading}")

    def o(C, V):
        return fl(dec("0.5") * Fr(C) * Fr(V) * Fr(V))
    run("capenergy", "em.CapacitorEnergy", EM, "Precision: exact (multiplication)", OPS(2, "C*V and (C*V)*V; 0.5*C is exact"), o, [
        ("typ-supercap", dict(C=1.0, V=2.7)),
        ("typ-electrolytic", dict(C=4.7e-3, V=400.0)),
        ("hard-pulse-bank", dict(C=0.01, V=2e4)),
        ("hard-pico", dict(C=1e-12, V=3.3)),
    ])

    def o(L, I):
        return fl(dec("0.5") * Fr(L) * Fr(I) * Fr(I))
    run("indenergy", "em.InductorEnergy", EM, "Precision: exact (multiplication)", OPS(2, "L*I and (L*I)*I; 0.5*L is exact"), o, [
        ("typ-choke", dict(L=1e-3, I=5.0)),
        ("typ-solenoid", dict(L=2.5, I=100.0)),
        ("hard-magnet", dict(L=10.0, I=1e5)),
        ("hard-micro", dict(L=1e-9, I=1e-3)),
    ])

    def o(R, C):
        return fl(Fr(R) * Fr(C))
    run("rc", "em.RCTimeConstant", EM, "Precision: exact (single multiplication)", EXACT1(), o, [
        ("typ-1k-1uf", dict(R=1e3, C=1e-6)),
        ("typ-10m-100pf", dict(R=1e7, C=1e-10)),
        ("hard-odd-values", dict(R=4.7e3, C=3.3e-9)),
        ("hard-leakage", dict(R=1e15, C=1e-3)),
        ("hard-subnormal-result", dict(R=1e-160, C=3.3e-150)),
    ])

    def o(L, C):
        return fl(1 / (2 * mp.pi * mp.sqrt(ex(L) * ex(C))))
    run("lc", "em.ResonantFrequencyLC", EM, "Precision: limited by float64 sqrt and pi (~15 significant digits)", D15(), o, [
        ("typ-audio", dict(L=1e-3, C=1e-6)),
        ("typ-fm-radio", dict(L=1e-6, C=1e-12)),
        ("hard-power-grid", dict(L=10.0, C=1e-3)),
        ("hard-femto", dict(L=1e-12, C=1e-15)),
        ("extreme-lc-underflow", dict(L=1e-200, C=1e-200)),
        ("extreme-lc-overflow", dict(L=1e200, C=1e200)),
    ])


# ---------------------------------------------------------------------------
# fluids
# ---------------------------------------------------------------------------

FL = "fluids/fluids.go"


def colebrook_f(Re, roughness, diameter):
    """The Colebrook-White friction factor solved to 50 digits, or 64/Re."""
    if Re < 2300:
        return fl(Fr(64) / Fr(Re))
    a = ex(roughness) / (ex(diameter) * ex(dec("3.7")))
    b = ex(dec("2.51")) / ex(Re)

    def g(x):
        # x = 1/sqrt(f) solves x = -2 log10(a + b x); g is increasing in x
        return x + 2 * mp.log10(a + b * x)
    lo, hi = mp.mpf("0.01"), mp.mpf(10000)
    for _ in range(200):
        mid = (lo + hi) / 2
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2
    for _ in range(8):  # Newton polish
        x = x - g(x) / (1 + 2 * b / (mp.log(10) * (a + b * x)))
    return fl(1 / (x * x))


def fluids():
    def o(rho, v, L, mu):
        return fl(Fr(rho) * Fr(v) * Fr(L) / Fr(mu))
    run("reynolds", "fluids.ReynoldsNumber", FL, "Precision: exact (multiplication and single division)", OPS(3, "rho*v, *L, /mu"), o, [
        ("typ-water-pipe", dict(rho=998.0, v=2.0, L=0.05, mu=1.0016e-3)),
        ("typ-air-wing", dict(rho=1.204, v=70.0, L=1.5, mu=1.825e-5)),
        ("hard-creeping", dict(rho=2500.0, v=1e-6, L=10.0, mu=1e5)),
        ("hard-ocean-liner", dict(rho=1025.0, v=12.0, L=300.0, mu=1.08e-3)),
    ])

    # BernoulliPressure: p1 + 0.5 rho (v1^2 - v2^2) + rho g (h1 - h2)
    def o(rho, v1, p1, h1, v2, h2, g):
        return fl(Fr(p1) + Fr(1, 2) * Fr(rho) * (Fr(v1) ** 2 - Fr(v2) ** 2) + Fr(rho) * Fr(g) * (Fr(h1) - Fr(h2)))
    run("bernoulli", "fluids.BernoulliPressure", FL, "Precision: exact (arithmetic only)",
        OPS(9, "v1*v1, v2*v2, their difference, the product with 0.5*rho, +p1, rho*g, h1-h2, their product, the final sum"), o, [
            ("typ-venturi", dict(rho=998.0, v1=1.5, p1=2e5, h1=0.0, v2=6.0, h2=0.0, g=9.80665)),
            ("typ-descending-pipe", dict(rho=998.0, v1=2.0, p1=3e5, h1=10.0, v2=2.5, h2=0.0, g=9.80665)),
            ("typ-air-nozzle", dict(rho=1.204, v1=5.0, p1=101325.0, h1=0.0, v2=60.0, h2=0.0, g=9.80665)),
            ("hard-near-equal-speeds-gauge-zero", dict(rho=998.0, v1=10.000001, p1=0.0, h1=0.0, v2=10.0, h2=0.0, g=9.80665)),
            ("hard-near-equal-speeds-atmosphere", dict(rho=998.0, v1=3.0000001, p1=101325.0, h1=0.0, v2=3.0, h2=0.0, g=9.80665)),
            ("hard-static-pressure-nearly-cancels", dict(rho=1.204, v1=0.0, p1=101325.0, h1=0.0, v2=410.2, h2=0.0, g=9.80665)),
            ("hard-height-cancels-speed", dict(rho=998.0, v1=0.0, p1=1000.0, h1=0.0, v2=0.0, h2=0.10204081632653061, g=9.80665)),
        ])

    # PipeFlowFriction
    ORACLES["fluids.PipeFlowFriction"] = colebrook_f
    pipe = "Precision: iterative solve to ~1e-10 relative change; Swamee–Jain seed"
    pipe_rule = Rule(1e-10, "rel", "the relative change at which the iteration stops bounds its error against the exact Colebrook-White root: relative 1e-10")
    lam_rule = EXACT1()
    for name, args in [
        ("typ-smooth-1e5", dict(Re=1e5, roughness=0.0, diameter=0.1)),
        ("typ-commercial-steel", dict(Re=5e5, roughness=4.5e-5, diameter=0.1)),
        ("typ-rough-concrete", dict(Re=2e6, roughness=3e-3, diameter=0.5)),
        ("hard-transition-start", dict(Re=2300.0, roughness=0.0, diameter=0.1)),
        ("hard-transition-start-rough", dict(Re=2300.0, roughness=2e-3, diameter=0.1)),
        ("hard-just-above-transition", dict(Re=2300.0000000000005, roughness=1e-4, diameter=0.05)),
        ("hard-fully-rough-moody-limit", dict(Re=1e7, roughness=5e-3, diameter=0.1)),
        ("hard-very-high-reynolds", dict(Re=1e10, roughness=1e-6, diameter=1.0)),
        ("hard-smooth-1e15", dict(Re=1e15, roughness=0.0, diameter=1.0)),
        ("hard-smooth-1e100", dict(Re=1e100, roughness=0.0, diameter=1.0)),
        ("extreme-smooth-1e200", dict(Re=1e200, roughness=0.0, diameter=1.0)),
        ("extreme-smooth-1e300", dict(Re=1e300, roughness=0.0, diameter=1.0)),
        ("hard-relative-roughness-0.3", dict(Re=1e4, roughness=0.03, diameter=0.1)),
    ]:
        add(f"pipefriction-{name}", "fluids.PipeFlowFriction", args, colebrook_f(**args), pipe_rule.tol, pipe_rule.kind,
            f"{FL} PipeFlowFriction: '{pipe}' -> {pipe_rule.reading}")
    for name, args in [
        ("typ-laminar-1000", dict(Re=1000.0, roughness=0.0, diameter=0.1)),
        ("hard-laminar-just-below-transition", dict(Re=2299.9999999999995, roughness=1e-3, diameter=0.1)),
        ("hard-laminar-creeping", dict(Re=1e-5, roughness=0.0, diameter=0.01)),
        ("hard-laminar-odd-quotient", dict(Re=777.0, roughness=0.0, diameter=0.01)),
    ]:
        add(f"pipefriction-{name}", "fluids.PipeFlowFriction", args, colebrook_f(**args), lam_rule.tol, lam_rule.kind,
            f"{FL} PipeFlowFriction: 'For laminar flow (Re < 2300), the exact Hagen–Poiseuille result is used: f = 64 / Re' -> {lam_rule.reading}")

    def o(f, L, D, rho, v):
        return fl(Fr(f) * (Fr(L) / Fr(D)) * (Fr(rho) * Fr(v) * Fr(v) / 2))
    run("darcy", "fluids.DarcyWeisbach", FL, "Precision: exact (arithmetic only)",
        OPS(5, "L/D, f*(L/D), rho*v, *v, the final product; /2 is exact"), o, [
            ("typ-water-main", dict(f=0.02, L=100.0, D=0.1, rho=998.0, v=2.0)),
            ("typ-oil-line", dict(f=0.035, L=2.5e4, D=0.3, rho=870.0, v=1.1)),
            ("hard-pipeline", dict(f=0.008, L=1e6, D=1.2, rho=720.0, v=3.3)),
            ("hard-capillary", dict(f=0.5, L=1e-2, D=1e-4, rho=1000.0, v=1e-3)),
        ])

    def o(Cd, rho, v, A):
        return fl(Fr(1, 2) * Fr(Cd) * Fr(rho) * Fr(v) * Fr(v) * Fr(A))
    run("drag", "fluids.DragForce", FL, "Precision: exact (multiplication only)",
        OPS(4, "*rho, *v, *v, *A; 0.5*Cd is exact"), o, [
            ("typ-car", dict(Cd=0.3, rho=1.204, v=30.0, A=2.2)),
            ("typ-sphere-water", dict(Cd=0.47, rho=998.0, v=2.0, A=0.01)),
            ("hard-rocket", dict(Cd=0.25, rho=0.0889, v=1500.0, A=10.5)),
            ("hard-negative-speed", dict(Cd=1.1, rho=1.204, v=-12.5, A=0.7)),
        ])

    def o(Cl, rho, v, A):
        return fl(Fr(1, 2) * Fr(Cl) * Fr(rho) * Fr(v) * Fr(v) * Fr(A))
    run("lift", "fluids.LiftForce", FL, "Precision: exact (multiplication only)",
        OPS(4, "*rho, *v, *v, *A; 0.5*Cl is exact"), o, [
            ("typ-airliner-wing", dict(Cl=1.2, rho=0.4135, v=250.0, A=120.0)),
            ("typ-glider", dict(Cl=0.9, rho=1.1, v=30.0, A=10.0)),
            ("hard-downforce", dict(Cl=-3.1, rho=1.204, v=90.0, A=1.5)),
            ("hard-small-uav", dict(Cl=0.7, rho=1.204, v=3.3, A=0.05)),
        ])

    def o(m, g, Cd, rho, A):
        return fl(mp.sqrt(2 * ex(m) * ex(g) / (ex(Cd) * ex(rho) * ex(A))))
    run("terminalv", "fluids.TerminalVelocity", FL, "Precision: exact (single sqrt)",
        OPS(3, "Cd*rho, *A, 2mg, the quotient: 4 roundings halved by the sqrt, plus its own"), o, [
            ("typ-skydiver", dict(m=80.0, g=9.80665, Cd=1.0, rho=1.204, A=0.7)),
            ("typ-raindrop", dict(m=3.35e-5, g=9.80665, Cd=0.45, rho=1.204, A=3.14e-6)),
            ("hard-micro-particle", dict(m=1e-12, g=9.80665, Cd=1.0, rho=1.204, A=1e-8)),
            ("hard-meteoroid", dict(m=1e5, g=9.80665, Cd=0.8, rho=1e-3, A=3.0)),
        ])

    def o(mu, r, v):
        return fl(6 * mp.pi * ex(mu) * ex(r) * ex(v))
    run("stokes", "fluids.StokesLaw", FL, "Precision: limited by float64 representation of π (~15 significant digits)", D15(), o, [
        ("typ-bead-glycerol", dict(mu=1.41, r=1e-3, v=0.01)),
        ("typ-pollen-air", dict(mu=1.825e-5, r=2.5e-5, v=0.03)),
        ("hard-nanoparticle", dict(mu=1.0016e-3, r=1e-9, v=1e-6)),
        ("hard-large-values", dict(mu=1e3, r=1e2, v=1e3)),
    ])

    def o(rho, v, A):
        return fl(Fr(rho) * Fr(v) * Fr(A))
    run("massflow", "fluids.MassFlowRate", FL, "Precision: exact (multiplication only)", OPS(2, "rho*v, *A"), o, [
        ("typ-water-main", dict(rho=998.0, v=2.0, A=7.85e-3)),
        ("typ-air-duct", dict(rho=1.204, v=8.0, A=0.09)),
        ("hard-river", dict(rho=998.0, v=1.7, A=5.4e3)),
        ("hard-micro", dict(rho=0.8, v=1e-3, A=1e-10)),
    ])

    def o(v, A):
        return fl(Fr(v) * Fr(A))
    run("volflow", "fluids.VolumetricFlowRate", FL, "Precision: exact (single multiplication)", EXACT1(), o, [
        ("typ-pipe", dict(v=2.0, A=7.85e-3)),
        ("typ-duct", dict(v=8.0, A=0.09)),
        ("hard-river", dict(v=1.7, A=5.4e3)),
        ("hard-odd", dict(v=0.1, A=0.3)),
        ("hard-subnormal-result", dict(v=1e-160, A=3.3e-150)),
    ])


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------

PMECH = "physics/mechanics.go"
PMAT = "physics/materials.go"
PTHERM = "physics/thermo.go"
POPT = "physics/optics.go"
G_NEWTON = dec("6.6743e-11")  # the library's documented CODATA 2018 value
R_GAS = dec("6.02214076e23") * dec("1.380649e-23")  # N_A * k_B, exact: 8.31446261815324
SIGMA_SB = 2 * mp.pi ** 5 * ex(dec("1.380649e-23")) ** 4 / (15 * ex(dec("6.62607015e-34")) ** 3 * 299792458 ** 2)


def fresnel_exact(n1, n2, theta):
    n1, n2, th = ex(n1), ex(n2), ex(theta)
    sT = n1 / n2 * mp.sin(th)
    if sT > 1 or sT < -1:
        return 1.0
    cI = mp.cos(th)
    cT = mp.sqrt(1 - sT ** 2)
    rs = ((n1 * cI - n2 * cT) / (n1 * cI + n2 * cT)) ** 2
    rp = ((n1 * cT - n2 * cI) / (n1 * cT + n2 * cI)) ** 2
    return fl((rs + rp) / 2)


def critical_theta(n1, n2, delta):
    """A float64 incidence angle whose exact refracted sine is about 1 - delta."""
    return float(mp.asin((1 - mp.mpf(delta)) * ex(n2) / ex(n1)))


def worst_rounded_partner(fixed, start, rounded, exact, steps=4000):
    """Scan the floats from `start` upward for the one at which the rounded
    evaluation `rounded(fixed, x)` of an exponent has the largest relative
    error against `exact(fixed, x)`. A hard input for a claim that the
    rounding of an exponent is amplified by the exponential: the worst case is
    the input to take, because a mild rounding error would hide the miss."""
    best, x = None, start
    for _ in range(steps):
        e = exact(fixed, x)
        err = abs(Fr(rounded(fixed, x)) - e) / abs(e)
        if best is None or err > best[0]:
            best = (err, x)
        x = math.nextafter(x, math.inf)
    return best[1]


def physics():
    # -- mechanics --------------------------------------------------------
    def o(F, m):
        return fl(Fr(F) / Fr(m))
    run("newton2", "physics.NewtonSecondLaw", PMECH, "Precision: exact (single division)", EXACT1(), o, [
        ("typ-10n-3kg", dict(F=10.0, m=3.0)),
        ("typ-weight", dict(F=-784.532, m=80.0)),
        ("hard-electron", dict(F=1e-3, m=9.1093837015e-31)),
        ("hard-sevenths", dict(F=1.0, m=7.0)),
        ("hard-subnormal-result", dict(F=1e-300, m=3e10)),
    ])

    proj = "Precision: limited by float64 trig (~15 significant digits)"
    t_land = 2 * 50.0 * math.sin(0.7) / 9.80665
    proj_items = [
        ("typ-45deg", dict(v0=50.0, theta=math.pi / 4, t=2.0, g=9.80665)),
        ("typ-30deg-long", dict(v0=120.0, theta=0.5235987755982988, t=3.7, g=9.80665)),
        ("hard-vertical-launch", dict(v0=50.0, theta=math.pi / 2, t=2.0, g=9.80665)),
        ("hard-launch-at-pi", dict(v0=50.0, theta=math.pi, t=2.0, g=9.80665)),
        ("hard-tiny-angle", dict(v0=300.0, theta=1e-300, t=5.0, g=9.80665)),
        ("hard-large-time", dict(v0=100.0, theta=0.3, t=1e6, g=9.80665)),
        ("hard-landing-cancellation", dict(v0=50.0, theta=0.7, t=t_land, g=9.80665)),
    ]

    def ox(v0, theta, t, g):
        return fl(ex(v0) * mp.cos(ex(theta)) * ex(t))

    def oy(v0, theta, t, g):
        return fl(ex(v0) * mp.sin(ex(theta)) * ex(t) - ex(g) * ex(t) ** 2 / 2)
    run("projectile-x", "physics.ProjectilePosition.x", PMECH, proj, D15(), ox, proj_items)
    run("projectile-y", "physics.ProjectilePosition.y", PMECH, proj, D15(), oy, proj_items)

    def o(m1, m2, r):
        return fl(G_NEWTON * Fr(m1) * Fr(m2) / Fr(r) ** 2)
    grav_rule = Rule(1e-14, "rel", "the uncertainty in G is a property of the constant; the arithmetic is compared with the formula at the documented G = 6.6743e-11, relative 1e-14 ('limited by float64' = ~15 digits with 10x slack)")
    run("gravity", "physics.GravitationalForce", PMECH, "Precision: limited by uncertainty in G (~2.2e-5 relative)", grav_rule, o, [
        ("typ-earth-moon", dict(m1=5.972e24, m2=7.342e22, r=3.844e8)),
        ("typ-unit", dict(m1=1.0, m2=1.0, r=1.0)),
        ("hard-electron-proton", dict(m1=9.1093837015e-31, m2=1.67262192369e-27, r=5.29177210903e-11)),
        ("hard-galaxies", dict(m1=1e42, m2=1e42, r=1e22)),
        ("extreme-masses-overflow", dict(m1=1e200, m2=1e200, r=1e100)),
        ("extreme-masses-underflow", dict(m1=1e-200, m2=1e-200, r=1e-100)),
    ])

    def o(M, r):
        return fl(mp.sqrt(ex(G_NEWTON) * ex(M) / ex(r)))
    run("orbitalv", "physics.OrbitalVelocity", PMECH, "Precision: limited by uncertainty in G (~2.2e-5 relative) and float64 sqrt", grav_rule, o, [
        ("typ-low-earth-orbit", dict(M=5.972e24, r=6.771e6)),
        ("typ-earth-around-sun", dict(M=1.989e30, r=1.496e11)),
        ("hard-tiny-body", dict(M=1e-3, r=1e-3)),
        ("hard-neutron-star", dict(M=2.8e30, r=1.2e4)),
        ("extreme-gm-over-r-overflow", dict(M=1e300, r=1e-300)),
    ])

    def o(k, x, c, v):
        return fl(-Fr(k) * Fr(x) - Fr(c) * Fr(v))
    run("spring", "physics.SpringForce", PMECH, "Precision: exact (two multiplications and a subtraction)",
        OPS(3, "k*x, c*v and the subtraction"), o, [
            ("typ-damped", dict(k=250.0, x=0.03, c=2.5, v=0.4)),
            ("typ-undamped", dict(k=1e4, x=-0.012, c=0.0, v=3.0)),
            ("hard-spring-balances-damper-1", dict(k=0.3, x=0.7, c=0.21, v=-1.0)),
            ("hard-spring-balances-damper-2", dict(k=1000.0, x=0.0123, c=10.0, v=-1.23)),
            ("hard-spring-balances-damper-3", dict(k=7.7, x=0.31, c=2.3, v=-float(Fr(77, 10) * Fr(31, 100) / Fr(23, 10)))),
        ])

    def oc1(m1, v1, m2, v2):
        return fl(((Fr(m1) - Fr(m2)) * Fr(v1) + 2 * Fr(m2) * Fr(v2)) / (Fr(m1) + Fr(m2)))

    def oc2(m1, v1, m2, v2):
        return fl(((Fr(m2) - Fr(m1)) * Fr(v2) + 2 * Fr(m1) * Fr(v1)) / (Fr(m1) + Fr(m2)))
    coll = "Precision: exact (arithmetic operations only)"
    coll_rule = OPS(6, "m1-m2, *v1, 2*m2*v2, their sum, m1+m2, the quotient")
    # speeds at which one body stops dead (the second body's v2 / the first body's v1 solve the zero of the numerator)
    stop_v2 = float(-(Fr(23, 10) - Fr(7, 10)) * Fr(19, 10) / (2 * Fr(7, 10)))
    stop_v1 = float((Fr(23, 10) - Fr(7, 10)) * Fr(19, 10) / (2 * Fr(23, 10)))
    coll_items = [
        ("typ-unequal", dict(m1=2.0, v1=3.0, m2=1.0, v2=-1.0)),
        ("typ-equal-masses", dict(m1=0.7, v1=1.9, m2=0.7, v2=-0.3)),
        ("hard-heavy-on-light", dict(m1=1e6, v1=2.5, m2=1e-3, v2=-7.0)),
        ("hard-light-on-heavy", dict(m1=1e-3, v1=40.0, m2=1e6, v2=0.0)),
        ("hard-first-body-stops", dict(m1=2.3, v1=1.9, m2=0.7, v2=stop_v2)),
        ("hard-second-body-stops", dict(m1=2.3, v1=stop_v1, m2=0.7, v2=1.9)),
    ]
    run("collision-v1", "physics.ElasticCollision.v1f", PMECH, coll, coll_rule, oc1, coll_items)
    run("collision-v2", "physics.ElasticCollision.v2f", PMECH, coll, coll_rule, oc2, coll_items)

    def o(theta, L, g, damping):
        return fl(-(ex(g) / ex(L)) * mp.sin(ex(theta)) - ex(damping) * mp.sin(ex(theta)))
    run("pendulum", "physics.Pendulum", PMECH, "Precision: limited by float64 sin (~15 significant digits)", D15(), o, [
        ("typ-small-angle", dict(theta=0.1, L=1.0, g=9.80665, damping=0.0)),
        ("typ-damped", dict(theta=0.5, L=0.25, g=9.80665, damping=0.1)),
        ("hard-theta-pi", dict(theta=math.pi, L=1.0, g=9.80665, damping=0.0)),
        ("hard-theta-10pi", dict(theta=10 * math.pi, L=1.0, g=9.80665, damping=0.2)),
        ("hard-theta-1000pi", dict(theta=1000 * math.pi, L=2.0, g=9.80665, damping=0.0)),
        ("hard-theta-355", dict(theta=355.0, L=1.0, g=9.80665, damping=0.0)),
        ("hard-theta-1e22", dict(theta=1e22, L=1.0, g=9.80665, damping=0.0)),
        ("hard-theta-max-float", dict(theta=1.7976931348623157e308, L=1.0, g=9.80665, damping=0.1)),
        ("hard-theta-pi-over-2", dict(theta=math.pi / 2, L=3.0, g=9.80665, damping=0.05)),
    ])

    def o(m, v):
        return fl(Fr(1, 2) * Fr(m) * Fr(v) * Fr(v))
    run("kinetic", "physics.KineticEnergy", PMECH, "Precision: exact (multiplication only)", OPS(2, "m*v and (m*v)*v; 0.5*m is exact"), o, [
        ("typ-car", dict(m=1500.0, v=27.0)),
        ("typ-bullet", dict(m=4e-3, v=900.0)),
        ("hard-relativistic-speed-scale", dict(m=1.0, v=2.9e8)),
        ("hard-negative-speed", dict(m=0.123, v=-17.3)),
    ])

    def o(m, g, h):
        return fl(Fr(m) * Fr(g) * Fr(h))
    run("potential", "physics.PotentialEnergy", PMECH, "Precision: exact (multiplication only)", OPS(2, "m*g and (m*g)*h"), o, [
        ("typ-lift", dict(m=75.0, g=9.80665, h=12.0)),
        ("typ-below-reference", dict(m=3.3, g=9.80665, h=-44.4)),
        ("hard-orbit-height", dict(m=1e3, g=8.7, h=4e5)),
        ("hard-tiny", dict(m=1e-9, g=9.80665, h=1e-6)),
    ])

    # -- materials --------------------------------------------------------
    def o(E, epsilon):
        return fl(Fr(E) * Fr(epsilon))
    run("hooke", "physics.HookesLaw", PMAT, "Precision: exact (single multiplication)", EXACT1(), o, [
        ("typ-steel", dict(E=200e9, epsilon=1.2e-3)),
        ("typ-rubber", dict(E=0.01e9, epsilon=0.5)),
        ("hard-odd", dict(E=70e9, epsilon=3.3e-4)),
        ("hard-compressive", dict(E=1.7e11, epsilon=-2.9e-3)),
        ("hard-subnormal-result", dict(E=1e-160, epsilon=3.3e-150)),
    ])

    def o(s1, s2, s3):
        return fl(mp.sqrt(ex(dec("0.5")) * ((ex(s1) - ex(s2)) ** 2 + (ex(s2) - ex(s3)) ** 2 + (ex(s3) - ex(s1)) ** 2)))
    run("vonmises", "physics.VonMisesStress", PMAT, "Precision: limited by float64 sqrt (~15 significant digits)", D15(), o, [
        ("typ-uniaxial", dict(s1=250e6, s2=0.0, s3=0.0)),
        ("typ-triaxial", dict(s1=250e6, s2=100e6, s3=-50e6)),
        ("hard-near-hydrostatic", dict(s1=1e9 + 1, s2=1e9, s3=1e9)),
        ("hard-deep-pressure-small-shear", dict(s1=1e10 + 3.5, s2=1e10, s3=1e10 - 1.25)),
        ("hard-pure-shear", dict(s1=100e6, s2=-100e6, s3=0.0)),
        ("extreme-squares-overflow", dict(s1=1e160, s2=0.0, s3=0.0)),
        ("extreme-squares-underflow", dict(s1=1e-170, s2=0.0, s3=0.0)),
    ])

    def o(s1, s2, s3):
        return fl((Fr(max(s1, s2, s3)) - Fr(min(s1, s2, s3))) / 2)
    run("tresca", "physics.TrescaStress", PMAT, "Precision: exact (comparisons and arithmetic)", EXACT1(), o, [
        ("typ-triaxial", dict(s1=250e6, s2=100e6, s3=-50e6)),
        ("typ-uniaxial", dict(s1=0.0, s2=3.1e8, s3=0.0)),
        ("hard-nearly-equal", dict(s1=1e9 + 3.1, s2=1e9, s3=1e9 + 1.7)),
        ("hard-odd-difference", dict(s1=0.1, s2=-0.7, s3=0.3)),
        ("extreme-difference-overflow", dict(s1=1.5e308, s2=0.0, s3=-1.5e308)),
    ])

    def o(sigma, a, Y):
        return fl(ex(Y) * ex(sigma) * mp.sqrt(mp.pi * ex(a)))
    run("sif", "physics.StressIntensityFactor", PMAT, "Precision: limited by float64 sqrt (~15 significant digits)", D15(), o, [
        ("typ-edge-crack", dict(sigma=100e6, a=0.01, Y=1.12)),
        ("typ-infinite-plate", dict(sigma=250e6, a=2e-3, Y=1.0)),
        ("hard-micro-crack", dict(sigma=1e9, a=1e-9, Y=1.0)),
        ("hard-long-crack", dict(sigma=50e6, a=10.0, Y=1.0)),
    ])

    def o(E, gamma, a):
        return fl(mp.sqrt(2 * ex(E) * ex(gamma) / (mp.pi * ex(a))))
    run("griffith", "physics.GriffithCriterion", PMAT, "Precision: limited by float64 sqrt (~15 significant digits)", D15(), o, [
        ("typ-glass", dict(E=70e9, gamma=1.0, a=1e-6)),
        ("typ-steel", dict(E=200e9, gamma=2.0, a=1e-3)),
        ("hard-nanoscale", dict(E=1e12, gamma=1.0, a=1e-12)),
        ("hard-large-flaw", dict(E=1e9, gamma=0.1, a=100.0)),
    ])

    def o(C, m, deltaK):
        return fl(ex(C) * mp.power(ex(deltaK), ex(m)))
    run("paris", "physics.ParisLaw", PMAT, "Precision: limited by float64 pow", D15(), o, [
        ("typ-steel-m3", dict(C=6.9e-12, m=3.0, deltaK=20.0)),
        ("typ-fractional-m", dict(C=1e-11, m=3.1, deltaK=12.5)),
        ("typ-aluminium", dict(C=1e-10, m=3.5, deltaK=8.0)),
        ("hard-threshold-region", dict(C=1e-11, m=3.0, deltaK=1e-3)),
        ("hard-large-range", dict(C=1e-13, m=4.7, deltaK=4e1)),
        ("hard-huge-argument", dict(C=1e-100, m=3.37, deltaK=1e30)),
    ])

    def o(epsilonF, c, Nf):
        return fl(ex(epsilonF) * mp.power(2 * ex(Nf), ex(c)))
    run("coffin", "physics.CoffinManson", PMAT, "Precision: limited by float64 pow", D15(), o, [
        ("typ-1000-cycles", dict(epsilonF=0.5, c=-0.6, Nf=1000)),
        ("typ-low-cycle", dict(epsilonF=0.8, c=-0.7, Nf=10)),
        ("typ-exponent-minus-half", dict(epsilonF=0.3, c=-0.5, Nf=1000000)),
        ("hard-one-cycle", dict(epsilonF=0.9, c=-0.66, Nf=1)),
        ("hard-trillion-cycles", dict(epsilonF=0.4, c=-0.123456, Nf=1 << 40)),
    ])

    def o(A, Q, R, T, sigma, n):
        return fl(ex(A) * mp.power(ex(sigma), ex(n)) * mp.exp(-ex(Q) / (ex(R) * ex(T))))
    gas = float(R_GAS)

    def t_creep(exponent):
        """The temperature, with Q/(R T) near `exponent`, at which float64 rounds that exponent worst.
        The exponent is rounded twice (R*T, then the quotient) and the exponential multiplies
        that relative error by its own size."""
        return worst_rounded_partner(
            2.7e5, 2.7e5 / (gas * exponent),
            lambda Qv, T: Qv / (gas * T),
            lambda Qv, T: Fr(Qv) / (Fr(gas) * Fr(T)))
    run("creep", "physics.CreepArrhenius", PMAT, "Precision: limited by float64 exp and pow", D15(), o, [
        ("typ-low-activation", dict(A=1e-30, Q=2e4, R=gas, T=300.0, sigma=1e8, n=4.0)),
        ("typ-steel-800k", dict(A=1e-30, Q=2e5, R=gas, T=800.0, sigma=1e8, n=4.0)),
        ("hard-activation-ratio-40", dict(A=1e-30, Q=3e5, R=gas, T=900.0, sigma=1e8, n=4.5)),
        ("hard-activation-ratio-108", dict(A=1e-30, Q=2.7e5, R=gas, T=300.0, sigma=1e8, n=4.0)),
        ("hard-activation-ratio-258-worst-rounding", dict(A=1e-30, Q=2.7e5, R=gas, T=t_creep(258.0), sigma=1e8, n=3.0)),
        ("hard-activation-ratio-513-worst-rounding", dict(A=1e-30, Q=2.7e5, R=gas, T=t_creep(513.0), sigma=1e8, n=3.0)),
        # exp(-738) is subnormal (about 250 grains) although 1e300*exp(-738) is an ordinary number
        ("extreme-exp-subnormal-before-the-product", dict(A=1e300, Q=2.7e5, R=gas, T=44.0, sigma=1.0, n=3.0)),
    ])

    def o(Vf, Ef, Em):
        return fl(Fr(Vf) * Fr(Ef) + (1 - Fr(Vf)) * Fr(Em))
    run("mixture", "physics.CompositeMixture", PMAT, "Precision: exact (arithmetic only)",
        OPS(4, "Vf*Ef, 1-Vf, *Em, the sum"), o, [
            ("typ-carbon-epoxy", dict(Vf=0.6, Ef=230e9, Em=3.5e9)),
            ("typ-glass-polyester", dict(Vf=0.3, Ef=70e9, Em=2.5e9)),
            ("hard-dilute-fibre", dict(Vf=1e-17, Ef=230e9, Em=3.5e9)),
            ("hard-almost-pure-fibre", dict(Vf=0.9999999999999999, Ef=230e9, Em=3.5e9)),
            ("hard-pure-matrix", dict(Vf=0.0, Ef=230e9, Em=3.5e9)),
        ])

    def o(E, I, L, K):
        return fl(mp.pi ** 2 * ex(E) * ex(I) / (ex(K) * ex(L)) ** 2)
    run("euler", "physics.EulerBuckling", PMAT, "Precision: limited by float64 representation of pi", D15(), o, [
        ("typ-steel-column", dict(E=200e9, I=8.33e-6, L=3.0, K=1.0)),
        ("typ-fixed-fixed", dict(E=70e9, I=1e-8, L=1.5, K=0.5)),
        ("hard-fixed-free-long", dict(E=1.1e11, I=1e-3, L=100.0, K=2.0)),
        ("hard-microbeam", dict(E=1.2e11, I=1e-24, L=1e-6, K=1.0)),
    ])

    def o(P, L, E, I):
        return fl(Fr(P) * Fr(L) ** 3 / (48 * Fr(E) * Fr(I)))
    run("beam", "physics.BeamDeflection", PMAT, "Precision: exact (arithmetic only)",
        OPS(6, "P*L, *L, *L, 48*E, *I, the quotient"), o, [
            ("typ-steel-beam", dict(P=10e3, L=4.0, E=200e9, I=8.33e-6)),
            ("typ-timber", dict(P=2e3, L=3.0, E=1.1e10, I=2.1e-4)),
            ("hard-microbeam", dict(P=1.0, L=1e-4, E=1.7e11, I=1e-20)),
            ("hard-bridge", dict(P=1e7, L=100.0, E=1e10, I=1e-3)),
        ])

    # -- thermodynamics ---------------------------------------------------
    def o(n, T, V):
        return fl(Fr(n) * R_GAS * Fr(T) / Fr(V))
    run("idealgas", "physics.IdealGas", PTHERM, "Precision: limited by float64 representation of R (~15 significant digits)", D15(), o, [
        ("typ-stp", dict(n=1.0, T=273.15, V=0.0224)),
        ("typ-tank", dict(n=2.5, T=300.0, V=0.05)),
        ("hard-micro-sample", dict(n=1e-12, T=1e-3, V=1e-9)),
        ("hard-stellar", dict(n=1e30, T=1.5e7, V=1e20)),
    ])

    def o(T, A, emissivity):
        return fl(ex(emissivity) * SIGMA_SB * ex(A) * ex(T) ** 4)
    run("stefan", "physics.StefanBoltzmann", PTHERM, "Precision: limited by float64 pow and sigma constant", D15(), o, [
        ("typ-sun", dict(T=5772.0, A=6.087e18, emissivity=1.0)),
        ("typ-room-radiator", dict(T=300.0, A=1.0, emissivity=0.95)),
        ("hard-cosmic-background", dict(T=2.725, A=1e3, emissivity=1.0)),
        ("hard-fusion-plasma", dict(T=1e8, A=1.0, emissivity=1.0)),
        ("hard-tiny", dict(T=1e-3, A=1e-6, emissivity=0.5)),
    ])

    def o(Th, Tc):
        return fl(1 - Fr(Tc) / Fr(Th))
    run("carnot", "physics.CarnotEfficiency", PTHERM, "Precision: exact (single division and subtraction)",
        OPS(2, "Tc/Th and 1-Tc/Th"), o, [
            ("typ-600-300", dict(Th=600.0, Tc=300.0)),
            ("typ-500-300", dict(Th=500.0, Tc=300.0)),
            ("typ-steam-plant", dict(Th=811.15, Tc=303.15)),
            ("hard-ocean-thermal", dict(Th=298.15, Tc=278.15)),
            ("hard-gradient-1k", dict(Th=300.0, Tc=299.0)),
            ("hard-gradient-100mk", dict(Th=300.0, Tc=299.9)),
            ("hard-gradient-1uk", dict(Th=300.0, Tc=299.999999)),
            ("hard-absolute-zero-cold", dict(Th=300.0, Tc=0.0)),
        ])

    def o(k, A, dTdx):
        return fl(-Fr(k) * Fr(A) * Fr(dTdx))
    run("fourier", "physics.FourierHeatConduction", PTHERM, "Precision: exact (multiplication only)", OPS(2, "k*A, *dTdx; the negation is exact"), o, [
        ("typ-copper-rod", dict(k=401.0, A=1e-4, dTdx=-50.0)),
        ("typ-wall", dict(k=0.04, A=12.0, dTdx=-25.0)),
        ("hard-micro", dict(k=1e-3, A=1e-12, dTdx=3.3e6)),
        ("hard-steep", dict(k=2.3e3, A=0.77, dTdx=-1.1e9)),
    ])

    def o(h, A, Ts, Tinf):
        return fl(Fr(h) * Fr(A) * (Fr(Ts) - Fr(Tinf)))
    run("cooling", "physics.NewtonCooling", PTHERM, "Precision: exact (multiplication only)", OPS(3, "h*A, Ts-Tinf, the product"), o, [
        ("typ-heated-plate", dict(h=10.0, A=2.0, Ts=350.0, Tinf=293.15)),
        ("typ-celsius", dict(h=25.0, A=0.5, Ts=20.1, Tinf=20.0)),
        ("hard-equal-to-1e-7", dict(h=10.0, A=2.0, Ts=300.0000001, Tinf=300.0)),
        ("hard-sub-ulp-gap", dict(h=7.0, A=3.0, Ts=1.0000000000000002, Tinf=1.0)),
    ])

    def o(L0, alpha, deltaT):
        return fl(Fr(L0) * Fr(alpha) * Fr(deltaT))
    run("expansion", "physics.ThermalExpansion", PTHERM, "Precision: exact (multiplication only)", OPS(2, "L0*alpha, *deltaT"), o, [
        ("typ-steel-rail", dict(L0=1000.0, alpha=12e-6, deltaT=40.0)),
        ("typ-aluminium-cooling", dict(L0=2.5, alpha=23e-6, deltaT=-60.0)),
        ("hard-invar", dict(L0=1e-3, alpha=1.2e-6, deltaT=1e-3)),
        ("hard-large-span", dict(L0=1e5, alpha=1.7e-5, deltaT=800.0)),
    ])

    # HeatEquation1DStep: the documented orders, measured on the discrete
    # sine mode u_i = sin(pi x_i), whose one-step factor is known exactly.
    heat = "Precision: O(dt) in time, O(dx^2) in space (first-order explicit scheme)"
    add("heat1d-order-time", "physics.HeatEquation1DStep.order_dt", dict(n=21, alpha=1.0, dt=1e-3), 1.0, 0.05, "abs",
        f"{PTHERM} HeatEquation1DStep: '{heat}' -> the truncation error per unit time, (g - exp(-alpha pi^2 dt))/dt for the "
        "measured one-step factor g of the sine mode, must be linear in dt: observed order log2((t(dt)-t(dt/2))/(t(dt/2)-t(dt/4))) "
        "within 0.05 of 1; the exact solution exp(-alpha pi^2 t) sin(pi x) is the reference")
    add("heat1d-order-space", "physics.HeatEquation1DStep.order_dx", dict(n=21, alpha=1.0, dt=1e-4), 2.0, 0.05, "abs",
        f"{PTHERM} HeatEquation1DStep: '{heat}' -> after Richardson extrapolation in dt removes the time error, the truncation "
        "error per unit time must fall by 4 when dx is halved: observed order within 0.05 of 2; the exact solution "
        "exp(-alpha pi^2 t) sin(pi x) is the reference")

    # -- optics -----------------------------------------------------------
    def o(n1, n2, thetaI):
        return fl(mp.asin(ex(n1) / ex(n2) * mp.sin(ex(thetaI))))
    snell = "Precision: limited by float64 trig (~15 significant digits)"
    run("snell", "physics.SnellRefraction", POPT, snell, D15(), o, [
        ("typ-air-to-glass-30deg", dict(n1=1.0, n2=1.5, thetaI=0.5235987755982988)),
        ("typ-water-to-air-20deg", dict(n1=1.333, n2=1.0, thetaI=0.3490658503988659)),
        ("hard-tiny-angle", dict(n1=1.0, n2=2.4, thetaI=1e-10)),
        ("hard-matched-indices-steep", dict(n1=1.3, n2=1.3, thetaI=1.5)),
        ("hard-grazing-incidence", dict(n1=1.0, n2=1.5, thetaI=math.pi / 2)),
        ("cond-near-critical-1e-9", dict(n1=1.5, n2=1.0, thetaI=critical_theta(1.5, 1.0, 1e-9))),
        ("cond-near-critical-1e-12", dict(n1=1.5, n2=1.0, thetaI=critical_theta(1.5, 1.0, 1e-12))),
        ("cond-near-critical-1e-14", dict(n1=1.5, n2=1.0, thetaI=critical_theta(1.5, 1.0, 1e-14))),
    ])

    def o(n1, n2, thetaI):
        return fresnel_exact(n1, n2, thetaI)
    fres = "Precision: limited by float64 trig (~15 significant digits)"
    run("fresnel", "physics.FresnelReflectance", POPT, fres, D15(), o, [
        ("typ-normal-glass", dict(n1=1.0, n2=1.5, thetaI=0.0)),
        ("typ-45deg-glass", dict(n1=1.0, n2=1.5, thetaI=math.pi / 4)),
        ("typ-water", dict(n1=1.0, n2=1.333, thetaI=0.8)),
        ("hard-brewster", dict(n1=1.0, n2=1.5, thetaI=math.atan(1.5))),
        ("hard-grazing", dict(n1=1.0, n2=1.5, thetaI=math.pi / 2)),
        ("hard-total-internal-reflection", dict(n1=1.5, n2=1.0, thetaI=1.0)),
        ("typ-crown-to-flint", dict(n1=1.5168, n2=1.62, thetaI=0.6)),
        ("hard-index-matched-1e-4", dict(n1=1.5, n2=1.5001, thetaI=0.7)),
        ("hard-index-matched-1e-6", dict(n1=1.5, n2=1.500001, thetaI=0.7)),
        ("hard-index-matched-1e-6-steep", dict(n1=1.0, n2=1.000001, thetaI=1.2)),
        ("cond-near-critical-1e-12", dict(n1=1.5, n2=1.0, thetaI=critical_theta(1.5, 1.0, 1e-12))),
        ("cond-near-critical-1e-14", dict(n1=1.5, n2=1.0, thetaI=critical_theta(1.5, 1.0, 1e-14))),
    ])

    def x_beer(mu, depth):
        """The path length, with mu*x near `depth`, at which float64 rounds the product mu*x worst;
        exp then multiplies that relative error by the optical depth."""
        return worst_rounded_partner(
            mu, depth / mu,
            lambda m, x: m * x,
            lambda m, x: Fr(m) * Fr(x))

    def o(I0, mu, x):
        return fl(ex(I0) * mp.exp(-ex(mu) * ex(x)))
    run("beer", "physics.BeerLambertLaw", POPT, "Precision: limited by float64 exp (~15 significant digits)", D15(), o, [
        ("typ-one-length", dict(I0=100.0, mu=0.1, x=10.0)),
        ("typ-optical-depth-5", dict(I0=1e3, mu=2.0, x=2.5)),
        ("hard-thin-1e-20", dict(I0=1.0, mu=1e-10, x=1e-10)),
        ("hard-optical-depth-30", dict(I0=1e6, mu=0.3, x=100.0)),
        ("hard-optical-depth-100", dict(I0=1e10, mu=0.7, x=142.85714285714286)),
        ("hard-optical-depth-258-worst-rounding", dict(I0=1e30, mu=1.7, x=x_beer(1.7, 258.0))),
        ("hard-optical-depth-513-worst-rounding", dict(I0=1e10, mu=3.3, x=x_beer(3.3, 513.0))),
        ("hard-no-attenuation", dict(I0=123.456, mu=0.0, x=1e3)),
        # exp(-740) is subnormal (about 90 grains) although 1e300*exp(-740) is an ordinary number
        ("extreme-exp-subnormal-before-the-product", dict(I0=1e300, mu=1.0, x=740.0)),
    ])


# ---------------------------------------------------------------------------
# color
# ---------------------------------------------------------------------------

CSP = "color/spaces.go"
CAD = "color/adapt.go"
CDF = "color/difference.go"
CSG = "color/spectral.go"

# The matrices exactly as the library writes them (decimal literals).
M_RGB2XYZ = [[dec("0.4124564"), dec("0.3575761"), dec("0.1804375")],
             [dec("0.2126729"), dec("0.7151522"), dec("0.0721750")],
             [dec("0.0193339"), dec("0.1191920"), dec("0.9503041")]]
M_BRADFORD = [[dec("0.8951000"), dec("0.2664000"), dec("-0.1614000")],
              [dec("-0.7502000"), dec("1.7135000"), dec("0.0367000")],
              [dec("0.0389000"), dec("-0.0685000"), dec("1.0296000")]]


def mat(rows):
    return mp.matrix([[ex(v) for v in row] for row in rows])


def matvec(m, v):
    out = m * mp.matrix([ex(x) for x in v])
    return [out[i] for i in range(3)]


def lab_f(t):
    delta = mp.mpf(6) / 29
    if t > delta ** 3:
        return mp.cbrt(t)
    return t / (3 * delta ** 2) + mp.mpf(4) / 29


def lab_finv(t):
    delta = mp.mpf(6) / 29
    if t > delta:
        return t ** 3
    return 3 * delta ** 2 * (t - mp.mpf(4) / 29)


def xyz_to_lab_exact(X, Y, Z, Xn, Yn, Zn):
    fx, fy, fz = lab_f(ex(X) / ex(Xn)), lab_f(ex(Y) / ex(Yn)), lab_f(ex(Z) / ex(Zn))
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def lab_to_xyz_exact(L, a, b, Xn, Yn, Zn):
    fy = (ex(L) + 16) / 116
    fx = ex(a) / 500 + fy
    fz = fy - ex(b) / 200
    return ex(Xn) * lab_finv(fx), ex(Yn) * lab_finv(fy), ex(Zn) * lab_finv(fz)


def hsv_exact(r, g, b):
    r, g, b = Fr(r), Fr(g), Fr(b)
    mx, mn = max(r, g, b), min(r, g, b)
    delta = mx - mn
    if mx == 0:
        return Fr(0), Fr(0), mx
    s = delta / mx
    if delta == 0:
        return Fr(0), s, mx
    if mx == r:
        h = 60 * (((g - b) / delta) % 6)
    elif mx == g:
        h = 60 * ((b - r) / delta + 2)
    else:
        h = 60 * ((r - g) / delta + 4)
    return h, s, mx


def hsv_to_rgb_exact(h, s, v):
    h, s, v = Fr(h), Fr(s), Fr(v)
    c = v * s
    hp = h / 60
    x = c * (1 - abs(hp % 2 - 1))
    m = v - c
    if hp < 1:
        rgb = (c, x, Fr(0))
    elif hp < 2:
        rgb = (x, c, Fr(0))
    elif hp < 3:
        rgb = (Fr(0), c, x)
    elif hp < 4:
        rgb = (Fr(0), x, c)
    elif hp < 5:
        rgb = (x, Fr(0), c)
    else:
        rgb = (c, Fr(0), x)
    return tuple(p + m for p in rgb)


def hue_deg(ap, b):
    if ap == 0 and b == 0:
        return mp.mpf(0)
    h = mp.degrees(mp.atan2(b, ap))
    return h + 360 if h < 0 else h


def de2000_exact(L1, a1, b1, L2, a2, b2):
    """CIEDE2000 as in Sharma, Wu, Dalal (2005), eqs. (2)-(22), kL = kC = kH = 1."""
    L1, a1, b1, L2, a2, b2 = [ex(v) for v in (L1, a1, b1, L2, a2, b2)]
    c1, c2 = mp.sqrt(a1 ** 2 + b1 ** 2), mp.sqrt(a2 ** 2 + b2 ** 2)
    cb = (c1 + c2) / 2
    c7 = cb ** 7
    g = (1 - mp.sqrt(c7 / (c7 + mp.mpf(25) ** 7))) / 2
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = mp.sqrt(a1p ** 2 + b1 ** 2), mp.sqrt(a2p ** 2 + b2 ** 2)
    h1p, h2p = hue_deg(a1p, b1), hue_deg(a2p, b2)
    dLp, dCp = L2 - L1, c2p - c1p
    tie = mp.mpf(10) ** -40  # an exact +-180 hue difference is decided by the rules for equality
    if c1p * c2p == 0:
        dhp = mp.mpf(0)
    else:
        dhp = h2p - h1p
        if dhp > 180 + tie:
            dhp -= 360
        elif dhp < -180 - tie:
            dhp += 360
        elif abs(abs(dhp) - 180) <= tie:
            dhp = mp.mpf(180) if dhp > 0 else mp.mpf(-180)
    dHp = 2 * mp.sqrt(c1p * c2p) * mp.sin(mp.radians(dhp / 2))
    Lb, cbp = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180 + tie:
        hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p - 360) / 2
    T = (1 - mp.mpf("0.17") * mp.cos(mp.radians(hbp - 30)) + mp.mpf("0.24") * mp.cos(mp.radians(2 * hbp))
         + mp.mpf("0.32") * mp.cos(mp.radians(3 * hbp + 6)) - mp.mpf("0.20") * mp.cos(mp.radians(4 * hbp - 63)))
    dtheta = 30 * mp.exp(-(((hbp - 275) / 25) ** 2))
    c7p = cbp ** 7
    rc = 2 * mp.sqrt(c7p / (c7p + mp.mpf(25) ** 7))
    sl = 1 + mp.mpf("0.015") * (Lb - 50) ** 2 / mp.sqrt(20 + (Lb - 50) ** 2)
    sc = 1 + mp.mpf("0.045") * cbp
    sh = 1 + mp.mpf("0.015") * cbp * T
    rt = -mp.sin(mp.radians(2 * dtheta)) * rc
    tl, tc, th = dLp / sl, dCp / sc, dHp / sh
    return mp.sqrt(tl ** 2 + tc ** 2 + th ** 2 + rt * tc * th)


# The 34 test pairs of Sharma, Wu and Dalal (2005), Table 1, with the published CIEDE2000 values.
SHARMA = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -1.1848, -84.8006), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -0.9009, -85.5211), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, -1.0, 2.0), (50.0, 0.0, 0.0), 2.3669),
    ((50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0009), 7.1792),
    ((50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0010), 7.1792),
    ((50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0011), 7.2195),
    ((50.0, 2.4900, -0.0010), (50.0, -2.4900, 0.0012), 7.2195),
    ((50.0, -0.0010, 2.4900), (50.0, 0.0009, -2.4900), 4.8045),
    ((50.0, -0.0010, 2.4900), (50.0, 0.0010, -2.4900), 4.8045),
    ((50.0, -0.0010, 2.4900), (50.0, 0.0011, -2.4900), 4.7461),
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((50.0, 2.5, 0.0), (56.0, -27.0, -3.0), 31.9030),
    ((50.0, 2.5, 0.0), (58.0, 24.0, 15.0), 19.4535),
    ((50.0, 2.5, 0.0), (50.0, 3.1736, 0.5854), 1.0000),
    ((50.0, 2.5, 0.0), (50.0, 3.2972, 0.0), 1.0000),
    ((50.0, 2.5, 0.0), (50.0, 1.8634, 0.5757), 1.0000),
    ((50.0, 2.5, 0.0), (50.0, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


def cie_table():
    """The CIE 1931 5 nm table, read from the library source (the formula's own data)."""
    import re
    rows = []
    with open(os.path.join("color", "spectral.go"), encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"\s*\{(\d+), ([0-9.]+), ([0-9.]+), ([0-9.]+)\},", line)
            if m:
                rows.append((int(m.group(1)), dec(m.group(2)), dec(m.group(3)), dec(m.group(4))))
    assert len(rows) == 81, len(rows)
    return rows


def blackbody_exact(T):
    """X/Y, 1, Z/Y of the documented 81-term Planck sum, exactly."""
    h, c, k = ex(dec("6.62607015e-34")), ex(299792458), ex(dec("1.380649e-23"))
    sx = sy = sz = mp.mpf(0)
    for nm, xb, yb, zb in cie_table():
        lam = ex(Fr(nm)) * ex(dec("1e-9"))
        planck = 1 / (lam ** 5 * (mp.exp(h * c / (lam * k * ex(T))) - 1))
        sx += planck * ex(xb)
        sy += planck * ex(yb)
        sz += planck * ex(zb)
    return sx / sy, mp.mpf(1), sz / sy


def color():
    rgb = "Precision: limited by float64 matrix multiplication"
    inv_fwd = mat(M_RGB2XYZ) ** -1

    # LinearRGBToXYZ with the matrix as written
    def xyz_k(i):
        def o(r, g, b):
            return fl(sum((M_RGB2XYZ[i][j] * Fr(c) for j, c in enumerate((r, g, b))), Fr(0)))
        return o
    items = [
        ("typ-white", dict(r=1.0, g=1.0, b=1.0)),
        ("typ-mid-gray", dict(r=0.5, g=0.5, b=0.5)),
        ("typ-saturated", dict(r=0.9, g=0.1, b=0.4)),
        ("hard-dark", dict(r=1e-5, g=2e-5, b=3e-5)),
        ("hard-near-zero-channel", dict(r=1.0, g=1e-12, b=1e-12)),
    ]
    for i, nm in enumerate("XYZ"):
        run(f"rgb2xyz-{nm}", f"color.LinearRGBToXYZ.{nm}", CSP, rgb, D15(), xyz_k(i), items)

    # XYZToLinearRGB: the inverse of the forward matrix
    def rgb_k(i):
        def o(X, Y, Z):
            return fl(matvec(inv_fwd, (X, Y, Z))[i])
        return o
    items = [
        ("typ-d65-white", dict(X=0.95047, Y=1.0, Z=1.08883)),
        ("typ-mid-gray", dict(X=0.475235, Y=0.5, Z=0.544415)),
        ("typ-saturated", dict(X=0.2, Y=0.3, Z=0.4)),
        ("hard-out-of-gamut", dict(X=1.2, Y=0.2, Z=0.1)),
        ("hard-dark", dict(X=1e-4, Y=2e-4, Z=3e-4)),
    ]
    for i, nm in enumerate("rgb"):
        run(f"xyz2rgb-{nm}", f"color.XYZToLinearRGB.{nm}", CSP,
            "Precision: limited by float64 matrix multiplication (the formula is [R,G,B] = M^{-1} * [X,Y,Z])", D15(), rgb_k(i), items)

    # BradfordAdapt: M^{-1} diag(ratios) M, with M^{-1} the exact inverse of the Bradford matrix
    MB = mat(M_BRADFORD)
    MBi = MB ** -1
    D65, D50, A = (0.3127, 0.3290), (0.3457, 0.3585), (0.4476, 0.4074)

    def wp(x, y):
        x, y = ex(x), ex(y)
        return [x / y, mp.mpf(1), (1 - x - y) / y]

    def bradford_k(i):
        def o(X, Y, Z, srcWPx, srcWPy, dstWPx, dstWPy):
            ratios = [d / s for d, s in zip(matvec(MB, wp(dstWPx, dstWPy)), matvec(MB, wp(srcWPx, srcWPy)))]
            cone = [c * r for c, r in zip(matvec(MB, (X, Y, Z)), ratios)]
            return fl((MBi * mp.matrix(cone))[i])
        return o

    def args(X, Y, Z, s, d):
        return dict(X=X, Y=Y, Z=Z, srcWPx=s[0], srcWPy=s[1], dstWPx=d[0], dstWPy=d[1])
    items = [
        ("typ-d65-to-d50-orange", args(0.5, 0.4, 0.1, D65, D50)),
        ("typ-d50-to-d65-teal", args(0.2, 0.3, 0.5, D50, D65)),
        ("typ-illuminant-a-to-d65", args(0.4, 0.4, 0.3, A, D65)),
        ("typ-d65-to-d50-white", args(0.95047, 1.0, 1.08883, D65, D50)),
        ("hard-same-white-identity", args(0.2, 0.3, 0.4, D65, D65)),
        ("hard-same-white-identity-2", args(0.7, 0.2, 0.05, D50, D50)),
        ("hard-dark", args(1e-4, 2e-4, 3e-4, D65, D50)),
    ]
    for i, nm in enumerate("XYZ"):
        run(f"bradford-{nm}", f"color.BradfordAdapt.{nm}", CAD, "Precision: limited by float64 matrix operations", D15(), bradford_k(i), items)

    # XYZToLab
    white = dict(Xn=0.95047, Yn=1.0, Zn=1.08883)
    lab_items = [
        ("typ-orange", dict(X=0.4, Y=0.3, Z=0.05, **white)),
        ("typ-blue", dict(X=0.15, Y=0.1, Z=0.6, **white)),
        ("typ-mid", dict(X=0.3, Y=0.35, Z=0.25, **white)),
        ("hard-hdr-bright", dict(X=95.0, Y=80.0, Z=30.0, **white)),
        ("hard-dark-linear-branch-1e-3", dict(X=0.95047e-3, Y=1.1e-3, Z=1.08883e-3, **white)),
        ("hard-dark-linear-branch-1e-6", dict(X=0.9e-6, Y=1.2e-6, Z=1.0e-6, **white)),
        ("hard-dark-linear-branch-1e-9", dict(X=0.8e-9, Y=1.3e-9, Z=1.0e-9, **white)),
        ("hard-branch-point-Y-below", dict(X=0.3, Y=0.00885, Z=0.2, **white)),
        ("hard-branch-point-Y-above", dict(X=0.3, Y=0.00886, Z=0.2, **white)),
        ("hard-branch-point-X-below", dict(X=0.00841, Y=0.5, Z=0.4, **white)),
        ("hard-branch-point-X-above", dict(X=0.00842, Y=0.5, Z=0.4, **white)),
        ("hard-near-neutral-1e-9", dict(X=0.5 * 0.95047 * (1 + 1e-9), Y=0.5, Z=0.5 * 1.08883, **white)),
        ("hard-near-neutral-1e-12", dict(X=0.5 * 0.95047 * (1 + 1e-12), Y=0.5, Z=0.5 * 1.08883 * (1 - 1e-12), **white)),
    ]
    for i, nm in enumerate(("L", "a", "b")):
        def o(X, Y, Z, Xn, Yn, Zn, i=i):
            return fl(xyz_to_lab_exact(X, Y, Z, Xn, Yn, Zn)[i])
        run(f"xyz2lab-{nm}", f"color.XYZToLab.{nm}", CSP, "Precision: limited by float64 cube root", D15(), o, lab_items)

    # LabToXYZ
    inv_items = [
        ("typ-orange", dict(L=50.0, a=20.0, b=-30.0, **white)),
        ("typ-green", dict(L=80.0, a=-40.0, b=60.0, **white)),
        ("typ-near-white", dict(L=99.0, a=1.0, b=-2.0, **white)),
        ("hard-large-chroma", dict(L=60.0, a=127.0, b=-128.0, **white)),
        ("hard-dark-L-1", dict(L=1.0, a=0.5, b=-0.5, **white)),
        ("hard-dark-L-0.01", dict(L=0.01, a=0.001, b=-0.001, **white)),
        ("hard-dark-L-1e-4", dict(L=1e-4, a=1e-5, b=-1e-5, **white)),
        ("hard-just-above-L8", dict(L=8.01, a=3.0, b=-3.0, **white)),
        ("hard-just-below-L8", dict(L=7.99, a=3.0, b=-3.0, **white)),
    ]
    for i, nm in enumerate("XYZ"):
        def o(L, a, b, Xn, Yn, Zn, i=i):
            return fl(lab_to_xyz_exact(L, a, b, Xn, Yn, Zn)[i])
        run(f"lab2xyz-{nm}", f"color.LabToXYZ.{nm}", CSP, "Precision: limited by float64 cube", D15(), o, inv_items)

    # RGBToHSV
    hsv_items = [
        ("typ-teal", dict(r=0.2, g=0.4, b=0.6)),
        ("typ-pink", dict(r=0.9, g=0.1, b=0.4)),
        ("typ-olive", dict(r=0.5, g=0.5, b=0.1)),
        ("hard-near-gray", dict(r=0.5, g=0.5, b=0.5000000000000001)),
        ("hard-hue-near-360", dict(r=1.0, g=0.5, b=0.5000000000000001)),
        ("hard-hue-near-zero", dict(r=1.0, g=0.5000000000000001, b=0.5)),
        ("hard-tiny-values", dict(r=1e-300, g=2e-300, b=3e-300)),
        ("hard-sector-boundary-yellow", dict(r=1.0, g=1.0, b=0.0)),
        ("hard-dark-saturated", dict(r=1e-9, g=3e-10, b=1e-10)),
    ]
    hsv_rule = OPS(6, "delta, g-b, the quotient, *60, the wrap +360 and one more")
    for i, nm in enumerate(("h", "s")):
        def o(r, g, b, i=i):
            return fl(hsv_exact(r, g, b)[i])
        run(f"rgb2hsv-{nm}", f"color.RGBToHSV.{nm}", CSP, "Precision: exact to float64", hsv_rule, o, hsv_items)

    # RGBToHSV: "H is in [0,360)". A hue just below 360 degrees is a float64 that rounds to 360.
    for name, a in [("typ-teal", dict(r=0.2, g=0.4, b=0.6)),
                    ("hard-red-with-tiny-blue-1e-17", dict(r=1.0, g=0.0, b=1e-17)),
                    ("hard-red-with-blue-one-ulp-above-green", dict(r=1.0, g=0.5, b=0.5000000000000001))]:
        h_true = hsv_exact(a["r"], a["g"], a["b"])[0]
        assert 0 <= h_true < 360
        add(f"rgb2hsv-range-{name}", "color.RGBToHSV.h_below_360", a, 1.0, 0.0, "abs",
            f"{CSP} RGBToHSV: 'converts RGB [0,1] to HSV where H is in [0,360)' -> the exact hue is {float(h_true)!r} < 360, "
            "so the returned H must be below 360 (the evaluator returns 1 when it is and 0 when it is not)")

    # HSVToRGB
    # h/60 is rounded, then 1-|mod(h/60,2)-1| cancels: the hue, just above a primary at 240, at which h/60 rounds worst
    hue_worst = worst_rounded_partner(60.0, 245.0, lambda d, h: h / d, lambda d, h: Fr(h) / Fr(d))
    back_items = [
        ("typ-orange", dict(h=30.0, s=0.6, v=0.8)),
        ("typ-blue", dict(h=200.0, s=0.9, v=0.5)),
        ("typ-lilac", dict(h=300.0, s=0.3, v=0.9)),
        ("hard-sector-boundary", dict(h=60.0, s=0.5, v=0.7)),
        ("hard-last-sector-edge", dict(h=359.99999999999994, s=0.8, v=0.6)),
        ("hard-nearly-saturated-1e-5", dict(h=30.0, s=0.99999, v=0.7)),
        ("hard-nearly-saturated-1e-9", dict(h=130.0, s=0.999999999, v=0.7)),
        ("hard-nearly-saturated-1e-13", dict(h=250.0, s=0.9999999999999, v=0.9)),
        ("hard-near-primary-red-0.001deg", dict(h=0.001, s=1.0, v=1.0)),
        ("hard-near-primary-green-120.001deg", dict(h=120.001, s=1.0, v=1.0)),
        ("hard-near-primary-blue-239.999deg", dict(h=239.999, s=1.0, v=1.0)),
        ("hard-near-primary-blue-240.001deg", dict(h=240.001, s=1.0, v=1.0)),
        ("hard-hue-245-worst-rounding", dict(h=hue_worst, s=1.0, v=1.0)),
    ]
    back_rule = OPS(7, "v*s, h/60, mod-1, 1-, *c, v-c, +m")
    for i, nm in enumerate("rgb"):
        def o(h, s, v, i=i):
            return fl(hsv_to_rgb_exact(h, s, v)[i])
        thin = {"r": ("hard-nearly-saturated-1e-13",), "g": (), "b": ("hard-nearly-saturated-1e-9",)}[nm]
        run(f"hsv2rgb-{nm}", f"color.HSVToRGB.{nm}", CSP, "Precision: exact to float64", back_rule, o, back_items, skip=thin)

    # DeltaE76
    def o(L1, a1, b1, L2, a2, b2):
        return fl(mp.sqrt((ex(L1) - ex(L2)) ** 2 + (ex(a1) - ex(a2)) ** 2 + (ex(b1) - ex(b2)) ** 2))
    run("de76", "color.DeltaE76", CDF, "Precision: exact to float64 sqrt precision",
        OPS(4, "three differences, squares, sums and the sqrt: 3.5 roundings"), o, [
            ("typ-sharma-1", dict(L1=50.0, a1=2.6772, b1=-79.7751, L2=50.0, a2=0.0, b2=-82.7485)),
            ("typ-far", dict(L1=20.0, a1=60.0, b1=-40.0, L2=85.0, a2=-30.0, b2=70.0)),
            ("hard-near-identical-1e-9", dict(L1=50.0, a1=10.0, b1=-10.0, L2=50.000000001, a2=10.000000001, b2=-10.000000001)),
            ("hard-one-axis", dict(L1=50.0, a1=0.0, b1=0.0, L2=50.0, a2=0.0, b2=1e-7)),
            ("hard-large-values", dict(L1=1e6, a1=-1e6, b1=2e6, L2=-3e6, a2=4e6, b2=-5e6)),
            ("extreme-squares-overflow", dict(L1=1e160, a1=0.0, b1=0.0, L2=0.0, a2=0.0, b2=0.0)),
            ("extreme-squares-underflow", dict(L1=1e-170, a1=0.0, b1=0.0, L2=0.0, a2=0.0, b2=0.0)),
        ])

    # DeltaE2000: the oracle reproduces the published values first
    for k, (p1, p2, published) in enumerate(SHARMA, 1):
        got = float(de2000_exact(*p1, *p2))
        assert abs(got - published) < 6e-5, (k, got, published)
    de_rule = Rule(1e-10, "abs", "'typically 1e-10' read as an absolute error of 1e-10 on a value that spans 0..100")
    de_quote = "Precision: limited by float64 trigonometric functions; typically 1e-10"

    def o(L1, a1, b1, L2, a2, b2):
        return fl(de2000_exact(L1, a1, b1, L2, a2, b2))

    def pair(p1, p2):
        return dict(L1=p1[0], a1=p1[1], b1=p1[2], L2=p2[0], a2=p2[1], b2=p2[2])
    sharma_items = [(f"typ-sharma-{k:02d}", pair(p1, p2)) for k, (p1, p2, _) in enumerate(SHARMA, 1)]
    run("de2000", "color.DeltaE2000", CDF, de_quote, de_rule, o, sharma_items)
    run("de2000", "color.DeltaE2000", CDF, de_quote, de_rule, o, [
        ("hard-identical", pair((50.0, 20.0, 10.0), (50.0, 20.0, 10.0))),
        ("hard-achromatic-pair", pair((30.0, 0.0, 0.0), (80.0, 0.0, 0.0))),
        ("hard-one-achromatic", pair((50.0, 0.0, 0.0), (50.0, 30.0, -20.0))),
        ("hard-near-identical-1e-9", pair((50.0, 20.0, 10.0), (50.000000001, 20.000000001, 10.000000001))),
        ("hard-opposite-hue-axis", pair((50.0, 30.0, 0.0), (50.0, -30.0, 0.0))),
        # A pair (L, a, b), (L, -a, -b) has a hue difference of exactly 180 degrees, where the standard's
        # rules take the "<= 180" branch. The pairs below come from scanning 1,200 such pairs: 203 of them
        # return the other branch on the default and the GOAMD64=v3 builds alike.
        ("hard-opposite-hue-exact-a", pair((16.92, 59.28, -51.41), (16.92, -59.28, 51.41))),
        ("hard-opposite-hue-exact-b", pair((33.68, -59.8, 59.44), (33.68, 59.8, -59.44))),
        ("hard-opposite-hue-exact-c", pair((32.24, -59.69, 40.41), (32.24, 59.69, -40.41))),
        ("hard-opposite-hue-exact-d", pair((20.69, 58.24, -41.25), (20.69, -58.24, 41.25))),
        ("hard-opposite-hue-exact-lightness-differs", pair((77.703, -50.942, -23.899), (13.236, 50.942, 23.899))),
        ("hard-hue-wrap-359-to-1", pair((50.0, 40.0, -0.5), (50.0, 40.0, 0.5))),
        ("hard-blue-region-rotation", pair((40.0, 5.0, -60.0), (45.0, 10.0, -55.0))),
        ("hard-very-high-chroma", pair((50.0, 120.0, -90.0), (50.0, 118.0, -95.0))),
        ("hard-lightness-50-minimum-weight", pair((50.0, 25.0, 25.0), (50.0, 27.0, 22.0))),
    ])

    # BlackbodyToXYZ: the documented 81-term sum evaluated exactly
    bb_rule = Rule(1e-12, "rel", "'limited by 5nm integration step and tabulated data' means float64 arithmetic is not a limiting factor: "
                   "relative 1e-12 against the documented sum evaluated exactly")
    bb_items = [("typ-illuminant-a-2856k", dict(T=2856.0)), ("typ-daylight-6500k", dict(T=6500.0)),
                ("typ-candle-1000k", dict(T=1000.0)), ("typ-hot-star-25000k", dict(T=25000.0)),
                ("hard-1e6k", dict(T=1e6)), ("hard-1e8k", dict(T=1e8)), ("hard-cold-30k", dict(T=30.0))]
    for i, nm in enumerate("XZ"):
        if nm == "Z":  # at 30 K only the 680-780 nm terms survive and the table has no z-bar there: Z is 2e-56 and is dropped
            bb_items = [it for it in bb_items if it[0] != "hard-cold-30k"]
        def o(T, i=i):
            return fl(blackbody_exact(T)[0 if i == 0 else 2])
        run(f"blackbody-{nm}", f"color.BlackbodyToXYZ.{nm}", CSG,
            "Precision: limited by 5nm integration step and tabulated data", bb_rule, o, bb_items)

    def o(T):
        return 1.0
    y_rule = Rule(0.0, "rel", "'The result is normalized so that Y=1' for every valid T > 0: Y must equal 1")
    run("blackbody-Y", "color.BlackbodyToXYZ.Y", CSG, "The result is normalized so that Y=1 for a perfect white diffuser; Valid range: T > 0 (Kelvin)", y_rule, o, [
        ("typ-6500k", dict(T=6500.0)), ("hard-cold-30k", dict(T=30.0)),
        ("extreme-below-26k-25k", dict(T=25.0)), ("extreme-below-26k-10k", dict(T=10.0)), ("extreme-below-26k-0.1k", dict(T=0.1))])

    # ToneMapReinhard: out = v (1 + v / wp^2) / (1 + v)
    def tm(v, wp):
        return fl(Fr(v) * (1 + Fr(v) / Fr(wp) ** 2) / (1 + Fr(v)))
    tm_items = [
        ("typ-mid-gray", dict(v=0.18, wp=4.0)),
        ("typ-bright", dict(v=3.0, wp=4.0)),
        ("typ-hdr", dict(v=100.0, wp=10.0)),
        ("hard-at-white-point-3", dict(v=3.0, wp=3.0)),
        ("hard-at-white-point-7", dict(v=7.0, wp=7.0)),
        ("hard-at-white-point-10", dict(v=10.0, wp=10.0)),
        ("hard-tiny", dict(v=1e-12, wp=2.0)),
        ("hard-huge-white-point", dict(v=0.5, wp=1e150)),
        ("hard-above-white", dict(v=1e12, wp=1e3)),
        ("hard-zero", dict(v=0.0, wp=4.0)),
        ("extreme-v-squared-overflow", dict(v=1e155, wp=1.0)),
    ]
    tm_rule = OPS(6, "wp*wp, v/wp2, 1+, v*(...), 1+v, the quotient")
    for k, nm in enumerate("rgb"):
        # one channel carries the case, the other two a fixed value
        def o(r, g, b, wp, k=k):
            return tm((r, g, b)[k], wp)
        items = []
        for name, a in tm_items:
            if k > 0 and not name.startswith("typ-"):
                continue
            args = dict(r=0.25, g=0.25, b=0.25, wp=a["wp"])
            args["rgb"[k]] = a["v"]
            items.append((name, args))
        run(f"tonemap-{nm}", f"color.ToneMapReinhard.{nm}", CSG,
            "Precision: exact to float64", tm_rule, o, items)


def main():
    acoustics()
    em()
    fluids()
    physics()
    color()
    doc = {
        "_comment": "Generated by tools/stressgolden/applied.py; do not edit by hand. See precision_applied_test.go.",
        "generator": {"mpmath": mp.__version__, "dps": mp.mp.dps},
        "cases": CASES,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"wrote {OUT}: {len(CASES)} cases")


if __name__ == "__main__":
    main()
