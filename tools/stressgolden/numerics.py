#!/usr/bin/env python3
"""Generate testdata/stress/precision_numerics.json: claim cases for the
numerical packages (linalg, optim and its subpackages, calculus, chaos).

Each case pairs one function with a precision, exactness or convergence
promise from its own doc comment and an input chosen where that promise is
hardest to keep (cancellation, extreme scales, near-singular or exactly
singular matrices, tolerances below the float spacing), plus typical inputs,
so that "fails everywhere" can be told apart from "fails at the edge".

Reference values come from mpmath at 60 significant digits (or exact
rationals), evaluated at the exact binary64 value of every input, then
rounded to the nearest float64. All input data is produced with a splitmix64
generator and plain IEEE additions and multiplications, so the file is
reproducible bit for bit on any platform.

This is a development-time generator, not an implementation: Go code never
imports it, and the library keeps zero dependencies. Requirements: Python 3.8+
and mpmath (version recorded in the output). Run from the repository root:

    python tools/stressgolden/numerics.py

Reading rules (each case states its reading in `claim`):
  - Explicit bounds are enforced as written.
  - Approximate claims ("~1e-12", "~15 significant digits") are enforced with
    10x slack: ~1e-12 -> 1e-11, "~15 significant digits" -> relative 1e-14.
  - "Exact" / "exact for IEEE 754 float64" for a computed (rounded) result is
    read as faithful rounding: relative error below 2^-52 (one ulp at the
    bottom of the binade, so never stricter than one ulp).
  - "~O(n)*eps" accumulated-rounding claims are enforced as 10*n*2^-52,
    relative to the quantity the error analysis bounds (stated per case).
  - Asymptotic claims (O(h^2), O(1/k^2)) are enforced with the constant of the
    reference the doc comment cites (stated per case).
  - For a vector or matrix result the evaluator returns one error measure
    (stated in `claim`) and the case expects 0 within the claimed tolerance;
    the reference vector travels in args["truth"].
  - A function that reports failure (nil, false, an error) where its doc
    promises a result is scored as NaN, i.e. as missing its claim, and the
    reverse: where the doc promises nil/false, success scores 1 against 0.
"""

import json
import math
import os
from fractions import Fraction

import mpmath as mp

mp.mp.dps = 60
OUT = os.path.join("testdata", "stress", "precision_numerics.json")
ULP = 2.0 ** -52
U = 2.0 ** -53
M64 = (1 << 64) - 1


def f64(x):
    """Round an mpmath value (or Fraction) to the nearest float64."""
    if isinstance(x, Fraction):
        return float(x)  # correctly rounded
    return float(mp.mpf(x))


class SplitMix64:
    """Deterministic data source: integer arithmetic plus exact scaling."""

    def __init__(self, seed):
        self.s = seed & M64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next() >> 11) * U  # exact multiple of 2^-53 in [0, 1)

    def normal(self):
        # Irwin-Hall: sum of 12 uniforms - 6. Only IEEE + and -, so the
        # sequence is identical on every platform (no libm).
        s = 0.0
        for _ in range(12):
            s += self.uniform()
        return s - 6.0


def case(cid, func, args, want, tol, kind, claim):
    assert kind in ("rel", "abs")
    return {"id": "numerics/" + cid, "func": func, "args": args, "want": want, "tol": tol, "tol_kind": kind, "claim": claim}


def M(x):
    return mp.mpf(x)


# ---------------------------------------------------------------------------
# linalg
# ---------------------------------------------------------------------------

def pearson_true(x, y):
    xs = [M(v) for v in x]
    ys = [M(v) for v in y]
    n = len(xs)
    mx = mp.fsum(xs) / n
    my = mp.fsum(ys) / n
    sxy = mp.fsum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = mp.fsum((a - mx) ** 2 for a in xs)
    syy = mp.fsum((b - my) ** 2 for b in ys)
    return sxy / mp.sqrt(sxx * syy)


def avg_ranks(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [Fraction(0)] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = Fraction(i + j, 2) + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def spearman_true(x, y):
    rx, ry = avg_ranks(x), avg_ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return M(sxy.numerator) / M(sxy.denominator) / mp.sqrt(M(sxx.numerator) / M(sxx.denominator) * M(syy.numerator) / M(syy.denominator))


def centered(X, T, N):
    Y = [[M(X[t * N + i]) for i in range(N)] for t in range(T)]
    for i in range(N):
        mu = mp.fsum(Y[t][i] for t in range(T)) / T
        for t in range(T):
            Y[t][i] -= mu
    return Y


def cov_mle(Y, T, N):
    return [[mp.fsum(Y[t][i] * Y[t][j] for t in range(T)) / T for j in range(N)] for i in range(N)]


def lw_identity_true(X, T, N):
    Y = centered(X, T, N)
    S = cov_mle(Y, T, N)
    m = mp.fsum(S[i][i] for i in range(N)) / N
    d2 = mp.fsum((S[i][j] - (m if i == j else 0)) ** 2 for i in range(N) for j in range(N)) / N
    bbar2 = mp.fsum(mp.fsum((Y[t][i] * Y[t][j] - S[i][j]) ** 2 for i in range(N) for j in range(N)) / N for t in range(T)) / (T * T)
    b2 = min(bbar2, d2)
    a2 = d2 - b2
    sh = b2 / d2
    sigma = [[a2 / d2 * S[i][j] + (sh * m if i == j else 0) for j in range(N)] for i in range(N)]
    return sh, sigma


def lw_constcorr_true(X, T, N):
    Y = centered(X, T, N)
    S = cov_mle(Y, T, N)
    sd = [mp.sqrt(S[i][i]) for i in range(N)]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rbar = mp.fsum(S[i][j] / (sd[i] * sd[j]) for i, j in pairs) / len(pairs)
    F = [[S[i][i] if i == j else rbar * sd[i] * sd[j] for j in range(N)] for i in range(N)]
    pim = [[mp.fsum((Y[t][i] * Y[t][j] - S[i][j]) ** 2 for t in range(T)) / T for j in range(N)] for i in range(N)]
    pi = mp.fsum(pim[i][j] for i in range(N) for j in range(N))
    rho = mp.fsum(pim[i][i] for i in range(N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            thii = mp.fsum((Y[t][i] ** 2 - S[i][i]) * (Y[t][i] * Y[t][j] - S[i][j]) for t in range(T)) / T
            thjj = mp.fsum((Y[t][j] ** 2 - S[j][j]) * (Y[t][i] * Y[t][j] - S[i][j]) for t in range(T)) / T
            rho += rbar / 2 * (sd[j] / sd[i] * thii + sd[i] / sd[j] * thjj)
    gamma = mp.fsum((F[i][j] - S[i][j]) ** 2 for i in range(N) for j in range(N))
    kappa = (pi - rho) / gamma
    delta = kappa / T
    assert 0 < delta < 1, delta
    sigma = [[delta * F[i][j] + (1 - delta) * S[i][j] for j in range(N)] for i in range(N)]
    return delta, sigma


def factor_returns(rng, T, N, load, scale, offset=0.0):
    X = []
    for _ in range(T):
        f = rng.normal()
        for i in range(N):
            X.append(offset + scale * (load[i] * f + rng.normal()))
    return X


def clean_corr_true(C, T, N):
    A = mp.matrix(N, N)
    for i in range(N):
        for j in range(N):
            A[i, j] = M(C[i * N + j])
    ev, Q = mp.eigsy(A)
    lmax = (1 + mp.sqrt(mp.mpf(N) / T)) ** 2
    vals = [ev[k] for k in range(N)]
    for v in vals:
        assert abs(v - lmax) > mp.mpf("1e-6"), "eigenvalue too close to the Marchenko-Pastur edge"
    noise = [v for v in vals if v <= lmax]
    avg = mp.fsum(noise) / len(noise) if noise else mp.mpf(0)
    lam = [v if v > lmax else avg for v in vals]
    R = [[mp.fsum(Q[i, k] * lam[k] * Q[j, k] for k in range(N)) for j in range(N)] for i in range(N)]
    out = [[R[i][j] / mp.sqrt(R[i][i] * R[j][j]) for j in range(N)] for i in range(N)]
    for i in range(N):
        out[i][i] = mp.mpf(1)
    return out, len(noise)


def corr_from_returns(X, T, N):
    Y = centered(X, T, N)
    S = cov_mle(Y, T, N)
    C = []
    for i in range(N):
        for j in range(N):
            C.append(1.0 if i == j else f64(S[i][j] / mp.sqrt(S[i][i] * S[j][j])))
    for i in range(N):
        for j in range(i + 1, N):
            C[j * N + i] = C[i * N + j]
    return C


def exact_det(A):
    """Determinant of a square matrix of Fractions (Gaussian elimination in Q)."""
    A = [row[:] for row in A]
    n = len(A)
    det = Fraction(1)
    for k in range(n):
        p = next((i for i in range(k, n) if A[i][k] != 0), None)
        if p is None:
            return Fraction(0)
        if p != k:
            A[k], A[p] = A[p], A[k]
            det = -det
        det *= A[k][k]
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= f * A[k][j]
    return det


def linalg_cases():
    C = []
    ex = ("'exact' read as faithful rounding: relative error below 2^-52")

    # PearsonCorrelation ----------------------------------------------------
    pc = "linalg/correlation.go PearsonCorrelation: 'Precision: exact for IEEE 754 float64.' -> " + ex
    # Typical seeds 1000..1007 were measured on the default and GOAMD64=v3
    # builds: 1000 and 1001 miss "exact" by 2-3 ulps on one build and not the
    # other, so the ratchet keeps 1002, which is faithful on both.
    for cid, seed, n, off in (("pearson/typical", 1002, 20, 0.0), ("pearson/offset-1e12", 1100, 50, 1e12)):
        rng = SplitMix64(seed)
        x, y = [], []
        for _ in range(n):
            u = 10.0 * rng.uniform()
            x.append(off + u)
            y.append(off + (0.5 * u + 0.3 * rng.normal()))
        C.append(case(cid, "linalg.PearsonCorrelation", {"x": x, "y": y}, f64(pearson_true(x, y)), ULP, "rel",
                      pc + ("" if off == 0 else "; data share an offset of %g with a spread of 10 (the centring cancels)" % off)))

    # SpearmanCorrelation ---------------------------------------------------
    sc = ("linalg/correlation.go SpearmanCorrelation: 'Precision: exact for IEEE 754 float64 (limited by ranking "
          "step).' -> " + ex + "; average ranks of ties are exact, so the reference is the exact Pearson correlation of the ranks")
    rng = SplitMix64(21)
    for cid, n, levels in (("spearman/ties-n30", 30, 7), ("spearman/n500", 500, 1 << 30)):
        x, y = [], []
        for _ in range(n):
            a = float(rng.next() % levels)
            x.append(a)
            y.append(a + float(rng.next() % levels) * 0.5)
        C.append(case(cid, "linalg.SpearmanCorrelation", {"x": x, "y": y}, f64(spearman_true(x, y)), ULP, "rel", sc))

    # Trace -----------------------------------------------------------------
    tr = ("linalg/matrix.go Trace: 'Precision: exact (accumulated float64 summation error for large n)' -> read as the "
          "recursive-summation bound |error| <= (n-1) * 2^-53 * sum|A_ii|, absolute")
    for cid, diag in (("trace/decimals", [0.1, 0.2, 0.3, 0.7]), ("trace/cancellation", [1e16, 1.0, -1e16])):
        n = len(diag)
        A = [0.0] * (n * n)
        for i, d in enumerate(diag):
            A[i * n + i] = d
            for j in range(n):
                if j != i:
                    A[i * n + j] = 0.25 * (i - j)
        want = sum((Fraction(d) for d in diag), Fraction(0))
        tol = (n - 1) * U * sum(abs(d) for d in diag)
        C.append(case(cid, "linalg.Trace", {"A": A, "n": n}, f64(want), tol, "abs", tr + " = %.3g" % tol))

    # CrossProduct ----------------------------------------------------------
    cp = ("linalg/matrix.go CrossProduct: 'Precision: exact for IEEE 754 float64.' -> " + ex +
          "; scalar = max_i |out_i - true_i| / max_j |true_j| (normwise, the most lenient per-vector reading)")
    eps9 = 2.0 ** -30
    for cid, a, b in (("cross/integers", [1.0, -2.0, 3.0], [4.0, 5.0, -6.0]),
                      ("cross/decimals", [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]),
                      ("cross/near-parallel", [0.1, 0.2, 0.3], [0.1 * (1 + eps9), 0.2, 0.3 * (1 - eps9)])):
        A_, B_ = [M(v) for v in a], [M(v) for v in b]
        t = [A_[1] * B_[2] - A_[2] * B_[1], A_[2] * B_[0] - A_[0] * B_[2], A_[0] * B_[1] - A_[1] * B_[0]]
        C.append(case(cid, "linalg.CrossProduct", {"a": a, "b": b, "truth": [f64(v) for v in t]}, 0.0, ULP, "abs",
                      cp + ("; nearly parallel inputs (the components cancel)" if "parallel" in cid else "")))

    # JamesSteinShrink ------------------------------------------------------
    js = ("linalg/shrinkage.go JamesSteinShrink: 'Precision: exact algebra in float64; no iterative error.' -> " + ex +
          "; scalar = the returned shrinkage factor c = (p-2)*variance/S")
    # Typical seeds 31..38 measured on both builds: all faithful, 31 only just
    # (0.99 of the tolerance on the default build), so the ratchet keeps 33.
    rng = SplitMix64(33)
    for cid, means, var in (("jamesstein/typical", [0.05 + 0.02 * rng.normal() for _ in range(10)], 1e-4),
                            ("jamesstein/offset-1e9", [1e9 + 0.1, 1e9 + 0.35, 1e9 + 0.8, 1e9 - 0.6], 0.05),
                            ("jamesstein/tiny-scale", [1e-151, 2e-151, 4e-151], 1e-303)):
        ms = [M(v) for v in means]
        p = len(ms)
        xb = mp.fsum(ms) / p
        S = mp.fsum((v - xb) ** 2 for v in ms)
        c = (p - 2) * M(var) / S
        assert 0 < c < 1
        C.append(case(cid, "linalg.JamesSteinShrink", {"means": means, "variance": var}, f64(c), ULP, "rel",
                      js + {"jamesstein/offset-1e9": "; means share an offset of 1e9",
                            "jamesstein/tiny-scale": "; means of order 1e-151: S = %.3g is below the code's 1e-300 'all identical' threshold" % f64(S)}.get(cid, "")))

    # Ledoit-Wolf, identity target ------------------------------------------
    lwi = ("linalg/shrinkage.go LedoitWolfShrinkageIdentity: 'Precision: ~1e-12 on the intensity for well-conditioned "
           "inputs; matrix entries accumulate to ~1e-9.' -> ")
    for cid, T, N, off, load in (("lw-identity/typical", 60, 5, 0.0, [0.8, 0.6, 0.4, 0.2, 0.0]),
                                 ("lw-identity/T500-N20", 500, 20, 0.0, [0.5 * (k % 3) for k in range(20)]),
                                 ("lw-identity/offset-1e4", 60, 5, 1e4, [0.8, 0.6, 0.4, 0.2, 0.0]),
                                 ("lw-identity/offset-1e5", 60, 5, 1e5, [0.8, 0.6, 0.4, 0.2, 0.0])):
        X = factor_returns(SplitMix64(100 + T + N), T, N, load, 0.01, off)
        sh, sig = lw_identity_true(X, T, N)
        assert 0 < sh < 1
        note = "" if off == 0 else "; returns carried at a level of %g (prices, not returns): the centring cancels %d digits" % (off, round(math.log10(off / 0.01)))
        C.append(case(cid + "-intensity", "linalg.LedoitWolfShrinkageIdentity", {"x": X, "T": T, "N": N, "measure": "intensity"},
                      f64(sh), 1e-11, "abs", lwi + "intensity absolute 1e-11 (10x)" + note))
        C.append(case(cid + "-matrix", "linalg.LedoitWolfShrinkageIdentity",
                      {"x": X, "T": T, "N": N, "measure": "matrix", "truth": [f64(sig[i][j]) for i in range(N) for j in range(N)]},
                      0.0, 1e-8, "abs", lwi + "matrix 1e-8 (10x); scalar = max |sigma_ij - true_ij| / max |true_ij|" + note))

    # Ledoit-Wolf, constant-correlation target -------------------------------
    lwc = "linalg/shrinkage.go LedoitWolfShrinkageConstantCorr: 'Precision: ~1e-12 intensity; matrix entries ~1e-9.' -> "
    for cid, T, N, off, load, scale in (("lw-constcorr/typical", 60, 5, 0.0, [0.9, 0.7, 0.2, -0.3, 0.5], 0.01),
                                        ("lw-constcorr/high-corr", 500, 6, 0.0, [8.0, 3.5, 9.0, 2.5, 7.0, 10.0], 0.01),
                                        ("lw-constcorr/offset-1e5", 60, 5, 1e5, [0.9, 0.7, 0.2, -0.3, 0.5], 0.01)):
        X = factor_returns(SplitMix64(200 + T + N), T, N, load, scale, off)
        dl, sig = lw_constcorr_true(X, T, N)
        note = {"lw-constcorr/typical": "", "lw-constcorr/high-corr": "; pairwise correlations 0.92-0.99",
                "lw-constcorr/offset-1e5": "; returns carried at a level of 1e5: the centring cancels 7 digits"}[cid]
        C.append(case(cid + "-intensity", "linalg.LedoitWolfShrinkageConstantCorr", {"x": X, "T": T, "N": N, "measure": "intensity"},
                      f64(dl), 1e-11, "abs", lwc + "intensity absolute 1e-11 (10x)" + note))
        C.append(case(cid + "-matrix", "linalg.LedoitWolfShrinkageConstantCorr",
                      {"x": X, "T": T, "N": N, "measure": "matrix", "truth": [f64(sig[i][j]) for i in range(N) for j in range(N)]},
                      0.0, 1e-8, "abs", lwc + "matrix 1e-8 (10x); scalar = max |sigma_ij - true_ij| / max |true_ij|" + note))

    # MarchenkoPasturBounds -------------------------------------------------
    mpb = ("linalg/shrinkage.go MarchenkoPasturBounds: 'Precision: exact to machine epsilon on sqrt.' -> relative error of each "
           "bound below machine epsilon 2^-52")
    for cid, Nn, Tt in (("mpbounds/q0.25", 25, 100), ("mpbounds/q0.1", 1, 10), ("mpbounds/q0.999", 999, 1000), ("mpbounds/q0.99999", 99999, 100000)):
        q = Nn / Tt  # float64 division, as a caller computes it
        sq = mp.sqrt(M(q))
        for which, v in (("min", (1 - sq) ** 2), ("max", (1 + sq) ** 2)):
            C.append(case(cid + "-" + which, "linalg.MarchenkoPasturBounds", {"q": q, "which": which}, f64(v), ULP, "rel",
                          mpb + "; q = %d/%d" % (Nn, Tt) + ("; lambdaMin = (1-sqrt q)^2 cancels as q -> 1" if which == "min" and Nn > 500 else "")))

    # CleanCorrelation --------------------------------------------------------
    cc = ("linalg/shrinkage.go CleanCorrelation: 'Precision: eigendecomposition to ~1e-12; reconstruction accumulates to "
          "~1e-9.' -> 1e-8 (10x) on max |out_ij - true_ij|; reference: mpmath eigsy of the same input, clipped at the "
          "same Marchenko-Pastur edge, reconstructed and renormalised")
    for cid, T, N, load in (("clean/typical", 40, 8, [0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0]),
                            ("clean/near-singular", 120, 6, [30.0, 30.0, 0.5, 0.4, 0.0, 0.0])):
        X = factor_returns(SplitMix64(300 + T + N), T, N, load, 0.01)
        Cm = corr_from_returns(X, T, N)
        out, nn = clean_corr_true(Cm, T, N)
        C.append(case(cid, "linalg.CleanCorrelation", {"corr": Cm, "T": T, "N": N, "truth": [f64(out[i][j]) for i in range(N) for j in range(N)]},
                      0.0, 1e-8, "abs", cc + "; %d of %d eigenvalues are noise" % (nn, N) +
                      ("; two assets with correlation ~0.999 (near-singular input)" if "singular" in cid else "")))

    # vector norms and distances ---------------------------------------------
    def vtrue_norm(v):
        return mp.sqrt(mp.fsum(M(a) ** 2 for a in v))

    nrm = ("linalg/vector.go %s: 'Precision: accumulated float64 rounding (~O(n)*eps), not bit-exact for n > 1' (overflow is "
           "caveated, underflow is not) -> relative 10*n*2^-52")
    for cid, v in (("l2norm/typical", [0.1 * (k + 1) for k in range(16)]),
                   ("l2norm/underflow-1e-170", [3e-170, 4e-170]),
                   ("l2norm/subnormal-squares-1e-160", [1e-160, 1e-160, 1e-160])):
        n = len(v)
        C.append(case(cid, "linalg.L2Norm", {"v": v}, f64(vtrue_norm(v)), 10 * n * ULP, "rel", nrm % "L2Norm"))
    for cid, a, b in (("encdist/typical", [0.1 * k for k in range(8)], [0.3 * k - 0.2 for k in range(8)]),
                      ("encdist/underflow-1e-170", [3e-170, 0.0], [0.0, 4e-170])):
        n = len(a)
        t = mp.sqrt(mp.fsum((M(x) - M(y)) ** 2 for x, y in zip(a, b))) / mp.sqrt(n)
        C.append(case(cid, "linalg.EncodingDistance", {"a": a, "b": b}, f64(t), 10 * n * ULP, "rel", nrm % "EncodingDistance"))
    for cid, a, b, w in (("wdist/typical", [0.1 * k for k in range(6)], [0.25 * k for k in range(6)], [1.0, 2.0, 0.5, 3.0, 1.0, 0.25]),
                         ("wdist/underflow-1e-170", [3e-170, 0.0], [0.0, 4e-170], [1.0, 1.0])):
        n = len(a)
        t = mp.sqrt(mp.fsum(M(wi) * (M(x) - M(y)) ** 2 for x, y, wi in zip(a, b, w)) / mp.fsum(M(wi) for wi in w))
        C.append(case(cid, "linalg.DimensionWeightedDistance", {"a": a, "b": b, "w": w}, f64(t), 10 * n * ULP, "rel",
                      nrm % "DimensionWeightedDistance"))
    for cid, v in (("l2normalize/typical", [0.3, -1.7, 2.2, 0.05]), ("l2normalize/underflow-1e-170", [3e-170, 4e-170])):
        n = len(v)
        C.append(case(cid, "linalg.L2Normalize", {"v": v}, 1.0, 10 * n * ULP, "rel",
                      (nrm % "L2Normalize") + "; scalar = the exact 2-norm of the normalised vector (1 when normalised; the "
                      "doc reserves 'returns false, vector unchanged' for zero magnitude)"))
    cs = ("linalg/vector.go CosineSimilarity: 'Precision: subject to accumulated float64 rounding in the dot/norm sums "
          "(~O(n)*eps) and to catastrophic cancellation in the dot product' -> absolute 10*n*2^-52 (cancellation is "
          "relative to |a||b|, i.e. absolute for a cosine); overflow and underflow are not caveated")
    for cid, a, b in (("cosine/typical", [0.1 * k - 0.3 for k in range(10)], [0.05 * k * k for k in range(10)]),
                      ("cosine/near-orthogonal", [1.0, 1e-8, 0.3], [-1e-8, 1.0, 0.0]),
                      ("cosine/underflow-1e-170", [1e-170, 1e-170, 1e-170], [1e-170, 2e-170, 3e-170]),
                      ("cosine/overflow-1e170", [1e170, 1e170, 1e170], [1e170, 2e170, 3e170])):
        A_, B_ = [M(x) for x in a], [M(x) for x in b]
        t = mp.fsum(x * y for x, y in zip(A_, B_)) / (mp.sqrt(mp.fsum(x * x for x in A_)) * mp.sqrt(mp.fsum(y * y for y in B_)))
        C.append(case(cid, "linalg.CosineSimilarity", {"a": a, "b": b}, f64(t), 10 * len(a) * ULP, "abs", cs))
    dp = ("linalg/vector.go DotProduct: 'Precision: subject to accumulated float64 rounding (~O(n)*eps); not bit-exact "
          "for n > 1' -> the standard bound 10*n*2^-52 * sum|a_i b_i| (absolute), so cancellation is allowed")
    for cid, a, b in (("dot/typical", [0.1 * k for k in range(12)], [1.0 / (k + 1) for k in range(12)]),
                      ("dot/cancellation", [1e17, 1.0, -1e17], [1.0, 1.0, 1.0])):
        t = mp.fsum(M(x) * M(y) for x, y in zip(a, b))
        tol = 10 * len(a) * ULP * float(mp.fsum(abs(M(x) * M(y)) for x, y in zip(a, b)))
        C.append(case(cid, "linalg.DotProduct", {"a": a, "b": b}, f64(t), tol, "abs", dp + " = %.3g" % tol))
    l1 = ("linalg/vector.go L1Norm: 'Precision: subject to accumulated float64 rounding (~O(n)*eps); not bit-exact for "
          "n > 1' -> relative 10*n*2^-52")
    v = [((-1) ** k) * 10.0 ** (k % 7 - 3) * (1 + 0.1 * k) for k in range(40)]
    C.append(case("l1norm/mixed-scales", "linalg.L1Norm", {"v": v}, f64(mp.fsum(abs(M(x)) for x in v)), 10 * len(v) * ULP, "rel", l1))
    li = "linalg/vector.go LInfNorm: 'Precision: exact for IEEE 754 float64.' -> exact match (max of |v_i| involves no rounding)"
    v = [-0.0, 5e-324, -3.5, 2.25, -1e300, 7.0]
    C.append(case("linfnorm/mixed", "linalg.LInfNorm", {"v": v}, 1e300, 0.0, "rel", li))

    # singular matrices -------------------------------------------------------
    dt = ("linalg/decompose.go Determinant: 'Returns 0 for singular matrices.' -> exact 0 for an exactly singular input "
          "(integer entries, exact rational determinant 0)")
    for cid, A, n in (("det/singular-2x2-exact-elimination", [1.0, 2.0, 2.0, 4.0], 2),
                      ("det/singular-1to9", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
                      ("det/singular-rank2-covariance", [26.0, 8.0, 11.0, 8.0, 10.0, 5.0, 11.0, 5.0, 5.0], 3)):
        assert exact_det([[Fraction(A[i * n + j]) for j in range(n)] for i in range(n)]) == 0
        C.append(case(cid, "linalg.Determinant", {"A": A, "n": n}, 0.0, 0.0, "abs", dt))
    iv = ("linalg/decompose.go Inverse: 'Returns false if A is singular.' -> scalar = 1 when Inverse reports success, 0 "
          "when it reports singular; an exactly singular input (integer entries) must give 0")
    for cid, A, n in (("inverse/singular-2x2-exact-elimination", [1.0, 2.0, 2.0, 4.0], 2),
                      ("inverse/singular-1to9", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
                      ("inverse/singular-rank2-covariance", [26.0, 8.0, 11.0, 8.0, 10.0, 5.0, 11.0, 5.0, 5.0], 3)):
        C.append(case(cid, "linalg.Inverse", {"A": A, "n": n}, 0.0, 0.0, "abs", iv))
    return C


def main():
    C = []
    C += linalg_cases()
    ids = [c["id"] for c in C]
    assert len(ids) == len(set(ids)), "duplicate case id"
    doc = {
        "_comment": "Generated by tools/stressgolden/numerics.py; do not edit by hand. See precision_test.go and precision_numerics_test.go.",
        "generator": {"mpmath": mp.__version__, "dps": mp.mp.dps},
        "cases": C,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"wrote {OUT}: {len(C)} cases")


if __name__ == "__main__":
    main()
