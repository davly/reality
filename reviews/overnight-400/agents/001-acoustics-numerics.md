# 001 — acoustics: numerical accuracy & IEEE-754 edge case audit

## Headline
The acoustics package is small, clean, and IEEE-754-friendly for the documented happy paths, but two literal physical constants are accuracy-limited (`AWeighting`'s `+2.0` offset and `SabineRT60`'s `0.161`) and several functions silently accept non-physical inputs (negative pressure, negative volume, negative length, paired-negative-cancellation tricks) that the docstrings claim to reject.

## Findings

### Constant-precision regressions

- `acoustics.go:195` (`AWeighting`): the offset literal `+ 2.0` is the IEC-61672 "A1000 normalisation". The exact value derived from the formula's own coefficients is `1.99985847230512...`, so the function returns `+1.4153e-4 dB` at 1 kHz instead of exactly `0.0`. The golden vector confirms this — `testdata/acoustics/a_weighting.json:7` pins `expected = 0.00014152769487818517` at 1 kHz. The bias is small (~0.00014 dB) but it is a documented inconsistency: the docstring at `acoustics.go:175` says "1 kHz is the reference; should be 0 dB" and the tests assert `0.0 ± 0.01`, which masks the true ~1.4e-4 dB systematic error. Replacing `+ 2.0` with the precomputed `-20*log10(R_A(1000))` = `1.9998584723051218` makes the 1 kHz value exact (within ULPs).

- `acoustics.go:102` (`SabineRT60`): the literal `0.161` is only 3 significant figures. The canonical value `24*ln(10)/c` at `c = 343 m/s` is `0.16111382574885452`, so all RT60 values carry a fixed `+7.06e-4` relative error (~+0.7 ms on a 1 s reverberation). The constant package already exposes `Ln10` and the speed of sound; computing `24*math.Ln10/343` once and storing in a `const` would buy ~12 more digits at zero runtime cost. The docstring at `acoustics.go:99` advertises "Precision: exact" — that claim is wrong as written.

### IEEE-754 edge cases that contradict documented behaviour

- `acoustics.go:164` (`WaveLength`): docstring at line 161 says "Returns +Inf if f == 0", but `WaveLength(-0.0, c)` returns `-Inf` (sign of -0 propagates through division). Likewise `WaveLength(0, 0)` returns NaN, not the documented +Inf. The signed-zero asymmetry is a real correctness gap for any code that reads sample data containing -0.0.

- `acoustics.go:101` (`SabineRT60`): docstring at line 97 says "Returns +Inf if A == 0". With `V = 0, A = 0` the function returns NaN (`0.161 * 0 / 0`). With `A = -0.0` it returns `-Inf`. Tests cover only the V>0, A=0 path (`acoustics_test.go:175`), missing both 0/0 and -0 forms.

- `acoustics.go:149` (`ResonantFrequency`): no input validation despite docstring "Valid range: L > 0, n >= 1, c > 0". `L = -1` returns -171.5 Hz, `L = -0` returns +Inf (not -Inf, because `2 * -0.0 = -0.0` but Go's runtime division of `+/-0` yields `±Inf` deterministically — confirmed empirically as +Inf), `n = 0` returns 0 (a non-resonance), and `n` accepts MaxInt32. None of these are flagged. Worse, the function takes `int` for `n` but `int` is platform-dependent (32/64-bit); a 64-bit platform overflowing into the float-conversion path is silent.

- `acoustics.go:65` (`DecibelSPL`): the doc says "NaN for negative p", but `DecibelSPL(-1, -1)` returns `0.0` (the two negatives cancel inside the ratio, then `log10(1) = 0`). The `negative ratio → NaN` invariant only holds when *exactly one* of {p, pRef} is negative. Same vulnerability in `DecibelFromIntensity` at line 81.

- `acoustics.go:29` (`SoundSpeed`): "NaN for negative arguments inside sqrt" — but `SoundSpeed(-1.4, R, T, -M)` (negative gamma + negative molar mass) returns a perfectly clean `342.93 m/s` because both negatives cancel inside `gamma*R*T/M` before the sqrt. Same pattern: `(-gamma) * R * T / (-M)` is positive. Three other sign-cancellation pairs exist with the four arguments.

- `acoustics.go:45` (`SoundIntensity`): squaring `r` discards the sign, so `SoundIntensity(P, -5)` equals `SoundIntensity(P, 5)`. Negative distance is non-physical but accepted. `SoundIntensity(-1, 1)` returns a *negative* intensity, which is also non-physical (intensity is a magnitude). Doc says only "Returns +Inf if r == 0".

### Numerical-method observations (not bugs, but precision-relevant)

- `acoustics.go:188` (`AWeighting`): the formula uses naive `f^4 * (...)` arithmetic. `num = 12194^2 * f^4` overflows to +Inf around `f ≈ 1e76`, after which `denom` also overflows and the function returns NaN. Above audio bandwidth (>100 kHz) the function still works correctly, but for any client passing unscaled PHY-level frequencies (e.g. RF in Hz), behaviour is silently NaN above ~1e76 Hz. Refactoring to log-space (`20*log10(num) - 20*log10(d1) - 20*log10(d2) - 10*log10(prod_d3)`) would extend the dynamic range to ~1e308 with the same final precision, at the cost of more transcendentals.

- `acoustics.go:128` (`DopplerShift`): the documented singularity at `c + vs == 0` is real, but the empirical behaviour around it is dramatic — at `vs = -c + 1e-12`, the result is `3.35e17` Hz; one ULP further negative, `-3.35e17` Hz. Catastrophic cancellation lives in the `c + vs` subtraction. There is no compensation and no warning; clients near Mach 1 source velocities will see meaningless answers without any indication. A simple `if math.Abs(c+vs) <= |c|*ulp` guard returning ±Inf would communicate the singularity.

- `acoustics.go:65` (`DecibelSPL` doubling test, `acoustics_test.go:137-142`): empirically `DecibelSPL(2, p) - DecibelSPL(1, p) = 6.020599913279625` versus exact `20*log10(2) = 6.020599913279624`. The 1-ULP difference is harmless and the existing `1e-10` tolerance covers it; calling out for completeness only.

- `acoustics.go:65` (`DecibelSPL`): subnormal pressure handling is correct — `DecibelSPL(5e-324, 2e-5)` returns `-6159.07` (the smallest subnormal pressure mapped to a finite, sensible dB). No log/exp pair would benefit from `log1p`/`expm1` since dB is intrinsically logarithmic.

### Tests / golden-file coverage gaps

- All four golden files contain only **3-4 test vectors**, well below the project's stated minimum of 20 (CLAUDE.md golden-files rule). `decibel_spl.json` has 3 cases, the rest 4. None test ±Inf, NaN, ±0, subnormals, or the documented singularities. `DecibelFromIntensity`, `SoundIntensity`, `SabineRT60`, `ResonantFrequency`, `WaveLength` have **no golden file at all** — they are tested only by Go-side unit tests.

- `acoustics_edge_test.go` is good for sign and proportionality checks but does not assert any exact bit-level values, NaN-vs-Inf distinctions for the boundary inputs above, or the documented "p=NaN, pRef=NaN" behaviour.

## Concrete recommendations

1. **AWeighting**: replace the literal `+ 2.0` at `acoustics.go:195` with a `const a1000Offset = 1.9998584723051218` (precomputed `-20*log10(R_A(1000))`). Update the golden vector at `testdata/acoustics/a_weighting.json:7` to `0.0`. Update the docstring precision claim accordingly.

2. **SabineRT60**: replace `0.161` at `acoustics.go:102` with `var sabineConstant = 24 * math.Ln10 / 343.0` (or accept `c` as a parameter to remove the hidden 343 m/s coupling). Either remove the "Precision: exact" claim at line 98 or qualify it as "exact in the constant chosen".

3. **WaveLength / SabineRT60**: explicitly normalize the divisor to absolute zero for documented +Inf behaviour. E.g. `if f == 0 { return math.Inf(1) }` ahead of the division at `acoustics.go:165`. Likewise for SabineRT60 at `acoustics.go:102` to handle 0/0 → +Inf as documented (or pick a different documented invariant and stick to it).

4. **DopplerShift**: add a guard at `acoustics.go:129` returning `math.Inf(...)` when `math.Abs(c+vs) <= 1e-12*math.Abs(c)`, to communicate the singularity rather than emit garbage 1e17 Hz answers. Tighten the docstring to say "behaviour within 1 ULP of `c+vs == 0` is undefined; do not rely on the sign".

5. **Sign-cancellation invariants**: either document that "negative inputs are accepted and may produce non-physical positive outputs via paired-negative cancellation" in `SoundSpeed` (`acoustics.go:24`), `DecibelSPL` (`acoustics.go:61`), `DecibelFromIntensity` (`acoustics.go:77`), and `SoundIntensity` (`acoustics.go:41`), or insert explicit `<0` guards at function entry. Pick one and make it consistent across the package.

6. **ResonantFrequency**: change the `n int` parameter at `acoustics.go:149` to `n float64` (or document that `int` is the only non-float64 parameter in the package and is intentional for harmonic indexing). Add a guard for `L <= 0`.

7. **Golden-file vectors**: bring all four existing files up to the documented minimum of 20 cases each, and create golden files for `SoundIntensity`, `DecibelFromIntensity`, `SabineRT60`, `ResonantFrequency`, `WaveLength`. Each set must include the IEEE-754 mandated edge cases per CLAUDE.md ("+Inf, -Inf, NaN, -0.0, subnormals"). The current 3-4-vector files would not satisfy any other reviewer in the 400-agent fleet who checks the golden-file rule.

8. **Document the A-weighting overflow**: the docstring at `acoustics.go:184` claims "Precision: limited by float64 log and sqrt (~12 significant digits at extreme frequencies)", but at `f ≥ 1e76` the function returns NaN due to numerator overflow. Either add the overflow note or refactor to log-space arithmetic.

## Sources
- C:\limitless\foundation\reality\acoustics\acoustics.go (lines 1-197, all functions)
- C:\limitless\foundation\reality\acoustics\acoustics_test.go (lines 1-322)
- C:\limitless\foundation\reality\acoustics\acoustics_edge_test.go (lines 1-216)
- C:\limitless\foundation\reality\acoustics\testdata\acoustics\a_weighting.json
- C:\limitless\foundation\reality\acoustics\testdata\acoustics\decibel_spl.json
- C:\limitless\foundation\reality\acoustics\testdata\acoustics\doppler.json
- C:\limitless\foundation\reality\acoustics\testdata\acoustics\sound_speed.json
- IEC 61672-1:2013 Annex E (cited in source) for canonical A-weighting normalisation derivation
- Numerical evidence collected by running an ad-hoc probe of all signed-zero, NaN, Inf, sub-normal, and overflow paths against the package formulas (probe deleted post-run)
