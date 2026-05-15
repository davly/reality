# 393 — meta-frame-conventions (frame/handedness/ordering across packages)

## Headline
Reality is implicitly right-handed Hamilton-quaternion `[w,x,y,z]` everywhere, but only `geometry/quaternion.go` documents the convention; every other 3D-aware package (orbital, em, physics, linalg) leaves frame, chirality, and active-vs-passive rotation undocumented — a single repo-wide `CONVENTIONS.md` plus per-function frame tags would close the gap with zero behavioural change.

## Per-package frame audit

| Package | Handedness | Up axis | Quat ordering | Active/passive | Documented? |
|---|---|---|---|---|---|
| geometry | RH (implied; `CrossProduct` uses RH formula in `linalg/matrix.go:205-207`, comment "right-hand rule" `matrix.go:195`) | none ("world" frame is unspecified, SDFs are frame-agnostic) | Hamilton `[w,x,y,z]` (`quaternion.go:7`) — `w` scalar first | active rotation of vectors, `v' = q·v·q*` (`quaternion.go:183`) | partial (quat ordering yes; handedness/active only via formula) |
| orbital | RH inertial, perifocal then 3-1-3 Euler `(Ω, i, ω)` (`orbital.go:38, 60`) | none — equatorial inertial frame implied by `i ∈ [0, π]` measured from "the" reference plane (`orbital.go:40`); ICRS/ECI/ecliptic distinction never made | n/a (no quaternion use) | passive frame rotation perifocal → inertial (orbital.go:38, "Rotation … to inertial frame") | no — frame name absent, equator vs ecliptic ambiguous, see slot 368 headline |
| em | n/a (scalar/1-D) — `CoulombForce`, `ElectricField` return magnitudes; no field vectors, no Poynting, no Lorentz cross product | n/a | n/a | n/a — only "passive sign convention" for circuits (`em.go:90`), unrelated to frame | n/a (no 3-D fields exist yet) |
| physics | implicit 2-D screen-style frame: `ProjectilePosition` returns `(x, y)` with gravity along `-y` (`mechanics.go:33-37`); no 3-D mechanics | y-up (gravity sign) | n/a | n/a | partial (sign of g implicit in formula, axis labels never stated) |
| linalg | RH (cross product right-hand rule, `matrix.go:195`), row-major flat slices (`vector.go:5-6`) | n/a (general n-D) | n/a | n/a (operator-level) | partial (storage order yes; chirality only via formula comment) |
| chaos | state-space, no physical frame — phase variables `(x, y, z)` are arbitrary (Lorenz, Van der Pol) | n/a | n/a | n/a | n/a (correctly frame-free) |
| signal | time/frequency domain — not a spatial frame; `signal.IFFT` ordering (k=0 DC at index 0) is the relevant convention | n/a | n/a | n/a | yes (DC bin documented in signal pkg) |
| acoustics | scalar magnitudes + `DopplerShift` with sign convention "positive vs source/receiver" (`acoustics.go:119`) — 1-D | n/a | n/a | n/a (sign convention is a 1-D approach/recede flag) | yes for Doppler signs |
| color | tristimulus + chromaticity, working frame is CIE XYZ with `Y` luminance up (`color/spaces.go`); not a physical frame | n/a | n/a | passive (matrix transforms `XYZ → RGB` are frame changes) | yes (each space named) |

## Implicit conventions, recovered by reading code

1. **Quaternion**: Hamilton `(w, x, y, z)` with `w` first (`geometry/quaternion.go:7, 21-23`). Equivalent to Eigen, ROS REP-103, Wikipedia. Opposite of JPL `(x, y, z, w)` convention used by Bullet, OpenCV, sometimes Unreal. **Stated.**
2. **Quaternion product**: Hamilton (i·j = k), see `QuatMul` `quaternion.go:69-76` — confirmed by slot 313 test `geometry_test.go:144-163` (i·j=k, j·k=i, k·i=j). **Implicit but verified by tests.**
3. **Quaternion-vector rotation**: active (rotates the vector inside a fixed frame) per `v' = q·v·q*` `quaternion.go:183`. Same direction as Eigen's `q * v`. **Implicit; needs note "active rotation".**
4. **Cross product**: right-hand rule `out = a × b` with standard determinant expansion `linalg/matrix.go:205-207`. Comment at `matrix.go:195` says "right-hand rule". **Stated.**
5. **Euler angles**: ZYX intrinsic (yaw-pitch-roll) `quaternion.go:213` — but parameter order is `(pitch, yaw, roll)` mapped to `(X, Y, Z)`. **Non-aerospace** (aerospace ISO 8855 is roll=X, pitch=Y, yaw=Z). Slot 313 already flagged this naming clash for the orbital boundary.
6. **Orbital rotation**: classic 3-1-3 Euler `(Ω, i, ω)` perifocal → inertial (`orbital.go:38, 60-65`). Sign of `z` (`orbital.go:65`) is `+sin(i)` — confirms RH inertial with `+z` along the angular momentum vector of the reference plane. **Implicit; reference frame name (ECI? J2000? ICRS?) not given — see slot 368.**
7. **Physics 2-D**: `ProjectilePosition` uses `y = v0 sin θ t − ½ g t²` (`mechanics.go:51`) — gravity along `-y`. Standard textbook convention; only matters because nothing says "y is up".
8. **Linalg storage**: row-major flat `[]float64` (`vector.go:5-6`). Different from BLAS/LAPACK column-major default. **Stated.**
9. **Vector convention**: column-vector semantics — `M·v` is the consistent operator order (`linalg/matrix.go` MatVec). **Implicit; not stated but consistent.**
10. **No left-handed code anywhere** — `CrossProduct`, `KeplerOrbit`, and `QuatRotateVec` all use the same RH sign. Would break golden files cross-language if a port flipped chirality.

## Risks & cross-language implications

- **Hamilton vs JPL drift in C++ port**: Eigen's `Quaterniond(w, x, y, z)` constructor takes Hamilton-`w`-first but its internal layout is `[x, y, z, w]` — anyone writing a C++ validator must use `q.coeffs()` carefully, or the golden file `[4]float64` ordering will silently swap `w`. Slot 313 noted no `QuatToMatrix`/`QuatToEuler` exist, so this only bites if/when those land.
- **Aerospace handoff**: `orbital` produces `(x, y, z)` in an unnamed inertial frame, and `geometry/quaternion` consumes ZYX (pitch-yaw-roll) Euler with non-aerospace parameter naming. A consumer chaining `orbital.KeplerOrbit → geometry.QuatFromEuler` to point a spacecraft has no guidance on which axis is "along velocity" vs "radial" vs "normal" (no RTN/LVLH helpers — see slot 368 headline).
- **Active vs passive ambiguity**: `QuatRotateVec` is active (rotates vector). A consumer wanting "rotate the body frame relative to world" (passive) must conjugate the quaternion first. The docstring says `v' = q·(0,v)·q*` — formula correct, intent unstated.
- **No 3-D EM**: `em` ships only scalar magnitudes. The day a `LorentzForce(q, v, E, B) [3]float64` lands, it MUST declare laboratory inertial frame and RH chirality, otherwise `B`-field sign convention becomes a portability hazard.
- **`physics` 2-D y-up vs graphics y-down**: Pistachio is the named consumer (`geometry` package doc says "Pistachio procgen pipeline at 60 FPS"). Most graphics pipelines are y-down screen / z-forward. If `physics.ProjectilePosition` results are dropped into Pistachio without a coordinate flip, gravity will appear to push projectiles upward on screen.

## Recommendation

1. **Ship `CONVENTIONS.md` at repo root.** One page. Sections: handedness, axis labels, quaternion ordering, rotation interpretation (active), orbital frame (declare it: "right-handed Cartesian inertial, +Z along reference-plane angular momentum, equatorial unless otherwise stated"), Euler convention. Cross-link from each package's `doc.go`.
2. **Add per-function `Frame:` doc tag** the way every function already has `Precision:` and `Reference:`. For example `KeplerOrbit` should say `Frame: right-handed inertial, +Z along reference-plane normal; perifocal → inertial via 3-1-3 Euler (Ω, i, ω)`. Cheap, no code churn.
3. **Rename `QuatFromEuler` parameters** or add a parallel aerospace constructor `QuatFromAerospaceEuler(roll, pitch, yaw)` that maps roll→X, pitch→Y, yaw→Z. Slot 313 already recommends this — flag it as a frame-convention bug, not just a naming nit.
4. **Add a `geometry.QuatFromHamiltonComponents` / `QuatFromJPLComponents` adapter pair** at the boundary. Internal Hamilton-only; expose adapters so future C++ validators can declare the input convention explicitly.
5. **Document `QuatRotateVec` as active** in one line: "rotates `v` by `q` within a fixed frame; for passive frame rotation use `QuatConjugate(q)` first."
6. **`physics` 2-D**: change package doc to state "2-D mechanics use right-handed `(x, y)` with gravity along `-y`". Single sentence in `mechanics.go:18` block.
7. **Defer 3-D EM frame docs** until a vector-valued EM function exists; right now there is nothing to document.
8. **Cross-test**: add a single repo-wide test `TestFrameConsistency_HandednessRH` that asserts `CrossProduct([1,0,0],[0,1,0]) == [0,0,1]` (linalg) and that `QuatRotateVec(QuatFromAxisAngle([0,0,1], π/2), [1,0,0])` is approximately `[0,1,0]` (geometry). Single golden vector each. Catches any future port that flips chirality.
9. **Slot 368 dependency**: the orbital frame question is mostly "ICRS vs J2000 vs ecliptic" and is owned by slot 368's recommendation to add a frame-rotation framework. This slot only asks that whatever frame the current Kepler returns be *named*, not that the framework be built today.

## Severity

Low/medium. No numerical bugs. All current code is internally consistent (RH everywhere, Hamilton everywhere). Cost is purely portability and consumer-onboarding: a new C++/Python validator MUST guess that `[4]float64` is `[w,x,y,z]` Hamilton (not `[x,y,z,w]` JPL) and that orbital `(x,y,z)` is RH inertial. Without `CONVENTIONS.md` the answer is "read all 22 packages and infer". With it, one page closes the question.

## Sources

- `C:\limitless\foundation\reality\geometry\quaternion.go` (lines 7, 21-23, 69-76, 183-203, 213-229)
- `C:\limitless\foundation\reality\geometry\sdf.go` (frame-free SDF primitives)
- `C:\limitless\foundation\reality\orbital\orbital.go` (lines 38, 44-67 — implicit inertial frame, 3-1-3 Euler)
- `C:\limitless\foundation\reality\em\em.go` (lines 1-21, 90 — "passive sign convention" applies to circuit power, not frame)
- `C:\limitless\foundation\reality\physics\mechanics.go` (lines 31-53 — 2-D y-up implicit)
- `C:\limitless\foundation\reality\linalg\matrix.go` (lines 185-208 — RH cross product, comment line 195)
- `C:\limitless\foundation\reality\linalg\vector.go` (lines 1-11 — row-major storage stated)
- `C:\limitless\foundation\reality\acoustics\acoustics.go` (line 119 — Doppler sign convention)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\313-dive-rotation-rep.md` (already documents Hamilton ordering, ZYX Euler naming clash, missing q→Euler/q→DCM)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\368-research-iau-frames.md` (already documents orbital frame is unnamed; recommends ICRS/ICRF3 framework)
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:419` (this slot)
