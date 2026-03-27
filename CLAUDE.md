# Reality

Universal truth encoded in code. Pure math, physics, constants. Zero dependencies. MIT open source.

## Quick Reference

- **Go module:** `github.com/davly/reality`
- **License:** MIT
- **Port:** None (library, not a service)
- **Status:** Design complete, implementation not yet started
- **Design doc:** `C:/CrossPollinationAnalysis/architecture/UNIVERSAL_TRUTH_FOUNDATION.md`
- **Review synthesis:** `C:/CrossPollinationAnalysis/reviews/reality-review/SYNTHESIS.md`
- **Context:** `CONTEXT.md` in this repo (read this for full background)

## Architecture

One repo. Sub-packages. Single Go module. Go is canonical; Python/C++/C# validate against golden files.

```
reality/
  linalg/       calculus/     stats/        physics/
  graph/        crypto/       geometry/     signal/
  constants/    bio/          color/        game/
  decision/     queuing/      geodesic/     sequence/
```

## Dependency Position

```
Consumer Apps -> Services -> AI (aicore) -> reality -> math stdlib
```

aicore imports reality. reality imports nothing.

## Key Design Rules

1. **Golden files are the proof.** Every function has golden-file test vectors (JSON, shared across 4 languages). Minimum 20 vectors per function, target 30. Per-function tolerance, not global.
2. **Zero dependencies.** Only the language's standard math library. No gonum, no numpy in the core path.
3. **No allocations in hot paths.** Functions accept output buffers. Pistachio calls these at 60 FPS.
4. **Every function cites its source.** Mathematical provenance as queryable metadata, not buried comments.
5. **Precision documented, not assumed.** Every function states valid input range, numerical precision, and failure modes.
6. **Reimplement from first principles.** Do not wrap existing libraries. Provide optional adapters separately.

## Building / Testing

Not yet implemented. When built:

```bash
# Run all Go tests
go test ./...

# Run golden-file validation
go test -run TestGolden ./...
```

## v1.0 Scope

~397 functions across 16 sub-packages. ~8,990 golden-file test vectors. 5 phases over ~20 weeks.

**Phase 0:** Golden-file infrastructure (weeks 1-2)
**Phase 1:** Extract existing aicore math (weeks 3-6)
**Phase 2:** Fill gaps across all domains (weeks 7-14)
**Phase 3:** Pistachio + RubberDuck integration (weeks 15-18)
**Phase 4:** Open source preparation (weeks 19-20)

## What This Is Not

- Not an AI library (no models, no tokens, no routing)
- Not a physics engine (no scene graph, no collision system, no GPU dispatch)
- Not a framework (pure functions -- numbers in, numbers out)
