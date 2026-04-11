# Fix: remove Conduit HTTP call from pure math function

**Date:** 2026-04-11
**File:** `prob/prob.go`
**Bug:** `ConfidenceFromPValue` contained a `conduit.EmitSampled()` HTTP call inside what should be a pure math function. This made the function impure (side-effecting), added a `context` dependency to a math library, and violated the "share the math, not the meaning" principle.

## Root cause

Wave 6.A5 added sampled Conduit observation to `ConfidenceFromPValue` as a "fail-silent" telemetry hook. The intent was to observe usage patterns. However, placing HTTP I/O inside a pure math function:
1. Makes the function non-deterministic (network timing).
2. Couples the math library to the Conduit infrastructure package.
3. Prevents use in contexts where network is unavailable (Rust no_std, ASM, embedded).
4. Violates the Layer 1 purity contract: math functions must be side-effect-free.

## Fix

- Removed the `conduit.EmitSampled(...)` call from `ConfidenceFromPValue`.
- Removed the now-unused imports: `"context"` and `"github.com/davly/reality/conduit"`.
- The function is now pure: `confidence = clamp(1 - pValue, 0, 1)`.

## Migration note

If Conduit observation of confidence-from-pvalue calls is desired, the emission should be done by the **caller**, not the math library. This follows the pattern used everywhere else in the ecosystem: math computes, callers observe.
