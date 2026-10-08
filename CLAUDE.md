# Reality

## Deep docs (start here)

- `ARCHITECTURE.md`: what the library is, its layers, design rules and the tests that enforce them, trust boundaries, known gaps
- `docs/CODE_MAP.md`: every package with its purpose, key files, exported-function count and internal imports
- `docs/API_SURFACE.md`: the `reality-compute` HTTP routes, the `conduit` outbound event, and the exported Go API per package
- `docs/DATA_MODEL.md`: no persistent state; process-wide and caller-owned in-memory state; golden and stress test data files
- `docs/OPERATIONS.md`: environment variables, `reality-compute` startup, CI jobs and the test ratchets that gate them
- `docs/IMPROVEMENTS.md`: ranked improvement proposals (P0/P1/P2) with receipts
- `docs/ARCHITECTURE_CHANGELOG.md`: what changed between documentation sweeps
