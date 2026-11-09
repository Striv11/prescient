# Work log

Newest entries at the bottom. Each entry records what actually changed, what was learned, and what comes next.

## 2026-11-09 (Day 1)
- Picked "Predictive-coding sequence learner" (Rust -> WASM) from the Cognition backlog: new domain and language versus the TypeScript-heavy recent projects.
- Built the Rust core: splitmix PRNG, dense matrix helpers, and a single-layer predictive-coding `Predictor` (iterative error relaxation for inference, local updates for learning), plus a CLI demo.
- Tests: 4-cycle is learned to >95%, a violated symbol spikes surprise >5x, deterministic per seed, valid distributions.
- Learned: the first model learned the 4-cycle immediately, but the 8-note melody with repeated notes plateaus around 75-88% in the CLI (the test with 600 reps passes >90%). Order-2 context needs the slower second layer: day 2.
- Next: two-timescale hierarchy and an ablation test.
