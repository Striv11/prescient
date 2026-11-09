# Plan

Seven-day arc. Every milestone has acceptance criteria that can be checked by a test or a command.

| Day | Milestone | Acceptance criteria | Status |
|-----|-----------|---------------------|--------|
| 1 | Rust core: PRNG, tensors, single-layer predictive-coding sequence model, CLI demo | `bash check.sh` passes; model reaches >95% next-symbol accuracy on a 4-cycle and spikes surprise >5x on a violated symbol (integration tests); runs are bit-for-bit deterministic per seed; `cargo run --release` prints a falling surprise curve | [x] |
| 2 | Context depth: two-layer hierarchy (fast/slow timescales) so order-2 melodies (`C E G E C E G A`) reach >95% accuracy | New test passes at >95% on the 8-note melody; hierarchy ablation test shows the single layer is worse; per-layer prediction-error stats exposed | [ ] |
| 3 | WASM bindings and a text/melody corpus API: `wasm-bindgen` wrapper (`observe`, `predict`, `state`, `save/load` weights as bytes) | `cargo build --target wasm32-unknown-unknown` succeeds in CI; `wasm-bindgen-test` (node) round-trips save/load with identical predictions; wasm size < 200 KB | [ ] |
| 4 | Web UI v1: piano-roll/text input, live prediction bars, surprise timeline, empty state, light/dark design system | Page renders from a static build; Playwright test: play a melody 20 times and the surprise chart falls; works at 375 px width | [ ] |
| 5 | Audio + interaction: Web Audio playback of notes and predictions, keyboard shortcuts, "break the pattern" button, text-mode (character) sequences | Playwright: shortcut keys trigger actions; violating the pattern visibly spikes surprise; text mode learns a repeated phrase to >90% accuracy | [ ] |
| 6 | Deploy and polish: GitHub Pages workflow, landing section, loading/error states, saved models in localStorage, perf benchmark with budgets | Pages workflow green; benchmark asserts >= 5k observe/s in WASM; reload restores model; screenshots in README | [ ] |
| 7 | Hardening and docs: property tests, accessibility pass, README write-up of the math and trade-offs, release v0.1.0 | Property tests (softmax invariants, determinism, save/load) pass; Lighthouse accessibility >= 90; README demo marked "live" | [ ] |

## Stretch
- Online anomaly highlighting on pasted text (per-character surprise heatmap).
- Compare against an n-gram baseline in the UI.
