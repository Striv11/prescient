# prescient

> A brain-inspired model that learns melodies and text patterns online and shows you what it expects to hear next.

![CI](https://github.com/Striv11/prescient/actions/workflows/ci.yml/badge.svg)

**Live demo:** _planned_ (GitHub Pages, arrives on day 6 of the plan).

## Problem
Most sequence models are trained offline, in batches, with backpropagation through time, and are opaque while they learn. There is little to *watch*: you cannot see a model form an expectation, get surprised, and adapt. `prescient` is a tiny sequence learner that learns from a single stream, one symbol at a time, and exposes its prediction and its surprise at every step, for anyone curious about how prediction-driven learning works.

## Motivation
Predictive coding says the brain continuously predicts its input and learns from the *prediction error*. Building it from scratch explores:
- inference by iterative error relaxation (instead of a single forward pass);
- learning with purely local updates (error x activity), no backprop through time;
- a deterministic, dependency-free Rust core that compiles to WebAssembly so the same code runs in a CLI, in tests and in the browser.

## Architecture
```
 previous hidden z(t-1) --A--> tanh --> prior z~(t)
                                           |
                       relax z(t) to reduce |  (z - z~)  and  (x - softmax(W z))
                                           v
 z(t) --W--> softmax --> predicted next-symbol distribution
```
- `src/rng.rs`: splitmix64 PRNG (reproducible everywhere).
- `src/tensor.rs`: small dense matrix helpers and a stable softmax.
- `src/model.rs`: `Predictor`: `predict()` gives the distribution over the next symbol; `observe(sym)` returns the surprise (-ln p), infers the hidden state and applies local weight updates.
- `src/main.rs`: CLI demo.
- Planned: two-timescale hierarchy, WASM bindings, web UI (see [PLAN.md](PLAN.md)).

Invariants: predictions are always a valid distribution; a given seed and input stream always give identical results.

## Setup
Prerequisite: a recent stable Rust toolchain.
```
cargo test          # unit and learning tests
bash check.sh       # fmt check + clippy + tests (same as CI)
```

## Demo
Status: CLI demo available now; browser demo **planned**.
```
cargo run --release
```
Prints the average surprise and next-note accuracy while the model repeatedly hears the melody C E G E C E G A. Example output:
```
 epoch   surprise  accuracy
     1      1.789       28%
    60      0.377       72%
```
Accuracy on this melody is limited because a single layer cannot disambiguate the repeated notes by itself; fixing that is day 2's hierarchy.

## Design decisions & trade-offs
- Softmax output with a cross-entropy-style error keeps the update rule local while giving real probabilities.
- f64 everywhere for determinism and simplicity; size and speed tuning come with the WASM work.
- Single layer first, so the hierarchy can be shown to help with an ablation.

## Roadmap
- [x] Day 1: core model, CLI, learning tests
- [ ] Day 2: two-layer hierarchy
- [ ] Day 3: WASM bindings
- [ ] Day 4-5: web UI, audio, shortcuts
- [ ] Day 6: GitHub Pages, benchmarks
- [ ] Day 7: hardening, docs, release

## License
MIT. See [LICENSE](LICENSE).
