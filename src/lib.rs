//! prescient: an online predictive-coding sequence learner.
//!
//! The model keeps a hidden state `z` that (a) predicts the next hidden state
//! through a recurrent prior and (b) generates a distribution over the next
//! symbol. Learning is driven only by local prediction errors.

pub mod model;
pub mod rng;
pub mod tensor;

pub use model::{Config, Predictor, StepStats};
