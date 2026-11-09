use crate::rng::Rng;
use crate::tensor::{softmax, Mat};

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: usize,
    pub hidden: usize,
    /// Relaxation iterations of the hidden state per observation.
    pub infer_steps: usize,
    pub infer_rate: f64,
    pub lr_out: f64,
    pub lr_trans: f64,
    pub seed: u64,
}

impl Config {
    pub fn new(vocab: usize) -> Self {
        Config {
            vocab,
            hidden: 24,
            infer_steps: 20,
            infer_rate: 0.1,
            lr_out: 0.05,
            lr_trans: 0.05,
            seed: 1,
        }
    }
}

/// What happened when one symbol was observed.
#[derive(Clone, Debug)]
pub struct StepStats {
    /// Distribution that was predicted *before* seeing the symbol.
    pub predicted: Vec<f64>,
    /// Surprise in nats: -ln p(symbol).
    pub surprise: f64,
    /// Whether the argmax prediction matched the symbol.
    pub correct: bool,
}

pub struct Predictor {
    cfg: Config,
    /// Output (generative) weights: hidden -> symbol logits.
    w: Mat,
    /// Transition weights: previous hidden -> prior for next hidden.
    a: Mat,
    z: Vec<f64>,
}

impl Predictor {
    pub fn new(cfg: Config) -> Self {
        assert!(cfg.vocab >= 2 && cfg.hidden >= 1);
        let mut rng = Rng::new(cfg.seed);
        let w = Mat::random(cfg.vocab, cfg.hidden, 0.1, &mut rng);
        let a = Mat::random(cfg.hidden, cfg.hidden, 0.1, &mut rng);
        let z = vec![0.0; cfg.hidden];
        Predictor { cfg, w, a, z }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Forget the current context (weights are kept).
    pub fn reset_state(&mut self) {
        self.z.iter_mut().for_each(|v| *v = 0.0);
    }

    fn prior(&self) -> Vec<f64> {
        self.a.mul_vec(&self.z).into_iter().map(f64::tanh).collect()
    }

    /// Distribution over the next symbol given the current context.
    pub fn predict(&self) -> Vec<f64> {
        softmax(&self.w.mul_vec(&self.prior()))
    }

    /// Observe the next symbol: infer the hidden state by relaxing prediction
    /// errors, then apply local weight updates.
    pub fn observe(&mut self, sym: usize) -> StepStats {
        assert!(sym < self.cfg.vocab, "symbol out of range");
        let z_prev = self.z.clone();
        let prior = self.prior();
        let predicted = softmax(&self.w.mul_vec(&prior));
        let surprise = -predicted[sym].max(1e-12).ln();
        let correct = argmax(&predicted) == sym;

        let mut x = vec![0.0; self.cfg.vocab];
        x[sym] = 1.0;

        // Inference: descend the free energy over z.
        let mut z = prior.clone();
        for _ in 0..self.cfg.infer_steps {
            let ex = sub(&x, &softmax(&self.w.mul_vec(&z)));
            let back = self.w.mul_t_vec(&ex);
            for i in 0..z.len() {
                z[i] += self.cfg.infer_rate * (back[i] - (z[i] - prior[i]));
            }
        }

        // Learning: local, error-times-activity updates.
        let ex = sub(&x, &softmax(&self.w.mul_vec(&z)));
        self.w.add_outer(self.cfg.lr_out, &ex, &z);
        let dz: Vec<f64> = z
            .iter()
            .zip(&prior)
            .map(|(zi, pi)| (zi - pi) * (1.0 - pi * pi))
            .collect();
        self.a.add_outer(self.cfg.lr_trans, &dz, &z_prev);

        self.z = z;
        StepStats {
            predicted,
            surprise,
            correct,
        }
    }
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

pub fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}
