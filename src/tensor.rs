//! Minimal dense matrix and vector helpers (row-major, f64).

use crate::rng::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize, scale: f64, rng: &mut Rng) -> Self {
        let data = (0..rows * cols).map(|_| rng.symmetric(scale)).collect();
        Mat { rows, cols, data }
    }

    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// y = M x
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        (0..self.rows)
            .map(|r| {
                self.data[r * self.cols..(r + 1) * self.cols]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    /// y = Mᵀ x
    pub fn mul_t_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.rows);
        let mut y = vec![0.0; self.cols];
        for (r, xr) in x.iter().enumerate() {
            for (c, yc) in y.iter_mut().enumerate() {
                *yc += self.data[r * self.cols + c] * xr;
            }
        }
        y
    }

    /// M += scale * a bᵀ
    pub fn add_outer(&mut self, scale: f64, a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), self.rows);
        assert_eq!(b.len(), self.cols);
        for (r, ar) in a.iter().enumerate() {
            for (c, bc) in b.iter().enumerate() {
                self.data[r * self.cols + c] += scale * ar * bc;
            }
        }
    }
}

/// Numerically stable softmax.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_and_transpose() {
        let m = Mat {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        assert_eq!(m.mul_vec(&[1.0, 0.0, -1.0]), vec![-2.0, -2.0]);
        assert_eq!(m.mul_t_vec(&[1.0, 1.0]), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn outer_product_update() {
        let mut m = Mat::zeros(2, 2);
        m.add_outer(2.0, &[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(m.data, vec![6.0, 8.0, 12.0, 16.0]);
        assert_eq!(m.at(1, 0), 12.0);
    }

    #[test]
    fn softmax_sums_to_one_and_is_stable() {
        let p = softmax(&[1000.0, 1000.0, 999.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[0] > p[2]);
    }
}
