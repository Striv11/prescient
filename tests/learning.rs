use prescient::{Config, Predictor};

fn run(p: &mut Predictor, seq: &[usize], reps: usize) -> (f64, f64) {
    let (mut s, mut c, mut n) = (0.0, 0, 0);
    for _ in 0..reps {
        for &x in seq {
            let st = p.observe(x);
            s += st.surprise;
            c += st.correct as usize;
            n += 1;
        }
    }
    (s / n as f64, c as f64 / n as f64)
}

#[test]
fn learns_a_cycle() {
    let seq = [0, 1, 2, 3];
    let mut p = Predictor::new(Config::new(4));
    let (s0, _) = run(&mut p, &seq, 5);
    run(&mut p, &seq, 300);
    let (s1, acc) = run(&mut p, &seq, 20);
    assert!(s1 < s0 * 0.2, "surprise {s0} -> {s1}");
    assert!(acc > 0.95, "accuracy {acc}");
}

#[test]
fn learns_melody_with_repeated_notes() {
    let seq = [0, 2, 4, 2, 0, 2, 4, 5];
    let mut p = Predictor::new(Config::new(6));
    run(&mut p, &seq, 600);
    let (_, acc) = run(&mut p, &seq, 20);
    assert!(acc > 0.9, "accuracy {acc}");
}

#[test]
fn surprise_spikes_on_violation() {
    let seq = [0, 1, 2, 3];
    let mut p = Predictor::new(Config::new(4));
    run(&mut p, &seq, 400);
    p.observe(0);
    let normal = p.observe(1).surprise;
    let violated = p.observe(3).surprise; // expected 2
    assert!(violated > normal * 5.0, "{normal} vs {violated}");
}

#[test]
fn deterministic_for_seed() {
    let seq = [0, 1, 0, 2];
    let mut a = Predictor::new(Config::new(3));
    let mut b = Predictor::new(Config::new(3));
    assert_eq!(run(&mut a, &seq, 50), run(&mut b, &seq, 50));
}

#[test]
fn predictions_are_distributions() {
    let mut p = Predictor::new(Config::new(5));
    p.observe(3);
    let d = p.predict();
    assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(d.iter().all(|x| *x >= 0.0));
}
