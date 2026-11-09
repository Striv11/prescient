use prescient::{Config, Predictor};

const MELODY: [usize; 8] = [0, 2, 4, 2, 0, 2, 4, 5]; // C E G E C E G A
const NAMES: [&str; 6] = ["C", "D", "E", "F", "G", "A"];

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "demo".into());
    if mode != "demo" {
        eprintln!("usage: prescient [demo]");
        std::process::exit(2);
    }
    let mut p = Predictor::new(Config::new(NAMES.len()));
    println!("Online learning of a repeating melody (C E G E C E G A)");
    println!("{:>6} {:>10} {:>9}", "epoch", "surprise", "accuracy");
    for epoch in 1..=60 {
        let (mut s, mut c) = (0.0, 0);
        let n = MELODY.len() * 4;
        for i in 0..n {
            let st = p.observe(MELODY[i % MELODY.len()]);
            s += st.surprise;
            c += st.correct as usize;
        }
        if epoch == 1 || epoch % 10 == 0 {
            println!(
                "{:>6} {:>10.3} {:>8.0}%",
                epoch,
                s / n as f64,
                100.0 * c as f64 / n as f64
            );
        }
    }
    let dist = p.predict();
    let top = prescient::model::argmax(&dist);
    println!(
        "next-note prediction: {} ({:.0}% confident)",
        NAMES[top],
        dist[top] * 100.0
    );
}
