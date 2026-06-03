//! Automated evaluation tuning (Texel method).
//!
//! ## Quick-start
//!
//! ```sh
//! # 1. Generate a self-play dataset (≈1000 games at depth 4, ~30 s).
//! checksmith datagen 1000 4 positions.txt
//!
//! # 2. Tune for up to 20 sweeps and print changed parameters.
//! checksmith tune positions.txt 20
//! ```
//!
//! The `tune` command prints a Rust snippet for every parameter that changed
//! during tuning — paste those values back into the source constants to apply
//! the improvement.
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`params`] | [`EvalParams`], [`EvalTrace`], `compute_trace`, `score_from_trace` |
//! | [`dataset`] | text-file loader / saver |
//! | [`datagen`] | self-play position generator |
//! | [`texel`] | sigmoid, MSE loss, `find_k`, coordinate descent |

pub mod datagen;
pub mod dataset;
pub mod params;
pub mod texel;

pub use params::{compute_trace, EvalParams, EvalTrace, N_PARAMS};

use std::path::Path;

// ─── CLI entry points ─────────────────────────────────────────────────────────

/// `checksmith datagen <games> <depth> <output-file>`
pub fn run_datagen(args: &[String]) {
    let usage = "usage: checksmith datagen <games> <depth> <output.txt>";
    let n_games: usize = args.get(0).and_then(|s| s.parse().ok())
        .unwrap_or_else(|| { eprintln!("{}", usage); std::process::exit(1); });
    let depth: u32 = args.get(1).and_then(|s| s.parse().ok())
        .unwrap_or_else(|| { eprintln!("{}", usage); std::process::exit(1); });
    let path = Path::new(args.get(2).unwrap_or_else(|| {
        eprintln!("{}", usage); std::process::exit(1);
    }));

    eprintln!("generating {} self-play games at depth {}…", n_games, depth);
    match datagen::generate(n_games, depth, path) {
        Ok(n) => println!("done: {} positions written to {}", n, path.display()),
        Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
    }
}

/// `checksmith tune <input-file> [max-sweeps]`
pub fn run_tune(args: &[String]) {
    let usage = "usage: checksmith tune <positions.txt> [max-sweeps=10]";
    let path = Path::new(args.get(0).unwrap_or_else(|| {
        eprintln!("{}", usage); std::process::exit(1);
    }));
    let max_sweeps: usize = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // Load dataset.
    eprintln!("loading dataset from {}…", path.display());
    let entries = match dataset::load(path) {
        Ok(e) if e.is_empty() => { eprintln!("error: empty dataset"); std::process::exit(1); }
        Ok(e) => e,
        Err(err) => { eprintln!("error loading dataset: {}", err); std::process::exit(1); }
    };

    // Split train / val.
    let (train, val) = texel::train_val_split(entries, 0.1);

    // Initialise parameters from the hand-crafted evaluator.
    let baseline = EvalParams::from_engine();
    let mut params = baseline.clone();

    // Find the optimal K scaling constant.
    eprintln!("finding K constant…");
    let k = texel::find_k(&train, &params, 0.1, 3.0);
    eprintln!("K = {:.4}", k);

    // Run tuning.
    eprintln!("running coordinate descent (max {} sweeps)…", max_sweeps);
    let sweeps = texel::tune(&train, &val, &mut params, k, max_sweeps);
    eprintln!("completed {} sweep(s)", sweeps);

    // Report changes.
    println!("\n=== Parameters that changed ===");
    params.print_diff(&baseline);

    println!("\n=== Rust snippet for changed scalar weights ===");
    print_rust_snippet(&params, &baseline);
}

fn print_rust_snippet(tuned: &EvalParams, baseline: &EvalParams) {
    macro_rules! maybe_print {
        ($name:expr, $t:expr, $b:expr) => {
            if $t != $b {
                println!("const {}: Score = Score::new({}, {});", $name, $t.mg, $t.eg);
            }
        };
    }
    maybe_print!("KNIGHT_MOBILITY", tuned.knight_mobility, baseline.knight_mobility);
    maybe_print!("BISHOP_MOBILITY", tuned.bishop_mobility, baseline.bishop_mobility);
    maybe_print!("ROOK_MOBILITY",   tuned.rook_mobility,   baseline.rook_mobility);
    maybe_print!("QUEEN_MOBILITY",  tuned.queen_mobility,  baseline.queen_mobility);
    maybe_print!("BISHOP_PAIR",     tuned.bishop_pair,     baseline.bishop_pair);
    maybe_print!("BAD_BISHOP_PAWN", tuned.bad_bishop_pawn, baseline.bad_bishop_pawn);
    maybe_print!("ROOK_OPEN_FILE",  tuned.rook_open_file,  baseline.rook_open_file);
    maybe_print!("ROOK_SEMI_OPEN",  tuned.rook_semi_open,  baseline.rook_semi_open);
    maybe_print!("ROOK_ON_SEVENTH", tuned.rook_on_seventh, baseline.rook_on_seventh);
    maybe_print!("KING_SHIELD_PAWN",     tuned.king_shield_pawn,     baseline.king_shield_pawn);
    maybe_print!("KING_OPEN_FILE",       tuned.king_open_file,       baseline.king_open_file);
    maybe_print!("KING_SEMI_OPEN",       tuned.king_semi_open,       baseline.king_semi_open);
    maybe_print!("KING_ATTACKER_KNIGHT", tuned.king_attacker_knight, baseline.king_attacker_knight);
    maybe_print!("KING_ATTACKER_BISHOP", tuned.king_attacker_bishop, baseline.king_attacker_bishop);
    maybe_print!("KING_ATTACKER_ROOK",   tuned.king_attacker_rook,   baseline.king_attacker_rook);
    maybe_print!("KING_ATTACKER_QUEEN",  tuned.king_attacker_queen,  baseline.king_attacker_queen);
    maybe_print!("OUTPOST_KNIGHT",  tuned.outpost_knight, baseline.outpost_knight);
    maybe_print!("OUTPOST_BISHOP",  tuned.outpost_bishop, baseline.outpost_bishop);
    maybe_print!("ISOLATED",        tuned.isolated,       baseline.isolated);
    maybe_print!("DOUBLED",         tuned.doubled,        baseline.doubled);
}
