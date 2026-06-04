//! SPSA (Simultaneous Perturbation Stochastic Approximation) search-parameter tuner.
//!
//! ## How it works
//!
//! SPSA estimates the gradient of the engine's win-rate with respect to each
//! search parameter by playing two short matches per iteration:
//!
//! 1. **Perturb** the current parameter vector θ in a random ±1 direction Δ
//!    (each component chosen uniformly at random).
//! 2. **Configure** two subprocess engines: one with θ+ = θ + c·Δ, the other
//!    with θ− = θ − c·Δ (component-wise, clamped to the parameter bounds).
//! 3. **Play** `games_per_iter` games between θ+ (White in even games) and
//!    θ− (Black in even games), alternating colors every game.
//! 4. **Estimate** the gradient: g = (score(θ+) − 0.5) × Δ / c.
//! 5. **Update** θ ← θ + a·g, then clamp to bounds.
//!
//! The sequences `a_k = a / (A + k + 1)^α` and `c_k = c / (k + 1)^γ` decay
//! over iterations according to the Spall (1992) schedule.
//!
//! ## CLI
//!
//! ```sh
//! checksmith tune-spsa                       # 10 games/iter, 100 iters, depth 5
//! checksmith tune-spsa 20 200 6             # 20 games/iter, 200 iters, depth 6
//! ```
//!
//! After every iteration the current parameter values are printed.
//! After all iterations a copy-pasteable Rust snippet is printed with the
//! `set_*` calls needed to apply the tuned values.
//!
//! ## Subprocess isolation
//!
//! Each iteration spawns two copies of the **current binary** via
//! `std::env::current_exe()`, communicates with them over stdin/stdout using
//! the UCI protocol, and terminates them with `quit` when the iteration ends.
//! This guarantees both engines run in fully isolated processes with their own
//! TTs, hash tables, and search state.
//!
//! ## Hyperparameter defaults
//!
//! | Symbol | Default | Meaning |
//! |--------|---------|---------|
//! | A      | 0.1 × iters | Stability offset in the a-sequence |
//! | a      | 0.5     | Initial learning rate |
//! | c      | 1.0     | Perturbation scaling (multiplied by `c_step` per param) |
//! | α      | 0.602   | Decay exponent for a-sequence (Spall 1992) |
//! | γ      | 0.101   | Decay exponent for c-sequence (Spall 1992) |

use crate::search::params::{ParamMeta, ALL_PARAMS};
use crate::selfplay::uci_process::UciProcess;
use crate::selfplay::{GameOutcome, MatchConfig, TimeControl, play_game_dyn};
use crate::selfplay::ExternalEngine;
use std::time::Instant;

// ── SPSA hyperparameters ──────────────────────────────────────────────────────

const ALPHA: f64 = 0.602;
const GAMMA: f64 = 0.101;

// ── Parameter state ───────────────────────────────────────────────────────────

/// Runtime state for a single tunable parameter during a SPSA run.
#[derive(Clone)]
struct Param {
    meta:  &'static ParamMeta,
    value: f64, // continuous-valued internal representation; rounded when applied
}

impl Param {
    fn from_meta(m: &'static ParamMeta) -> Self {
        Param { meta: m, value: m.default as f64 }
    }

    fn clamped_int(&self) -> i32 {
        self.value.round() as i32
    }

    fn apply_perturbed(&self, delta: f64, c_k: f64) -> i32 {
        let v = self.value + delta * self.meta.c_step * c_k;
        v.round().clamp(self.meta.min as f64, self.meta.max as f64) as i32
    }

    fn update(&mut self, gradient: f64, a_k: f64) {
        self.value += a_k * gradient * self.meta.c_step;
        self.value = self.value.clamp(self.meta.min as f64, self.meta.max as f64);
    }
}

// ── SPSA runner ───────────────────────────────────────────────────────────────

/// Run a full SPSA tuning session.
///
/// `games_per_iter` — games played per iteration (split evenly between the
///   two perturbed variants).
/// `iterations` — total SPSA iterations.
/// `depth` — per-move search depth for every game.
pub fn run(games_per_iter: usize, iterations: usize, depth: u32) {
    let exe = match std::env::current_exe() {
        Ok(p)  => p,
        Err(e) => { eprintln!("cannot find current executable: {}", e); return; }
    };
    let exe_str = exe.to_string_lossy().to_string();

    let mut params: Vec<Param> = ALL_PARAMS.iter().map(Param::from_meta).collect();

    // SPSA hyperparameters.
    let big_a = (0.1 * iterations as f64).max(1.0);
    let a_init = 0.5_f64;   // learning rate scale
    let c_init = 1.0_f64;   // perturbation scale (multiplied by c_step)

    let config = MatchConfig {
        games: games_per_iter,
        tc: TimeControl::FixedDepth(depth),
        paired: true, // alternate colors per game
        ..Default::default()
    };

    println!(
        "Checksmith SPSA  params={}  games/iter={}  iters={}  depth={}",
        params.len(), games_per_iter, iterations, depth,
    );
    println!("Parameters:");
    for p in &params {
        println!("  {:20}  current={:4}  range=[{},{}]  step={}",
            p.meta.name, p.clamped_int(), p.meta.min, p.meta.max, p.meta.c_step);
    }
    println!();

    let total_start = Instant::now();

    for iter in 0..iterations {
        let k   = iter as f64;
        let a_k = a_init / (big_a + k + 1.0).powf(ALPHA);
        let c_k = c_init / (k + 1.0).powf(GAMMA);

        // Random ±1 perturbation vector (one sign per parameter).
        let deltas: Vec<f64> = params.iter().map(|_| {
            if pseudo_rand(iter, 0) { 1.0 } else { -1.0 }
        }).collect();

        // Compute θ+ and θ- for each param (integer-rounded, clamped).
        let vals_plus:  Vec<i32> = params.iter().zip(&deltas)
            .map(|(p, &d)| p.apply_perturbed(d, c_k)).collect();
        let vals_minus: Vec<i32> = params.iter().zip(&deltas)
            .map(|(p, &d)| p.apply_perturbed(-d, c_k)).collect();

        // Spawn two subprocess engines and configure their parameters.
        let mut eng_plus  = match configure_engine(&exe_str, &params, &vals_plus) {
            Ok(e)  => e,
            Err(e) => { eprintln!("iter {}: cannot launch engine+: {}", iter+1, e); continue; }
        };
        let mut eng_minus = match configure_engine(&exe_str, &params, &vals_minus) {
            Ok(e)  => e,
            Err(e) => { eprintln!("iter {}: cannot launch engine-: {}", iter+1, e); continue; }
        };

        // Play the mini-match and measure W/D/L from eng_plus's perspective.
        let (wins, draws, losses) = play_mini_match(
            &mut eng_plus, &mut eng_minus, &config, iter,
        );
        let n = (wins + draws + losses) as f64;
        if n == 0.0 { continue; }
        let score = (wins as f64 + draws as f64 * 0.5) / n;

        // Gradient estimate and parameter update.
        for (i, param) in params.iter_mut().enumerate() {
            let gradient = deltas[i] * (score - 0.5) / c_k;
            param.update(gradient, a_k);
        }

        let elapsed = total_start.elapsed().as_secs_f64();
        println!(
            "Iter {:>4}/{}  W:{} D:{} L:{}  score:{:.3}  a_k:{:.4}  c_k:{:.4}  {:.1}s",
            iter + 1, iterations, wins, draws, losses, score, a_k, c_k, elapsed,
        );
        // Print current parameter values every 10 iterations.
        if (iter + 1) % 10 == 0 || iter + 1 == iterations {
            println!("  Current params:");
            for p in &params {
                println!("    {:20} = {}", p.meta.name, p.clamped_int());
            }
        }
    }

    // Final report.
    println!();
    println!("===========================");
    println!("SPSA tuning complete.  Apply these values:");
    println!();
    for p in &params {
        let v = p.clamped_int();
        if v != p.meta.default {
            println!("  setoption name {} value {}  // was {}", p.meta.name, v, p.meta.default);
        }
    }
    println!("===========================");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Launch a subprocess engine and send setoption commands for perturbed values.
fn configure_engine(
    exe: &str,
    params: &[Param],
    values: &[i32],
) -> std::io::Result<ExternalEngine> {
    let mut process = UciProcess::launch(exe)?;
    for (param, &val) in params.iter().zip(values) {
        process.send_setoption(param.meta.name, val)?;
    }
    Ok(ExternalEngine::from_process(process))
}

/// Play `config.games` games between `plus` (even games = White) and `minus`.
/// Returns `(wins, draws, losses)` from `plus`'s perspective.
fn play_mini_match(
    plus:   &mut ExternalEngine,
    minus:  &mut ExternalEngine,
    config: &MatchConfig,
    iter:   usize,
) -> (u64, u64, u64) {
    let (mut w, mut d, mut l) = (0u64, 0u64, 0u64);

    for game in 0..config.games {
        let fen = config.openings.pick(iter * config.games + game);
        let plus_is_white = game % 2 == 0;

        let record = if plus_is_white {
            play_game_dyn(fen, config, plus, minus)
        } else {
            play_game_dyn(fen, config, minus, plus)
        };

        let outcome_for_plus = if plus_is_white {
            record.outcome
        } else {
            match record.outcome {
                GameOutcome::WhiteWins => GameOutcome::BlackWins,
                GameOutcome::BlackWins => GameOutcome::WhiteWins,
                o => o,
            }
        };

        match outcome_for_plus {
            GameOutcome::WhiteWins => w += 1,
            GameOutcome::BlackWins => l += 1,
            GameOutcome::Draw(_)   => d += 1,
            GameOutcome::Aborted(_)=> {}
        }
    }

    (w, d, l)
}

/// Deterministic pseudo-random bool from (iter, index) — avoids a rand dep.
fn pseudo_rand(iter: usize, idx: usize) -> bool {
    // Xorshift-inspired: mix iter and idx, check the low bit.
    let mut x = (iter.wrapping_mul(2654435761) ^ idx.wrapping_mul(1234567891)) as u64;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x & 1 == 0
}

// ── CLI entry ─────────────────────────────────────────────────────────────────

/// `checksmith tune-spsa [games_per_iter] [iterations] [depth]`
pub fn run_spsa_command(args: &[String]) {
    let games  = args.first().and_then(|s| s.parse().ok()).unwrap_or(10usize);
    let iters  = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100usize);
    let depth  = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5u32);
    run(games, iters, depth);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_from_meta_uses_default() {
        for m in ALL_PARAMS {
            let p = Param::from_meta(m);
            assert_eq!(p.clamped_int(), m.default);
        }
    }

    #[test]
    fn apply_perturbed_clamps_to_range() {
        let m = &ALL_PARAMS[0]; // FutilityMargin1: default=250, min=100, max=500
        let p = Param::from_meta(m);
        // Large positive perturbation should be clamped to max.
        let v = p.apply_perturbed(1.0, 1_000_000.0);
        assert_eq!(v, m.max);
        // Large negative perturbation should be clamped to min.
        let v = p.apply_perturbed(-1.0, 1_000_000.0);
        assert_eq!(v, m.min);
    }

    #[test]
    fn update_moves_value_in_gradient_direction() {
        let m = &ALL_PARAMS[0];
        let mut p = Param::from_meta(m);
        let before = p.value;
        p.update(1.0, 0.5); // positive gradient → value increases
        assert!(p.value > before, "positive gradient should increase value");
    }

    #[test]
    fn pseudo_rand_produces_both_values() {
        let values: Vec<bool> = (0..100).map(|i| pseudo_rand(i, 0)).collect();
        assert!(values.contains(&true),  "pseudo_rand must produce true");
        assert!(values.contains(&false), "pseudo_rand must produce false");
    }

    #[test]
    fn pseudo_rand_is_deterministic() {
        for i in 0..20 {
            assert_eq!(pseudo_rand(i, 0), pseudo_rand(i, 0));
        }
    }
}
