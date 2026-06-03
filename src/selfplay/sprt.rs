//! Sequential Probability Ratio Test (SPRT) for engine testing.
//!
//! ## What SPRT is
//!
//! SPRT is a statistical test that terminates a match as soon as there is
//! enough evidence to accept or reject a hypothesis.  For engine testing the
//! hypotheses are:
//!
//! * **H0**: the test engine has Elo gain ≤ `elo0` vs the baseline (no improvement).
//! * **H1**: the test engine has Elo gain ≥ `elo1` vs the baseline (improvement).
//!
//! Standard parameters: `elo0 = 0`, `elo1 = 5`, `alpha = beta = 0.05`.
//! This means we accept at most a 5% false-positive rate and a 5% false-negative rate.
//!
//! ## Formula
//!
//! Given W wins, D draws, L losses:
//! ```text
//! N  = W + D + L
//! s  = (W + D/2) / N           (observed score)
//! p0 = expected_score(elo0)    (expected score under H0)
//! p1 = expected_score(elo1)    (expected score under H1)
//!
//! LLR = N × [s × ln(p1/p0) + (1−s) × ln((1−p1)/(1−p0))]
//!
//! lo  = ln(β / (1−α))          (accept H0 when LLR < lo)
//! hi  = ln((1−β) / α)          (accept H1 when LLR > hi)
//! ```
//!
//! ## Cute Chess CLI integration
//!
//! For process-level testing against other engines, Cute Chess CLI has built-in
//! SPRT support:
//!
//! ```sh
//! cutechess-cli \
//!   -engine cmd=./target/release/checksmith name=dev \
//!   -engine cmd=./target/release/checksmith_base name=base \
//!   -each  proto=uci tc=10+0.1 \
//!   -games 10000 -concurrency 4 \
//!   -openings file=openings.epd format=epd order=random \
//!   -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
//!   -ratinginterval 100
//! ```
//!
//! [`SprtTest`] can be used independently of Cute Chess to evaluate results
//! from any match source — internal self-play, exported PGN, or manual input.

// ─── Elo math ─────────────────────────────────────────────────────────────────

/// Convert a win-rate `score ∈ (0, 1)` to an Elo difference.
///
/// Returns `±∞` at the extremes (to signal a perfect or perfectly losing record).
pub fn elo_from_score(score: f64) -> f64 {
    if score <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if score >= 1.0 {
        return f64::INFINITY;
    }
    -400.0 * (1.0 / score - 1.0).log10()
}

/// Convert an Elo difference to an expected score (logistic model).
///
/// `expected_score(0) == 0.5`, `expected_score(+∞) → 1`.
pub fn expected_score(elo_diff: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-elo_diff / 400.0))
}

// ─── SprtConfig ───────────────────────────────────────────────────────────────

/// Configuration for one sequential probability ratio test.
#[derive(Clone, Debug)]
pub struct SprtConfig {
    /// Elo gain under the null hypothesis (typically 0.0 — no improvement).
    pub elo0: f64,
    /// Elo gain under the alternative hypothesis (typically 5.0 — detectable gain).
    pub elo1: f64,
    /// False-positive rate: P(accept H1 | H0 true).  Standard: 0.05.
    pub alpha: f64,
    /// False-negative rate: P(accept H0 | H1 true).  Standard: 0.05.
    pub beta: f64,
}

impl Default for SprtConfig {
    fn default() -> Self {
        SprtConfig {
            elo0:  0.0,
            elo1:  5.0,
            alpha: 0.05,
            beta:  0.05,
        }
    }
}

// ─── SprtOutcome ──────────────────────────────────────────────────────────────

/// Decision from the current SPRT state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SprtOutcome {
    /// Neither boundary has been crossed.  Play more games.
    Continue,
    /// LLR ≤ lo: H0 accepted (no meaningful improvement detected).
    H0Accepted,
    /// LLR ≥ hi: H1 accepted (improvement is statistically significant).
    H1Accepted,
}

// ─── SprtTest ─────────────────────────────────────────────────────────────────

/// An incremental SPRT test.
///
/// Feed game results with [`update`](SprtTest::update) and query
/// [`outcome`](SprtTest::outcome) after each batch.
///
/// # Example
///
/// ```rust,ignore
/// let mut test = SprtTest::new(SprtConfig::default());
/// test.update(5, 8, 7);
/// println!("{}", test); // SPRT(0/5) LLR=… [lo, hi]  W:5 D:8 L:7  Elo:…
/// ```
#[derive(Clone, Debug)]
pub struct SprtTest {
    pub config: SprtConfig,
    pub wins:   u64,
    pub draws:  u64,
    pub losses: u64,
    /// Current log-likelihood ratio.  Updated by every [`update`](SprtTest::update) call.
    pub llr: f64,
}

impl SprtTest {
    /// Create a new test with the given configuration and zero results.
    pub fn new(config: SprtConfig) -> Self {
        SprtTest { config, wins: 0, draws: 0, losses: 0, llr: 0.0 }
    }

    /// Add `w` wins, `d` draws, and `l` losses to the running totals and
    /// recompute the LLR.
    pub fn update(&mut self, w: u64, d: u64, l: u64) {
        self.wins   += w;
        self.draws  += d;
        self.losses += l;
        self.llr = self.compute_llr();
    }

    /// Current test decision.
    pub fn outcome(&self) -> SprtOutcome {
        if self.llr <= self.lo_bound() {
            SprtOutcome::H0Accepted
        } else if self.llr >= self.hi_bound() {
            SprtOutcome::H1Accepted
        } else {
            SprtOutcome::Continue
        }
    }

    /// Lower LLR boundary.  Accept H0 when LLR falls below this.
    pub fn lo_bound(&self) -> f64 {
        (self.config.beta / (1.0 - self.config.alpha)).ln()
    }

    /// Upper LLR boundary.  Accept H1 when LLR rises above this.
    pub fn hi_bound(&self) -> f64 {
        ((1.0 - self.config.beta) / self.config.alpha).ln()
    }

    /// Total games counted.
    pub fn games(&self) -> u64 {
        self.wins + self.draws + self.losses
    }

    /// Observed score for the test engine: `(W + D/2) / N`.
    pub fn score(&self) -> f64 {
        let n = self.games();
        if n == 0 {
            return 0.5;
        }
        (self.wins as f64 + self.draws as f64 / 2.0) / n as f64
    }

    /// Elo estimate from the observed score.
    pub fn elo_estimate(&self) -> f64 {
        elo_from_score(self.score())
    }

    fn compute_llr(&self) -> f64 {
        let n = self.games();
        if n == 0 {
            return 0.0;
        }
        let s = self.score();
        if s <= 0.0 || s >= 1.0 {
            return 0.0;
        }
        let p0 = expected_score(self.config.elo0);
        let p1 = expected_score(self.config.elo1);
        // Guard against degenerate configs (elo0 == elo1 → p0 == p1 → ln(1) = 0).
        if (p0 - p1).abs() < 1e-12 || p0 <= 0.0 || p1 <= 0.0 || p0 >= 1.0 || p1 >= 1.0 {
            return 0.0;
        }
        n as f64 * (s * (p1 / p0).ln() + (1.0 - s) * ((1.0 - p1) / (1.0 - p0)).ln())
    }
}

impl std::fmt::Display for SprtTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SPRT({:.0}/{:.0})  LLR={:.2} [{:.2}, {:.2}]  W:{} D:{} L:{}  Elo:{:+.1}",
            self.config.elo0,
            self.config.elo1,
            self.llr,
            self.lo_bound(),
            self.hi_bound(),
            self.wins,
            self.draws,
            self.losses,
            self.elo_estimate(),
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_score_at_zero_elo() {
        let s = expected_score(0.0);
        assert!((s - 0.5).abs() < 1e-10, "expected_score(0) should be 0.5, got {}", s);
    }

    #[test]
    fn expected_score_increases_with_elo() {
        let s100 = expected_score(100.0);
        let s200 = expected_score(200.0);
        assert!(s100 > 0.5, "positive Elo → score > 0.5");
        assert!(s200 > s100, "more Elo → higher score");
    }

    #[test]
    fn elo_from_score_round_trips() {
        for score in [0.3, 0.5, 0.6, 0.75, 0.9] {
            let elo = elo_from_score(score);
            let back = expected_score(elo);
            assert!((back - score).abs() < 1e-9, "round-trip failed for score={}", score);
        }
    }

    #[test]
    fn elo_from_score_extremes() {
        assert!(elo_from_score(0.0).is_infinite());
        assert!(elo_from_score(1.0).is_infinite());
        assert_eq!(elo_from_score(0.5), 0.0);
    }

    #[test]
    fn sprt_bounds_with_default_config() {
        let test = SprtTest::new(SprtConfig::default());
        let lo = test.lo_bound();
        let hi = test.hi_bound();
        // Standard α=β=0.05 → lo ≈ −2.944, hi ≈ 2.944
        assert!(lo < 0.0, "lo must be negative");
        assert!(hi > 0.0, "hi must be positive");
        assert!((lo + hi).abs() < 1e-9, "symmetric bounds when alpha == beta");
    }

    #[test]
    fn sprt_starts_as_continue() {
        let test = SprtTest::new(SprtConfig::default());
        assert_eq!(test.outcome(), SprtOutcome::Continue);
        assert_eq!(test.games(), 0);
    }

    #[test]
    fn sprt_accepts_h1_after_many_wins() {
        let mut test = SprtTest::new(SprtConfig::default());
        // Simulate a very strong engine (score ≈ 0.8 → Elo +240).
        // After 1000 games, LLR should exceed hi bound.
        test.update(800, 100, 100);
        assert_eq!(
            test.outcome(),
            SprtOutcome::H1Accepted,
            "strong result should accept H1; LLR={:.2}",
            test.llr
        );
    }

    #[test]
    fn sprt_accepts_h0_after_many_losses() {
        let mut test = SprtTest::new(SprtConfig::default());
        // Score ≈ 0.2 → clearly negative Elo → accept H0.
        test.update(100, 100, 800);
        assert_eq!(
            test.outcome(),
            SprtOutcome::H0Accepted,
            "weak result should accept H0; LLR={:.2}",
            test.llr
        );
    }

    #[test]
    fn sprt_display_contains_key_fields() {
        let mut test = SprtTest::new(SprtConfig::default());
        test.update(10, 5, 5);
        let s = test.to_string();
        assert!(s.contains("SPRT(0/5)"), "display should show elo0/elo1");
        assert!(s.contains("LLR="), "display should show LLR");
        assert!(s.contains("W:10"), "display should show wins");
    }

    #[test]
    fn sprt_incremental_matches_batch() {
        let config = SprtConfig::default();
        let mut incremental = SprtTest::new(config.clone());
        for _ in 0..10 {
            incremental.update(6, 2, 2);
        }
        let mut batch = SprtTest::new(config);
        batch.update(60, 20, 20);
        assert!((incremental.llr - batch.llr).abs() < 1e-9, "incremental vs batch LLR mismatch");
    }
}
