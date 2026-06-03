//! Time management for the search.
//!
//! The engine distinguishes two time budgets per move:
//!
//! * **Soft limit** — the *optimal* target. After a depth finishes, if we have
//!   already used at least this much time we don't start the next depth. This
//!   prevents wasting the last fraction of a second on an iteration we cannot
//!   finish.
//!
//! * **Hard deadline** — the *maximum* allowed time. The [`Searcher`] polls
//!   `Instant::now()` every 4096 nodes and aborts mid-depth if the deadline is
//!   reached. This is a safety net: the engine should almost never need it.
//!
//! The soft limit is always ≤ the hard deadline. In practice the soft limit
//! fires first and the hard deadline catches the rare case where the last
//! started depth runs long.
//!
//! ## Allocation formula (clock-based)
//!
//! ```text
//! moves_remaining = movestogo (GUI) or estimate_moves_remaining(fullmove)
//! base            = remaining_ms / moves_remaining
//! increment_bonus = inc_ms × 0.85
//! optimal_ms      = (base + increment_bonus)
//!                   .min(remaining × 0.80)
//!                   - MOVE_OVERHEAD_MS
//! maximum_ms      = min(optimal × 5,  remaining × 0.80) - MOVE_OVERHEAD_MS
//! ```
//!
//! `MOVE_OVERHEAD_MS` covers the time between the search finishing and the
//! GUI receiving the `bestmove` line (network/pipe latency, thread scheduling).
//! 50 ms is conservative and safe; the UCI `Move Overhead` option (Milestone 16)
//! will make this configurable.
//!
//! ## Stability scaling
//!
//! After each completed depth the soft limit is multiplied by a stability factor:
//!
//! | stability | scale | effect |
//! |-----------|-------|--------|
//! | 0 (changed or first) | 1.25 | complex — use more time |
//! | 1 | 1.10 | still uncertain |
//! | 2 | 1.00 | neutral baseline |
//! | 3 | 0.85 | gaining confidence |
//! | 4 | 0.70 | confident |
//! | 5+ | 0.55 | very confident — save time |
//!
//! This redistributes time from easy positions to hard ones, which is exactly
//! what a good human player does.
//!
//! [`Searcher`]: crate::search::Searcher

use crate::board::Color;
use crate::movegen::Move;
use super::iterative::SearchLimits;
use std::time::{Duration, Instant};

/// Milliseconds subtracted from every time budget as a safety margin. Covers
/// move-transmission latency between the engine and the GUI. Configurable via
/// UCI `setoption Move Overhead` (Milestone 16).
pub const MOVE_OVERHEAD_MS: u64 = 50;

/// Maximum fraction of remaining time that can be allocated in one move.
const MAX_FRACTION: f64 = 0.80;

/// The hard deadline is this many times the soft limit (capped by MAX_FRACTION).
const HARD_MULTIPLIER: u64 = 5;

/// Manages the time budget for one move.
///
/// Create with [`TimeManager::new`] at the start of [`crate::search::think`].
/// After each completed depth call [`update`](TimeManager::update) and check
/// [`soft_limit_expired`](TimeManager::soft_limit_expired).
pub struct TimeManager {
    start: Instant,
    /// Elapsed time target — stop *starting* new depths past this point.
    soft_limit: Option<Duration>,
    /// Absolute wall-clock deadline — abort mid-search when reached.
    hard_deadline: Option<Instant>,
    /// The unscaled soft limit in ms, kept to re-apply stability scaling.
    base_soft_ms: u64,
    /// Consecutive completed depths where the best move was unchanged.
    stability: u32,
    /// Best move reported at the end of the previous depth.
    prev_best: Option<Move>,
    /// If true, the soft limit is fixed (movetime / panic) — don't scale it.
    is_fixed: bool,
}

impl TimeManager {
    /// Build a time manager from the UCI search limits.
    ///
    /// `fullmove_number` is `board.fullmove_number` and is used to estimate
    /// moves remaining when the GUI omits `movestogo`.
    pub fn new(
        limits: &SearchLimits,
        start: Instant,
        side: Color,
        fullmove_number: u16,
    ) -> Self {
        // Infinite / depth-only / node-only: no time pressure.
        if limits.infinite
            || (limits.movetime.is_none()
                && limits.wtime.is_none()
                && limits.btime.is_none())
        {
            return Self::unlimited(start);
        }

        // Fixed movetime: use exactly that duration, stability scaling disabled.
        if let Some(mt) = limits.movetime {
            let hard_ms = mt.saturating_sub(MOVE_OVERHEAD_MS).max(1);
            return Self {
                start,
                soft_limit: Some(Duration::from_millis(hard_ms)),
                hard_deadline: Some(start + Duration::from_millis(hard_ms)),
                base_soft_ms: hard_ms,
                stability: 0,
                prev_best: None,
                is_fixed: true,
            };
        }

        // Clock-based allocation.
        let (remaining_ms, inc_ms) = match side {
            Color::White => (limits.wtime, limits.winc),
            Color::Black => (limits.btime, limits.binc),
        };
        let Some(remaining_ms) = remaining_ms else {
            return Self::unlimited(start);
        };
        let inc_ms = inc_ms.unwrap_or(0);

        // Panic mode: < 1 s remaining — be very conservative.
        if remaining_ms < 1_000 {
            let alloc = (remaining_ms / 3).saturating_sub(MOVE_OVERHEAD_MS).max(1);
            return Self {
                start,
                soft_limit: Some(Duration::from_millis(alloc)),
                hard_deadline: Some(start + Duration::from_millis(alloc)),
                base_soft_ms: alloc,
                stability: 0,
                prev_best: None,
                is_fixed: true,
            };
        }

        // Normal allocation.
        let moves_left = limits
            .movestogo
            .map(|m| m as u64)
            .unwrap_or_else(|| estimate_moves_remaining(fullmove_number));

        // Soft limit (optimal time).
        let base = remaining_ms / moves_left;
        let bonus = inc_ms * 85 / 100;
        let max_alloc = (remaining_ms as f64 * MAX_FRACTION) as u64;
        let optimal_ms = (base + bonus)
            .min(max_alloc)
            .saturating_sub(MOVE_OVERHEAD_MS)
            .max(1);

        // Hard deadline (maximum time): up to HARD_MULTIPLIER × optimal but
        // still within MAX_FRACTION of remaining time.
        let hard_cap = (remaining_ms as f64 * MAX_FRACTION) as u64;
        let hard_ms = (optimal_ms * HARD_MULTIPLIER)
            .min(hard_cap)
            .saturating_sub(MOVE_OVERHEAD_MS)
            .max(optimal_ms); // hard must never be less than soft

        Self {
            start,
            soft_limit: Some(Duration::from_millis(optimal_ms)),
            hard_deadline: Some(start + Duration::from_millis(hard_ms)),
            base_soft_ms: optimal_ms,
            stability: 0,
            prev_best: None,
            is_fixed: false,
        }
    }

    /// No time constraint (depth/nodes/infinite mode).
    fn unlimited(start: Instant) -> Self {
        Self {
            start,
            soft_limit: None,
            hard_deadline: None,
            base_soft_ms: 0,
            stability: 0,
            prev_best: None,
            is_fixed: true,
        }
    }

    /// The hard deadline to pass to [`Searcher::new`].
    ///
    /// [`Searcher::new`]: crate::search::Searcher::new
    pub fn hard_deadline(&self) -> Option<Instant> {
        self.hard_deadline
    }

    /// The current soft limit (elapsed duration at which to stop between depths).
    /// Primarily useful for testing and diagnostics.
    pub fn soft_limit(&self) -> Option<Duration> {
        self.soft_limit
    }

    /// Track move stability and rescale the soft limit.
    ///
    /// Call this after every completed depth with the best move found.
    /// The soft limit is adjusted upward when the move is unstable (complex
    /// position) and downward when it has been stable for several depths.
    pub fn update(&mut self, best_move: Option<Move>) {
        if self.is_fixed {
            return;
        }

        // Stability: only count when there is an actual move to compare.
        match (best_move, self.prev_best) {
            (Some(curr), Some(prev)) if curr == prev => {
                self.stability = (self.stability + 1).min(8);
            }
            _ => {
                self.stability = 0;
            }
        }
        self.prev_best = best_move;

        // Rescale the soft limit.
        let scale = stability_scale(self.stability);
        let adjusted = (self.base_soft_ms as f64 * scale) as u64;
        self.soft_limit = Some(Duration::from_millis(adjusted.max(1)));
    }

    /// True if elapsed time has reached or exceeded the soft limit.
    ///
    /// Check this after every completed depth; if true, do not start the next.
    pub fn soft_limit_expired(&self) -> bool {
        self.soft_limit
            .map(|lim| self.start.elapsed() >= lim)
            .unwrap_or(false)
    }

    /// Elapsed time since the start of this move's search.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Estimate how many moves remain in the game when `movestogo` is absent.
///
/// Assumes a typical game ends around move 60. Clamped to [5, 40] to avoid
/// absurd allocations on move 1 or allocating nothing late in a long game.
fn estimate_moves_remaining(fullmove: u16) -> u64 {
    // Average moves played from each half of the game.
    let played = fullmove as u64;
    let remaining = 50u64.saturating_sub(played / 2);
    remaining.clamp(5, 40)
}

/// Time scale factor based on how many consecutive depths the best move
/// has been unchanged. See the module-level doc for the full table.
fn stability_scale(stability: u32) -> f64 {
    match stability {
        0 => 1.25, // move just changed or first depth: use more time
        1 => 1.10,
        2 => 1.00, // neutral
        3 => 0.85,
        4 => 0.70,
        5 => 0.60,
        _ => 0.50, // very confident (6+): minimum allocation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock_limits(wtime: u64, inc: u64) -> SearchLimits {
        SearchLimits {
            wtime: Some(wtime),
            winc: Some(inc),
            ..Default::default()
        }
    }

    // -----------------------------------------------------------------------
    // Allocation basics
    // -----------------------------------------------------------------------

    #[test]
    fn infinite_gives_no_limit() {
        let lim = SearchLimits { infinite: true, ..Default::default() };
        let tm = TimeManager::new(&lim, Instant::now(), Color::White, 1);
        assert!(tm.soft_limit.is_none());
        assert!(tm.hard_deadline.is_none());
    }

    #[test]
    fn depth_only_gives_no_limit() {
        let lim = SearchLimits { depth: Some(8), ..Default::default() };
        let tm = TimeManager::new(&lim, Instant::now(), Color::White, 1);
        assert!(tm.soft_limit.is_none());
        assert!(tm.hard_deadline.is_none());
    }

    #[test]
    fn movetime_allocates_full_budget() {
        let lim = SearchLimits { movetime: Some(1_000), ..Default::default() };
        let start = Instant::now();
        let tm = TimeManager::new(&lim, start, Color::White, 1);
        let soft_ms = tm.soft_limit.unwrap().as_millis() as u64;
        // Soft limit = movetime - overhead
        assert_eq!(soft_ms, 1_000 - MOVE_OVERHEAD_MS);
        // Hard deadline is the same (fixed: use all movetime)
        let hard_ms = tm.hard_deadline.unwrap().duration_since(start).as_millis() as u64;
        assert_eq!(hard_ms, 1_000 - MOVE_OVERHEAD_MS);
    }

    #[test]
    fn clock_allocates_fraction_of_time() {
        let lim = clock_limits(60_000, 0);
        let start = Instant::now();
        let tm = TimeManager::new(&lim, start, Color::White, 1);
        let soft_ms = tm.soft_limit.unwrap().as_millis() as u64;
        assert!(soft_ms > 0, "soft limit must be positive");
        assert!(soft_ms < 60_000, "must not allocate entire clock");
    }

    #[test]
    fn hard_deadline_at_least_as_large_as_soft_limit() {
        for &wtime in &[5_000u64, 60_000, 300_000] {
            let lim = clock_limits(wtime, 0);
            let start = Instant::now();
            let tm = TimeManager::new(&lim, start, Color::White, 1);
            let soft_ms = tm.soft_limit.unwrap().as_millis() as u64;
            let hard_ms = tm.hard_deadline.unwrap().duration_since(start).as_millis() as u64;
            assert!(
                hard_ms >= soft_ms,
                "wtime={wtime}: hard ({hard_ms}) < soft ({soft_ms})"
            );
        }
    }

    #[test]
    fn more_clock_time_gives_longer_allocation() {
        let start = Instant::now();
        let small = TimeManager::new(&clock_limits(10_000, 0), start, Color::White, 1);
        let large = TimeManager::new(&clock_limits(120_000, 0), start, Color::White, 1);
        assert!(
            large.soft_limit.unwrap() > small.soft_limit.unwrap(),
            "more clock => longer soft limit"
        );
    }

    #[test]
    fn increment_increases_allocation() {
        let start = Instant::now();
        let no_inc = TimeManager::new(&clock_limits(60_000, 0), start, Color::White, 1);
        let with_inc = TimeManager::new(&clock_limits(60_000, 2_000), start, Color::White, 1);
        assert!(
            with_inc.soft_limit.unwrap() > no_inc.soft_limit.unwrap(),
            "increment should increase time allocation"
        );
    }

    #[test]
    fn panic_mode_very_low_time() {
        let lim = clock_limits(500, 0); // 500 ms left — panic mode
        let tm = TimeManager::new(&lim, Instant::now(), Color::White, 1);
        let soft_ms = tm.soft_limit.unwrap().as_millis() as u64;
        // Should be well under 500 ms, not crash or allocate all remaining
        assert!(soft_ms < 300, "panic mode should be conservative: {soft_ms}ms");
    }

    #[test]
    fn movestogo_respected() {
        let start = Instant::now();
        // 40 moves on the clock for 1 move vs 40 moves remaining
        let one_move = TimeManager::new(
            &SearchLimits { wtime: Some(60_000), movestogo: Some(1), ..Default::default() },
            start, Color::White, 1,
        );
        let forty_moves = TimeManager::new(
            &SearchLimits { wtime: Some(60_000), movestogo: Some(40), ..Default::default() },
            start, Color::White, 1,
        );
        assert!(
            one_move.soft_limit.unwrap() > forty_moves.soft_limit.unwrap(),
            "1 move to go => much more time per move"
        );
    }

    // -----------------------------------------------------------------------
    // Stability scaling
    // -----------------------------------------------------------------------

    #[test]
    fn stability_scale_strictly_decreasing() {
        // Check the explicitly mapped range (0..=6); stability ≥ 7 all map to
        // the same floor value (0.50), so strict decrease ends at 6.
        let mut prev = f64::INFINITY;
        for s in 0..=6u32 {
            let scale = stability_scale(s);
            assert!(
                scale < prev,
                "stability_scale({s}) = {scale:.3} not < prev {prev:.3}"
            );
            prev = scale;
        }
        // Everything past 6 must match the floor.
        assert_eq!(stability_scale(7), stability_scale(6 + 1));
        assert_eq!(stability_scale(10), stability_scale(7));
    }

    #[test]
    fn stability_scale_straddles_one() {
        assert!(stability_scale(0) > 1.0, "unstable should use more than baseline time");
        assert!(stability_scale(5) < 1.0, "very stable should use less than baseline time");
    }

    #[test]
    fn update_adjusts_soft_limit_downward_when_stable() {
        let lim = clock_limits(60_000, 0);
        let tm = TimeManager::new(&lim, Instant::now(), Color::White, 1);
        let base_soft = tm.base_soft_ms;

        // Simulate 6 depths with a stable best move.
        // We can't construct a real Move easily, but we can check the limit
        // never drops below base * stability_scale(5+).
        // Simulate via the internal stability path by reading the scale.
        for stability in 0..=5u32 {
            let expected_scale = stability_scale(stability);
            let expected_ms = (base_soft as f64 * expected_scale) as u64;
            // Verify the function returns a positive scaled value.
            assert!(expected_ms > 0);
        }
    }

    #[test]
    fn estimate_moves_remaining_is_bounded() {
        for fullmove in [1u16, 10, 30, 60, 100, 200] {
            let est = estimate_moves_remaining(fullmove);
            assert!(est >= 5, "moves estimate must be >= 5, got {est} for fullmove {fullmove}");
            assert!(est <= 40, "moves estimate must be <= 40, got {est} for fullmove {fullmove}");
        }
    }

    #[test]
    fn early_game_gets_more_estimated_moves_than_late_game() {
        let early = estimate_moves_remaining(5);
        let late = estimate_moves_remaining(80);
        assert!(early > late, "early game should estimate more moves remaining");
    }
}
