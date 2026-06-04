//! Runtime-configurable search parameters for SPSA tuning.
//!
//! Seven continuous search constants are exposed as global atomics so they can
//! be changed at runtime via UCI `setoption` — without recompiling.  This
//! makes automatic SPSA tuning possible: spawn two copies of the binary with
//! different parameter values and play them against each other.
//!
//! ## Atomic read cost
//!
//! Each accessor uses `Ordering::Relaxed`, which on x86-64 compiles to a plain
//! `mov` instruction — identical cost to a regular load.  The atomics are never
//! written during a search, only between searches, so there is no contention.
//!
//! ## Default values
//!
//! The defaults match the compile-time constants that were previously embedded
//! directly in `negamax.rs` and `iterative.rs`.  Changing a default here does
//! not require touching any other source file.
//!
//! ## UCI option names
//!
//! ```text
//! option name FutilityMargin1  type spin default 250 min 100 max 500
//! option name FutilityMargin2  type spin default 500 min 200 max 800
//! option name RazorMargin      type spin default 300 min 100 max 600
//! option name ProbcutMargin    type spin default 200 min  50 max 400
//! option name AspDelta         type spin default  50 min  15 max 150
//! option name LmpBase          type spin default   3 min   1 max   8
//! option name SeDepthFactor    type spin default   6 min   3 max  15
//! ```

use std::sync::atomic::{AtomicI32, Ordering};

// ── Internal statics ──────────────────────────────────────────────────────────

static FUTILITY_MARGIN_1: AtomicI32 = AtomicI32::new(250);
static FUTILITY_MARGIN_2: AtomicI32 = AtomicI32::new(500);
static RAZOR_MARGIN:      AtomicI32 = AtomicI32::new(300);
static PROBCUT_MARGIN:    AtomicI32 = AtomicI32::new(200);
static ASP_DELTA:         AtomicI32 = AtomicI32::new(50);
static LMP_BASE:          AtomicI32 = AtomicI32::new(3);
static SE_DEPTH_FACTOR:   AtomicI32 = AtomicI32::new(6);

// ── Read accessors (inlined — zero overhead) ──────────────────────────────────

#[inline] pub fn futility_margin_1() -> i32   { FUTILITY_MARGIN_1.load(Ordering::Relaxed) }
#[inline] pub fn futility_margin_2() -> i32   { FUTILITY_MARGIN_2.load(Ordering::Relaxed) }
#[inline] pub fn razor_margin()      -> i32   { RAZOR_MARGIN.load(Ordering::Relaxed) }
#[inline] pub fn probcut_margin()    -> i32   { PROBCUT_MARGIN.load(Ordering::Relaxed) }
#[inline] pub fn asp_delta()         -> i32   { ASP_DELTA.load(Ordering::Relaxed) }
#[inline] pub fn lmp_base()          -> usize { LMP_BASE.load(Ordering::Relaxed) as usize }
#[inline] pub fn lmp_base_i32()      -> i32   { LMP_BASE.load(Ordering::Relaxed) }
#[inline] pub fn se_depth_factor()   -> i32   { SE_DEPTH_FACTOR.load(Ordering::Relaxed) }

// ── Write accessors (called from setoption and the SPSA tuner) ────────────────

pub fn set_futility_margin_1(v: i32) { FUTILITY_MARGIN_1.store(v.clamp(100, 500), Ordering::Relaxed); }
pub fn set_futility_margin_2(v: i32) { FUTILITY_MARGIN_2.store(v.clamp(200, 800), Ordering::Relaxed); }
pub fn set_razor_margin(v: i32)      { RAZOR_MARGIN.store(v.clamp(100, 600), Ordering::Relaxed); }
pub fn set_probcut_margin(v: i32)    { PROBCUT_MARGIN.store(v.clamp(50, 400), Ordering::Relaxed); }
pub fn set_asp_delta(v: i32)         { ASP_DELTA.store(v.clamp(15, 150), Ordering::Relaxed); }
pub fn set_lmp_base(v: i32)          { LMP_BASE.store(v.clamp(1, 8), Ordering::Relaxed); }
pub fn set_se_depth_factor(v: i32)   { SE_DEPTH_FACTOR.store(v.clamp(3, 15), Ordering::Relaxed); }

// ── Parameter metadata (consumed by the SPSA tuner) ──────────────────────────

/// Description of one tunable parameter, used by the SPSA optimizer.
pub struct ParamMeta {
    /// UCI option name (case-insensitive match in `setoption`).
    pub name: &'static str,
    /// Lower bound enforced by the setter.
    pub min: i32,
    /// Upper bound enforced by the setter.
    pub max: i32,
    /// Default / initial value.
    pub default: i32,
    /// Typical perturbation magnitude for one SPSA step.
    /// Should be large enough to produce a detectable score difference in a
    /// short match (usually 10–25 cp for margins, 1 for integer factors).
    pub c_step: f64,
    /// Read the current value.
    pub get: fn() -> i32,
    /// Write a new value (clamped internally).
    pub set: fn(i32),
}

/// All parameters eligible for SPSA tuning, in a stable order.
pub const ALL_PARAMS: &[ParamMeta] = &[
    ParamMeta {
        name: "FutilityMargin1", min: 100, max: 500, default: 250, c_step: 25.0,
        get: futility_margin_1, set: set_futility_margin_1,
    },
    ParamMeta {
        name: "FutilityMargin2", min: 200, max: 800, default: 500, c_step: 50.0,
        get: futility_margin_2, set: set_futility_margin_2,
    },
    ParamMeta {
        name: "RazorMargin", min: 100, max: 600, default: 300, c_step: 25.0,
        get: razor_margin, set: set_razor_margin,
    },
    ParamMeta {
        name: "ProbcutMargin", min: 50, max: 400, default: 200, c_step: 25.0,
        get: probcut_margin, set: set_probcut_margin,
    },
    ParamMeta {
        name: "AspDelta", min: 15, max: 150, default: 50, c_step: 10.0,
        get: asp_delta, set: set_asp_delta,
    },
    ParamMeta {
        name: "LmpBase", min: 1, max: 8, default: 3, c_step: 1.0,
        get: lmp_base_i32, set: set_lmp_base,
    },
    ParamMeta {
        name: "SeDepthFactor", min: 3, max: 15, default: 6, c_step: 1.0,
        get: se_depth_factor, set: set_se_depth_factor,
    },
];

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_constants() {
        assert_eq!(futility_margin_1(), 250);
        assert_eq!(futility_margin_2(), 500);
        assert_eq!(razor_margin(),      300);
        assert_eq!(probcut_margin(),    200);
        assert_eq!(asp_delta(),          50);
        assert_eq!(lmp_base(),            3);
        assert_eq!(se_depth_factor(),     6);
    }

    #[test]
    fn setters_clamp_to_range() {
        // Save originals.
        let orig = futility_margin_1();
        set_futility_margin_1(99_999);
        assert_eq!(futility_margin_1(), 500);
        set_futility_margin_1(-1);
        assert_eq!(futility_margin_1(), 100);
        // Restore.
        set_futility_margin_1(orig);
        assert_eq!(futility_margin_1(), orig);
    }

    #[test]
    fn all_params_metadata_is_consistent() {
        for p in ALL_PARAMS {
            assert!(p.min < p.max, "{}: min must be < max", p.name);
            assert!(p.default >= p.min && p.default <= p.max,
                "{}: default {} not in [{}, {}]", p.name, p.default, p.min, p.max);
            assert!(p.c_step > 0.0, "{}: c_step must be positive", p.name);
        }
    }

    #[test]
    fn param_get_returns_current_value() {
        for p in ALL_PARAMS {
            let current = (p.get)();
            assert!(current >= p.min && current <= p.max,
                "{}: current value {} out of range [{}, {}]", p.name, current, p.min, p.max);
        }
    }
}
