//! Tapered scores.
//!
//! A single number can't capture that a knight on the rim is bad in the
//! middlegame but the king belongs in the corner *then* and in the center
//! *later*. So every evaluation term carries two numbers — a **middlegame** (mg)
//! value and an **endgame** (eg) value — bundled in a [`Score`].
//!
//! At the end we collapse the two into one centipawn number by **tapering**:
//! interpolating between mg and eg according to how much material is left on the
//! board (the "game phase"). A full board uses the mg value; a bare-kings
//! endgame uses the eg value; everything between is a weighted blend. This makes
//! the engine's priorities shift smoothly as pieces come off.

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// The maximum game phase (full starting material). See [`taper`].
///
/// Knight/bishop = 1, rook = 2, queen = 4; two of each per side -> 24.
pub const TOTAL_PHASE: i32 = 24;

/// A paired middlegame / endgame score, in centipawns.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Score {
    pub mg: i32,
    pub eg: i32,
}

impl Score {
    pub const ZERO: Score = Score { mg: 0, eg: 0 };

    #[inline]
    pub const fn new(mg: i32, eg: i32) -> Score {
        Score { mg, eg }
    }
}

/// Collapse a paired score into a single centipawn value for the given game
/// `phase` (0 = pure endgame .. [`TOTAL_PHASE`] = full middlegame).
#[inline]
pub fn taper(score: Score, phase: i32) -> i32 {
    let phase = phase.clamp(0, TOTAL_PHASE);
    (score.mg * phase + score.eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE
}

impl Add for Score {
    type Output = Score;
    #[inline]
    fn add(self, rhs: Score) -> Score {
        Score::new(self.mg + rhs.mg, self.eg + rhs.eg)
    }
}
impl Sub for Score {
    type Output = Score;
    #[inline]
    fn sub(self, rhs: Score) -> Score {
        Score::new(self.mg - rhs.mg, self.eg - rhs.eg)
    }
}
impl Neg for Score {
    type Output = Score;
    #[inline]
    fn neg(self) -> Score {
        Score::new(-self.mg, -self.eg)
    }
}
/// Scale a score by an integer count (e.g. per mobility square).
impl Mul<i32> for Score {
    type Output = Score;
    #[inline]
    fn mul(self, rhs: i32) -> Score {
        Score::new(self.mg * rhs, self.eg * rhs)
    }
}
impl AddAssign for Score {
    #[inline]
    fn add_assign(&mut self, rhs: Score) {
        self.mg += rhs.mg;
        self.eg += rhs.eg;
    }
}
impl SubAssign for Score {
    #[inline]
    fn sub_assign(&mut self, rhs: Score) {
        self.mg -= rhs.mg;
        self.eg -= rhs.eg;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taper_interpolates() {
        let s = Score::new(100, 200);
        assert_eq!(taper(s, TOTAL_PHASE), 100); // full middlegame -> mg
        assert_eq!(taper(s, 0), 200); // pure endgame -> eg
        assert_eq!(taper(s, TOTAL_PHASE / 2), 150); // halfway -> blend
    }

    #[test]
    fn arithmetic() {
        let a = Score::new(10, 20);
        let b = Score::new(3, 4);
        assert_eq!(a + b, Score::new(13, 24));
        assert_eq!(a - b, Score::new(7, 16));
        assert_eq!(b * 5, Score::new(15, 20));
        assert_eq!(-a, Score::new(-10, -20));
    }
}
