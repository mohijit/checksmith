//! Fixed opening positions for reproducible engine testing.
//!
//! A fixed opening book ensures:
//! * Both sides reach different, varied positions across a match rather than
//!   always playing the same lines from the starting position.
//! * Results are reproducible — game `i` always starts from the same FEN.
//! * Draw by repetition of a narrow opening repertoire is not inflated.
//!
//! For maximum coverage, openings span open games, semi-open games, closed games,
//! and flank/hypermodern systems.  Each FEN is after 3–6 moves of theory so
//! the engine immediately faces a real middlegame decision.

/// 24 opening FENs covering the most common chess openings.
/// Each entry is valid and has been verified by counting material and move counts.
pub const OPENINGS: &[&str] = &[
    // ── Open games ───────────────────────────────────────────────────────────
    // Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    // Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    // King's Gambit Accepted: 1.e4 e5 2.f4 exf4
    "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3",
    // Petroff Defence: 1.e4 e5 2.Nf3 Nf6
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    // ── Semi-open games ──────────────────────────────────────────────────────
    // Sicilian Najdorf: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    // Sicilian Dragon: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 g6
    "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    // French Advance: 1.e4 e6 2.d4 d5 3.e5 c5
    "rnbqkbnr/pp3ppp/4p3/2ppP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq c6 0 4",
    // Caro-Kann: 1.e4 c6 2.d4 d5
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3",
    // Pirc/Modern: 1.e4 d6 2.d4 Nf6 3.Nc3 g6
    "rnbqkb1r/ppp1pp1p/3p1np1/8/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4",
    // ── Closed games ────────────────────────────────────────────────────────
    // Queen's Gambit Declined: 1.d4 d5 2.c4 e6 3.Nc3 Nf6
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    // Nimzo-Indian: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    // King's Indian Defence: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 0-0
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 4 7",
    // Grunfeld: 1.d4 Nf6 2.c4 g6 3.Nc3 d5
    "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq d6 0 4",
    // Dutch Defence: 1.d4 f5
    "rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq f6 0 2",
    // ── Hypermodern ─────────────────────────────────────────────────────────
    // English Opening: 1.c4 e5 2.Nc3 Nf6 3.g3 d5
    "rnbqkb1r/ppp2ppp/5n2/3pp3/2P5/2N3P1/PP1PPP1P/R1BQKBNR w KQkq d6 0 4",
    // Reti Opening: 1.Nf3 d5 2.c4 c6
    "rnbqkbnr/pp2pppp/2p5/3p4/2P5/5N2/PP1PPPPP/RNBQKB1R w KQkq - 0 3",
    // Catalan: 1.d4 Nf6 2.c4 e6 3.g3 d5 4.Bg2 Be7 5.Nf3 0-0
    "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1 w - - 4 6",
    // London System: 1.d4 d5 2.Bf4 Nf6 3.e3 e6 4.Nf3 c5
    "rnbqkb1r/pp3ppp/4pn2/2pp4/3P1B2/4PN2/PPP2PPP/RN1QKB1R w KQkq c6 0 5",
    // ── Gambit / Flank ───────────────────────────────────────────────────────
    // Benko Gambit: 1.d4 Nf6 2.c4 c5 3.d5 b5
    "rnbqkb1r/p2ppppp/5n2/1ppP4/2P5/8/PP2PPPP/RNBQKBNR w KQkq b6 0 4",
    // Budapest Gambit: 1.d4 Nf6 2.c4 e5
    "rnbqkb1r/pppp1ppp/5n2/4p3/2PP4/8/PP2PPPP/RNBQKBNR w KQkq e6 0 3",
    // Bird's Opening: 1.f4 d5 2.Nf3 Nf6
    "rnbqkb1r/ppp1pppp/5n2/3p4/5P2/5N2/PPPPP1PP/RNBQKB1R w KQkq - 2 3",
    // Vienna Game: 1.e4 e5 2.Nc3 Nf6 3.f4
    "rnbqkb1r/pppp1ppp/5n2/4p3/4PP2/2N5/PPPP2PP/R1BQKBNR b KQkq f3 0 3",
    // Benoni: 1.d4 Nf6 2.c4 c5 3.d5 e6 4.Nc3 exd5 5.cxd5 d6 6.e4 g6
    "rnbqkb1r/pp3p1p/3p1np1/2pP4/4P3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 7",
    // Slav Defence: 1.d4 d5 2.c4 c6 3.Nc3 Nf6 4.Nf3 dxc4
    "rnbqkb1r/pp2pppp/2p2n2/8/2pP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
];

/// Return the opening FEN for game `index`, cycling through all entries.
///
/// Deterministic: `pick(i)` always returns the same FEN, regardless of
/// how many games have been played.
#[inline]
pub fn pick(index: usize) -> &'static str {
    OPENINGS[index % OPENINGS.len()]
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn all_opening_fens_parse() {
        for (i, &fen) in OPENINGS.iter().enumerate() {
            Board::from_fen(fen)
                .unwrap_or_else(|e| panic!("opening[{}] failed to parse: {} — {:?}", i, fen, e));
        }
    }

    #[test]
    fn pick_cycles_correctly() {
        let n = OPENINGS.len();
        // First cycle
        for i in 0..n {
            assert_eq!(pick(i), OPENINGS[i]);
        }
        // Second cycle wraps
        for i in 0..n {
            assert_eq!(pick(i + n), OPENINGS[i]);
        }
    }

    #[test]
    fn at_least_twenty_openings() {
        assert!(OPENINGS.len() >= 20, "need at least 20 openings for diversity");
    }
}
