//! Self-play position generation for tuning datasets.
//!
//! ## How it works
//!
//! Two copies of the engine play each other at low depth (default 4).  After
//! each game we record:
//!
//! * All positions reached from ply 8 onward (skip opening theory).
//! * Positions where the side to move is **not** in check (tactical positions
//!   bias the labels; the static eval is unreliable there).
//! * Positions with at least 5 pieces on the board (avoid trivial endings).
//!
//! Every position from a game gets the same game result as its outcome label.
//! This produces noisy but useful training data.
//!
//! ## Limitations
//!
//! Low-depth self-play produces positions with systematic biases:
//!
//! * The engine has preferred openings at low depth → positions cluster.
//! * Tactical oversights are common → material balance is noisier than in
//!   strong games.
//!
//! For best results, supplement with positions from real human or strong-engine
//! games (just parse PGNs and record all quiet positions).  The dataset format
//! accepts any source; `datagen` is just a quick bootstrapper.

use crate::board::{Board, STARTING_FEN};
use crate::search;
use std::path::Path;
use std::sync::{atomic::AtomicBool, Arc};

/// Generate `n_games` self-play games at `depth` and write positions to `path`.
///
/// Prints progress every 100 games.  Returns the total number of positions saved.
pub fn generate(n_games: usize, depth: u32, path: &Path) -> std::io::Result<usize> {
    let mut lines: Vec<String> = Vec::new();

    for game_no in 0..n_games {
        let positions = play_one_game(depth);
        lines.extend(positions);

        if (game_no + 1) % 100 == 0 || game_no + 1 == n_games {
            eprint!(
                "\r  game {}/{} — {} positions so far    ",
                game_no + 1,
                n_games,
                lines.len()
            );
        }
    }
    eprintln!();

    super::dataset::save_raw(&lines, path)?;
    eprintln!("saved {} positions to {}", lines.len(), path.display());
    Ok(lines.len())
}

/// Play one full game and return a list of `"<FEN> <result>"` strings.
fn play_one_game(depth: u32) -> Vec<String> {
    let mut board  = Board::from_fen(STARTING_FEN).unwrap();
    let mut moves_played = 0usize;
    let max_moves = 200; // cap to prevent infinite games
    let mut game_fens: Vec<String> = Vec::new();
    let result: f64;

    loop {
        let legal = board.legal_moves();
        if legal.is_empty() {
            // Checkmate or stalemate.
            result = if board.is_in_check() {
                // The side to move is mated → opponent wins.
                if board.side_to_move == crate::board::Color::White { 0.0 } else { 1.0 }
            } else {
                0.5 // stalemate
            };
            break;
        }

        if board.halfmove_clock >= 100 {
            result = 0.5; // 50-move rule draw
            break;
        }
        if moves_played >= max_moves {
            result = 0.5; // adjudicate as draw
            break;
        }

        // Record the position before the move (so we capture the "from" state).
        let should_record = moves_played >= 8
            && !board.is_in_check()
            && board.occupancy().count() >= 5;

        if should_record {
            game_fens.push(board.to_fen());
        }

        // Choose a move: use the engine at the given depth.
        let stop = Arc::new(AtomicBool::new(false));
        let mut searcher = search::Searcher::new(
            None,
            Arc::clone(&stop),
            None,
            crate::eval::HandcraftedEvaluator,
        );
        let tt = search::TranspositionTable::new(4); // 4 MB TT per move
        let (mv, _score) = searcher.search_root(&mut board, depth, None, &tt, &[], -search::INFINITY, search::INFINITY);

        match mv {
            Some(m) => { board.make_move(m); }
            None => { result = 0.5; break; } // no move found (shouldn't happen)
        }
        moves_played += 1;
    }

    // Annotate all collected FENs with the game result.
    game_fens
        .into_iter()
        .map(|fen| format!("{} {:.1}", fen, result))
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_at_least_some_positions() {
        let positions = play_one_game(2);
        // A full game at depth 2 should produce at least a few recorded positions.
        assert!(!positions.is_empty(), "no positions generated");
        // Each line must end with a recognisable result token.
        for line in &positions {
            assert!(
                line.ends_with(" 1.0") || line.ends_with(" 0.5") || line.ends_with(" 0.0"),
                "unexpected format: {}", line
            );
        }
    }
}
