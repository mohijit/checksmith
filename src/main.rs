//! Driver for Checksmith.
//!
//! With no arguments (or `uci`), it speaks the UCI protocol on stdin/stdout —
//! this is how chess GUIs launch it. The other subcommands are dev conveniences:
//!
//!   checksmith                              # enter the UCI loop (default)
//!   checksmith uci                          # enter the UCI loop
//!   checksmith show [<fen>]                 # print a position
//!   checksmith perft <depth> [<fen>]        # run a perft divide
//!   checksmith go <depth> [<fen>]           # fixed-depth search, print best move
//!   checksmith bench [depth]                # node-count benchmark (default depth 10)
//!   checksmith test  [max-depth]            # tactical test suite (default depth 8)
//!   checksmith epd <file.epd> [depth]       # run an EPD test suite (default depth 8)
//!   checksmith nps                          # quick perft-5 NPS measurement
//!   checksmith verify [depth]               # search correctness verifier (default depth 5)
//!   checksmith match [games] [depth]        # internal self-play regression match
//!   checksmith datagen <games> <d> <out>    # generate self-play training positions
//!   checksmith tune <positions.txt> [iters] # Texel-tune evaluation parameters
//!   checksmith tune-spsa [games] [iters]    # SPSA search-parameter tuning

use checksmith::board::{Board, STARTING_FEN};
use checksmith::search::{is_mate_score, mate_distance_plies, search};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(String::as_str) {
        None | Some("uci") => checksmith::uci::run(),
        Some("perft")   => run_perft(&args[2..]),
        Some("go")      => run_search(&args[2..]),
        Some("show")    => run_show(&args[2..]),
        Some("bench")   => checksmith::bench::run(&args[2..]),
        Some("test")    => checksmith::bench::run_test(&args[2..]),
        Some("epd")     => checksmith::epd::run_epd_command(&args[2..]),
        Some("nps")     => {
            let nps = checksmith::bench::measure_nps();
            println!("NPS: {} ({:.1} Mnps)", nps, nps as f64 / 1_000_000.0);
        }
        Some("verify")  => checksmith::debug::run_verify(&args[2..]),
        Some("match")   => checksmith::selfplay::run_match(&args[2..]),
        Some("datagen")   => checksmith::tune::run_datagen(&args[2..]),
        Some("tune")      => checksmith::tune::run_tune(&args[2..]),
        Some("tune-spsa") => checksmith::tune::spsa::run_spsa_command(&args[2..]),
        // Back-compat: a bare FEN argument prints the position.
        Some(_) => run_show(&args[1..]),
    }
}

/// Handle `show [<fen>]`: print a board, its FEN, and legal-move count.
fn run_show(args: &[String]) {
    let fen = if args.is_empty() {
        STARTING_FEN.to_string()
    } else {
        args.join(" ")
    };
    match Board::from_fen(&fen) {
        Ok(board) => {
            println!("{}", board);
            println!();
            println!("FEN (round-trip): {}", board.to_fen());
            println!("Legal moves     : {}", board.legal_moves().len());
        }
        Err(err) => {
            eprintln!("Failed to parse FEN: {}", err);
            std::process::exit(1);
        }
    }
}

/// Handle `go <depth> [<fen>]`: fixed-depth search, print score and best move.
fn run_search(args: &[String]) {
    let depth: u32 = match args.first().and_then(|d| d.parse().ok()) {
        Some(d) => d,
        None => {
            eprintln!("usage: checksmith go <depth> [<fen>]");
            std::process::exit(1);
        }
    };
    let fen = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        STARTING_FEN.to_string()
    };

    let mut board = match Board::from_fen(&fen) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("Failed to parse FEN: {}", err);
            std::process::exit(1);
        }
    };

    let start = Instant::now();
    let result = search(&mut board, depth);
    let elapsed = start.elapsed();

    // Format the score the way a UCI engine would: mate distance or centipawns.
    let score_str = if is_mate_score(result.score) {
        let plies = mate_distance_plies(result.score);
        // Convert plies to "moves to mate", signed for who is winning.
        let moves = (plies + 1) / 2;
        format!("mate {}", if result.score > 0 { moves } else { -moves })
    } else {
        format!("cp {}", result.score)
    };

    let secs = elapsed.as_secs_f64();
    let nps = if secs > 0.0 {
        result.nodes as f64 / secs
    } else {
        0.0
    };
    println!(
        "depth {}  score {}  nodes {}  nps {:.0}  time {:.3}s",
        result.depth, score_str, result.nodes, nps, secs
    );
    match result.best_move {
        Some(mv) => println!("bestmove {}", mv.to_uci()),
        None => println!("bestmove (none)"),
    }
}

/// Handle `perft <depth> [<fen>]`.
fn run_perft(args: &[String]) {
    let depth: u32 = match args.first().and_then(|d| d.parse().ok()) {
        Some(d) => d,
        None => {
            eprintln!("usage: checksmith perft <depth> [<fen>]");
            std::process::exit(1);
        }
    };
    let fen = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        STARTING_FEN.to_string()
    };

    let mut board = match Board::from_fen(&fen) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("Failed to parse FEN: {}", err);
            std::process::exit(1);
        }
    };

    let start = Instant::now();
    let (total, breakdown) = board.perft_divide(depth);
    let elapsed = start.elapsed();

    // Sorted output matches the conventional `perft divide` format for easy
    // diffing against reference engines like Stockfish.
    let mut breakdown = breakdown;
    breakdown.sort_by_key(|(mv, _)| mv.to_uci());
    for (mv, count) in &breakdown {
        println!("{}: {}", mv.to_uci(), count);
    }
    println!();
    println!("Nodes searched: {}", total);
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        println!(
            "Time: {:.3}s  ({:.2} Mnps)",
            secs,
            total as f64 / secs / 1_000_000.0
        );
    }
}
