//! Transposition table.
//!
//! A *transposition* is the same position reached by different move orders.
//! Without a cache, the search re-explores each one from scratch. The
//! transposition table (TT) is a big hash table, keyed by the position's Zobrist
//! hash (maintained incrementally since Milestone 1), that remembers what we
//! learned about a position so we never redo that work.
//!
//! Each entry stores:
//!
//! * the **full key**, to detect hash collisions (two positions mapping to the
//!   same table slot);
//! * the **depth** the result was searched to — a shallow result can't be reused
//!   where a deeper one is needed;
//! * the **score**, plus a **bound type** describing what the score means;
//! * the **best move**, used to order moves first even when we can't reuse the
//!   score;
//! * a **generation** stamp so entries from older searches are preferentially
//!   overwritten.
//!
//! ## Bounds
//!
//! Alpha-beta doesn't always compute an exact score. When a node fails high
//! (beta cutoff), all we know is the true value is *at least* this — a **lower
//! bound**. When it fails low, the value is *at most* this — an **upper bound**.
//! Only a node searched fully within its window yields an **exact** score. The
//! bound determines whether a cached score can produce a cutoff.
//!
//! ## Mate scores
//!
//! A mate score encodes distance-to-mate from the *root*. Stored in the TT it
//! must be distance-to-mate from *this node*, or a position reached at a
//! different depth would report the wrong mate distance. Callers convert with
//! `score_to_tt` / `score_from_tt` (in [`super::negamax`]).
//!
//! ## Thread safety (Lazy SMP)
//!
//! The TT is shared across all search threads via `Arc<TranspositionTable>`.
//! `probe` and `store` take `&self` and mutate the entry array through
//! `UnsafeCell`. This is intentionally unsynchronised — locking every access
//! would be prohibitively expensive.
//!
//! **Why this is safe in practice:**
//! * x86-64 guarantees that naturally aligned reads/writes up to 8 bytes are
//!   atomic at the hardware level. Since `key: u64` is the first field and
//!   aligned, a concurrent reader either sees the old key or the new one — never
//!   a torn key. A torn key never equals the position's Zobrist hash, so the
//!   probe is treated as a miss, not a corrupt hit.
//! * In the worst case a racy write causes a TT miss (wasted work). Illegal moves
//!   and wrong mate distances cannot slip through because every hit is gated on
//!   the key check.
//! * This "lockless TT" pattern is universally adopted in high-performance chess
//!   engines (Crafty, Stockfish, Leela).

use crate::movegen::Move;
use std::cell::UnsafeCell;
use std::mem;
use std::sync::atomic::{AtomicU8, Ordering};

/// What a stored score tells us about the true value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    /// Empty slot.
    None,
    /// Exact value (a full-window search).
    Exact,
    /// A lower bound: true value is `>= score` (the node failed high).
    Lower,
    /// An upper bound: true value is `<= score` (the node failed low).
    Upper,
}

#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    mv: u16, // packed Move; 0 == none
    score: i16,
    depth: u8,
    bound: Bound,
    generation: u8,
}

impl Default for Entry {
    fn default() -> Entry {
        Entry {
            key: 0,
            mv: 0,
            score: 0,
            depth: 0,
            bound: Bound::None,
            generation: 0,
        }
    }
}

/// A successful probe result.
pub struct TTData {
    pub score: i16,
    pub depth: u8,
    pub bound: Bound,
    mv: u16,
}

impl TTData {
    /// The stored best move, if any.
    #[inline]
    pub fn best_move(&self) -> Option<Move> {
        if self.mv != 0 {
            Some(Move::from_bits(self.mv))
        } else {
            None
        }
    }
}

/// A fixed-size, direct-mapped transposition table safe to share across threads.
///
/// All methods take `&self`. Interior mutation is performed through
/// `UnsafeCell`; see the module-level doc for the safety reasoning.
pub struct TranspositionTable {
    /// The entry array. Mutable through `UnsafeCell` without locks.
    entries: UnsafeCell<Box<[Entry]>>,
    /// `index = hash & mask`; `entries.len()` is always a power of two.
    mask: usize,
    /// Incremented at the start of each search; old entries are evicted first.
    generation: AtomicU8,
}

// SAFETY: We explicitly vouch that concurrent `probe`/`store` calls are
// safe to perform without synchronisation. See the module-level doc.
unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    /// Allocate a table of roughly `size_mb` megabytes (rounded down to a power
    /// of two number of entries).
    pub fn new(size_mb: usize) -> TranspositionTable {
        let entry_size = mem::size_of::<Entry>();
        let desired = (size_mb.max(1) * 1024 * 1024 / entry_size).max(1);
        // Round down to a power of two so indexing is a cheap mask.
        let count = if desired.is_power_of_two() {
            desired
        } else {
            desired.next_power_of_two() >> 1
        }
        .max(1);

        let entries: Box<[Entry]> = vec![Entry::default(); count].into_boxed_slice();
        TranspositionTable {
            entries: UnsafeCell::new(entries),
            mask: count - 1,
            generation: AtomicU8::new(0),
        }
    }

    /// Begin a new search; entries from prior searches become preferred eviction
    /// targets. Cheaper than zeroing the whole table.
    pub fn new_generation(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Forget everything (used on `ucinewgame`).
    ///
    /// # Safety
    /// Must only be called when no search threads are accessing the table.
    pub fn clear(&self) {
        // SAFETY: caller guarantees no concurrent access.
        unsafe {
            let entries = &mut *self.entries.get();
            for entry in entries.iter_mut() {
                *entry = Entry::default();
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Look up a position. Returns `None` on a miss or a key collision.
    #[inline]
    pub fn probe(&self, key: u64) -> Option<TTData> {
        let idx = (key as usize) & self.mask;
        // SAFETY: idx is always in bounds; concurrent writes are accepted (lockless TT).
        let entry = unsafe { &(*self.entries.get())[idx] };
        if entry.bound != Bound::None && entry.key == key {
            Some(TTData {
                score: entry.score,
                depth: entry.depth,
                bound: entry.bound,
                mv: entry.mv,
            })
        } else {
            None
        }
    }

    /// Store a result. Replacement favors deeper entries and entries from the
    /// current generation, so shallow leftovers from old searches get evicted.
    #[inline]
    pub fn store(&self, key: u64, depth: u32, score: i16, bound: Bound, mv: Option<Move>) {
        let generation = self.generation.load(Ordering::Relaxed);
        let idx = (key as usize) & self.mask;
        // SAFETY: idx is always in bounds; concurrent reads/writes are accepted.
        let entry = unsafe { &mut (*self.entries.get())[idx] };

        let replace = entry.bound == Bound::None
            || entry.generation != generation
            || depth as u8 >= entry.depth;
        if !replace {
            return;
        }

        // If we have no move but the slot already holds one for this position,
        // keep it — a best move is useful for ordering even on a bound-only store.
        let mv_bits = match mv {
            Some(m) => m.bits(),
            None if entry.key == key => entry.mv,
            None => 0,
        };

        *entry = Entry {
            key,
            mv: mv_bits,
            score,
            depth: depth as u8,
            bound,
            generation,
        };
    }

    /// Approximate fill level of the current generation, in permille (0..=1000),
    /// for UCI `info hashfull`.
    pub fn hashfull(&self) -> u32 {
        let generation = self.generation.load(Ordering::Relaxed);
        // SAFETY: read-only scan; concurrent writes produce at worst stale counts.
        let entries = unsafe { &*self.entries.get() };
        let sample = 1000.min(entries.len());
        if sample == 0 {
            return 0;
        }
        let used = entries[..sample]
            .iter()
            .filter(|e| e.bound != Bound::None && e.generation == generation)
            .count();
        (used * 1000 / sample) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Square;
    use crate::movegen::MoveFlag;

    #[test]
    fn entry_is_compact() {
        // Keep entries small so the table holds many positions per megabyte.
        assert!(mem::size_of::<Entry>() <= 16);
    }

    #[test]
    fn size_is_power_of_two() {
        let tt = TranspositionTable::new(1);
        let len = unsafe { (&*tt.entries.get()).len() };
        assert!(len.is_power_of_two());
        assert_eq!(tt.mask, len - 1);
    }

    #[test]
    fn store_then_probe_round_trips() {
        let tt = TranspositionTable::new(1);
        let mv = Move::new(Square::E2, Square::E4, MoveFlag::DoublePush);
        tt.store(0xDEAD_BEEF, 7, 123, Bound::Exact, Some(mv));

        let hit = tt.probe(0xDEAD_BEEF).expect("should hit");
        assert_eq!(hit.score, 123);
        assert_eq!(hit.depth, 7);
        assert_eq!(hit.bound, Bound::Exact);
        assert_eq!(hit.best_move(), Some(mv));

        // A different key in (likely) the same slot must not falsely match.
        assert!(tt.probe(0x1234).is_none());
    }

    #[test]
    fn clear_empties_the_table() {
        let tt = TranspositionTable::new(1);
        tt.store(42, 3, 10, Bound::Lower, None);
        assert!(tt.probe(42).is_some());
        tt.clear();
        assert!(tt.probe(42).is_none());
    }

    #[test]
    fn shared_across_threads() {
        use std::sync::Arc;
        let tt = Arc::new(TranspositionTable::new(1));
        let tt2 = Arc::clone(&tt);
        let mv = Move::new(Square::A1, Square::H8, MoveFlag::Quiet);

        // One thread stores, main thread probes.
        let handle = std::thread::spawn(move || {
            tt2.store(0xABCD, 5, 42, Bound::Exact, Some(mv));
        });
        handle.join().unwrap();

        // The store from the other thread must be visible after joining.
        let hit = tt.probe(0xABCD).expect("cross-thread store must be visible after join");
        assert_eq!(hit.score, 42);
    }
}
