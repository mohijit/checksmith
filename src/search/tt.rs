//! Transposition table — 4-way set-associative (clustered) design.
//!
//! A *transposition* is the same position reached by different move orders.
//! Without a cache, the search re-explores each one from scratch. The
//! transposition table (TT) is a hash table, keyed by the position's Zobrist
//! hash, that remembers what we learned about a position so we never redo
//! that work.
//!
//! Each entry stores:
//!
//! * the **full key**, to detect hash collisions;
//! * the **depth** the result was searched to;
//! * the **score** plus a **bound type** describing what it means;
//! * the **best move**, used for move ordering even when the score can't be reused;
//! * a **generation** stamp so entries from older searches are preferentially
//!   overwritten.
//!
//! ## 4-way clustering
//!
//! The table is divided into fixed-size **clusters** of 4 entries each.
//! A position with hash `H` is stored in one of the 4 slots of cluster
//! `H & mask`.  Compared to direct-mapping (1 entry per slot), clustering
//! reduces eviction collisions: a new entry only displaces an existing one
//! when all four slots in its cluster are occupied.  This raises the effective
//! TT hit rate, especially in multi-PV searches.
//!
//! Each cluster is exactly 64 bytes — one x86-64 cache line.
//!
//! ## TT prefetching
//!
//! Call [`TranspositionTable::prefetch`] with the child position's hash
//! *before* descending into the recursive search call.  The CPU starts
//! fetching the cluster while the engine is still generating moves; by the
//! time the child probes the TT, the data is already in L1 cache.
//!
//! ## Bounds
//!
//! Alpha-beta doesn't always compute an exact score.  When a node fails high
//! (beta cutoff), all we know is the true value is *at least* this — a
//! **lower bound**. When it fails low, the value is *at most* this — an
//! **upper bound**.  Only a full-window node yields an **exact** score.
//!
//! ## Mate scores
//!
//! A mate score encodes distance-to-mate from the *root*.  Stored in the TT
//! it must be distance-to-mate from *this node*, or a position reached at a
//! different depth would report the wrong mate distance.  Callers convert with
//! `score_to_tt` / `score_from_tt` (in [`super::negamax`]).
//!
//! ## Thread safety (Lazy SMP)
//!
//! The TT is shared across all search threads via `Arc<TranspositionTable>`.
//! `probe` and `store` take `&self` and mutate entries through `UnsafeCell`.
//! This is intentionally unsynchronised — locking every access would be
//! prohibitively expensive.
//!
//! **Why this is safe in practice:**
//! * x86-64 guarantees that naturally aligned reads/writes up to 8 bytes are
//!   atomic at the hardware level.  A torn key never matches the position's
//!   Zobrist hash, so the probe is treated as a miss, not a corrupt hit.
//! * In the worst case a racy write causes a TT miss (wasted work). Illegal
//!   moves and wrong mate distances cannot slip through because every hit is
//!   gated on the key check.
//! * This "lockless TT" pattern is universally adopted in high-performance
//!   chess engines (Crafty, Stockfish, Leela).

use crate::movegen::Move;
use std::cell::UnsafeCell;
use std::mem;
use std::sync::atomic::{AtomicU8, Ordering};

/// Number of entries per cluster (= one cache line at 16 bytes/entry).
pub const CLUSTER_SIZE: usize = 4;

/// What a stored score tells us about the true value.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    /// Empty slot.
    None  = 0,
    /// Exact value (a full-window search).
    Exact = 1,
    /// A lower bound: true value is `>= score` (the node failed high).
    Lower = 2,
    /// An upper bound: true value is `<= score` (the node failed low).
    Upper = 3,
}

/// One table entry — 16 bytes, so four fit exactly in a 64-byte cache line.
#[derive(Clone, Copy)]
struct Entry {
    key:        u64,    // 8 — Zobrist key for collision detection
    mv:         u16,    // 2 — packed Move (0 = none)
    score:      i16,    // 2 — stored score
    depth:      u8,     // 1 — search depth
    bound:      Bound,  // 1 — u8 under the hood (repr(u8))
    generation: u8,     // 1 — search generation stamp
    _pad:       u8,     // 1 — explicit padding to reach 16 bytes
}

impl Default for Entry {
    fn default() -> Self {
        Entry { key: 0, mv: 0, score: 0, depth: 0,
                bound: Bound::None, generation: 0, _pad: 0 }
    }
}

/// A cluster of [`CLUSTER_SIZE`] entries — exactly one 64-byte cache line.
struct Cluster {
    entries: [Entry; CLUSTER_SIZE],
}

impl Default for Cluster {
    fn default() -> Self {
        Cluster { entries: [Entry::default(); CLUSTER_SIZE] }
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
        if self.mv != 0 { Some(Move::from_bits(self.mv)) } else { None }
    }
}

/// A 4-way set-associative transposition table safe to share across threads.
///
/// All methods take `&self`.  Interior mutation is performed through
/// `UnsafeCell`; see the module-level doc for the safety reasoning.
pub struct TranspositionTable {
    /// The cluster array.  Mutable through `UnsafeCell` without locks.
    entries:    UnsafeCell<Box<[Cluster]>>,
    /// `cluster_index = hash & mask`; `entries.len()` is always a power of two.
    mask:       usize,
    /// Incremented at the start of each search; old entries are evicted first.
    generation: AtomicU8,
}

// SAFETY: We explicitly vouch that concurrent `probe`/`store` calls are safe
// without synchronisation.  See the module-level doc.
unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    /// Allocate a table of roughly `size_mb` megabytes (rounded down to a
    /// power-of-two number of clusters).
    pub fn new(size_mb: usize) -> TranspositionTable {
        let cluster_bytes = mem::size_of::<Cluster>();
        let desired = (size_mb.max(1) * 1024 * 1024 / cluster_bytes).max(1);
        let count = if desired.is_power_of_two() {
            desired
        } else {
            desired.next_power_of_two() >> 1
        }
        .max(1);

        let clusters: Box<[Cluster]> = (0..count)
            .map(|_| Cluster::default())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        TranspositionTable {
            entries:    UnsafeCell::new(clusters),
            mask:       count - 1,
            generation: AtomicU8::new(0),
        }
    }

    /// Begin a new search; entries from prior searches become preferred
    /// eviction targets.  Cheaper than zeroing the whole table.
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
            for cluster in (*self.entries.get()).iter_mut() {
                *cluster = Cluster::default();
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Look up a position.  Returns `None` on a miss or a key collision.
    #[inline]
    pub fn probe(&self, key: u64) -> Option<TTData> {
        let idx = (key as usize) & self.mask;
        // SAFETY: idx is always in bounds; concurrent writes are accepted.
        let cluster = unsafe { &(*self.entries.get())[idx] };
        for entry in &cluster.entries {
            if entry.bound != Bound::None && entry.key == key {
                return Some(TTData {
                    score: entry.score,
                    depth: entry.depth,
                    bound: entry.bound,
                    mv:    entry.mv,
                });
            }
        }
        None
    }

    /// Store a result.
    ///
    /// Replacement policy (in priority order):
    /// 1. Update the existing entry for this key (preserves TT moves).
    /// 2. Fill an empty slot.
    /// 3. Evict the lowest-quality entry: `depth − age_penalty`, where
    ///    `age_penalty` grows for entries from older generations.
    #[inline]
    pub fn store(&self, key: u64, depth: u32, score: i16, bound: Bound, mv: Option<Move>) {
        let generation = self.generation.load(Ordering::Relaxed);
        let idx = (key as usize) & self.mask;
        // SAFETY: idx is always in bounds; concurrent reads/writes are accepted.
        let cluster = unsafe { &mut (*self.entries.get())[idx] };

        // Pass 1: prefer same-key slot or first empty slot.
        for entry in cluster.entries.iter_mut() {
            if entry.key == key || entry.bound == Bound::None {
                let mv_bits = match mv {
                    Some(m)  => m.bits(),
                    // Preserve existing move when the key matches and no new move.
                    None if entry.key == key => entry.mv,
                    None     => 0,
                };
                *entry = Entry {
                    key, mv: mv_bits, score,
                    depth: depth as u8, bound, generation, _pad: 0,
                };
                return;
            }
        }

        // Pass 2: all slots occupied — evict the one with lowest quality.
        // quality = depth − (age_in_generations × 4)
        let worst_idx = cluster.entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| {
                let age = generation.wrapping_sub(e.generation);
                e.depth as i16 - age as i16 * 4
            })
            .map(|(i, _)| i)
            .unwrap(); // safe: CLUSTER_SIZE > 0

        let mv_bits = mv.map(|m| m.bits()).unwrap_or(0);
        cluster.entries[worst_idx] = Entry {
            key, mv: mv_bits, score,
            depth: depth as u8, bound, generation, _pad: 0,
        };
    }

    /// Prefetch the cluster for `key` into L1 cache.
    ///
    /// Call this with the child position's hash *before* making the move and
    /// recursing.  On x86-64 this emits a `PREFETCHT0` instruction; on other
    /// architectures it is a no-op.
    #[inline]
    pub fn prefetch(&self, key: u64) {
        let idx = (key as usize) & self.mask;
        // SAFETY: idx is in bounds; we only use the address, not the value.
        let ptr = unsafe {
            (*self.entries.get())[idx].entries.as_ptr() as *const i8
        };
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(ptr, _MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// Approximate fill level of the current generation, in permille (0..=1000),
    /// for UCI `info hashfull`.
    pub fn hashfull(&self) -> u32 {
        let generation = self.generation.load(Ordering::Relaxed);
        // SAFETY: read-only scan; concurrent writes produce at worst stale counts.
        let clusters = unsafe { &*self.entries.get() };
        // Sample the first 250 clusters (= 1000 entries) for a fast estimate.
        let sample = 250.min(clusters.len());
        if sample == 0 { return 0; }
        let used = clusters[..sample]
            .iter()
            .flat_map(|c| c.entries.iter())
            .filter(|e| e.bound != Bound::None && e.generation == generation)
            .count();
        (used * 1000 / (sample * CLUSTER_SIZE)) as u32
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Square;
    use crate::movegen::MoveFlag;

    #[test]
    fn entry_is_compact() {
        assert_eq!(mem::size_of::<Entry>(), 16,
            "Entry must be exactly 16 bytes so a cluster fits in one cache line");
    }

    #[test]
    fn cluster_is_cache_line_sized() {
        assert_eq!(mem::size_of::<Cluster>(), 64,
            "Cluster must be 64 bytes (one x86-64 cache line)");
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

        // A different key in the same cluster must not falsely match.
        assert!(tt.probe(0x1234).is_none());
    }

    #[test]
    fn cluster_holds_multiple_keys() {
        // Store CLUSTER_SIZE distinct keys that all hash to the same cluster.
        let tt = TranspositionTable::new(1);
        let num_clusters = unsafe { (&*tt.entries.get()).len() } as u64;

        // Keys that share the same cluster: differ only in bits above the mask.
        let base: u64 = 0xABCD_1234;
        let keys: Vec<u64> = (0..CLUSTER_SIZE as u64)
            .map(|i| base.wrapping_add(i * num_clusters))
            .collect();

        for (i, &k) in keys.iter().enumerate() {
            tt.store(k, (i + 1) as u32, (i * 10) as i16, Bound::Exact, None);
        }
        for (i, &k) in keys.iter().enumerate() {
            let hit = tt.probe(k).expect("all cluster entries should be retrievable");
            assert_eq!(hit.score, (i * 10) as i16,
                "key {} should return score {}", k, i * 10);
        }
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

        let handle = std::thread::spawn(move || {
            tt2.store(0xABCD, 5, 42, Bound::Exact, Some(mv));
        });
        handle.join().unwrap();

        let hit = tt.probe(0xABCD)
            .expect("cross-thread store must be visible after join");
        assert_eq!(hit.score, 42);
    }
}
