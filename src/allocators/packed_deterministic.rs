//! Compared to the scalar deterministic allocator, the packed deterministic
//! allocator packs busy flags into an integer bitfield. This can speed up
//! searches for slots, at the cost of increasing contention.

use ::IndexedAllocator;
use ::utilities::new_boxed_slice;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};


#[cfg(target_pointer_width="32")]
const BITS_IN_USIZE: usize = 32;

#[cfg(target_pointer_width="64")]
const BITS_IN_USIZE: usize = 64;


/// Packed deterministic allocator
#[derive(Debug)]
pub struct Allocator<T> {
    /// Flags telling whether each of the data blocks is in use, in packed form
    in_use: Box<[AtomicUsize]>,

    /// The data blocks themselves, with matching indices
    data: Box<[UnsafeCell<T>]>,

    /// Suggestion for the client of the next data block to be tried. Must be
    /// incremented via fetch-add on every read to remain a good suggestion.
    /// Value must be wrapped around modulo data.len() to make any sense.
    next_index: AtomicUsize,
}

impl<T: Default> Allocator<T> {
    /// Constructor for default-constructible types
    pub fn new(size: usize) -> Self {
        let packed_size = size / BITS_IN_USIZE + 
                            (if size % BITS_IN_USIZE != 0 { 1 } else { 0 });
        Self {
            in_use: new_boxed_slice(|| AtomicUsize::new(0usize), packed_size),
            data: new_boxed_slice(|| UnsafeCell::new(T::default()), size),
            next_index: AtomicUsize::new(0),
        }
    }
}

impl<T> IndexedAllocator for Allocator<T> {
    type Data = T;

    fn raw_allocate(&self) -> Option<usize> {
        // Look for an unused data block, allowing for a full storage scan
        // before giving up and bailing
        let size = self.data.len();
        for _ in 0..size {
            // Get a suggestion of a data block to try out next
            let index = self.next_index.fetch_add(1, Ordering::Relaxed) % size;

            // Decompose the data block index into a word index and a bit index
            let (word, bit) = (index / BITS_IN_USIZE, index % BITS_IN_USIZE);

            // Prepare a bit mask for reserving the data block
            let mask = 1usize << bit;

            // If that data block is free, reserve it and return it. An Acquire
            // memory barrier is needed to make sure that the current thread
            // gets a consistent view of the freshly allocated block's contents.
            //
            // Assuming that the allocator is well-dimensioned and the overall
            // allocation pattern meets our FIFO expectations, this will succeed
            // most of the time. So the usual failure case optimizations that
            // pre-check the flag before swapping it or that delay the Acquire
            // barrier until success is confirmed are unlikely to be worthwhile.
            //
            let old_value = self.in_use[word].fetch_or(mask, Ordering::Acquire);
            if old_value | mask == 0 {
                return Some(index);
            } else {
                // TODO: Find the next bit worth trying
                let next_mask = usize::max_value() << (bit + 1);
                let next_available = !old_value | next_mask;
                unimplemented!()
            }
        }

        // Too many failed attempts to allocate, storage is likely full
        None
    }

    #[inline]
    unsafe fn raw_get(&self, index: usize) -> &T {
        & *self.data[index].get()
    }

    #[inline]
    unsafe fn raw_get_mut(&self, index: usize) -> &mut T {
        &mut *self.data[index].get()
    }

    #[inline]
    unsafe fn raw_deallocate(&self, index: usize) {
        // A Release memory barrier is needed to make sure that the next thread
        // which allocates this memory location will see consistent data in it.
        self.in_use[index].store(false, Ordering::Release);
    }
}

unsafe impl<T> Sync for Allocator<T> {}


allocator_tests_benches! { Allocator }