//! The scalar deterministic allocator uses scalar (Boolean) busy flags and a
//! deterministic counter for slot selection. It is the simplest implementation,
//! against which more complex implementations in this crate can be compared.

use ::IndexedAllocator;
use ::utilities::new_boxed_slice;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};


/// Scalar deterministic allocator
#[derive(Debug)]
pub struct Allocator<T> {
    /// Flags telling whether each of the data blocks is in use
    in_use: Box<[AtomicBool]>,

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
        Self {
            in_use: new_boxed_slice(|| AtomicBool::new(false), size),
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
            if self.in_use[index].swap(true, Ordering::Acquire) == false {
                return Some(index);
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