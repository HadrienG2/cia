//! Concurrent Indexed Allocator (CIA)
//!
//! This crate solves a problem which does not come up every day, but which I
//! have faced from time to time: given an array of N objects of a certain type
//! T, and an unknown amount of threads making allocation requests, answer each
//! of these allocation requests with either a storage buffer of type T which
//! can be referred to using an integer index in [0..N[, or a notification that
//! the available storage is full.
//!
//!
//! # Design space
//!
//! ## Data layout
//!
//! One first choice that must be made in when implementing this allocator is
//! the data layout. There are essentially two key design choices to be made:
//!
//! - Whether to store data in array-of-structure (one array of (flag, data)
//!   tuples) or structure-of-array layout (one array of flags and one of data)
//! - Whether to have one "scalar" AtomicBool per allocatable object, or to pack
//!   these bools together in a "packed" AtomicUsize bitfield
//!
//! The tradeoff is between using a minimal amount of memory, promoting true
//! sharing and avoiding false sharing, and being able to quickly scan the list
//! of allocations when looking for a storage slot:
//!
//! - Scalar layouts have less cache contention and simpler code, packed layouts
//!   allow for faster scans in the uncontended regime.
//! - AoS layouts have reduced false sharing, but poor alignment properties.
//!   Moreover, the flags and data are accessed at different points in time, so
//!   keeping them separate makes sense.
//!
//! I think SoA makes more sense, but I am less sure about scalar vs packed. I
//! will probably start with scalar, as that is simpler to write, and try packed
//! later on as an alternate implementation.
//!
//! ## Allocation algorithm
//!
//! Another design choice has to do with the algorithm used to explore storage
//! slots. Here, I essentially see two possibilities:
//!
//! - Use an atomic "current index" counter. Guarantees sensible exploration of
//!   the storage slots, that is optimal if the allocation/liberation pattern is
//!   FIFO, but requires extra synchronization.
//! - Pick a storage slot at random. Requires no synchronization, but may take a
//!   lot of iterations to find a storage slot if most of them are busy, and is
//!   hard on the CPU cache.
//!
//! I will initially go with the atomic counter, as that seems generally better,
//! and then try the alternate strategy to see if it handles contention better.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};


/// I've wanted a utility function like this since... forever.
fn new_boxed_slice<F, T>(mut generator: F, size: usize) -> Box<[T]>
    where F: FnMut() -> T
{
    let mut acc = Vec::with_capacity(size);
    for _ in 0..size {
        acc.push(generator())
    }
    acc.into_boxed_slice()
}


/// This is an implementation of the concurrent indexed allocator concept
pub struct ConcurrentIndexedAllocator<T> {
    /// Flags telling whether each of the data blocks is in use
    in_use: Box<[AtomicBool]>,

    /// The data blocks themselves, with matching indices
    data: Box<[UnsafeCell<T>]>,

    /// Suggestion for the client of the next data block to be tried. Must be
    /// incremented via fetch-add on every read to remain a good suggestion.
    /// Value must be wrapped around modulo data.len() to make any sense.
    next_index: AtomicUsize,
}

impl<T: Default> ConcurrentIndexedAllocator<T> {
    /// Constructor for default-constructible types
    pub fn new(size: usize) -> Self {
        Self {
            in_use: new_boxed_slice(|| AtomicBool::new(false), size),
            data: new_boxed_slice(|| UnsafeCell::new(T::default()), size),
            next_index: AtomicUsize::new(0),
        }
    }
}

impl<T> ConcurrentIndexedAllocator<T> {
    /// Attempt to allocate a new indexed data block
    pub fn allocate<'a>(&'a self) -> Option<Allocation<'a, T>> {
        // Look for an unused data block, allowing for a full storage scan
        // before giving up and bailing
        let size = self.data.len();
        for _ in 0..size {
            // Get a suggestion of a data block to try out next
            let i = self.next_index.fetch_add(1, Ordering::Relaxed) % size;

            // If that data block is free, reserve it and return it. Need an
            // Acquire memory barrier for the data to be consistent when
            // receiving a concurrently deallocated data block.
            if self.in_use[i].swap(true, Ordering::Acquire) == false {
                return Some(Allocation { allocator: self, index: i });
            }
        }

        // Too many failed attempts to allocate, storage is likely full
        None
    }

    /// Deallocating by index is unsafe, because it can cause a data race if the
    /// wrong data block is accidentally liberated. In any case, we recommend
    /// that you liberate the data via the Allocation RAII interface.
    unsafe fn deallocate(&self, index: usize) {
        self.in_use[index].store(false, Ordering::Release);
    }
}


/// Proxy object representing a successfully allocated data block
pub struct Allocation<'a, T: 'a> {
    /// Allocator which we got the data from
    allocator: &'a ConcurrentIndexedAllocator<T>,

    /// Index of the data within the allocator
    index: usize,
}

// Data can be accessed via the usual Deref/DerefMut smart pointer interface...
impl<'a, T> Deref for Allocation<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        let target_ptr = self.allocator.data[self.index].get();
        unsafe { & *target_ptr }
    }
}
//
impl<'a, T> DerefMut for Allocation<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let target_ptr = self.allocator.data[self.index].get();
        unsafe { &mut *target_ptr }
    }
}

// ...and will be automatically liberated on drop in the usual RAII way.
impl<'a, T> Drop for Allocation<'a, T> {
    fn drop(&mut self) {
        // This is safe because we trust that the inner index is valid
        unsafe { self.allocator.deallocate(self.index) };
    }
}

impl<'a, T> Allocation<'a, T> {
    /// The ability to extract the index of the allocation is a critical part of
    /// this abstraction. It is what allows an allocation to be used in
    /// space-contrained scenarios, such as within the NaN bits of a float.
    ///
    /// However, this is a lossy operation. Converting to an allocator + index
    /// is safe to do, as it's just exposing information to the outside world...
    ///
    pub fn into_raw(self) -> (&'a ConcurrentIndexedAllocator<T>, usize) {
        (self.allocator, self.index)
    }

    /// ...but going back to the Allocation abstraction after that is unsafe,
    /// because we have no way to check that the (allocator, index) pair which
    /// you give back is the same that you received from us. And if it's not,
    /// a data race disaster will likely ensue.
    ///
    pub unsafe fn from_raw(allocator: &'a ConcurrentIndexedAllocator<T>,
                           index: usize) -> Self {
        Self {
            allocator,
            index
        }
    }
}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
