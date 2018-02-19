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


fn new_boxed_slice<F, T>(mut generator: F, size: usize) -> Box<[T]>
    where F: FnMut() -> T
{
    let mut acc = Vec::with_capacity(size);
    for _ in 0..size {
        acc.push(generator())
    }
    acc.into_boxed_slice()
}


pub struct ConcurrentIndexedAllocator<T> {
    in_use: Box<[AtomicBool]>,
    data: Box<[UnsafeCell<T>]>,
    next_index: AtomicUsize,
}

impl<T: Default> ConcurrentIndexedAllocator<T> {
    fn new(size: usize) -> Self {
        Self {
            in_use: new_boxed_slice(|| AtomicBool::new(false), size),
            data: new_boxed_slice(|| UnsafeCell::new(T::default()), size),
            next_index: AtomicUsize::new(0),
        }
    }
}

impl<T> ConcurrentIndexedAllocator<T> {
    fn allocate<'a>(&'a self) -> Option<Allocation<'a, T>> {
        // Look for an unused allocation, allowing a full storage scan
        for _ in 0..self.data.len() {
            let i = self.next_index.fetch_add(1, Ordering::Relaxed);
            if self.in_use[i].swap(true, Ordering::Acquire) == false {
                return Some(Allocation { allocator: self, index: i });
            }
        }

        // Too many attempts, storage is likely full
        None
    }

    // Make sure that you target the right index when calling this method...
    unsafe fn deallocate(&self, index: usize) {
        self.in_use[index].store(false, Ordering::Release);
    }
}


pub struct Allocation<'a, T: 'a> {
    allocator: &'a ConcurrentIndexedAllocator<T>,
    index: usize,
}

impl<'a, T> Deref for Allocation<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        let target_ptr = self.allocator.data[self.index].get();
        unsafe { & *target_ptr }
    }
}

impl<'a, T> DerefMut for Allocation<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let target_ptr = self.allocator.data[self.index].get();
        unsafe { &mut *target_ptr }
    }
}

impl<'a, T> Drop for Allocation<'a, T> {
    fn drop(&mut self) {
        // This is safe because we trust that the inner index is valid
        unsafe { self.allocator.deallocate(self.index) };
    }
}

impl<'a, T> Allocation<'a, T> {
    // This is a lossy conversion. We know that the allocation index which you
    // will get here is valid...
    fn into_raw(self) -> (&'a ConcurrentIndexedAllocator<T>, usize) {
        (self.allocator, self.index)
    }

    // ...but we rely on your care for passing back a matching
    // (allocator, index) pair to this fn, or else disaster will ensue.
    unsafe fn from_raw(allocator: &'a ConcurrentIndexedAllocator<T>,
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
