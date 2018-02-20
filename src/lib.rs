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
use std::mem;
use std::ptr;
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
#[derive(Debug)]
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
    /// Attempt to allocate a new indexed data block. The data block is provided
    /// to you in the state where the last client left it: you can only assume
    /// that it is a valid value of type T, and if you need more cleanup you
    /// will have to do it yourself.
    pub fn allocate<'a>(&'a self) -> Option<Allocation<'a, T>> {
        // Look for an unused data block, allowing for a full storage scan
        // before giving up and bailing
        let size = self.data.len();
        for _ in 0..size {
            // Get a suggestion of a data block to try out next
            let index = self.next_index.fetch_add(1, Ordering::Relaxed) % size;

            // If that data block is free, reserve it and return it. Need an
            // Acquire memory barrier for the data to be consistent when
            // receiving a concurrently deallocated data block.
            if self.in_use[index].swap(true, Ordering::Acquire) == false {
                return Some(Allocation { allocator: self, index });
            }
        }

        // Too many failed attempts to allocate, storage is likely full
        None
    }

    /// Access indexed data. This is unsafe because using the wrong index can
    /// cause a data race with another thread concurrently accessing that index
    unsafe fn get(&self, index: usize) -> &T {
        & *self.data[index].get()
    }

    /// Mutably access indexed data. This is unsafe for the same reason that
    /// the get() method is: badly used, it will cause a data race.
    unsafe fn get_mut(&self, index: usize) -> &mut T {
        &mut *self.data[index].get()
    }

    /// Deallocating by index is unsafe, because it can cause a data race if the
    /// wrong data block is accidentally liberated.
    unsafe fn deallocate(&self, index: usize) {
        self.in_use[index].store(false, Ordering::Release);
    }
}


/// Proxy object representing a successfully allocated data block
#[derive(Debug)]
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
        unsafe { self.allocator.get(self.index) }
    }
}
//
impl<'a, T> DerefMut for Allocation<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.allocator.get_mut(self.index) }
    }
}

// ...and will be automatically liberated on drop in the usual RAII way.
impl<'a, T> Drop for Allocation<'a, T> {
    fn drop(&mut self) {
        // This is safe because we trust that the inner index is valid
        unsafe { self.allocator.deallocate(self.index) };
    }
}

// Two allocations are equal if they originate from the same allocator and
// target the same index. As allocations model owned values, this should never
// happen, and indicates a bug in the allocator.
impl<'a, T> PartialEq for Allocation<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.allocator, other.allocator) && (self.index == other.index)
    }
}
//
impl<'a, T> Eq for Allocation<'a, T> {}

impl<'a, T> Allocation<'a, T> {
    /// The ability to extract the index of the allocation is a critical part of
    /// this abstraction. It is what allows an allocation to be used in
    /// space-contrained scenarios, such as within the NaN bits of a float.
    ///
    /// However, this is a lossy operation. Converting to an allocator + index
    /// is safe to do, as it's just exposing information to the outside world...
    ///
    pub fn into_raw(self) -> (&'a ConcurrentIndexedAllocator<T>, usize) {
        let result = (self.allocator, self.index);
        mem::forget(self);
        result
    }

    /// ...but going back to the Allocation abstraction after that is unsafe,
    /// because we have no way to check that the (allocator, index) pair which
    /// you give back is the same that you received from us, and that you do not
    /// sneakily attempt to create multiple Allocations from it. If you do this,
    /// the program will head straight into undefined behaviour territory.
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
    use std::mem::ManuallyDrop;
    use super::*;


    /// Check that allocators are created in a correct state
    #[test]
    fn new() {
        let allocator = ConcurrentIndexedAllocator::<u8>::new(42);
        assert_eq!(allocator.in_use.len(), 42);
        assert!(allocator.in_use.iter().all(|b| !b.load(Ordering::Relaxed)));
        assert_eq!(allocator.data.len(), 42);
    }

    /// Check that basic allocation and Allocation comparison works
    #[test]
    fn allocation_eq() {
        // Perform two allocations and check that the results make sense
        let allocator = ConcurrentIndexedAllocator::<String>::new(2);
        let alloc1 = allocator.allocate().unwrap();
        let alloc2 = allocator.allocate().unwrap();
        assert!(((alloc1.index == 0) && (alloc2.index == 1)) ||
                ((alloc1.index == 1) && (alloc2.index == 0)));

        // Check that a third allocation will fail
        assert_eq!(allocator.allocate(), None);

        // Check the Allocation equality operator. This involves the dangerous
        // creation of a duplicate allocation, which should never be used and
        // whose destructor should never be run, as it violates the type's basic
        // assumptions and can easily cause various undefined behaviour.
        let alloc1bis = ManuallyDrop::new(
            unsafe {
                Allocation::from_raw(&allocator, alloc1.index)
            }
        );
        assert_eq!(alloc1, alloc1);
        assert_eq!(alloc1, *alloc1bis);
        assert!(alloc1 != alloc2);
    }

    /// Do more extensive check of allocation in the sequential case
    #[test]
    fn allocate() {
        const CAPACITY: usize = 15;
        let allocator = ConcurrentIndexedAllocator::<f64>::new(CAPACITY);
        let mut allocations = Vec::with_capacity(CAPACITY);
        for _ in 0..CAPACITY {
            let allocation = allocator.allocate().unwrap();
            assert!(ptr::eq(&allocator, allocation.allocator));
            assert!(allocation.index < CAPACITY);
            assert!(!allocations.contains(&allocation));
            allocations.push(allocation);
        }
        assert_eq!(allocator.allocate(), None);
    }

    /// Check that we can read and write to an allocation
    #[test]
    fn read_write() {
        let allocator = ConcurrentIndexedAllocator::<char>::new(1);
        let mut allocation = allocator.allocate().unwrap();
        *allocation = '@';
        assert_eq!(*allocation, '@');
    }

    /// Check that we can deallocate data by dropping the Allocation
    #[test]
    fn deallocate() {
        let allocator = ConcurrentIndexedAllocator::<isize>::new(1);
        {
            let _allocation = allocator.allocate().unwrap();
            assert_eq!(allocator.allocate(), None);
        }
        assert!(allocator.allocate().is_some());
    }

    /// Check that we can destructure an Allocation, then put it back together,
    /// without it being freed by the allocator.
    #[test]
    fn split_and_merge() {
        let allocator = ConcurrentIndexedAllocator::<&str>::new(1);
        let allocation = allocator.allocate().unwrap();
        let (_, index) = allocation.into_raw();
        assert_eq!(allocator.allocate(), None);
        let _allocation = unsafe { Allocation::from_raw(&allocator, index) };
        assert_eq!(allocator.allocate(), None);
    }
}
