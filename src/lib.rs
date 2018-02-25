//! Concurrent Indexed Allocators (CIA)
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

extern crate testbench;

pub mod allocators;
pub(crate) mod utilities;

use std::ops::{Deref, DerefMut};
use std::mem;


/// An indexed allocator is a a mechanism for distributing a finite pool of
/// objects of a certain type across multi-threaded clients, in such a fashion
/// that each allocation may be uniquely identified by an integer index in the
/// range 0..N where N is the capacity of the allocator.
///
/// Allocated data blocks are provided to you in the state where the last client
/// left them: you can only assume that it is a valid value of type T, and if
/// you need more cleanup you will have to do it yourself.
///
pub trait IndexedAllocator: Sync {
    // Type of data being managed by the allocator
    type Data;

    /// Request an object from the allocator. This is the recommended high-level
    /// API, which provides a maximally safe wrapper around the allocation.
    #[inline]
    fn allocate(&self) -> Option<Allocation<Self>> {
        self.raw_allocate().map(|index| unsafe {
            Allocation::from_raw(self, index)
        })
    }

    /// Request a raw indexed allocation, without using the safe wrapper.
    fn raw_allocate(&self) -> Option<usize>;

    /// Access the content of a certain allocation. This is unsafe because there
    /// is no safeguard against incorrect aliasing and data races. In fact, we
    /// have no way to check that you even own this allocation...
    unsafe fn raw_get(&self, index: usize) -> &Self::Data;

    /// Mutably access the content of a certain allocation. This is unsafe
    /// because there is no safeguard against incorrect aliasing and data races.
    /// In fact, we have no way to check that you even own this allocation...
    unsafe fn raw_get_mut(&self, index: usize) -> &mut Self::Data;

    /// Liberate an allocation. This is unsafe because at this layer of the
    /// interface, we have no way to ensure that you will not try to reuse the
    /// allocation after freeing it.
    unsafe fn raw_deallocate(&self, index: usize);
}


/// Proxy object providing a safe interface to an allocated data block
#[derive(Debug)]
pub struct Allocation<'a, Allocator: 'a + IndexedAllocator + ?Sized> {
    /// Allocator which we got the data from
    allocator: &'a Allocator,

    /// Index identifying the data within the allocator
    index: usize,
}
//
// Data can be accessed via the usual Deref/DerefMut smart pointer interface...
impl<'a, Allocator> Deref for Allocation<'a, Allocator>
    where Allocator: IndexedAllocator + ?Sized
{
    type Target = Allocator::Data;

    fn deref(&self) -> &Self::Target {
        unsafe { self.allocator.raw_get(self.index) }
    }
}
//
impl<'a, Allocator> DerefMut for Allocation<'a, Allocator>
    where Allocator: IndexedAllocator + ?Sized
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.allocator.raw_get_mut(self.index) }
    }
}
//
// ...and will be automatically liberated on drop in the usual RAII way.
impl<'a, Allocator> Drop for Allocation<'a, Allocator>
    where Allocator: IndexedAllocator + ?Sized
{
    fn drop(&mut self) {
        unsafe { self.allocator.raw_deallocate(self.index) };
    }
}
//
impl<'a, Allocator> Allocation<'a, Allocator>
    where Allocator: IndexedAllocator + ?Sized
{
    /// The ability to extract the index of the allocation is a critical part of
    /// this abstraction. It is what allows an allocation to be used in
    /// space-contrained scenarios, such as within the NaN bits of a float.
    ///
    /// However, this is a lossy operation. Converting to an allocator + index
    /// is safe to do, as it's just exposing information to the outside world...
    ///
    pub fn into_raw(self) -> usize {
        let result = self.index;
        mem::forget(self);
        result
    }

    /// ...but going back to the Allocation abstraction after that is unsafe,
    /// because we have no way to check that the (allocator, index) pair which
    /// you give back is consistent with what you received, and that you do not
    /// sneakily attempt to create multiple Allocations from it. If you do this,
    /// the program will head straight into undefined behaviour territory...
    ///
    pub unsafe fn from_raw(allocator: &'a Allocator, index: usize) -> Self {
        Self {
            allocator,
            index
        }
    }
}