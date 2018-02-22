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

extern crate testbench;

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::mem;
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
//
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
//
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
    #[inline]
    unsafe fn get(&self, index: usize) -> &T {
        & *self.data[index].get()
    }

    /// Mutably access indexed data. This is unsafe for the same reason that
    /// the get() method is: badly used, it will cause a data race.
    #[inline]
    unsafe fn get_mut(&self, index: usize) -> &mut T {
        &mut *self.data[index].get()
    }

    /// Deallocating by index is unsafe, because it can cause a data race if the
    /// wrong data block is accidentally liberated.
    #[inline]
    unsafe fn deallocate(&self, index: usize) {
        self.in_use[index].store(false, Ordering::Release);
    }
}
//
unsafe impl<T> Sync for ConcurrentIndexedAllocator<T> {}


/// Proxy object representing a successfully allocated data block
#[derive(Debug)]
pub struct Allocation<'a, T: 'a> {
    /// Allocator which we got the data from
    allocator: &'a ConcurrentIndexedAllocator<T>,

    /// Index of the data within the allocator
    index: usize,
}
//
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
//
// ...and will be automatically liberated on drop in the usual RAII way.
impl<'a, T> Drop for Allocation<'a, T> {
    fn drop(&mut self) {
        // This is safe because we trust that the inner index is valid
        unsafe { self.allocator.deallocate(self.index) };
    }
}
//
impl<'a, T> Allocation<'a, T> {
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
    use std::ptr;
    use std::sync::{Arc, Mutex};
    use super::*;
    use testbench;
    use testbench::race_cell::{UsizeRaceCell, Racey};


    /// Check that allocators are created in a correct state
    #[test]
    fn new() {
        let allocator = ConcurrentIndexedAllocator::<u8>::new(42);
        assert_eq!(allocator.in_use.len(), 42);
        assert!(allocator.in_use.iter().all(|b| !b.load(Ordering::Relaxed)));
        assert_eq!(allocator.data.len(), 42);
    }

    /// Check that basic allocation works
    #[test]
    fn basic_allocation() {
        // Perform two allocations and check that the results make sense
        let allocator = ConcurrentIndexedAllocator::<String>::new(2);
        let alloc1 = allocator.allocate().unwrap();
        let alloc2 = allocator.allocate().unwrap();
        assert!(((alloc1.index == 0) && (alloc2.index == 1)) ||
                ((alloc1.index == 1) && (alloc2.index == 0)));

        // Check that a third allocation will fail
        assert!(allocator.allocate().is_none());
    }

    /// Do more extensive check of allocation in the sequential case
    #[test]
    fn more_allocations() {
        const CAPACITY: usize = 15;
        let allocator = ConcurrentIndexedAllocator::<f64>::new(CAPACITY);
        let mut allocations = Vec::with_capacity(CAPACITY);
        for _ in 0..CAPACITY {
            // Request an allocation (should succeed)
            let allocation = allocator.allocate().unwrap();

            // Check that the allocator field is correct
            assert!(ptr::eq(&allocator, allocation.allocator));

            // Check that the index makes sense and has not been seen before
            assert!(allocation.index < CAPACITY);
            assert!(!allocations.contains(&allocation.index));

            // Extract index with into_raw(), check that it works
            let (index1, index2) = (allocation.index, allocation.into_raw());
            assert_eq!(index1, index2);

            // Record the index of the new allocation
            allocations.push(index2);
        }

        // At the end, the allocator should have no capacity left
        assert!(allocator.allocate().is_none());
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
            assert!(allocator.allocate().is_none());
        }
        assert!(allocator.allocate().is_some());
    }

    /// Check that we can destructure an Allocation, then put it back together,
    /// without it being freed by the allocator.
    #[test]
    fn split_and_merge() {
        let allocator = ConcurrentIndexedAllocator::<&str>::new(1);
        let allocation = allocator.allocate().unwrap();
        let index = allocation.into_raw();
        assert!(allocator.allocate().is_none());
        let _allocation = unsafe { Allocation::from_raw(&allocator, index) };
        assert!(allocator.allocate().is_none());
    }

    /// Run two threads in parallel, allocating data until the capacity of the
    /// allocator is exhausted. Then check if the allocation pattern was correct
    ///
    /// This test should be run on a release build and with --test-threads=1
    /// in order to maximize concurrency (and thus odds of bug detection).
    ///
    #[test]
    #[ignore]
    fn concurrent_alloc() {
        // We will allocate LOTS of booleans
        const CAPACITY: usize = 30_000_000;
        type BoolAllocator = ConcurrentIndexedAllocator<bool>;
        let allocator_1 = Arc::new(BoolAllocator::new(CAPACITY));
        let allocator_2 = allocator_1.clone();
        let allocator_3 = allocator_1.clone();

        // Each thread will track the index of the allocations that it received
        let local_indices_1 = new_boxed_slice(|| false, CAPACITY);
        let local_indices_2 = local_indices_1.clone();

        // Once the allocator's capacity has been used up, threads will collect
        // and check a global view of which indices were observed overall.
        let global_indices_1 = Arc::new(Mutex::new(None));
        let global_indices_2 = global_indices_1.clone();

        // Here is what each thread will do
        fn worker(allocator: Arc<BoolAllocator>,
                  mut local_indices: Box<[bool]>,
                  global_indices: Arc<Mutex<Option<Box<[bool]>>>>)
        {
            // As long as allocation succeeeds, record the allocated indices.
            // Make sure that no index appears twice: it means that a certain
            // piece of data was allocated twice to a thread, which is wrong.
            while let Some(allocation) = allocator.allocate() {
                let index = allocation.into_raw();
                assert!(!mem::replace(&mut local_indices[index], true));
            }

            // At the end, threads race for the global index store...
            let mut indices_option = global_indices.lock().unwrap();
            if indices_option.is_none() {
                // The one that gets there first imposes its world view of index
                // allocations to the other thread...
                *indices_option = Some(local_indices);
            } else if let Some(ref indices) = *indices_option {
                // ...which must check that overall, each available storage
                // index was allocated exactly once
                for (local, global) in local_indices.iter()
                                                    .zip(indices.iter()) {
                    assert!(local ^ global);
                }
            }
        }

        // Run both threads to completion
        testbench::concurrent_test_2(move || worker(allocator_1,
                                                    local_indices_1,
                                                    global_indices_1),
                                     move || worker(allocator_2,
                                                    local_indices_2,
                                                    global_indices_2));

        // At the end, all allocator capacity should have been used up
        assert!(allocator_3.allocate().is_none());
    }

    /// Run two threads in parallel in an alloc/read/write/dealloc loop. Check
    /// if inconsistent data or impossible allocation patterns are observed.
    ///
    /// This test should be run on a release build and with --test-threads=1
    /// in order to maximize concurrency (and thus odds of bug detection).
    ///
    #[test]
    #[ignore]
    fn concurrent_access() {
        // Each thread will go through this number of iterations
        const ITERATIONS: usize = 30_000_000;

        // In this test, we use a single-cell allocator contention to make sure
        // that threads hit the same allocation yet play nicely with each other.
        type TimestampAllocator = ConcurrentIndexedAllocator<UsizeRaceCell>;
        let allocator_1 = Arc::new(TimestampAllocator::new(1));
        let allocator_2 = allocator_1.clone();
        let allocator_3 = allocator_1.clone();

        // The allocator's single cell holds a timestamp counter. On each
        // iteration, a thread reads the counter, increments it, stores the new
        // timestamp in the cell, then liberates it. Each thread also keeps a
        // record of which timestamp values it has seen in this process.
        let local_timestamps_1 = new_boxed_slice(|| false, ITERATIONS);
        let local_timestamps_2 = local_timestamps_1.clone();

        // Once all iterations have been executed, the threads synchronize to
        // check if each timestamp value has been seen once and only once.
        let global_timestamps_1 = Arc::new(Mutex::new(None));
        let global_timestamps_2 = global_timestamps_1.clone();

        // Here is what each thread will do
        fn worker(allocator: Arc<TimestampAllocator>,
                  mut local_timestamps: Box<[bool]>,
                  global_timestamps: Arc<Mutex<Option<Box<[bool]>>>>)
        {
            // Iterate until the maximal timestamp is reached
            let mut num_allocations = 0usize;
            loop {
                // Try to get access to the allocator's single cell
                if let Some(allocation) = allocator.allocate() {
                    // Extract the current timestamp
                    let timestamp = match allocation.get() {
                        // Allocator cell should be seen in a consistent state
                        Racey::Inconsistent => {
                            panic!("Inconsistent allocator state observed");
                        },

                        // Record the timestamps that are observed
                        Racey::Consistent(timestamp) => timestamp,
                    };

                    // Exit the loop when the maximal timestamp is observed
                    if timestamp == ITERATIONS { break; }

                    // Record which timestamps were observed by this thread, and
                    // make sure that a given timestamp is never observed twice.
                    assert!(
                        !mem::replace(&mut local_timestamps[timestamp], true)
                    );

                    // Increment the timestamp counter and liberate the cell
                    allocation.set(timestamp + 1);

                    // Record the amount of successful allocation
                    num_allocations += 1;
                }
            }

            // The test did not work properly if one single thread did most of
            // the allocations: we wanted concurrent allocator access!
            assert!(num_allocations > ITERATIONS/10);

            // At the end, threads race for the global timestamp store...
            let mut timestamps_option = global_timestamps.lock().unwrap();
            if timestamps_option.is_none() {
                // The one that gets there first imposes its world view of
                // observed timestamps to the other thread...
                *timestamps_option = Some(local_timestamps);
            } else if let Some(ref timestamps) = *timestamps_option {
                // ...which must check that overall, each possible timestamp
                // was observed exactly once, by one thread or the other
                for (local, global) in local_timestamps.iter()
                                                       .zip(timestamps.iter()) {
                    assert!(local ^ global);
                }
            }
        }

        // Run both threads to completion
        testbench::concurrent_test_2(move || worker(allocator_1,
                                                    local_timestamps_1,
                                                    global_timestamps_1),
                                     move || worker(allocator_2,
                                                    local_timestamps_2,
                                                    global_timestamps_2));

        // At the end, the cell should be available and follow expectations
        let timestamp_cell = allocator_3.allocate();
        let cell_contents = timestamp_cell.expect("Cell should be available");
        assert_eq!(cell_contents.get(), Racey::Consistent(ITERATIONS));
    }
}


/// Performance benchmarks
///
/// These benchmarks masquerading as tests are a stopgap solution until
/// benchmarking lands in Stable Rust. They should be compiled in release mode,
/// and run with only one OS thread. In addition, the default behaviour of
/// swallowing test output should obviously be suppressed.
///
/// TL;DR: cargo test --release -- --ignored --nocapture --test-threads=1
///
/// TODO: Switch to standard Rust benchmarks once they are stable
///
#[cfg(test)]
mod benchmarks {
    use std::mem;
    use std::sync::Arc;
    use super::*;
    use testbench;

    /// We'll benchmark the worst case: an allocator of tiny booleans
    type BoolAllocator = ConcurrentIndexedAllocator<bool>;

    /// Benchmark of (sequential) allocation performance
    #[test]
    #[ignore]
    fn alloc() {
        // Get ready to allocate a lot of stuff
        const ITERATIONS: u32 = 150_000_000;
        const CAPACITY: usize = ITERATIONS as usize;
        let allocator = BoolAllocator::new(CAPACITY);

        // Perform the allocations, leaking them after the fact
        testbench::benchmark(ITERATIONS, || {
            let allocation = allocator.allocate().unwrap();
            assert!(allocation.index < CAPACITY);
            mem::forget(allocation);
        });
    }

    /// Benchmark of (sequential) allocation + liberation performance
    #[test]
    #[ignore]
    fn alloc_free() {
        // Get ready to allocate a lot of stuff
        const ITERATIONS: u32 = 150_000_000;
        const CAPACITY: usize = ITERATIONS as usize;
        let allocator = BoolAllocator::new(CAPACITY);

        // Perform the allocations, dropping them after the fact
        testbench::benchmark(ITERATIONS, || {
            let allocation = allocator.allocate().unwrap();
            assert!(allocation.index < CAPACITY);
        });
    }

    /// Benchmark of (sequential) allocation + data readout performance
    #[test]
    #[ignore]
    fn alloc_read_free() {
        // Get ready to allocate a lot of stuff
        const ITERATIONS: u32 = 150_000_000;
        const CAPACITY: usize = ITERATIONS as usize;
        let allocator = BoolAllocator::new(CAPACITY);

        // Perform the allocations, read the data, and leak
        testbench::benchmark(ITERATIONS, || {
            let allocation = allocator.allocate().unwrap();
            assert!(*allocation == false);
        });
    }

    /// Benchmark of (sequential) allocation + data read/write performance
    #[test]
    #[ignore]
    fn alloc_read_write_free() {
        // Get ready to allocate a lot of stuff
        const ITERATIONS: u32 = 150_000_000;
        const CAPACITY: usize = ITERATIONS as usize;
        let allocator = BoolAllocator::new(CAPACITY);

        // Perform the allocations, read the data, modify it, and leak
        testbench::benchmark(ITERATIONS, || {
            let mut allocation = allocator.allocate().unwrap();
            assert!(*allocation == false);
            *allocation = true;
        });
    }

    /// Benchmark of parallel allocation performance
    #[test]
    #[ignore]
    fn concurrent_alloc() {
        // Get ready to allocate a lot of stuff
        const ITERATIONS: u32 = 150_000_000;
        const CAPACITY: usize = 10 * (ITERATIONS as usize);
        let allocator = Arc::new(BoolAllocator::new(CAPACITY));
        let allocator2 = allocator.clone();

        // Perform leaking allocations concurrently
        testbench::concurrent_benchmark(ITERATIONS, || {
            let allocation = allocator.allocate().unwrap();
            assert!(allocation.index < CAPACITY);
            mem::forget(allocation);
        }, move || {
            mem::forget(allocator2.allocate().unwrap());
        });
    }

    /// Benchmark of parallel allocation + liberation performance
    #[test]
    #[ignore]
    fn concurrent_alloc_free() {
        // Get ready to allocate a lot of stuff
        const ITERATIONS: u32 = 150_000_000;
        const CAPACITY: usize = ITERATIONS as usize;
        let allocator = Arc::new(BoolAllocator::new(CAPACITY));
        let allocator2 = allocator.clone();

        // Perform allocations and liberations concurrently
        testbench::concurrent_benchmark(ITERATIONS, || {
            let allocation = allocator.allocate().unwrap();
            assert!(allocation.index < CAPACITY);
        }, move || {
            allocator2.allocate().unwrap();
        });
    }

    /// Benchmark of parallel allocation + data readout performance
    #[test]
    #[ignore]
    fn concurrent_alloc_read_free() {
        // Get ready to allocate a lot of stuff
        const ITERATIONS: u32 = 150_000_000;
        const CAPACITY: usize = ITERATIONS as usize;
        let allocator = Arc::new(BoolAllocator::new(CAPACITY));
        let allocator2 = allocator.clone();

        // Perform leaking allocations concurrently
        testbench::concurrent_benchmark(ITERATIONS, || {
            let allocation = allocator.allocate().unwrap();
            assert!(*allocation == false);
        }, move || {
            let allocation = allocator2.allocate().unwrap();
            assert!(*allocation == false);
        });
    }

    /// Benchmark of parallel allocation + data read/write performance
    #[test]
    #[ignore]
    fn concurrent_alloc_read_write_free() {
        // Get ready to allocate a lot of stuff
        const ITERATIONS: u32 = 150_000_000;
        const CAPACITY: usize = ITERATIONS as usize;
        let allocator = Arc::new(BoolAllocator::new(CAPACITY));
        let allocator2 = allocator.clone();

        // Perform leaking allocations concurrently
        testbench::concurrent_benchmark(ITERATIONS, || {
            let mut allocation = allocator.allocate().unwrap();
            assert!(*allocation == false);
            *allocation = allocation.index >= CAPACITY;
        }, move || {
            let mut allocation = allocator2.allocate().unwrap();
            assert!(*allocation == false);
            *allocation = allocation.index >= CAPACITY;
        });
    }
}