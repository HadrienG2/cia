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
        let allocator = IndexedAllocator::<u8>::new(42);
        assert_eq!(allocator.in_use.len(), 42);
        assert!(allocator.in_use.iter().all(|b| !b.load(Ordering::Relaxed)));
        assert_eq!(allocator.data.len(), 42);
    }

    /// Check that basic allocation works
    #[test]
    fn basic_allocation() {
        // Perform two allocations and check that the results make sense
        let allocator = IndexedAllocator::<String>::new(2);
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
        let allocator = IndexedAllocator::<f64>::new(CAPACITY);
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
        let allocator = IndexedAllocator::<char>::new(1);
        let mut allocation = allocator.allocate().unwrap();
        *allocation = '@';
        assert_eq!(*allocation, '@');
    }

    /// Check that we can deallocate data by dropping the Allocation
    #[test]
    fn deallocate() {
        let allocator = IndexedAllocator::<isize>::new(1);
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
        let allocator = IndexedAllocator::<&str>::new(1);
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
        type BoolAllocator = IndexedAllocator<bool>;
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
        type TimestampAllocator = IndexedAllocator<UsizeRaceCell>;
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
    type BoolAllocator = IndexedAllocator<bool>;

    /// Benchmark of an unrealistically optimal allocation/liberation scenario
    #[test]
    #[ignore]
    fn best_case_alloc() {
        // Get ready to allocate many times
        const ITERATIONS: u32 = 100_000_000;
        let allocator = BoolAllocator::new(1);

        // Perform an allocation/liberation cycle
        testbench::benchmark(ITERATIONS, || {
            let allocation = allocator.allocate().unwrap();
            assert!(allocation.index == 0);
        });
    }

    /// Like best_case, but with some busy cells. Allows studying their impact.
    #[test]
    #[ignore]
    fn busy_cell_alloc() {
        // Get ready to allocate many times
        const ITERATIONS: u32 = 100_000_000;
        let allocator = BoolAllocator::new(2);

        // Clog up the first cell of the allocator, to study the impact
        mem::forget(allocator.allocate().unwrap());

        // Perform an allocation/liberation cycle
        testbench::benchmark(ITERATIONS, || {
            let allocation = allocator.allocate().unwrap();
            assert!(allocation.index == 1);
        });
    }

    /// Benchmark of (sequential) allocation performance
    #[test]
    #[ignore]
    fn alloc() {
        // Get ready to allocate a lot of stuff
        const ITERATIONS: u32 = 100_000_000;
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
        const ITERATIONS: u32 = 100_000_000;
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
        const ITERATIONS: u32 = 100_000_000;
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
        const ITERATIONS: u32 = 100_000_000;
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
        const ITERATIONS: u32 = 100_000_000;
        const CAPACITY: usize = 8 * (ITERATIONS as usize);
        let allocator = Arc::new(BoolAllocator::new(CAPACITY));
        let allocator2 = allocator.clone();

        // Perform leaking allocations concurrently
        testbench::concurrent_benchmark(ITERATIONS, || {
            let allocation = allocator.allocate().unwrap();
            assert!(allocation.index < CAPACITY);
            mem::forget(allocation);
        }, move || {
            allocator2.allocate().unwrap();
        });
    }

    /// Benchmark of parallel allocation + liberation performance
    #[test]
    #[ignore]
    fn concurrent_alloc_free() {
        // Get ready to allocate a lot of stuff
        const ITERATIONS: u32 = 100_000_000;
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
        const ITERATIONS: u32 = 100_000_000;
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
        const ITERATIONS: u32 = 100_000_000;
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
            *allocation = *allocation || (allocation.index >= CAPACITY);
        });
    }
}