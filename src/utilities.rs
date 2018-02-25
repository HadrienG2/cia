//! Some utility functions used throughout this crate

/// I've wanted a utility function like this since... forever.
pub(crate) fn new_boxed_slice<F, T>(mut generator: F, size: usize) -> Box<[T]>
    where F: FnMut() -> T
{
    let mut acc = Vec::with_capacity(size);
    for _ in 0..size {
        acc.push(generator())
    }
    acc.into_boxed_slice()
}