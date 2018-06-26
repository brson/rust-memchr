/*!
This crate defines two functions, `memchr` and `memrchr`, which expose a safe
interface to the corresponding functions in `libc`.
*/
#![feature(core_intrinsics)]
#![deny(missing_docs)]
#![allow(unused_imports)]
#![doc(html_root_url = "https://docs.rs/memchr/2.0.0")]

#![cfg_attr(not(feature = "use_std"), no_std)]

#[cfg(all(test, not(feature = "use_std")))]
#[macro_use]
extern crate std;

#[cfg(all(feature = "libc", not(target_arch = "wasm32")))]
extern crate libc;

#[macro_use]
#[cfg(test)]
extern crate quickcheck;

#[cfg(all(feature = "libc", not(target_arch = "wasm32")))]
use libc::c_void;
#[cfg(all(feature = "libc", not(target_arch = "wasm32")))]
use libc::{c_int, size_t};

#[cfg(feature = "use_std")]
use std::cmp;
#[cfg(not(feature = "use_std"))]
use core::cmp;

const LO_U64: u64 = 0x0101010101010101;
const HI_U64: u64 = 0x8080808080808080;

// use truncation
const LO_USIZE: usize = LO_U64 as usize;
const HI_USIZE: usize = HI_U64 as usize;

#[cfg(target_pointer_width = "32")]
const USIZE_BYTES: usize = 4;
#[cfg(target_pointer_width = "64")]
const USIZE_BYTES: usize = 8;

/// Return `true` if `x` contains any zero byte.
///
/// From *Matters Computational*, J. Arndt
///
/// "The idea is to subtract one from each of the bytes and then look for
/// bytes where the borrow propagated all the way to the most significant
/// bit."
#[inline]
fn contains_zero_byte(x: usize) -> bool {
    x.wrapping_sub(LO_USIZE) & !x & HI_USIZE != 0
}

#[cfg(target_pointer_width = "32")]
#[inline]
fn repeat_byte(b: u8) -> usize {
    let mut rep = (b as usize) << 8 | b as usize;
    rep = rep << 16 | rep;
    rep
}

#[cfg(target_pointer_width = "64")]
#[inline]
fn repeat_byte(b: u8) -> usize {
    let mut rep = (b as usize) << 8 | b as usize;
    rep = rep << 16 | rep;
    rep = rep << 32 | rep;
    rep
}

macro_rules! iter_next {
    // Common code for the memchr iterators:
    // update haystack and position and produce the index
    //
    // self: &mut Self where Self is the iterator
    // search_result: Option<usize> which is the result of the corresponding
    // memchr function.
    //
    // Returns Option<usize> (the next iterator element)
    ($self_:expr, $search_result:expr) => {
        $search_result.map(move |index| {
            // split and take the remaining back half
            $self_.haystack = $self_.haystack.split_at(index + 1).1;
            let found_position = $self_.position + index;
            $self_.position = found_position + 1;
            found_position
        })
    }
}

macro_rules! iter_next_back {
    ($self_:expr, $search_result:expr) => {
        $search_result.map(move |index| {
            // split and take the remaining front half
            $self_.haystack = $self_.haystack.split_at(index).0;
            $self_.position + index
        })
    }
}

/// An iterator for memchr
pub struct Memchr<'a> {
    needle: u8,
    // The haystack to iterate over
    haystack: &'a [u8],
    // The index
    position: usize,
}

impl<'a> Memchr<'a> {
    /// Creates a new iterator that yields all positions of needle in haystack.
    pub fn new(needle: u8, haystack: &[u8]) -> Memchr {
        Memchr {
            needle: needle,
            haystack: haystack,
            position: 0,
        }
    }
}

impl<'a> Iterator for Memchr<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        iter_next!(self, memchr(self.needle, self.haystack))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.haystack.len()))
    }
}

impl<'a> DoubleEndedIterator for Memchr<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        iter_next_back!(self, memrchr(self.needle, self.haystack))
    }
}

/// A safe interface to `memchr`.
///
/// Returns the index corresponding to the first occurrence of `needle` in
/// `haystack`, or `None` if one is not found.
///
/// memchr reduces to super-optimized machine code at around an order of
/// magnitude faster than `haystack.iter().position(|&b| b == needle)`.
/// (See benchmarks.)
///
/// # Example
///
/// This shows how to find the first position of a byte in a byte string.
///
/// ```rust
/// use memchr::memchr;
///
/// let haystack = b"the quick brown fox";
/// assert_eq!(memchr(b'k', haystack), Some(8));
/// ```
#[inline(always)] // reduces constant overhead
pub fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    // libc memchr
    #[cfg(all(feature = "libc",
              not(target_arch = "wasm32"),
              any(not(target_os = "windows"),
                  not(any(target_pointer_width = "32",
                          target_pointer_width = "64")))))]
    #[inline(always)] // reduces constant overhead
    fn memchr_specific(needle: u8, haystack: &[u8]) -> Option<usize> {
        use libc::memchr as libc_memchr;

        let p = unsafe {
            libc_memchr(haystack.as_ptr() as *const c_void,
                        needle as c_int,
                        haystack.len() as size_t)
        };
        if p.is_null() {
            None
        } else {
            Some(p as usize - (haystack.as_ptr() as usize))
        }
    }

    // use fallback on windows, since it's faster
    // use fallback on wasm32, since it doesn't have libc
    #[cfg(all(any(not(feature = "libc"), target_os = "windows", target_arch = "wasm32"),
              any(target_pointer_width = "32",
                  target_pointer_width = "64")))]
    fn memchr_specific(needle: u8, haystack: &[u8]) -> Option<usize> {
        fallback::memchr(needle, haystack)
    }

    // For the rare case of neither 32 bit nor 64-bit platform.
    #[cfg(all(any(not(feature = "libc"), target_os = "windows"),
              not(target_pointer_width = "32"),
              not(target_pointer_width = "64")))]
    fn memchr_specific(needle: u8, haystack: &[u8]) -> Option<usize> {
        haystack.iter().position(|&b| b == needle)
    }

    memchr_specific(needle, haystack)
}

/// A safe interface to `memrchr`.
///
/// Returns the index corresponding to the last occurrence of `needle` in
/// `haystack`, or `None` if one is not found.
///
/// # Example
///
/// This shows how to find the last position of a byte in a byte string.
///
/// ```rust
/// use memchr::memrchr;
///
/// let haystack = b"the quick brown fox";
/// assert_eq!(memrchr(b'o', haystack), Some(17));
/// ```
#[inline(always)] // reduces constant overhead
pub fn memrchr(needle: u8, haystack: &[u8]) -> Option<usize> {

    #[cfg(all(feature = "libc", target_os = "linux"))]
    #[inline(always)] // reduces constant overhead
    fn memrchr_specific(needle: u8, haystack: &[u8]) -> Option<usize> {
        // GNU's memrchr() will - unlike memchr() - error if haystack is empty.
        if haystack.is_empty() {
            return None;
        }
        let p = unsafe {
            libc::memrchr(haystack.as_ptr() as *const c_void,
                          needle as c_int,
                          haystack.len() as size_t)
        };
        if p.is_null() {
            None
        } else {
            Some(p as usize - (haystack.as_ptr() as usize))
        }
    }

    #[cfg(all(not(all(feature = "libc", target_os = "linux")),
              any(target_pointer_width = "32", target_pointer_width = "64")))]
    fn memrchr_specific(needle: u8, haystack: &[u8]) -> Option<usize> {
        fallback::memrchr(needle, haystack)
    }

    // For the rare case of neither 32 bit nor 64-bit platform.
    #[cfg(all(not(all(feature = "libc", target_os = "linux")),
              not(target_pointer_width = "32"),
              not(target_pointer_width = "64")))]
    fn memrchr_specific(needle: u8, haystack: &[u8]) -> Option<usize> {
        haystack.iter().rposition(|&b| b == needle)
    }

    memrchr_specific(needle, haystack)
}

/// An iterator for Memchr2
pub struct Memchr2<'a> {
    needle1: u8,
    needle2: u8,
    // The haystack to iterate over
    haystack: &'a [u8],
    // The index
    position: usize,
}

impl<'a> Memchr2<'a> {
    /// Creates a new iterator that yields all positions of needle in haystack.
    pub fn new(needle1: u8, needle2: u8, haystack: &[u8]) -> Memchr2 {
        Memchr2 {
            needle1: needle1,
            needle2: needle2,
            haystack: haystack,
            position: 0,
        }
    }
}

impl<'a> Iterator for Memchr2<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        iter_next!(self, memchr2(self.needle1, self.needle2, self.haystack))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.haystack.len()))
    }
}


/// Like `memchr`, but searches for two bytes instead of one.
pub fn memchr2(needle1: u8, needle2: u8, haystack: &[u8]) -> Option<usize> {
    fn slow(b1: u8, b2: u8, haystack: &[u8]) -> Option<usize> {
        haystack.iter().position(|&b| b == b1 || b == b2)
    }

    let len = haystack.len();
    let ptr = haystack.as_ptr();
    let align = (ptr as usize) & (USIZE_BYTES - 1);
    let mut i = 0;
    if align > 0 {
        i = cmp::min(USIZE_BYTES - align, len);
        if let Some(found) = slow(needle1, needle2, &haystack[..i]) {
            return Some(found);
        }
    }
    let repeated_b1 = repeat_byte(needle1);
    let repeated_b2 = repeat_byte(needle2);
    if len >= USIZE_BYTES {
        while i <= len - USIZE_BYTES {
            unsafe {
                let u = *(ptr.offset(i as isize) as *const usize);
                let found_ub1 = contains_zero_byte(u ^ repeated_b1);
                let found_ub2 = contains_zero_byte(u ^ repeated_b2);
                if found_ub1 || found_ub2 {
                    break;
                }
            }
            i += USIZE_BYTES;
        }
    }
    slow(needle1, needle2, &haystack[i..]).map(|pos| i + pos)
}

/// An iterator for Memchr3
pub struct Memchr3<'a> {
    needle1: u8,
    needle2: u8,
    needle3: u8,
    // The haystack to iterate over
    haystack: &'a [u8],
    // The index
    position: usize,
}

impl<'a> Memchr3<'a> {
    /// Create a new `Memchr3` that's initialized to zero with a haystack
    pub fn new(
        needle1: u8,
        needle2: u8,
        needle3: u8,
        haystack: &[u8],
    ) -> Memchr3 {
        Memchr3 {
            needle1: needle1,
            needle2: needle2,
            needle3: needle3,
            haystack: haystack,
            position: 0,
        }
    }
}

impl<'a> Iterator for Memchr3<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        iter_next!(
            self,
            memchr3(self.needle1, self.needle2, self.needle3, self.haystack)
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.haystack.len()))
    }
}

/// Like `memchr`, but searches for three bytes instead of one.
pub fn memchr3(
    needle1: u8,
    needle2: u8,
    needle3: u8,
    haystack: &[u8],
) -> Option<usize> {
    fn slow(b1: u8, b2: u8, b3: u8, haystack: &[u8]) -> Option<usize> {
        haystack.iter().position(|&b| b == b1 || b == b2 || b == b3)
    }

    let len = haystack.len();
    let ptr = haystack.as_ptr();
    let align = (ptr as usize) & (USIZE_BYTES - 1);
    let mut i = 0;
    if align > 0 {
        i = cmp::min(USIZE_BYTES - align, len);
        if let Some(found) = slow(needle1, needle2, needle3, &haystack[..i]) {
            return Some(found);
        }
    }
    let repeated_b1 = repeat_byte(needle1);
    let repeated_b2 = repeat_byte(needle2);
    let repeated_b3 = repeat_byte(needle3);
    if len >= USIZE_BYTES {
        while i <= len - USIZE_BYTES {
            unsafe {
                let u = *(ptr.offset(i as isize) as *const usize);
                let found_ub1 = contains_zero_byte(u ^ repeated_b1);
                let found_ub2 = contains_zero_byte(u ^ repeated_b2);
                let found_ub3 = contains_zero_byte(u ^ repeated_b3);
                if found_ub1 || found_ub2 || found_ub3 {
                    break;
                }
            }
            i += USIZE_BYTES;
        }
    }
    slow(needle1, needle2, needle3, &haystack[i..]).map(|pos| i + pos)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(missing_docs)]
pub mod avx2 {
    #[cfg(all(feature = "use_std", target_arch = "x86_64"))]
    use std::arch::x86_64::*;
    #[cfg(all(not(feature = "use_std"), target_arch = "x86_64"))]
    use core::arch::x86_64::*;
    #[cfg(all(feature = "use_std", target_arch = "x86"))]
    use std::arch::x86::*;
    #[cfg(all(not(feature = "use_std"), target_arch = "x86"))]
    use core::arch::x86::*;

    include!("memchr_avx2_call_table.rs");

    #[inline(always)]
    pub fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
        unsafe { memchr_avx2(needle, haystack) }
    }

    // NB: We don't want to inline this because then the optimizer won't be able
    // to turn the tail call into a jump table. Effectively, the other,
    // non-inline avx functions are inlined into this one. TODO: verify this.
    #[inline(never)]
    unsafe fn memchr_avx2(needle: u8, haystack: &[u8]) -> Option<usize> {
        use std::intrinsics::{likely};

        // FIXME: assembly for this is reloading dil into edi unnecessarily...
        let len = haystack.len();
        // FIXME: Do this jump unconditionally and then jump at the end of
        // each function to the > 256 case.
        if likely(len < 256) {
            AVX2FNS[len as u8 as usize](needle, haystack)
        } else {
            memchr_avx2_ge256(needle, haystack)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn memchr_avx2_0(_needle: u8, haystack: &[u8]) -> Option<usize> {
        debug_assert_eq!(haystack.len(), 0);

        None
    }

    #[target_feature(enable = "avx2")]
    unsafe fn memchr_avx2_1(needle: u8, haystack: &[u8]) -> Option<usize> {
         debug_assert_eq!(haystack.len(), 1);

        let p: *const u8 = haystack.as_ptr();
        if *p.offset(0) == needle { return Some(0); }
        return None;
    }

    #[target_feature(enable = "avx2")]
    unsafe fn memchr_avx2_2(needle: u8, haystack: &[u8]) -> Option<usize> {
        debug_assert_eq!(haystack.len(), 2);

        let p: *const u8 = haystack.as_ptr();
        if *p.offset(0) == needle { return Some(0); }
        if *p.offset(1) == needle { return Some(1); }
        return None;
    }

    #[target_feature(enable = "avx2")]
    unsafe fn memchr_avx2_3(needle: u8, haystack: &[u8]) -> Option<usize> {
        //use std::intrinsics::cttz_nonzero;

        debug_assert_eq!(haystack.len(), 3);

        // FIXME: Hand-implement this like the autovectorizor
        // Hand-implement lt32 like this is autovectorizered
        /*let p: *const u8 = haystack.as_ptr();
        let r1 = *p.offset(0) == needle;
        let r2 = *p.offset(1) == needle;
        let r3 = *p.offset(2) == needle;
        let mut mask = 0u8;
        mask |= (r1 as u8) << 0;
        mask |= (r2 as u8) << 1;
        mask |= (r3 as u8) << 2;
        if mask != 0 {
            return Some(cttz_nonzero(mask as usize));
        }
        return None;*/

        let p: *const u8 = haystack.as_ptr();
        if *p.offset(0) == needle { return Some(0); }
        if *p.offset(1) == needle { return Some(1); }
        if *p.offset(2) == needle { return Some(2); }
        return None;
    }

    #[target_feature(enable = "avx2")]
    unsafe fn memchr_avx2_lt32(needle: u8, haystack: &[u8]) -> Option<usize> {
        debug_assert!(haystack.len() < 32);

        let p: *const u8 = haystack.as_ptr();
        let len = haystack.len() as isize;
        let q_x15 = _mm256_set1_epi8(needle as i8);
        return do_tail(p, len, 0, q_x15);
    }

    #[target_feature(enable = "avx2")]
    unsafe fn memchr_avx2_lt64(needle: u8, haystack: &[u8]) -> Option<usize> {
        debug_assert!(haystack.len() >= 32);
        debug_assert!(haystack.len() < 64);

        let p: *const u8 = haystack.as_ptr();
        let len = haystack.len() as isize;
        let q_x15 = _mm256_set1_epi8(needle as i8);

        if let Some(r) = cmp(q_x15, p, 0, 0) {
            return Some(r);
        }

        return do_tail(p, len, 32, q_x15);
    }

    #[target_feature(enable = "avx2")]
    unsafe fn memchr_avx2_lt256(needle: u8, haystack: &[u8]) -> Option<usize> {
        debug_assert!(haystack.len() >= 64);
        debug_assert!(haystack.len() < 256);

        let p: *const u8 = haystack.as_ptr();
        let len = haystack.len() as isize;
        let q_x15 = _mm256_set1_epi8(needle as i8);

        if let Some(r) = cmp(q_x15, p, 0, 0) {
            return Some(r);
        }
        if let Some(r) = cmp(q_x15, p, 32, 0) {
            return Some(r);
        }

        let mut i = 64;
        let len_minus = len - 32;

        while i <= len_minus {
            if let Some(r) = cmp(q_x15, p, i, 0) {
                return Some(r);
            }
            i += 32;
        }

        debug_assert!(len - i < 32);
        return do_tail(p, len, i, q_x15);
    }

    #[target_feature(enable = "avx2")]
    unsafe fn memchr_avx2_ge256(needle: u8, haystack: &[u8]) -> Option<usize> {
        debug_assert!(haystack.len() >= 256);

        use std::intrinsics::{likely, unlikely};

        let p: *const u8 = haystack.as_ptr();
        let len = haystack.len() as isize;
        debug_assert!(haystack.len() <= isize::max_value() as usize);

        if unlikely(len < 288) {
            return memchr_avx2_lt288(needle, haystack);
        }

        let mut i = 0;
        let q_x15 = _mm256_set1_epi8(needle as i8);

        // TODO consider stream_load
        // consider testc_si256 / testnzc_si256 / testz
        // investigate permute + bmi2 pext
        // https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
        // TODO: Add tests for finding in haystack more than 256 bytes

        let len_minus = len - 288;

        while i <= len_minus {
            let j = i;
            let loadcmp = |o| {
                let x = load(p, j + o);
                let x = _mm256_cmpeq_epi8(x, q_x15);
                x
            };

            let x0 = loadcmp(0);
            let x1 = loadcmp(32);
            let x2 = loadcmp(64);
            let x3 = loadcmp(96);
            let x4 = loadcmp(128);
            let x5 = loadcmp(160);
            let x6 = loadcmp(192);
            let x7 = loadcmp(224);
            let x8 = loadcmp(256);

            // TODO: check perf of max_epu for all these ors
            // LLVM actually optimizes these ors from 9 to 8 and eliminates the
            // setzero.
            let sum_01_x8 = _mm256_or_si256(x0, x1);
            let sum_23_x9 = _mm256_or_si256(x2, x3);
            let sum_45_x10 = _mm256_or_si256(x4, x5);
            let sum_67_x11 = _mm256_or_si256(x6, x7);

            let sum_init_x12 = _mm256_setzero_si256();
            let sum_01_x12 = _mm256_or_si256(sum_init_x12, sum_01_x8);
            let sum_03_x12 = _mm256_or_si256(sum_01_x12, sum_23_x9);
            let sum_05_x12 = _mm256_or_si256(sum_03_x12, sum_45_x10);
            let sum_07_x12 = _mm256_or_si256(sum_05_x12, sum_67_x11);
            let sum_08_x12 = _mm256_or_si256(sum_07_x12, x8);

            // Just to make it clear we're done with these
            drop(sum_init_x12);
            drop(sum_01_x12);
            drop(sum_03_x12);
            drop(sum_05_x12);
            drop(sum_07_x12);

            // Seems to be slightly faster than vptest, though
            // it uses one more instruction
            let sum = _mm256_movemask_epi8(sum_08_x12);
            if sum == 0 {
                i += 288;
                continue;
            }

            // NB: The assembly code for resolving the match is expected to
            // be straightforword (looking just much as the intrinsics
            // read), but LLVM is spewing some AVX vomit that I don't
            // understand. For long searches that doesn't matter much, but
            // the overhead matters for early matches. Would be good to
            // resolve.

            let offset = None
                .or_else(|| check_match(i + 0, sum_01_x8, x0, x1, false))
                .or_else(|| check_match(i + 64, sum_23_x9, x2, x3, false))
                .or_else(|| check_match(i + 128, sum_45_x10, x4, x5, false))
                .or_else(|| check_match(i + 192, sum_67_x11, x6, x7, false))
                .or_else(|| {
                    let matches = _mm256_movemask_epi8(x8);
                    debug_assert!(matches != 0);
                    return off(i + 256, matches);
                });

            debug_assert!(offset.is_some());
            return offset;
        }

        let len_minus = len - 32;
        while i <= len_minus  {
            if let Some(r) = cmp(q_x15, p, i, 0) {
                return Some(r);
            }

            i += 32;
        }

        if i < len {
            debug_assert!(len - i < 32);
            return do_tail(p, len, i, q_x15);
        }

        None
    }

    #[target_feature(enable = "avx2")]
    unsafe fn memchr_avx2_lt288(needle: u8, haystack: &[u8]) -> Option<usize> {
        debug_assert!(haystack.len() >= 256);

        use std::intrinsics::{likely, unlikely};

        let p: *const u8 = haystack.as_ptr();
        let len = haystack.len() as isize;
        debug_assert!(haystack.len() <= isize::max_value() as usize);

        let mut i = 0;
        let q_x15 = _mm256_set1_epi8(needle as i8);

        let j = i;
        let loadcmp = |o| {
            let x = load(p, j + o);
            let x = _mm256_cmpeq_epi8(x, q_x15);
            x
        };

        let x0 = loadcmp(0);
        let x1 = loadcmp(32);
        let x2 = loadcmp(64);
        let x3 = loadcmp(96);
        let x4 = loadcmp(128);
        let x5 = loadcmp(160);
        let x6 = loadcmp(192);
        let x7 = loadcmp(224);

        let sum_01_x8 = _mm256_or_si256(x0, x1);
        let sum_23_x9 = _mm256_or_si256(x2, x3);
        let sum_45_x10 = _mm256_or_si256(x4, x5);
        let sum_67_x11 = _mm256_or_si256(x6, x7);

        let sum_init_x12 = _mm256_setzero_si256();
        let sum_01_x12 = _mm256_or_si256(sum_init_x12, sum_01_x8);
        let sum_03_x12 = _mm256_or_si256(sum_01_x12, sum_23_x9);
        let sum_05_x12 = _mm256_or_si256(sum_03_x12, sum_45_x10);
        let sum_07_x12 = _mm256_or_si256(sum_05_x12, sum_67_x11);

        // Just to make it clear we're done with these
        drop(sum_init_x12);
        drop(sum_01_x12);
        drop(sum_03_x12);
        drop(sum_05_x12);

        let sum = _mm256_movemask_epi8(sum_07_x12);
        if sum != 0 {
            let offset = None
                .or_else(|| check_match(i + 0, sum_01_x8, x0, x1, false))
                .or_else(|| check_match(i + 64, sum_23_x9, x2, x3, false))
                .or_else(|| check_match(i + 128, sum_45_x10, x4, x5, false))
                .or_else(|| check_match(i + 192, sum_67_x11, x6, x7, false));

            debug_assert!(offset.is_some());
            return offset;
        }

        i += 256;

        if i < len {
            debug_assert!(len - i < 32);
            return do_tail(p, len, i, q_x15);
        }

        None
    }

    #[inline(always)]
    unsafe fn check_match(o: isize, sumv: __m256i,
                          v0: __m256i, v1: __m256i,
                          contains_needle: bool) -> Option<usize> {

        debug_assert!(!contains_needle || _mm256_movemask_epi8(sumv) != 0);

        // This movemask is the thing LLVM is generating many extra
        // instructions for. Using vptest instead doesn't do any
        // better.
        // NB: This movemask will be optimized away when contains_needle
        // is true.
        let matches = _mm256_movemask_epi8(sumv);
        if contains_needle || matches != 0 {
            let matches_0 = _mm256_movemask_epi8(v0);
            if matches_0 != 0 {
                return off(o + 0, matches_0)
            };
            let matches_1 = _mm256_movemask_epi8(v1);
            debug_assert!(matches_1 != 0);
            return off(o + 32, matches_1);
        }
        None
    }

    #[inline(always)]
    unsafe fn off(offset: isize, bitmask: i32) -> Option<usize> {
        use std::intrinsics::cttz_nonzero;
        Some((offset + cttz_nonzero(bitmask) as isize) as usize)
    }

    #[inline(always)]
    unsafe fn do_tail(p: *const u8, len: isize,
                      mut i: isize, q: __m256i) -> Option<usize> {

        use std::intrinsics::{likely, unlikely};

        // TODO: fall back to sse when appropriate

        let rem = len - i;
        debug_assert!(rem < 32);

        let align_mask = 32 - 1;
        let overalignment = (p.offset(i) as usize & align_mask) as isize;
        debug_assert!(overalignment < 32);
        //println!("over {}", overalignment);

        // Unlikely because byte arrays aren't generally aligned to 32-byte
        // boundaries
        if unlikely(overalignment == 0) {
            //println!("no overalignment");
            // TODO: extract this into another function
            let o = i + 0;
            let x = _mm256_load_si256(p.offset(o) as *const __m256i);
            let r = _mm256_cmpeq_epi8(x, q);
            let z = _mm256_movemask_epi8(r);
            let garbage_mask = {
                let ones = u32::max_value();
                let mask = ones << rem;
                let mask = !mask;
                mask as i32
            };
            let z = z & garbage_mask;
            if z != 0 {
                return off(o, z);
            }

            return None;
        }

        let readable_before = 32 - overalignment;
        let good_bytes_before = ::std::cmp::min(rem, readable_before);
        let good_bytes_after = rem - good_bytes_before;
        debug_assert!(good_bytes_before < 32);
        debug_assert!(overalignment < 32);
        //println!("gbb {} gba {}", good_bytes_before, good_bytes_after);

        i -= overalignment;

        // TODO: simd threshold
        // TODO: for the footer call we don't need to mask out
        // the bytes already read
        if good_bytes_before > 0 {
            let o = i + 0;
            let x = _mm256_load_si256(p.offset(o) as *const __m256i);
            let r = _mm256_cmpeq_epi8(x, q);
            let z = _mm256_movemask_epi8(r);
            let garbage_mask = {
                let ones = u32::max_value();
                let mask = ones << good_bytes_before;
                let mask = !mask;
                let mask = mask << overalignment;
                mask as i32
            };
            let z = z & garbage_mask;
            if z != 0 {
                return off(o, z);
            }
        }

        i += 32;

        debug_assert!(i >= len);
        debug_assert!(i + 32 >= len, good_bytes_after > 0);

        // TODO: simd threshold
        if good_bytes_after > 0 {
            let o = i + 0;
            let x = _mm256_load_si256(p.offset(o) as *const __m256i);
            let r = _mm256_cmpeq_epi8(x, q);
            let z = _mm256_movemask_epi8(r);
            let garbage_mask = {
                let ones = u32::max_value();
                let mask = ones << good_bytes_after;
                let mask = !mask;
                mask as i32
            };
            let z = z & garbage_mask;
            if z != 0 {
                return off(o, z);
            }
        }

        if cfg!(debug) || cfg!(test) {
            i += good_bytes_after;
        }

        debug_assert_eq!(i, len);

        return None;
    }

    // TODO remove
    #[inline(always)]
    unsafe fn load(p: *const u8, o: isize) -> __m256i {
        _mm256_loadu_si256(p.offset(o) as *const __m256i)
    }

    #[inline(always)]
    unsafe fn cmp(q: __m256i, p: *const u8, i: isize, o: isize) -> Option<usize> {
        let o = i + o;
        let x = load(p, o);
        let r = _mm256_cmpeq_epi8(x, q);
        let z = _mm256_movemask_epi8(r);
        if z != 0 {
            return off(o, z);
        }
        None
    }

    #[target_feature(enable = "avx2")]
    #[allow(unused)]
    unsafe fn memchr_avx2_(needle: u8, haystack: &[u8]) -> Option<usize> {
        use std::intrinsics::{likely, unlikely, cttz_nonzero};

        let p: *const u8 = haystack.as_ptr();
        let len = haystack.len() as isize;
        debug_assert!(haystack.len() <= isize::max_value() as usize);

        match len {
            0 => {
                return None;
            }
            1 => {
                if *p.offset(0) == needle { return Some(0); }
                return None;
            }
            2 => {
                if *p.offset(0) == needle { return Some(0); }
                if *p.offset(1) == needle { return Some(1); }
                return None;
            }
            3 => {
                if *p.offset(0) == needle { return Some(0); }
                if *p.offset(1) == needle { return Some(1); }
                if *p.offset(2) == needle { return Some(2); }
                return None;
            }
            _ if len < 32 => {
                let q_x15 = _mm256_set1_epi8(needle as i8);

                return do_tail(p, len, 0, q_x15);
            }
            _ if len < 64 => {
                let mut i = 0;
                let q_x15 = _mm256_set1_epi8(needle as i8);

                if let Some(r) = cmp(q_x15, p, i, 0) {
                    return Some(r);
                }
                i += 32;

                debug_assert!(len - i < 32);
                return do_tail(p, len, i, q_x15);
            }
            _ if len < 256 => {
                let mut i = 0;
                let q_x15 = _mm256_set1_epi8(needle as i8);

                let len_minus = len - 32;
                while i <= len_minus {
                    if let Some(r) = cmp(q_x15, p, i, 0) {
                        return Some(r);
                    }
                    i += 32;
                }

                debug_assert!(len - i < 32);
                return do_tail(p, len, i, q_x15);
            }
            _ => { }
        }

        #[inline(always)]
        unsafe fn off(offset: isize, bitmask: i32) -> Option<usize> {
            Some((offset + cttz_nonzero(bitmask) as isize) as usize)
        }

        #[inline(always)]
        unsafe fn do_tail(p: *const u8, len: isize,
                          mut i: isize, q: __m256i) -> Option<usize> {

            // TODO: fall back to sse when appropriate

            let rem = len - i;
            debug_assert!(rem < 32);

            if rem == 0 { return None }

            let align_mask = 32 - 1;
            let overalignment = (p.offset(i) as usize & align_mask) as isize;
            debug_assert!(overalignment < 32);

            if overalignment == 0 {
                // TODO: extract this into another function
                let o = i + 0;
                let x = _mm256_load_si256(p.offset(o) as *const __m256i);
                let r = _mm256_cmpeq_epi8(x, q);
                let z = _mm256_movemask_epi8(r);
                let garbage_mask = {
                    let ones = u32::max_value();
                    let mask = ones << rem;
                    let mask = !mask;
                    mask as i32
                };
                let z = z & garbage_mask;
                if z != 0 {
                    return off(o, z);
                }

                return None;
            }

            let readable_before = 32 - overalignment;
            let good_bytes_before = ::std::cmp::min(rem, readable_before);
            let good_bytes_after = rem - good_bytes_before;
            debug_assert!(good_bytes_before < 32);
            debug_assert!(overalignment < 32);

            i -= overalignment;

            // TODO: simd threshold
            // TODO: for the footer call we don't need to mask out
            // the bytes already read
            if good_bytes_before > 0 {
                let o = i + 0;
                let x = _mm256_load_si256(p.offset(o) as *const __m256i);
                let r = _mm256_cmpeq_epi8(x, q);
                let z = _mm256_movemask_epi8(r);
                let garbage_mask = {
                    let ones = u32::max_value();
                    let mask = ones << good_bytes_before;
                    let mask = !mask;
                    let mask = mask << overalignment;
                    mask as i32
                };
                let z = z & garbage_mask;
                if z != 0 {
                    return off(o, z);
                }
            }

            i += 32;

            if i >= len {
                if cfg!(debug) || cfg!(test) {
                    i += overalignment;
                    i += len - i;
                }

                debug_assert_eq!(i, len);
                return None;
            }

            debug_assert!(i + 32 > len);

            // TODO: simd threshold
            if good_bytes_after > 0 {
                let o = i + 0;
                let x = _mm256_load_si256(p.offset(o) as *const __m256i);
                let r = _mm256_cmpeq_epi8(x, q);
                let z = _mm256_movemask_epi8(r);
                let garbage_mask = {
                    let ones = u32::max_value();
                    let mask = ones << good_bytes_after;
                    let mask = !mask;
                    mask as i32
                };
                let z = z & garbage_mask;
                if z != 0 {
                    return off(o, z);
                }
            }

            if cfg!(debug) || cfg!(test) {
                i += good_bytes_after;
            }

            debug_assert_eq!(i, len);

            return None;
        }

        // TODO remove
        #[inline(always)]
        unsafe fn load(p: *const u8, o: isize) -> __m256i {
            _mm256_loadu_si256(p.offset(o) as *const __m256i)
        }

        #[inline(always)]
        unsafe fn cmp(q: __m256i, p: *const u8, i: isize, o: isize) -> Option<usize> {
            let o = i + o;
            let x = load(p, o);
            let r = _mm256_cmpeq_epi8(x, q);
            let z = _mm256_movemask_epi8(r);
            if z != 0 {
                return off(o, z);
            }
            None
        }

        let mut i = 0;
        let q_x15 = _mm256_set1_epi8(needle as i8);

        // TODO consider stream_load
        // consider testc_si256 / testnzc_si256 / testz
        // investigate permute + bmi2 pext
        // https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
        // TODO: Add tests for finding in haystack more than 256 bytes

        let len_minus = len - 288;

        while i <= len_minus {
            let j = i;
            let loadcmp = |o| {
                let x = load(p, j + o);
                let x = _mm256_cmpeq_epi8(x, q_x15);
                x
            };

            let x0 = loadcmp(0);
            let x1 = loadcmp(32);
            let x2 = loadcmp(64);
            let x3 = loadcmp(96);
            let x4 = loadcmp(128);
            let x5 = loadcmp(160);
            let x6 = loadcmp(192);
            let x7 = loadcmp(224);
            let x8 = loadcmp(256);

            // TODO: check perf of max_epu for all these ors
            // LLVM actually optimizes these ors from 9 to 8 and eliminates the
            // setzero.
            let sum_01_x8 = _mm256_or_si256(x0, x1);
            let sum_23_x9 = _mm256_or_si256(x2, x3);
            let sum_45_x10 = _mm256_or_si256(x4, x5);
            let sum_67_x11 = _mm256_or_si256(x6, x7);

            let sum_init_x12 = _mm256_setzero_si256();
            let sum_01_x12 = _mm256_or_si256(sum_init_x12, sum_01_x8);
            let sum_03_x12 = _mm256_or_si256(sum_01_x12, sum_23_x9);
            let sum_05_x12 = _mm256_or_si256(sum_03_x12, sum_45_x10);
            let sum_07_x12 = _mm256_or_si256(sum_05_x12, sum_67_x11);
            let sum_08_x12 = _mm256_or_si256(sum_07_x12, x8);

            // Just to make it clear we're done with these
            drop(sum_init_x12);
            drop(sum_01_x12);
            drop(sum_03_x12);
            drop(sum_05_x12);
            drop(sum_07_x12);

            // Seems to be slightly faster than vptest, though
            // it uses one more instruction
            let sum = _mm256_movemask_epi8(sum_08_x12);
            if sum == 0 {
                i += 288;
                continue;
            }

            // NB: The assembly code for resolving the match is expected to
            // be straightforword (looking just much as the intrinsics
            // read), but LLVM is spewing some AVX vomit that I don't
            // understand. For long searches that doesn't matter much, but
            // the overhead matters for early matches. Would be good to
            // resolve.

            #[inline(always)]
            unsafe fn check_match(o: isize, sumv: __m256i,
                                  v0: __m256i, v1: __m256i,
                                  contains_needle: bool) -> Option<usize> {

                debug_assert!(!contains_needle || _mm256_movemask_epi8(sumv) != 0);

                // This movemask is the thing LLVM is generating many extra
                // instructions for. Using vptest instead doesn't do any
                // better.
                // NB: This movemask will be optimized away when contains_needle
                // is true.
                let matches = _mm256_movemask_epi8(sumv);
                if contains_needle || matches != 0 {
                    let matches_0 = _mm256_movemask_epi8(v0);
                    if matches_0 != 0 {
                        return off(o + 0, matches_0)
                    };
                    let matches_1 = _mm256_movemask_epi8(v1);
                    debug_assert!(matches_1 != 0);
                    return off(o + 32, matches_1);
                }
                None
            }

            let offset = None
                .or_else(|| check_match(i + 0, sum_01_x8, x0, x1, false))
                .or_else(|| check_match(i + 64, sum_23_x9, x2, x3, false))
                .or_else(|| check_match(i + 128, sum_45_x10, x4, x5, false))
                .or_else(|| check_match(i + 192, sum_67_x11, x6, x7, false))
                .or_else(|| {
                    let matches = _mm256_movemask_epi8(x8);
                    debug_assert!(matches != 0);
                    return off(i + 256, matches);
                });

            debug_assert!(offset.is_some());
            return offset;
        }

        let len_minus = len - 32;
        while i <= len_minus  {
            if let Some(r) = cmp(q_x15, p, i, 0) {
                return Some(r);
            }

            i += 32;
        }

        debug_assert!(len - i < 32);
        do_tail(p, len, i, q_x15)
    }

    pub fn memrchr(needle: u8, haystack: &[u8]) -> Option<usize> {
        ::fallback::memrchr(needle, haystack)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(missing_docs)]
pub mod sse {
    #[cfg(all(feature = "use_std", target_arch = "x86_64"))]
    use std::arch::x86_64::*;
    #[cfg(all(not(feature = "use_std"), target_arch = "x86_64"))]
    use core::arch::x86_64::*;
    #[cfg(all(feature = "use_std", target_arch = "x86"))]
    use std::arch::x86::*;
    #[cfg(all(not(feature = "use_std"), target_arch = "x86"))]
    use core::arch::x86::*;

    #[inline(always)]
    pub fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
        unsafe { memchr_sse(needle, haystack) }
    }

    #[target_feature(enable = "sse3")]
    pub unsafe fn memchr_sse(needle: u8, haystack: &[u8]) -> Option<usize> {
        use std::intrinsics::{likely, unlikely, cttz_nonzero};

        if haystack.is_empty() { return None }
        if unlikely(haystack[0] == needle) {
            return Some(0);
        }

        let start = haystack.as_ptr();
        let end = start.offset(haystack.len() as isize);

        // TODO try not incrementing i
        let mut i: *const u8 = start;

        /*if haystack.len() < 16 {
            let mask = 4 - 1;
            while i as usize & mask != 0 && i < end {
                if unlikely(*i == needle) {
                    return Some(i as usize - start as usize);
                }
                i = i.offset(1);
            }
            let n32 = needle as u32;
            let mut qq = n32 | n32 << 8;
            qq = qq | qq << 16;
            while i.offset(4) < end {
                let ii = i as *const u32;
                let j = *ii ^ qq;
                let jj = &j as *const u32 as *const u8;
                #[inline]
                fn contains_zero_byte_(x: u32) -> bool {
                    const LO_U32: u32 = 0x01010101;
                    const HI_U32: u32 = 0x80808080;
                    x.wrapping_sub(LO_U32) & !x & HI_U32 != 0
                }
                if unlikely(contains_zero_byte_(j)) {
                    if unlikely(*jj.offset(0) == 0) {
                        return Some(i as usize - start as usize);
                    }
                    if unlikely(*jj.offset(1) == 0) {
                        return Some(i.offset(1) as usize - start as usize);
                    }
                    if unlikely(*jj.offset(2) == 0) {
                        return Some(i.offset(2) as usize - start as usize);
                    }
                    if unlikely(*jj.offset(3) == 0) {
                        return Some(i.offset(3) as usize - start as usize);
                    }
                    // unreachable
                }
                i = i.offset(4);
            }
            while i < end {
                if *i == needle {
                    return Some(i as usize - start as usize);
                }
                i = i.offset(1);
            }
            return None;
        }*/

        let q = _mm_set1_epi8(needle as i8);

        // sse performs better with properly aligned data, so let's work through the
        // initial bytes until we reach an alignment
        /*let align_mask: usize = 0b1111;
        let is_aligned = i as usize & align_mask == 0;
        if likely(!is_aligned) {
            // First do an unaligned simd search
            let x = _mm_lddqu_si128(i as *const __m128i);
            let r = _mm_cmpeq_epi8(x, q);
            let z = _mm_movemask_epi8(r);
            if unlikely(z != 0) {
                return Some(i as usize - start as usize + cttz_nonzero(z) as usize);
            }
            i = i.offset(16);

            // The least-significant 4 bytes of an address must be zeroed to be aligned
            let align_mask: usize = !0b1111;
            i = (i as usize & align_mask) as *const _;
            // Now i is aligned, everything before it is scanned (and maybe some after it).
        }*/

        // TODO: Find a cutoff point for haystack.len() cutoff point
        // for doing simd search
        // TODO: figure out if this extra upfront aligned short buffer
        // branch is worth it.
        
        let align_mask = 16 - 1;
        let overalignment = (i as usize & align_mask) as isize;
        let aligned = overalignment == 0;

        /*if haystack.len() < 16 && aligned {
            let o = 0;
            let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
            let r = _mm_cmpeq_epi8(x, q);
            let z = _mm_movemask_epi8(r);
            let extra_bytes = i as usize + 16 - end as usize;
            let garbage_mask = {
                let ones = u32::max_value();
                let mask = ones << (16 - extra_bytes);
                let mask = !mask;
                mask as i32
            };
            let z = z & garbage_mask;
            if unlikely(z != 0) {
                return Some(i.offset(o) as usize + cttz_nonzero(z) as usize - start as usize);
            }
        }*/

        // Handle unaligned haystacks or haystacks less than 16 bytes. This
        // searches bytes both prior to and beyond the haystack, but not
        // beyond the cacheline or page boundary.
        // TODO figure out if doing the more complex masking and
        // pointer math is worth the aligned load.
        if !aligned {
            // haystack.len() may be < 16
            i = i.wrapping_offset(-overalignment);
            let o = 0;
            let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
            let r = _mm_cmpeq_epi8(x, q);
            let z = _mm_movemask_epi8(r);
            let garbage_mask = {
                debug_assert!(overalignment < 16);
                let max_hi_bytes = ::std::cmp::min(haystack.len(), 16);
                let ones = u32::max_value();
                let mask = ones << max_hi_bytes;
                let mask = !mask;
                let mask = mask << overalignment;
                mask as i32
            };
            let z = z & garbage_mask;
            if unlikely(z != 0) {
                return Some(i.offset(o) as usize + cttz_nonzero(z) as usize - start as usize);
            }

            i = i.offset(16);

            if i >= end {
                return None;
            }

            if i.offset(16) > end {
                let o = 0;
                let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
                let r = _mm_cmpeq_epi8(x, q);
                let z = _mm_movemask_epi8(r);
                let extra_bytes = i as usize + 16 - end as usize;
                let garbage_mask = {
                    let ones = u32::max_value();
                    let mask = ones << (16 - extra_bytes);
                    let mask = !mask;
                    mask as i32
                };
                let z = z & garbage_mask;
                if unlikely(z != 0) {
                    return Some(i.offset(o) as usize + cttz_nonzero(z) as usize - start as usize);
                }

                return None;
            }
        }

        // TODO confirm this performs better than the max_epu8 strategy
        /*if likely(i.offset(64) <= end) {
            let o = 0;
            let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
            let r = _mm_cmpeq_epi8(x, q);
            let z = _mm_movemask_epi8(r);
            if unlikely(z != 0) {
                return Some(i.offset(o) as usize - start as usize + cttz_nonzero(z) as usize);
            }

            let o = 16;
            let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
            let r = _mm_cmpeq_epi8(x, q);
            let z = _mm_movemask_epi8(r);
            if unlikely(z != 0) {
                return Some(i.offset(o) as usize - start as usize + cttz_nonzero(z) as usize);
            }

            let o = 32;
            let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
            let r = _mm_cmpeq_epi8(x, q);
            let z = _mm_movemask_epi8(r);
            if unlikely(z != 0) {
                return Some(i.offset(o) as usize - start as usize + cttz_nonzero(z) as usize);
            }

            let o = 48;
            let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
            let r = _mm_cmpeq_epi8(x, q);
            let z = _mm_movemask_epi8(r);
            if unlikely(z != 0) {
                return Some(i.offset(o) as usize - start as usize + cttz_nonzero(z) as usize);
            }

            i = i.offset(64);
        }*/

        if likely(i.offset(128) <= end) {
            let mut ii = i;
            let mut found = false;
            let end_minus = end.offset(-128);
            let mut r0_ = ::std::mem::uninitialized();
            let mut r1_ = ::std::mem::uninitialized();
            let mut r2_ = ::std::mem::uninitialized();
            let mut r3_ = ::std::mem::uninitialized();
            let mut r4_ = ::std::mem::uninitialized();
            let mut r5_ = ::std::mem::uninitialized();
            let mut r6_ = ::std::mem::uninitialized();
            while ii <= end_minus {
                let j = ii;
                let mut x0 = _mm_load_si128(j.offset(0) as *const __m128i);
                let mut x1 = _mm_load_si128(j.offset(16) as *const __m128i);
                let mut x2 = _mm_load_si128(j.offset(32) as *const __m128i);
                let mut x3 = _mm_load_si128(j.offset(48) as *const __m128i);
                let mut x4 = _mm_load_si128(j.offset(64) as *const __m128i);
                let mut x5 = _mm_load_si128(j.offset(80) as *const __m128i);
                let mut x6 = _mm_load_si128(j.offset(96) as *const __m128i);
                let mut x7 = _mm_load_si128(j.offset(112) as *const __m128i);

                x0 = _mm_cmpeq_epi8(x0, q);
                x1 = _mm_cmpeq_epi8(x1, q);
                x2 = _mm_cmpeq_epi8(x2, q);
                x3 = _mm_cmpeq_epi8(x3, q);
                x4 = _mm_cmpeq_epi8(x4, q);
                x5 = _mm_cmpeq_epi8(x5, q);
                x6 = _mm_cmpeq_epi8(x6, q);
                x7 = _mm_cmpeq_epi8(x7, q);

                x7 = _mm_max_epu8(x0, x7);
                x7 = _mm_max_epu8(x1, x7);
                x7 = _mm_max_epu8(x2, x7);
                x7 = _mm_max_epu8(x3, x7);
                x7 = _mm_max_epu8(x4, x7);
                x7 = _mm_max_epu8(x5, x7);
                x7 = _mm_max_epu8(x6, x7);

                let m = _mm_movemask_epi8(x7);

                if m == 0 {
                    ii = ii.offset(128);
                    continue;
                } else {
                    r0_ = x0;
                    r1_ = x1;
                    r2_ = x2;
                    r3_ = x3;
                    r4_ = x4;
                    r5_ = x5;
                    r6_ = x6;
                    found = true;
                    break;
                 }
            }

            i = ii;

            if found {
                let j = i;
                let s = j as usize - start as usize;

                let z = _mm_movemask_epi8(r0_);
                if unlikely(z != 0) {
                    return Some(0 + s + cttz_nonzero(z) as usize);
                }
                let z = _mm_movemask_epi8(r1_);
                if unlikely(z != 0) {
                    return Some(16 + s + cttz_nonzero(z) as usize);
                }
                let z = _mm_movemask_epi8(r2_);
                if unlikely(z != 0) {
                    return Some(32 + s + cttz_nonzero(z) as usize);
                }
                let z = _mm_movemask_epi8(r3_);
                if unlikely(z != 0) {
                    return Some(48 + s + cttz_nonzero(z) as usize);
                }
                let z = _mm_movemask_epi8(r4_);
                if unlikely(z != 0) {
                    return Some(64 + s + cttz_nonzero(z) as usize);
                }
                let z = _mm_movemask_epi8(r5_);
                if unlikely(z != 0) {
                    return Some(80 + s + cttz_nonzero(z) as usize);
                }
                let z = _mm_movemask_epi8(r6_);
                if unlikely(z != 0) {
                    return Some(96 + s + cttz_nonzero(z) as usize);
                }

                let i2 = i;
                let j = ::std::ptr::read_volatile(&i2);
                let x7 = _mm_load_si128(j.offset(112) as *const __m128i);
                let r7 = _mm_cmpeq_epi8(x7, q);
                let z = _mm_movemask_epi8(r7);
                debug_assert!(z != 0);
                return Some(112 + s + cttz_nonzero(z) as usize);
            }
        }

        loop {
            if i.offset(32) <= end {
                let o = 0;
                let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
                let r = _mm_cmpeq_epi8(x, q);
                let z = _mm_movemask_epi8(r);
                if unlikely(z != 0) {
                    return Some(i.offset(o) as usize - start as usize + cttz_nonzero(z) as usize);
                }

                let o = 16;
                let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
                let r = _mm_cmpeq_epi8(x, q);
                let z = _mm_movemask_epi8(r);
                if unlikely(z != 0) {
                    return Some(i.offset(o) as usize - start as usize + cttz_nonzero(z) as usize);
                }

                i = i.offset(32);
            } else {
                if i.offset(16) <= end {
                    let o = 0;
                    let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
                    let r = _mm_cmpeq_epi8(x, q);
                    let z = _mm_movemask_epi8(r);
                    if unlikely(z != 0) {
                        return Some(i.offset(o) as usize - start as usize + cttz_nonzero(z) as usize);
                    }

                    i = i.offset(16);
                }

                break;
            }
        }

        // TODO: do this without the unaligned load
        /*debug_assert!((end as usize - i as usize) < 16);
        if i < end {
            debug_assert!(haystack.len() >= 16);
            i = end.offset(-16);
            let x = _mm_lddqu_si128(i as *const __m128i);
            let r = _mm_cmpeq_epi8(x, q);
            let z = _mm_movemask_epi8(r);
            if unlikely(z != 0) {
                return Some(i.offset(0) as usize - start as usize + cttz_nonzero(z) as usize);
            }
        }*/

        // Just like with the prologue, this search will scan past the end of
        // the buffer (but not a cacheline or page boundary), then mask out the
        // garbage results.
        if i < end {
            let o = 0;
            let x = _mm_load_si128(i.offset(o) as usize as *const __m128i);
            let r = _mm_cmpeq_epi8(x, q);
            let z = _mm_movemask_epi8(r);
            let extra_bytes = o as usize + 16 - end as usize;
            let garbage_mask = {
                let ones = u32::max_value();
                let mask = ones << (16 - extra_bytes);
                let mask = !mask;
                mask as i32
            };
            let z = z & garbage_mask;
            if unlikely(z != 0) {
                return Some(i.offset(o) as usize + cttz_nonzero(z) as usize - start as usize);
            }

            if cfg!(debug) || cfg!(test) {
                i = i.offset(16).offset(-(extra_bytes as isize));
            }
        }

        debug_assert_eq!(i, end);

        None
    }

    pub fn memrchr(needle: u8, haystack: &[u8]) -> Option<usize> {
        ::fallback::memrchr(needle, haystack)
    }
}

#[allow(dead_code)]
//#[cfg(any(test, not(feature = "libc"), all(not(target_os = "linux"),
//          any(target_pointer_width = "32", target_pointer_width = "64"))))]
#[allow(missing_docs)]
pub mod fallback {
    #[cfg(feature = "use_std")]
    use std::cmp;
    #[cfg(not(feature = "use_std"))]
    use core::cmp;

    use super::{
        LO_U64, HI_U64, LO_USIZE, HI_USIZE, USIZE_BYTES,
        contains_zero_byte, repeat_byte,
    };

    pub fn memchr_simple_unsafe(x: u8, text: &[u8]) -> Option<usize> {
        unsafe { memchr_simple(x, text) }
    }

    // TODO: I had this autovectorizing beautifully for a moment,
    // but can't figure out how
    pub unsafe fn memchr_simple(x: u8, text: &[u8]) -> Option<usize> {
        let len = text.len();
        let ptr = text.as_ptr();

        let mut i = 0;

        while i < len {
            if *ptr.offset(i as isize) == x {
                break;
            }
            i += 1;
        }

        if i != len {
            Some(i)
        } else {
            None
        }
    }

    /// Return the first index matching the byte `x` in `text`.
    pub fn memchr(x: u8, text: &[u8]) -> Option<usize> {
        // Scan for a single byte value by reading two `usize` words at a time.
        //
        // Split `text` in three parts
        // - unaligned initial part, before first word aligned address in text
        // - body, scan by 2 words at a time
        // - the last remaining part, < 2 word size
        let len = text.len();
        let ptr = text.as_ptr();

        // search up to an aligned boundary
        let align = (ptr as usize) & (USIZE_BYTES - 1);
        let mut offset;
        if align > 0 {
            offset = cmp::min(USIZE_BYTES - align, len);
            let pos = text[..offset].iter().position(|elt| *elt == x);
            if let Some(index) = pos {
                return Some(index);
            }
        } else {
            offset = 0;
        }

        // search the body of the text
        let repeated_x = repeat_byte(x);

        if len >= 2 * USIZE_BYTES {
            while offset <= len - 2 * USIZE_BYTES {
                debug_assert_eq!((ptr as usize + offset) % USIZE_BYTES, 0);
                unsafe {
                    let u = *(ptr.offset(offset as isize) as *const usize);
                    let v = *(ptr.offset((offset + USIZE_BYTES) as isize) as *const usize);

                    // break if there is a matching byte
                    let zu = contains_zero_byte(u ^ repeated_x);
                    let zv = contains_zero_byte(v ^ repeated_x);
                    if zu || zv {
                        break;
                    }
                }
                offset += USIZE_BYTES * 2;
            }
        }

        // find the byte after the point the body loop stopped
        text[offset..].iter().position(|elt| *elt == x).map(|i| offset + i)
    }

    /// Return the last index matching the byte `x` in `text`.
    pub fn memrchr(x: u8, text: &[u8]) -> Option<usize> {
        // Scan for a single byte value by reading two `usize` words at a time.
        //
        // Split `text` in three parts
        // - unaligned tail, after the last word aligned address in text
        // - body, scan by 2 words at a time
        // - the first remaining bytes, < 2 word size
        let len = text.len();
        let ptr = text.as_ptr();

        // search to an aligned boundary
        let end_align = (ptr as usize + len) & (USIZE_BYTES - 1);
        let mut offset;
        if end_align > 0 {
            offset = if end_align >= len { 0 } else { len - end_align };
            let pos = text[offset..].iter().rposition(|elt| *elt == x);
            if let Some(index) = pos {
                return Some(offset + index);
            }
        } else {
            offset = len;
        }

        // search the body of the text
        let repeated_x = repeat_byte(x);

        while offset >= 2 * USIZE_BYTES {
            debug_assert_eq!((ptr as usize + offset) % USIZE_BYTES, 0);
            unsafe {
                let u = *(ptr.offset(offset as isize - 2 * USIZE_BYTES as isize) as *const usize);
                let v = *(ptr.offset(offset as isize - USIZE_BYTES as isize) as *const usize);

                // break if there is a matching byte
                let zu = contains_zero_byte(u ^ repeated_x);
                let zv = contains_zero_byte(v ^ repeated_x);
                if zu || zv {
                    break;
                }
            }
            offset -= 2 * USIZE_BYTES;
        }

        // find the byte before the point the body loop stopped
        text[..offset].iter().rposition(|elt| *elt == x)
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::v1::*;
    use quickcheck;

    use super::{memchr, memrchr, memchr2, memchr3, Memchr, Memchr2, Memchr3};
    // Use a macro to test both native and fallback impls on all configurations
    macro_rules! memchr_tests {
        ($mod_name:ident, $memchr:path, $memrchr:path) => {
            mod $mod_name {
            use std::prelude::v1::*;
            use quickcheck;
            #[test]
            fn matches_one() {
                assert_eq!(Some(0), $memchr(b'a', b"a"));
            }

            #[test]
            fn matches_begin() {
                assert_eq!(Some(0), $memchr(b'a', b"aaaa"));
            }

            #[test]
            fn matches_end() {
                assert_eq!(Some(4), $memchr(b'z', b"aaaaz"));
            }

            #[test]
            fn matches_nul() {
                assert_eq!(Some(4), $memchr(b'\x00', b"aaaa\x00"));
            }

            #[test]
            fn matches_past_nul() {
                assert_eq!(Some(5), $memchr(b'z', b"aaaa\x00z"));
            }

            #[test]
            fn no_match_empty() {
                assert_eq!(None, $memchr(b'a', b""));
            }

            #[test]
            fn no_match() {
                assert_eq!(None, $memchr(b'a', b"xyz"));
            }

            #[test]
            fn qc_never_fail() {
                fn prop(needle: u8, haystack: Vec<u8>) -> bool {
                    $memchr(needle, &haystack); true
                }
                quickcheck::quickcheck(prop as fn(u8, Vec<u8>) -> bool);
            }

            #[test]
            fn matches_one_reversed() {
                assert_eq!(Some(0), $memrchr(b'a', b"a"));
            }

            #[test]
            fn matches_begin_reversed() {
                assert_eq!(Some(3), $memrchr(b'a', b"aaaa"));
            }

            #[test]
            fn matches_end_reversed() {
                assert_eq!(Some(0), $memrchr(b'z', b"zaaaa"));
            }

            #[test]
            fn matches_nul_reversed() {
                assert_eq!(Some(4), $memrchr(b'\x00', b"aaaa\x00"));
            }

            #[test]
            fn matches_past_nul_reversed() {
                assert_eq!(Some(0), $memrchr(b'z', b"z\x00aaaa"));
            }

            #[test]
            fn no_match_empty_reversed() {
                assert_eq!(None, $memrchr(b'a', b""));
            }

            #[test]
            fn no_match_reversed() {
                assert_eq!(None, $memrchr(b'a', b"xyz"));
            }

            #[test]
            fn qc_never_fail_reversed() {
                fn prop(needle: u8, haystack: Vec<u8>) -> bool {
                    $memrchr(needle, &haystack); true
                }
                quickcheck::quickcheck(prop as fn(u8, Vec<u8>) -> bool);
            }

            #[test]
            fn qc_correct_memchr() {
                fn prop(v: Vec<u8>, offset: u8) -> bool {
                    // test all pointer alignments
                    let uoffset = (offset & 0xF) as usize;
                    let data = if uoffset <= v.len() {
                        &v[uoffset..]
                    } else {
                        &v[..]
                    };
                    for byte in 0..256u32 {
                        let byte = byte as u8;
                        let pos = data.iter().position(|elt| *elt == byte);
                        if $memchr(byte, &data) != pos {
                            return false;
                        }
                    }
                    true
                }
                quickcheck::quickcheck(prop as fn(Vec<u8>, u8) -> bool);
            }

            #[test]
            fn qc_correct_memrchr() {
                fn prop(v: Vec<u8>, offset: u8) -> bool {
                    // test all pointer alignments
                    let uoffset = (offset & 0xF) as usize;
                    let data = if uoffset <= v.len() {
                        &v[uoffset..]
                    } else {
                        &v[..]
                    };
                    for byte in 0..256u32 {
                        let byte = byte as u8;
                        let pos = data.iter().rposition(|elt| *elt == byte);
                        if $memrchr(byte, &data) != pos {
                            return false;
                        }
                    }
                    true
                }
                quickcheck::quickcheck(prop as fn(Vec<u8>, u8) -> bool);
            }
            }
        }
    }

    memchr_tests! { native, ::memchr, ::memrchr }
    memchr_tests! { fallback, ::fallback::memchr, ::fallback::memrchr }
    memchr_tests! { avx2, ::avx2::memchr, ::avx2::memrchr }
    memchr_tests! { sse, ::sse::memchr, ::sse::memrchr }

    #[test]
    fn memchr2_matches_one() {
        assert_eq!(Some(0), memchr2(b'a', b'b', b"a"));
        assert_eq!(Some(0), memchr2(b'a', b'b', b"b"));
        assert_eq!(Some(0), memchr2(b'b', b'a', b"a"));
        assert_eq!(Some(0), memchr2(b'b', b'a', b"b"));
    }

    #[test]
    fn memchr2_matches_begin() {
        assert_eq!(Some(0), memchr2(b'a', b'b', b"aaaa"));
        assert_eq!(Some(0), memchr2(b'a', b'b', b"bbbb"));
    }

    #[test]
    fn memchr2_matches_end() {
        assert_eq!(Some(4), memchr2(b'z', b'y', b"aaaaz"));
        assert_eq!(Some(4), memchr2(b'z', b'y', b"aaaay"));
    }

    #[test]
    fn memchr2_matches_nul() {
        assert_eq!(Some(4), memchr2(b'\x00', b'z', b"aaaa\x00"));
        assert_eq!(Some(4), memchr2(b'z', b'\x00', b"aaaa\x00"));
    }

    #[test]
    fn memchr2_matches_past_nul() {
        assert_eq!(Some(5), memchr2(b'z', b'y', b"aaaa\x00z"));
        assert_eq!(Some(5), memchr2(b'y', b'z', b"aaaa\x00z"));
    }

    #[test]
    fn memchr2_no_match_empty() {
        assert_eq!(None, memchr2(b'a', b'b', b""));
        assert_eq!(None, memchr2(b'b', b'a', b""));
    }

    #[test]
    fn memchr2_no_match() {
        assert_eq!(None, memchr2(b'a', b'b', b"xyz"));
    }

    #[test]
    fn qc_never_fail_memchr2() {
        fn prop(needle1: u8, needle2: u8, haystack: Vec<u8>) -> bool {
            memchr2(needle1, needle2, &haystack);
            true
        }
        quickcheck::quickcheck(prop as fn(u8, u8, Vec<u8>) -> bool);
    }

    #[test]
    fn memchr3_matches_one() {
        assert_eq!(Some(0), memchr3(b'a', b'b', b'c', b"a"));
        assert_eq!(Some(0), memchr3(b'a', b'b', b'c', b"b"));
        assert_eq!(Some(0), memchr3(b'a', b'b', b'c', b"c"));
    }

    #[test]
    fn memchr3_matches_begin() {
        assert_eq!(Some(0), memchr3(b'a', b'b', b'c', b"aaaa"));
        assert_eq!(Some(0), memchr3(b'a', b'b', b'c', b"bbbb"));
        assert_eq!(Some(0), memchr3(b'a', b'b', b'c', b"cccc"));
    }

    #[test]
    fn memchr3_matches_end() {
        assert_eq!(Some(4), memchr3(b'z', b'y', b'x', b"aaaaz"));
        assert_eq!(Some(4), memchr3(b'z', b'y', b'x', b"aaaay"));
        assert_eq!(Some(4), memchr3(b'z', b'y', b'x', b"aaaax"));
    }

    #[test]
    fn memchr3_matches_nul() {
        assert_eq!(Some(4), memchr3(b'\x00', b'z', b'y', b"aaaa\x00"));
        assert_eq!(Some(4), memchr3(b'z', b'\x00', b'y', b"aaaa\x00"));
        assert_eq!(Some(4), memchr3(b'z', b'y', b'\x00', b"aaaa\x00"));
    }

    #[test]
    fn memchr3_matches_past_nul() {
        assert_eq!(Some(5), memchr3(b'z', b'y', b'x', b"aaaa\x00z"));
        assert_eq!(Some(5), memchr3(b'y', b'z', b'x', b"aaaa\x00z"));
        assert_eq!(Some(5), memchr3(b'y', b'x', b'z', b"aaaa\x00z"));
    }

    #[test]
    fn memchr3_no_match_empty() {
        assert_eq!(None, memchr3(b'a', b'b', b'c', b""));
        assert_eq!(None, memchr3(b'b', b'a', b'c', b""));
        assert_eq!(None, memchr3(b'c', b'b', b'a', b""));
    }

    #[test]
    fn memchr3_no_match() {
        assert_eq!(None, memchr3(b'a', b'b', b'c', b"xyz"));
    }

    // return an iterator of the 0-based indices of haystack that match the
    // needle
    fn positions1<'a>(needle: u8, haystack: &'a [u8])
        -> Box<DoubleEndedIterator<Item=usize> + 'a>
    {
        Box::new(haystack.iter()
                         .enumerate()
                         .filter(move |&(_, &elt)| elt == needle)
                         .map(|t| t.0))
    }

    fn positions2<'a>(needle1: u8, needle2: u8, haystack: &'a [u8])
        -> Box<DoubleEndedIterator<Item=usize> + 'a>
    {
        Box::new(haystack
            .iter()
            .enumerate()
            .filter(move |&(_, &elt)| elt == needle1 || elt == needle2)
            .map(|t| t.0))
    }

    fn positions3<'a>(
        needle1: u8,
        needle2: u8,
        needle3: u8,
        haystack: &'a [u8],
    ) -> Box<DoubleEndedIterator<Item=usize> + 'a> {
        Box::new(haystack
            .iter()
            .enumerate()
            .filter(move |&(_, &elt)| {
                elt == needle1 || elt == needle2 || elt == needle3
            })
            .map(|t| t.0))
    }

    #[test]
    fn memchr_iter() {
        let haystack = b"aaaabaaaab";
        let mut memchr_iter = Memchr::new(b'b', haystack);
        let first = memchr_iter.next();
        let second = memchr_iter.next();
        let third = memchr_iter.next();

        let mut answer_iter = positions1(b'b', haystack);
        assert_eq!(answer_iter.next(), first);
        assert_eq!(answer_iter.next(), second);
        assert_eq!(answer_iter.next(), third);
    }

    #[test]
    fn memchr2_iter() {
        let haystack = b"axxb";
        let mut memchr_iter = Memchr2::new(b'a', b'b', haystack);
        let first = memchr_iter.next();
        let second = memchr_iter.next();
        let third = memchr_iter.next();

        let mut answer_iter = positions2(b'a', b'b', haystack);
        assert_eq!(answer_iter.next(), first);
        assert_eq!(answer_iter.next(), second);
        assert_eq!(answer_iter.next(), third);
    }

    #[test]
    fn memchr3_iter() {
        let haystack = b"axxbc";
        let mut memchr_iter = Memchr3::new(b'a', b'b', b'c', haystack);
        let first = memchr_iter.next();
        let second = memchr_iter.next();
        let third = memchr_iter.next();
        let fourth = memchr_iter.next();

        let mut answer_iter = positions3(b'a', b'b', b'c', haystack);
        assert_eq!(answer_iter.next(), first);
        assert_eq!(answer_iter.next(), second);
        assert_eq!(answer_iter.next(), third);
        assert_eq!(answer_iter.next(), fourth);
    }

    #[test]
    fn memchr_reverse_iter() {
        let haystack = b"aaaabaaaabaaaab";
        let mut memchr_iter = Memchr::new(b'b', haystack);
        let first = memchr_iter.next();
        let second = memchr_iter.next_back();
        let third = memchr_iter.next();
        let fourth = memchr_iter.next_back();

        let mut answer_iter = positions1(b'b', haystack);
        assert_eq!(answer_iter.next(), first);
        assert_eq!(answer_iter.next_back(), second);
        assert_eq!(answer_iter.next(), third);
        assert_eq!(answer_iter.next_back(), fourth);
    }

    #[test]
    fn memrchr_iter(){
        let haystack = b"aaaabaaaabaaaab";
        let mut memchr_iter = Memchr::new(b'b', haystack);
        let first = memchr_iter.next_back();
        let second = memchr_iter.next_back();
        let third = memchr_iter.next_back();
        let fourth = memchr_iter.next_back();

        let mut answer_iter = positions1(b'b', haystack);
        assert_eq!(answer_iter.next_back(), first);
        assert_eq!(answer_iter.next_back(), second);
        assert_eq!(answer_iter.next_back(), third);
        assert_eq!(answer_iter.next_back(), fourth);

    }

    #[test]
    fn qc_never_fail_memchr3() {
        fn prop(
            needle1: u8,
            needle2: u8,
            needle3: u8,
            haystack: Vec<u8>,
        ) -> bool {
            memchr3(needle1, needle2, needle3, &haystack);
            true
        }
        quickcheck::quickcheck(prop as fn(u8, u8, u8, Vec<u8>) -> bool);
    }

    #[test]
    fn qc_correct_memchr() {
        fn prop(v: Vec<u8>, offset: u8) -> bool {
            // test all pointer alignments
            let uoffset = (offset & 0xF) as usize;
            let data = if uoffset <= v.len() {
                &v[uoffset..]
            } else {
                &v[..]
            };
            for byte in 0..256u32 {
                let byte = byte as u8;
                let pos = data.iter().position(|elt| *elt == byte);
                if memchr(byte, &data) != pos {
                    return false;
                }
            }
            true
        }
        quickcheck::quickcheck(prop as fn(Vec<u8>, u8) -> bool);
    }

    #[test]
    fn qc_correct_memrchr() {
        fn prop(v: Vec<u8>, offset: u8) -> bool {
            // test all pointer alignments
            let uoffset = (offset & 0xF) as usize;
            let data = if uoffset <= v.len() {
                &v[uoffset..]
            } else {
                &v[..]
            };
            for byte in 0..256u32 {
                let byte = byte as u8;
                let pos = data.iter().rposition(|elt| *elt == byte);
                if memrchr(byte, &data) != pos {
                    return false;
                }
            }
            true
        }
        quickcheck::quickcheck(prop as fn(Vec<u8>, u8) -> bool);
    }

    #[test]
    fn qc_correct_memchr2() {
        fn prop(v: Vec<u8>, offset: u8) -> bool {
            // test all pointer alignments
            let uoffset = (offset & 0xF) as usize;
            let data = if uoffset <= v.len() {
                &v[uoffset..]
            } else {
                &v[..]
            };
            for b1 in 0..256u32 {
                for b2 in 0..256u32 {
                    let (b1, b2) = (b1 as u8, b2 as u8);
                    let expected = data
                        .iter()
                        .position(|&b| b == b1 || b == b2);
                    let got = memchr2(b1, b2, &data);
                    if expected != got {
                        return false;
                    }
                }
            }
            true
        }
        quickcheck::quickcheck(prop as fn(Vec<u8>, u8) -> bool);
    }

    // take items from a DEI, taking front for each true and back for each
    // false. Return a vector with the concatenation of the fronts and the
    // reverse of the backs.
    fn double_ended_take<I, J>(mut iter: I, take_side: J) -> Vec<I::Item>
        where I: DoubleEndedIterator,
              J: Iterator<Item=bool>,
    {
        let mut found_front = Vec::new();
        let mut found_back = Vec::new();

        for take_front in take_side {
            if take_front {
                if let Some(pos) = iter.next() {
                    found_front.push(pos);
                } else {
                    break;
                }
            } else {
                if let Some(pos) = iter.next_back() {
                    found_back.push(pos);
                } else {
                    break;
                }
            };
        }

        let mut all_found = found_front;
        all_found.extend(found_back.into_iter().rev());
        all_found
    }


    quickcheck! {
        fn qc_memchr_double_ended_iter(needle: u8, data: Vec<u8>,
                                       take_side: Vec<bool>) -> bool
        {
            // make nonempty
            let mut take_side = take_side;
            if take_side.is_empty() { take_side.push(true) };

            let iter = Memchr::new(needle, &data);
            let all_found = double_ended_take(
                iter, take_side.iter().cycle().cloned());

            all_found.iter().cloned().eq(positions1(needle, &data))
        }

        fn qc_memchr1_iter(data: Vec<u8>) -> bool {
            let needle = 0;
            let answer = positions1(needle, &data);
            answer.eq(Memchr::new(needle, &data))
        }

        fn qc_memchr1_rev_iter(data: Vec<u8>) -> bool {
            let needle = 0;
            let answer = positions1(needle, &data);
            answer.rev().eq(Memchr::new(needle, &data).rev())
        }

        fn qc_memchr2_iter(data: Vec<u8>) -> bool {
            let needle1 = 0;
            let needle2 = 1;
            let answer = positions2(needle1, needle2, &data);
            answer.eq(Memchr2::new(needle1, needle2, &data))
        }

        fn qc_memchr3_iter(data: Vec<u8>) -> bool {
            let needle1 = 0;
            let needle2 = 1;
            let needle3 = 2;
            let answer = positions3(needle1, needle2, needle3, &data);
            answer.eq(Memchr3::new(needle1, needle2, needle3, &data))
        }

        fn qc_memchr1_iter_size_hint(data: Vec<u8>) -> bool {
            // test that the size hint is within reasonable bounds
            let needle = 0;
            let mut iter = Memchr::new(needle, &data);
            let mut real_count = data
                .iter()
                .filter(|&&elt| elt == needle)
                .count();

            while let Some(index) = iter.next() {
                real_count -= 1;
                let (lower, upper) = iter.size_hint();
                assert!(lower <= real_count);
                assert!(upper.unwrap() >= real_count);
                assert!(upper.unwrap() <= data.len() - index);
            }
            true
        }
    }
}
