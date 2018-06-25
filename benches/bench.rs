#![feature(test)]
#![feature(slice_internals)]

extern crate memchr;
extern crate test;

use std::iter;
use std::collections::VecDeque;
use test::black_box;

fn bench_data() -> Vec<u8> { iter::repeat(b'z').take(10000).collect() }

fn bench_data_realbig() -> Vec<u8> { iter::repeat(b'z').take(100000).collect() }

fn bench_data_63() -> Vec<u8> {
    let v: Vec<u8> = iter::repeat(b'z').take(63).collect();
    assert_eq!(v.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_15() -> Vec<u8> {
    let v: Vec<u8> = iter::repeat(b'z').take(15).collect();
    assert_eq!(v.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_15_unaligned() -> VecDeque<u8> {
    let mut v: VecDeque<_> = iter::repeat(b'z').take(16).collect();
    v.pop_front();
    assert_ne!(v.as_slices().0.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_15_unaligned_found() -> VecDeque<u8> {
    let mut v: VecDeque<_> = iter::repeat(b'z').take(15).chain(iter::repeat(b'a').take(1)).collect();
    v.pop_front();
    assert_ne!(v.as_slices().0.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_16_unaligned() -> VecDeque<u8> {
    let mut v: VecDeque<_> = iter::repeat(b'z').take(17).collect();
    v.pop_front();
    assert_ne!(v.as_slices().0.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_16_unaligned_found() -> VecDeque<u8> {
    let mut v: VecDeque<_> = iter::repeat(b'z').take(16).chain(iter::repeat(b'a').take(1)).collect();
    v.pop_front();
    assert_ne!(v.as_slices().0.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_31_overaligned_31_found_31() -> VecDeque<u8> {
    let mut v: VecDeque<_> = iter::repeat(b'z').take(31 + 30).chain(iter::repeat(b'a').take(1)).collect();
    assert_eq!(v.as_slices().0.as_ptr() as usize & (16 - 1), 0);
    for _ in 0..31 {
        v.pop_front();
    }
    assert_ne!(v.as_slices().0.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_16_overaligned_8_found_16() -> VecDeque<u8> {
    let mut v: VecDeque<_> = iter::repeat(b'z').take(16 + 15).chain(iter::repeat(b'a').take(1)).collect();
    assert_eq!(v.as_slices().0.as_ptr() as usize & (32 - 1), 0);
    for _ in 0..16 {
        v.pop_front();
    }
    assert_ne!(v.as_slices().0.as_ptr() as usize & (32 - 1), 0);
    v
}

fn aligned_buffer() -> Vec<u8> {
    let mut v: Vec<u128> = Vec::with_capacity(1024);
    let p = v.as_mut_ptr();
    let len = v.len();
    let cap = v.capacity();
    ::std::mem::forget(v);
    let p = p as *mut u8;
    let cap = cap / ::std::mem::size_of::<u128>();
    let v = unsafe { Vec::from_raw_parts(p, len, cap) };
    v
}

fn bench_data_1_found() -> Vec<u8> {
    let mut v = aligned_buffer();
    v.extend(iter::repeat(b'a').take(1));
    assert_eq!(v.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_128_found_last() -> Vec<u8> {
    let v: Vec<u8> = iter::repeat(b'z').take(127).chain(iter::repeat(b'a').take(1)).collect();
    assert_eq!(v.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_128_found_first() -> Vec<u8> {
    let v: Vec<u8> = iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(127)).collect();
    assert_eq!(v.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_128_found_64() -> Vec<u8> {
    let v: Vec<u8> = iter::repeat(b'z').take(63).chain(iter::repeat(b'a').take(1)).chain(iter::repeat(b'z').take(1)).collect();
    assert_eq!(v.as_ptr() as usize & (16 - 1), 0);
    v
}

fn bench_data_empty() -> Vec<u8> { vec![] }

#[ignore]
#[bench]
fn iterator_memchr(b: &mut test::Bencher) {
    let haystack = bench_data();
    let needle = b'a';
    b.iter(|| {
        assert!(haystack.iter().position(|&b| b == needle).is_none());
    });
    b.bytes = haystack.len() as u64;
}

extern crate core;

#[bench]
fn core_memchr(b: &mut test::Bencher) {
    use core::slice::memchr;
    let haystack = bench_data();
    let needle = b'a';
    b.iter(|| {
        assert!(memchr::memchr(needle, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn optimized_memchr_libc_big(b: &mut test::Bencher) {
    let haystack = bench_data();
    let needle = b'a';
    b.iter(|| {
        assert!(memchr::memchr(needle, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn optimized_memchr_libc_realbig(b: &mut test::Bencher) {
    let haystack = bench_data_realbig();
    let needle = b'a';
    b.iter(|| {
        assert!(memchr::memchr(needle, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn optimized_memchr_libc_63(b: &mut test::Bencher) {
    let haystack = bench_data_63();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn optimized_memchr_libc_15(b: &mut test::Bencher) {
    let haystack = bench_data_15();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn optimized_memchr_libc_15_unaligned(b: &mut test::Bencher) {
    let haystack = bench_data_15_unaligned();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn optimized_memchr_libc_15_unaligned_found(b: &mut test::Bencher) {
    let haystack = bench_data_15_unaligned_found();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack) == Some(14)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn optimized_memchr_libc_16_unaligned(b: &mut test::Bencher) {
    let haystack = bench_data_16_unaligned();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn optimized_memchr_libc_16_unaligned_found(b: &mut test::Bencher) {
    let haystack = bench_data_16_unaligned_found();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack) == Some(15)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn optimized_memchr_31_overaligned_31_found_31(b: &mut test::Bencher) {
    let haystack = bench_data_31_overaligned_31_found_31();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack) == Some(30)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn optimized_memchr_16_overaligned_8_found_16(b: &mut test::Bencher) {
    let haystack = bench_data_16_overaligned_8_found_16();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack) == Some(15)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn optimized_memchr_libc_1_found(b: &mut test::Bencher) {
    let haystack = bench_data_1_found();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack) == Some(0)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn optimized_memchr_libc_128_found_last(b: &mut test::Bencher) {
    let haystack = bench_data_128_found_last();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack) == Some(127)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn optimized_memchr_libc_128_found_first(b: &mut test::Bencher) {
    let haystack = bench_data_128_found_first();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack).is_some()));
        }
    });
    b.bytes = 100;
}

#[bench]
fn optimized_memchr_libc_128_found_64(b: &mut test::Bencher) {
    let haystack = bench_data_128_found_64();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack).is_some()));
        }
    });
    b.bytes = 100;
}

#[bench]
fn optimized_memchr_libc_128_empty(b: &mut test::Bencher) {
    let haystack = bench_data_empty();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn avx2_memchr_big(b: &mut test::Bencher) {
    let haystack = bench_data();
    let needle = b'a';
    b.iter(|| {
        assert!(memchr::avx2::memchr(needle, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn avx2_memchr_realbig(b: &mut test::Bencher) {
    let haystack = bench_data_realbig();
    let needle = b'a';
    b.iter(|| {
        assert!(memchr::avx2::memchr(needle, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn avx2_memchr_63(b: &mut test::Bencher) {
    let haystack = bench_data_63();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn avx2_memchr_15(b: &mut test::Bencher) {
    let haystack = bench_data_15();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn avx2_memchr_15_unaligned(b: &mut test::Bencher) {
    let haystack = bench_data_15_unaligned();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn avx2_memchr_15_unaligned_found(b: &mut test::Bencher) {
    let haystack = bench_data_15_unaligned_found();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack) == Some(14)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn avx2_memchr_16_unaligned(b: &mut test::Bencher) {
    let haystack = bench_data_16_unaligned();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn avx2_memchr_16_unaligned_found(b: &mut test::Bencher) {
    let haystack = bench_data_16_unaligned_found();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack) == Some(15)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn avx2_memchr_31_overaligned_31_found_31(b: &mut test::Bencher) {
    let haystack = bench_data_31_overaligned_31_found_31();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack) == Some(30)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn avx2_memchr_16_overaligned_8_found_16(b: &mut test::Bencher) {
    let haystack = bench_data_16_overaligned_8_found_16();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack) == Some(15)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn avx2_memchr_1_found(b: &mut test::Bencher) {
    let haystack = bench_data_1_found();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack) == Some(0)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn avx2_memchr_128_found_last(b: &mut test::Bencher) {
    let haystack = bench_data_128_found_last();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack) == Some(127)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn avx2_memchr_128_found_first(b: &mut test::Bencher) {
    let haystack = bench_data_128_found_first();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack).is_some()));
        }
    });
    b.bytes = 100;
}

#[bench]
fn avx2_memchr_128_found_64(b: &mut test::Bencher) {
    let haystack = bench_data_128_found_64();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack).is_some()));
        }
    });
    b.bytes = 100;
}

#[bench]
fn avx2_memchr_128_empty(b: &mut test::Bencher) {
    let haystack = bench_data_empty();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::avx2::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn sse_memchr_big(b: &mut test::Bencher) {
    let haystack = bench_data();
    let needle = b'a';
    b.iter(|| {
        assert!(memchr::sse::memchr(needle, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn sse_memchr_realbigbig(b: &mut test::Bencher) {
    let haystack = bench_data_realbig();
    let needle = b'a';
    b.iter(|| {
        assert!(memchr::sse::memchr(needle, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn sse_memchr_63(b: &mut test::Bencher) {
    let haystack = bench_data_63();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn sse_memchr_15(b: &mut test::Bencher) {
    let haystack = bench_data_15();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn sse_memchr_15_unaligned(b: &mut test::Bencher) {
    let haystack = bench_data_15_unaligned();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn sse_memchr_15_unaligned_found(b: &mut test::Bencher) {
    let haystack = bench_data_15_unaligned_found();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack) == Some(14)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn sse_memchr_16_unaligned(b: &mut test::Bencher) {
    let haystack = bench_data_16_unaligned();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn sse_memchr_16_unaligned_found(b: &mut test::Bencher) {
    let haystack = bench_data_16_unaligned_found();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack) == Some(15)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn sse_memchr_31_overaligned_31_found_31(b: &mut test::Bencher) {
    let haystack = bench_data_31_overaligned_31_found_31();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack) == Some(30)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn sse_memchr_16_overaligned_8_found_16(b: &mut test::Bencher) {
    let haystack = bench_data_16_overaligned_8_found_16();
    let (haystack, not_haystack) = haystack.as_slices();
    assert!(not_haystack.is_empty());
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack) == Some(15)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn sse_memchr_1_found(b: &mut test::Bencher) {
    let haystack = bench_data_1_found();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack) == Some(0)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn sse_memchr_128_found_last(b: &mut test::Bencher) {
    let haystack = bench_data_128_found_last();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack) == Some(127)));
        }
    });
    b.bytes = haystack.len() as u64 * 100;
}

#[bench]
fn sse_memchr_128_found_first(b: &mut test::Bencher) {
    let haystack = bench_data_128_found_first();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack).is_some()));
        }
    });
    b.bytes = 100;
}

#[bench]
fn sse_memchr_128_found_64(b: &mut test::Bencher) {
    let haystack = bench_data_128_found_64();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack).is_some()));
        }
    });
    b.bytes = 100;
}

#[bench]
fn sse_memchr_128_empty(b: &mut test::Bencher) {
    let haystack = bench_data_empty();
    let needle = b'a';
    b.iter(|| {
        for _ in 0..100 {
            assert!(black_box(memchr::sse::memchr(needle, &haystack).is_none()));
        }
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn iterator_memrchr(b: &mut test::Bencher) {
    let haystack = bench_data();
    let needle = b'a';
    b.iter(|| {
        assert!(haystack.iter().rposition(|&b| b == needle).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[bench]
fn optimized_memrchr(b: &mut test::Bencher) {
    let haystack = bench_data();
    let needle = b'a';
    b.iter(|| {
        assert!(memchr::memrchr(needle, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[ignore]
#[bench]
fn iterator_memchr2(b: &mut test::Bencher) {
    let haystack = bench_data();
    let (needle1, needle2) = (b'a', b'b');
    b.iter(|| {
        assert!(haystack.iter().position(|&b| {
            b == needle1 || b == needle2
        }).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[ignore]
#[bench]
fn manual_memchr2(b: &mut test::Bencher) {
    fn find_singles(
        sparse: &[bool],
        text: &[u8],
    ) -> Option<(usize, usize)> {
        for (hi, &b) in text.iter().enumerate() {
            if sparse[b as usize] {
                return Some((hi, hi+1));
            }
        }
        None
    }

    let haystack = bench_data();
    let mut sparse = vec![false; 256];
    sparse[b'a' as usize] = true;
    sparse[b'b' as usize] = true;
    b.iter(|| {
        assert!(find_singles(&sparse, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[ignore]
#[bench]
fn optimized_memchr2(b: &mut test::Bencher) {
    let haystack = bench_data();
    let (needle1, needle2) = (b'a', b'b');
    b.iter(|| {
        assert!(memchr::memchr2(needle1, needle2, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[ignore]
#[bench]
fn iterator_memchr3(b: &mut test::Bencher) {
    let haystack = bench_data();
    let (needle1, needle2, needle3) = (b'a', b'b', b'c');
    b.iter(|| {
        assert!(haystack.iter().position(|&b| {
            b == needle1 || b == needle2 || b == needle3
        }).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[ignore]
#[bench]
fn optimized_memchr3(b: &mut test::Bencher) {
    let haystack = bench_data();
    let (needle1, needle2, needle3) = (b'a', b'b', b'c');
    b.iter(|| {
        assert!(memchr::memchr3(
            needle1, needle2, needle3, &haystack).is_none());
    });
    b.bytes = haystack.len() as u64;
}
