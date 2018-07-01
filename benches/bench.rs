#![feature(test)]
#![feature(slice_internals)]

extern crate memchr;
extern crate test;

macro_rules! memchr_benches {
    ($mod_name:ident, $memchr:path) => {
        mod $mod_name {
            use ::memchr;
            use ::test;
            use std::iter;
            use test::black_box;

            use std::collections::VecDeque;

            fn aligned_buffer() -> Vec<u8> {
                let mut v: Vec<u64> = Vec::with_capacity(1024);
                let p = v.as_mut_ptr();
                let len = v.len();
                let cap = v.capacity();
                ::std::mem::forget(v);
                let p = p as *mut u8;
                let cap = cap / ::std::mem::size_of::<u64>();
                let v = unsafe { Vec::from_raw_parts(p, len, cap) };
                assert_aligned(&v);
                v
            }

            fn assert_overaligned(v: &[u8], align: usize) {
                // TODO: seems like jemalloc will give us 64-byte aligned buffers
                // but libc malloc gives us 16-byte aligned buffers
                let alignment = 16;
                let align = align % alignment;
                assert_eq!(v.as_ptr() as usize & (alignment - 1), align);
            }

            fn assert_aligned(v: &[u8]) {
                assert_overaligned(&v, 0)
            }

            fn bench_data_000_empty() -> Vec<u8> { vec![] }

            #[bench]
            fn memchr_000_empty(b: &mut test::Bencher) {
                let haystack = bench_data_000_empty();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack).is_none()));
                    }
                });
                b.bytes = haystack.len() as u64;
            }

            fn bench_data_001_found() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1));
                v
            }

            #[bench]
            fn memchr_001_found(b: &mut test::Bencher) {
                let haystack = bench_data_001_found();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_002_found() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(1).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_002_found(b: &mut test::Bencher) {
                let haystack = bench_data_002_found();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(1)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_003_found() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(2).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_003_found(b: &mut test::Bencher) {
                let haystack = bench_data_003_found();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(2)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_004_found() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(3).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_004_found(b: &mut test::Bencher) {
                let haystack = bench_data_004_found();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(3)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_005_found() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(4).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_005_found(b: &mut test::Bencher) {
                let haystack = bench_data_005_found();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(4)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_007_found() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(6).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_007_found(b: &mut test::Bencher) {
                let haystack = bench_data_007_found();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(6)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_008_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(7)));
                v
            }

            #[bench]
            fn memchr_008_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_008_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_008_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(7).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_008_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_008_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(7)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_015_aligned_notfound() -> Vec<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(15));
                v
            }

            #[bench]
            fn memchr_015_aligned_notfound(b: &mut test::Bencher) {
                let haystack = bench_data_015_aligned_notfound();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack).is_none()));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_015_overaligned_1_notfound() -> VecDeque<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(16));
                let mut v = VecDeque::from(v);
                v.pop_front();
                assert_overaligned(v.as_slices().0, 1);
                v
            }

            #[bench]
            fn memchr_015_overaligned_1_notfound(b: &mut test::Bencher) {
                let haystack = bench_data_015_overaligned_1_notfound();
                let (haystack, not_haystack) = haystack.as_slices();
                assert!(not_haystack.is_empty());
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack).is_none()));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_015_overaligned_1_found_14() -> VecDeque<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(15).chain(iter::repeat(b'a').take(1)));
                let mut v = VecDeque::from(v);
                v.pop_front();
                assert_overaligned(&v.as_slices().0, 1);
                v
            }

            #[bench]
            fn memchr_015_overaligned_1_found_14(b: &mut test::Bencher) {
                let haystack = bench_data_015_overaligned_1_found_14();
                let (haystack, not_haystack) = haystack.as_slices();
                assert!(not_haystack.is_empty());
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(14)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_016_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(15)));
                v
            }

            #[bench]
            fn memchr_016_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_016_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_016_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(15).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_016_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_016_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(15)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_016_aligned_found_8() -> Vec<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(8));
                v.extend(iter::repeat(b'a').take(1));
                v.extend(iter::repeat(b'a').take(7));
                v
            }

            #[bench]
            fn memchr_016_aligned_found_8(b: &mut test::Bencher) {
                let haystack = bench_data_016_aligned_found_8();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(8)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_016_overaligned_1_notfound() -> VecDeque<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(17));
                let mut v = VecDeque::from(v);
                v.pop_front();
                assert_overaligned(&v.as_slices().0, 1);
                v
            }

            #[bench]
            fn memchr_016_overaligned_1_notfound(b: &mut test::Bencher) {
                let haystack = bench_data_016_overaligned_1_notfound();
                let (haystack, not_haystack) = haystack.as_slices();
                assert!(not_haystack.is_empty());
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack).is_none()));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_016_overaligned_1_found_15() -> VecDeque<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(16).chain(iter::repeat(b'a').take(1)));
                let mut v = VecDeque::from(v);
                v.pop_front();
                assert_overaligned(&v.as_slices().0, 1);
                v
            }

            #[bench]
            fn memchr_016_overaligned_1_found_15(b: &mut test::Bencher) {
                let haystack = bench_data_016_overaligned_1_found_15();
                let (haystack, not_haystack) = haystack.as_slices();
                assert!(not_haystack.is_empty());
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(15)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_016_overaligned_8_found_15() -> VecDeque<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(8 + 15).chain(iter::repeat(b'a').take(1)));
                let mut v = VecDeque::from(v);
                for _ in 0..8 {
                    v.pop_front();
                }
                assert_overaligned(&v.as_slices().0, 8);
                v
            }

            #[bench]
            fn memchr_016_overaligned_8_found_15(b: &mut test::Bencher) {
                let haystack = bench_data_016_overaligned_8_found_15();
                let (haystack, not_haystack) = haystack.as_slices();
                assert!(not_haystack.is_empty());
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(15)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_017_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(16)));
                v
            }

            #[bench]
            fn memchr_017_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_017_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_017_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(16).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_017_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_017_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(16)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_031_aligned_found_30() -> Vec<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(30).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_031_aligned_found_30(b: &mut test::Bencher) {
                let haystack = bench_data_031_aligned_found_30();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(30)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_031_overaligned_1_found_30() -> VecDeque<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(1 + 30).chain(iter::repeat(b'a').take(1)));
                let mut v = VecDeque::from(v);
                v.pop_front();
                assert_overaligned(&v.as_slices().0, 1);
                v
            }

            #[bench]
            fn memchr_031_overaligned_1_found_30(b: &mut test::Bencher) {
                let haystack = bench_data_031_overaligned_1_found_30();
                let (haystack, not_haystack) = haystack.as_slices();
                assert!(not_haystack.is_empty());
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(30)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_031_overaligned_31_found_30() -> VecDeque<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(31 + 30).chain(iter::repeat(b'a').take(1)));
                let mut v = VecDeque::from(v);
                for _ in 0..31 {
                    v.pop_front();
                }
                assert_overaligned(&v.as_slices().0, 31);
                v
            }

            #[bench]
            fn memchr_031_overaligned_31_found_30(b: &mut test::Bencher) {
                let haystack = bench_data_031_overaligned_31_found_30();
                let (haystack, not_haystack) = haystack.as_slices();
                assert!(not_haystack.is_empty());
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(30)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_032_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(31)));
                v
            }

            #[bench]
            fn memchr_032_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_032_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_032_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(31).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_032_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_032_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(31)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_033_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(32)));
                v
            }

            #[bench]
            fn memchr_033_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_033_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_033_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(32).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_033_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_033_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(32)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_063_aligned_notfound() -> Vec<u8> {
                let mut v: Vec<u8> = aligned_buffer();
                v.extend(iter::repeat(b'z').take(63));
                v
            }

            #[bench]
            fn memchr_063_aligned_notfound(b: &mut test::Bencher) {
                let haystack = bench_data_063_aligned_notfound();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack).is_none()));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_064_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(63)));
                v
            }

            #[bench]
            fn memchr_064_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_064_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_064_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(63).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_064_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_064_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(63)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_065_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(64)));
                v
            }

            #[bench]
            fn memchr_065_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_065_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_065_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(64).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_065_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_065_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(64)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_127_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(126)));
                v
            }

            #[bench]
            fn memchr_127_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_127_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_127_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(126).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_127_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_127_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(126)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_128_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(127)));
                v
            }

            #[bench]
            fn memchr_128_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_128_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = 100;
            }

            fn bench_data_128_found_64() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(64).chain(iter::repeat(b'a').take(1)).chain(iter::repeat(b'z').take(63)));
                v
            }

            #[bench]
            fn memchr_128_found_64(b: &mut test::Bencher) {
                let haystack = bench_data_128_found_64();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(64)));
                    }
                });
                b.bytes = 100;
            }

            fn bench_data_128_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(127).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_128_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_128_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(127)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_255_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(254)));
                v
            }

            #[bench]
            fn memchr_255_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_255_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_255_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(254).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_255_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_255_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(254)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_256_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(255)));
                v
            }

            #[bench]
            fn memchr_256_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_256_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_256_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(255).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_256_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_256_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(255)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_287_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(286)));
                v
            }

            #[bench]
            fn memchr_287_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_287_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_287_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(286).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_287_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_287_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(286)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_288_found_first() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'a').take(1).chain(iter::repeat(b'z').take(287)));
                v
            }

            #[bench]
            fn memchr_288_found_first(b: &mut test::Bencher) {
                let haystack = bench_data_288_found_first();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(0)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_288_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(287).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_288_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_288_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(287)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_320_found_last() -> Vec<u8> {
                let mut v = aligned_buffer();
                v.extend(iter::repeat(b'z').take(319).chain(iter::repeat(b'a').take(1)));
                v
            }

            #[bench]
            fn memchr_320_found_last(b: &mut test::Bencher) {
                let haystack = bench_data_320_found_last();
                let needle = b'a';
                b.iter(|| {
                    for _ in 0..100 {
                        assert!(black_box($memchr(needle, &haystack) == Some(319)));
                    }
                });
                b.bytes = haystack.len() as u64 * 100;
            }

            fn bench_data_big() -> Vec<u8> { iter::repeat(b'z').take(10000).collect() }

            #[bench]
            fn memchr_big(b: &mut test::Bencher) {
                let haystack = bench_data_big();
                let needle = b'a';
                b.iter(|| {
                    assert!(black_box($memchr(needle, &haystack).is_none()));
                });
                b.bytes = haystack.len() as u64;
            }

            fn bench_data_realbig() -> Vec<u8> { iter::repeat(b'z').take(100000).collect() }

            #[bench]
            fn memchr_realbig(b: &mut test::Bencher) {
                let haystack = bench_data_realbig();
                let needle = b'a';
                b.iter(|| {
                    assert!(black_box($memchr(needle, &haystack).is_none()));
                });
                b.bytes = haystack.len() as u64;
            }

            #[bench]
            fn memchr_icache_thrasher(b: &mut test::Bencher) {
                let needle = b'a';
                let haystack_000_empty = bench_data_000_empty();
                let haystack_001_found = bench_data_001_found();
                let haystack_002_found = bench_data_002_found();
                let haystack_003_found = bench_data_003_found();
                let haystack_004_found = bench_data_004_found();
                let haystack_005_found = bench_data_005_found();
                let haystack_007_found = bench_data_007_found();
                let haystack_008_found_first = bench_data_008_found_first();
                let haystack_008_found_last = bench_data_008_found_last();
                let haystack_015_aligned_notfound = bench_data_015_aligned_notfound();
                let haystack_015_overaligned_1_notfound = bench_data_015_overaligned_1_notfound();
                let haystack_015_overaligned_1_notfound = haystack_015_overaligned_1_notfound.as_slices().0;
                let haystack_015_overaligned_1_found_14 = bench_data_015_overaligned_1_found_14();
                let haystack_015_overaligned_1_found_14 = haystack_015_overaligned_1_found_14.as_slices().0;
                let haystack_016_found_first = bench_data_016_found_first();
                let haystack_016_found_last = bench_data_016_found_last();
                let haystack_016_aligned_found_8 = bench_data_016_aligned_found_8();
                let haystack_016_overaligned_1_notfound = bench_data_016_overaligned_1_notfound();
                let haystack_016_overaligned_1_notfound = haystack_016_overaligned_1_notfound.as_slices().0;
                let haystack_016_overaligned_1_found_15 = bench_data_016_overaligned_1_found_15();
                let haystack_016_overaligned_1_found_15 = haystack_016_overaligned_1_found_15.as_slices().0;
                let haystack_016_overaligned_8_found_15 = bench_data_016_overaligned_8_found_15();
                let haystack_016_overaligned_8_found_15 = haystack_016_overaligned_8_found_15.as_slices().0;
                let haystack_017_found_first = bench_data_017_found_first();
                let haystack_017_found_last = bench_data_017_found_last();
                let haystack_031_aligned_found_30 = bench_data_031_aligned_found_30();
                let haystack_031_overaligned_1_found_30 = bench_data_031_overaligned_1_found_30();
                let haystack_031_overaligned_1_found_30 = haystack_031_overaligned_1_found_30.as_slices().0;
                let haystack_031_overaligned_31_found_30 = bench_data_031_overaligned_31_found_30();
                let haystack_031_overaligned_31_found_30 = haystack_031_overaligned_31_found_30.as_slices().0;
                let haystack_032_found_first = bench_data_032_found_first();
                let haystack_032_found_last = bench_data_032_found_last();
                let haystack_033_found_first = bench_data_033_found_first();
                let haystack_033_found_last = bench_data_033_found_last();
                let haystack_063_aligned_notfound = bench_data_063_aligned_notfound();
                let haystack_064_found_first = bench_data_064_found_first();
                let haystack_064_found_last = bench_data_064_found_last();
                let haystack_065_found_first = bench_data_065_found_first();
                let haystack_065_found_last = bench_data_065_found_last();
                let haystack_127_found_first = bench_data_127_found_first();
                let haystack_127_found_last = bench_data_127_found_last();
                let haystack_128_found_first = bench_data_128_found_first();
                let haystack_128_found_64 = bench_data_128_found_64();
                let haystack_128_found_last = bench_data_128_found_last();
                let haystack_255_found_first = bench_data_255_found_first();
                let haystack_255_found_last = bench_data_255_found_last();
                let haystack_256_found_first = bench_data_256_found_first();
                let haystack_256_found_last = bench_data_256_found_last();
                let haystack_287_found_first = bench_data_287_found_first();
                let haystack_287_found_last = bench_data_287_found_last();
                let haystack_288_found_first = bench_data_288_found_first();
                let haystack_288_found_last = bench_data_288_found_last();
                let haystack_320_found_last = bench_data_320_found_last();
                let haystack_big = bench_data_big();
                let haystack_realbig = bench_data_realbig();

                b.iter(|| {
                    assert!(black_box($memchr(needle, &haystack_000_empty).is_none()));
                    assert!(black_box($memchr(needle, &haystack_001_found) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_002_found) == Some(1)));
                    assert!(black_box($memchr(needle, &haystack_003_found) == Some(2)));
                    assert!(black_box($memchr(needle, &haystack_004_found) == Some(3)));
                    assert!(black_box($memchr(needle, &haystack_005_found) == Some(4)));
                    assert!(black_box($memchr(needle, &haystack_007_found) == Some(6)));
                    assert!(black_box($memchr(needle, &haystack_008_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_008_found_last) == Some(7)));
                    assert!(black_box($memchr(needle, &haystack_015_aligned_notfound).is_none()));
                    assert!(black_box($memchr(needle, &haystack_015_overaligned_1_notfound).is_none()));
                    assert!(black_box($memchr(needle, &haystack_015_overaligned_1_found_14) == Some(14)));
                    assert!(black_box($memchr(needle, &haystack_016_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_016_found_last) == Some(15)));
                    assert!(black_box($memchr(needle, &haystack_016_aligned_found_8) == Some(8)));
                    assert!(black_box($memchr(needle, &haystack_016_overaligned_1_notfound).is_none()));
                    assert!(black_box($memchr(needle, &haystack_016_overaligned_1_found_15) == Some(15)));
                    assert!(black_box($memchr(needle, &haystack_016_overaligned_8_found_15) == Some(15)));
                    assert!(black_box($memchr(needle, &haystack_017_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_017_found_last) == Some(16)));
                    assert!(black_box($memchr(needle, &haystack_031_aligned_found_30) == Some(30)));
                    assert!(black_box($memchr(needle, &haystack_031_overaligned_1_found_30) == Some(30)));
                    assert!(black_box($memchr(needle, &haystack_031_overaligned_31_found_30) == Some(30)));
                    assert!(black_box($memchr(needle, &haystack_032_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_032_found_last) == Some(31)));
                    assert!(black_box($memchr(needle, &haystack_033_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_033_found_last) == Some(32)));
                    assert!(black_box($memchr(needle, &haystack_063_aligned_notfound).is_none()));
                    assert!(black_box($memchr(needle, &haystack_064_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_064_found_last) == Some(63)));
                    assert!(black_box($memchr(needle, &haystack_065_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_065_found_last) == Some(64)));
                    assert!(black_box($memchr(needle, &haystack_127_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_127_found_last) == Some(126)));
                    assert!(black_box($memchr(needle, &haystack_128_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_128_found_64) == Some(64)));
                    assert!(black_box($memchr(needle, &haystack_128_found_last) == Some(127)));
                    assert!(black_box($memchr(needle, &haystack_255_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_255_found_last) == Some(254)));
                    assert!(black_box($memchr(needle, &haystack_256_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_256_found_last) == Some(255)));
                    assert!(black_box($memchr(needle, &haystack_287_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_287_found_last) == Some(286)));
                    assert!(black_box($memchr(needle, &haystack_288_found_first) == Some(0)));
                    assert!(black_box($memchr(needle, &haystack_288_found_last) == Some(287)));
                    assert!(black_box($memchr(needle, &haystack_320_found_last) == Some(319)));
                    assert!(black_box($memchr(needle, &haystack_big).is_none()));
                    assert!(black_box($memchr(needle, &haystack_realbig).is_none()));
                });
            }
        }
    }
}

memchr_benches! { libc, memchr::memchr }
memchr_benches! { avx2, memchr::avx2::memchr }
//memchr_benches! { sse, memchr::sse::memchr }

use std::iter;

fn bench_data() -> Vec<u8> { iter::repeat(b'z').take(10000).collect() }

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

#[ignore]
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

#[ignore]
#[bench]
fn iterator_memrchr(b: &mut test::Bencher) {
    let haystack = bench_data();
    let needle = b'a';
    b.iter(|| {
        assert!(haystack.iter().rposition(|&b| b == needle).is_none());
    });
    b.bytes = haystack.len() as u64;
}

#[ignore]
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
