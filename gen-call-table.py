#!/usr/bin/env python3

print("type MemchrFn = unsafe fn (u8, &[u8]) -> Option<usize>;")

print("const MEMCHR_AVX2FNS: [MemchrFn; 256] = [")

for i in range(0, 256):
    if i <= 3:
        j = str(i)
    elif i < 16:
        j = "lt16"
    elif i == 16:
        j = "eq16"
    elif i < 32:
        j = "lt32"
    elif i < 64:
        j = "lt64"
    elif i < 256:
        j = "lt256"
    print("memchr_avx2_{},".format(j))

print("];")

print("const MEMRCHR_AVX2FNS: [MemchrFn; 256] = [")

for i in range(0, 256):
    if i <= 3:
        j = str(i)
    elif i < 16:
        j = "lt16"
    elif i == 16:
        j = "eq16"
    elif i < 32:
        j = "lt32"
    elif i < 64:
        j = "lt64"
    elif i < 256:
        j = "lt256"
    j = "full"
    print("memrchr_avx2_{},".format(j))

print("];")
