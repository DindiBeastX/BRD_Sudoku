/*
 * secp256k1_ptx.h — PTX-level 256-bit Montgomery modular multiplication
 *                   for Ada Lovelace (sm_89) and Blackwell consumer (sm_120).
 *
 * This is the most performance-critical function in the entire key-search
 * pipeline.  Every point addition requires 3+ field multiplications, and
 * every hash verification feeds from a point addition.
 *
 * Approach: 256-bit Montgomery multiplication using the CIOS (Coarsely
 * Integrated Operand Scanning) algorithm, implemented with explicit PTX
 * madc.lo.cc / madc.hi.cc chains.  This is what the secp8x32 forks use and
 * delivers the tightest instruction schedule on NVIDIA hardware.
 *
 * The secp256k1 Montgomery constant: R = 2^256 mod p
 * Montgomery inverse: p' s.t. p*p' ≡ -1 (mod 2^32) → p' = 0xD2253531
 *
 * Usage:
 *   fe256 r, a, b;
 *   mont_mul(r, a, b);   // r = (a * b * R^{-1}) mod p
 *
 * To convert from/to Montgomery form:
 *   mont_mul(a_mont, a, R2);       // R2 = R^2 mod p (constant below)
 *   mont_mul(a_normal, a_mont, 1); // back to normal
 */

#pragma once
#include <cstdint>

// Montgomery constant R^2 mod p (precomputed)
__constant__ uint32_t MONT_R2[8] = {
    0x000E90A1u, 0x000007A2u, 0x00000001u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u
};

// p' = -p^{-1} mod 2^32 for secp256k1
#define MONT_PINV 0xD2253531u

// ─────────────────────────────────────────────────────────────────────────────
// 8-limb (256-bit) × 8-limb → 16-limb product via PTX madc chains
//
// The outer loop is over i (8 iterations); inner loop is over j (8 iters).
// Each outer iteration also performs one Montgomery reduction step (adds
// m * p to cancel the lowest limb), keeping the accumulator width bounded.
//
// The resulting 9 upper limbs are then compared to p for final reduction.
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void mont_mul(
    uint32_t       r[8],
    const uint32_t a[8],
    const uint32_t b[8])
{
    // 9-limb accumulator (256 bits + 1 overflow bit)
    uint32_t t[9] = {};
    uint32_t c;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        // ── Step 1: accumulate a[i] * b[j] into t ─────────────────────────
        asm volatile(
            "mad.lo.cc.u32  %0, %9, %10, %0;  \n"  // t[0] += a[i]*b[0] lo
            "madc.hi.cc.u32 %1, %9, %10, %1;  \n"  // t[1] += a[i]*b[0] hi
            "madc.lo.cc.u32 %1, %9, %11, %1;  \n"  // t[1] += a[i]*b[1] lo
            "madc.hi.cc.u32 %2, %9, %11, %2;  \n"
            "madc.lo.cc.u32 %2, %9, %12, %2;  \n"
            "madc.hi.cc.u32 %3, %9, %12, %3;  \n"
            "madc.lo.cc.u32 %3, %9, %13, %3;  \n"
            "madc.hi.cc.u32 %4, %9, %13, %4;  \n"
            "madc.lo.cc.u32 %4, %9, %14, %4;  \n"
            "madc.hi.cc.u32 %5, %9, %14, %5;  \n"
            "madc.lo.cc.u32 %5, %9, %15, %5;  \n"
            "madc.hi.cc.u32 %6, %9, %15, %6;  \n"
            "madc.lo.cc.u32 %6, %9, %16, %6;  \n"
            "madc.hi.cc.u32 %7, %9, %16, %7;  \n"
            "madc.lo.cc.u32 %7, %9, %17, %7;  \n"
            "madc.hi.u32    %8, %9, %17, %8;  \n"
            : "+r"(t[0]),"+r"(t[1]),"+r"(t[2]),"+r"(t[3]),
              "+r"(t[4]),"+r"(t[5]),"+r"(t[6]),"+r"(t[7]),"+r"(t[8])
            : "r"(a[i]),
              "r"(b[0]),"r"(b[1]),"r"(b[2]),"r"(b[3]),
              "r"(b[4]),"r"(b[5]),"r"(b[6]),"r"(b[7])
        );

        // ── Step 2: Montgomery reduction of lowest limb ────────────────────
        // m = t[0] * p'  mod 2^32
        uint32_t m;
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(m) : "r"(t[0]), "r"(MONT_PINV));

        // t += m * p  (m * p cancels t[0])
        // secp256k1 p = [0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF×6]
        asm volatile(
            "mad.lo.cc.u32  %0, %9, %10, %0;  \n"
            "madc.hi.cc.u32 %1, %9, %10, %1;  \n"
            "madc.lo.cc.u32 %1, %9, %11, %1;  \n"
            "madc.hi.cc.u32 %2, %9, %11, %2;  \n"
            "madc.lo.cc.u32 %2, %9, %12, %2;  \n"
            "madc.hi.cc.u32 %3, %9, %12, %3;  \n"
            "madc.lo.cc.u32 %3, %9, %12, %3;  \n"
            "madc.hi.cc.u32 %4, %9, %12, %4;  \n"
            "madc.lo.cc.u32 %4, %9, %12, %4;  \n"
            "madc.hi.cc.u32 %5, %9, %12, %5;  \n"
            "madc.lo.cc.u32 %5, %9, %12, %5;  \n"
            "madc.hi.cc.u32 %6, %9, %12, %6;  \n"
            "madc.lo.cc.u32 %6, %9, %12, %6;  \n"
            "madc.hi.cc.u32 %7, %9, %12, %7;  \n"
            "madc.lo.cc.u32 %7, %9, %12, %7;  \n"
            "madc.hi.u32    %8, %9, %12, %8;  \n"
            : "+r"(t[0]),"+r"(t[1]),"+r"(t[2]),"+r"(t[3]),
              "+r"(t[4]),"+r"(t[5]),"+r"(t[6]),"+r"(t[7]),"+r"(t[8])
            : "r"(m),
              "r"(0xFFFFFC2Fu),  // p[0]
              "r"(0xFFFFFFFEu),  // p[1]
              "r"(0xFFFFFFFFu)   // p[2..7]
        );

        // Shift accumulator right by one limb (discard t[0] which is now 0)
        t[0]=t[1]; t[1]=t[2]; t[2]=t[3]; t[3]=t[4];
        t[4]=t[5]; t[5]=t[6]; t[6]=t[7]; t[7]=t[8];
        t[8]=0;
    }

    // Final conditional subtraction: if t >= p, subtract p
    // (t fits in 8 limbs after 8 reduction steps)
    // Compare t vs p
    uint32_t borrow = 0;
    uint32_t tmp[8];
    asm(
        "sub.cc.u32  %0, %8,  0xFFFFFC2F; \n"
        "subc.cc.u32 %1, %9,  0xFFFFFFFE; \n"
        "subc.cc.u32 %2, %10, 0xFFFFFFFF; \n"
        "subc.cc.u32 %3, %11, 0xFFFFFFFF; \n"
        "subc.cc.u32 %4, %12, 0xFFFFFFFF; \n"
        "subc.cc.u32 %5, %13, 0xFFFFFFFF; \n"
        "subc.cc.u32 %6, %14, 0xFFFFFFFF; \n"
        "subc.cc.u32 %7, %15, 0xFFFFFFFF; \n"
        "subc.u32    %16, 0, 0;           \n"
        : "=r"(tmp[0]),"=r"(tmp[1]),"=r"(tmp[2]),"=r"(tmp[3]),
          "=r"(tmp[4]),"=r"(tmp[5]),"=r"(tmp[6]),"=r"(tmp[7]),"=r"(borrow)
        : "r"(t[0]),"r"(t[1]),"r"(t[2]),"r"(t[3]),
          "r"(t[4]),"r"(t[5]),"r"(t[6]),"r"(t[7])
    );

    // Select tmp (subtracted) if no borrow, else keep t
    #pragma unroll
    for (int i = 0; i < 8; i++)
        r[i] = borrow ? t[i] : tmp[i];
}

// Convenience: modular square via Montgomery multiply
__device__ __forceinline__ void mont_sqr(uint32_t r[8], const uint32_t a[8]) {
    mont_mul(r, a, a);
}
