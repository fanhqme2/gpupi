#include <cuda.h>
#include <cuda_runtime.h>
#include "batch_mul_direct.h"

__global__ void batch_mul_direct_kernel(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L){
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        // Pointers to the start of the current instance
        //uint32_t * a = A + idx * L;
        uint32_t * b = B + idx * L;
        //uint32_t * r = ret + idx * (L * 2);
        uint32_t r[BATCH_MUL_DIRECT_L_MAX * 2]; // local result array in registers, max size is 128 words (512 bytes)
        uint32_t a[BATCH_MUL_DIRECT_L_MAX]; // local copy of a
        for (int i = 0; i < L; i++) {
            a[i] = A[idx * L + i];
        }

        // Initialize result to zero
        for (int i = 0; i < L * 2; i++) {
            r[i] = 0;
        }

        // Schoolbook multiplication
        for (int i = 0; i < L; i++) {
            uint64_t carry = 0;
            uint32_t bi = b[i];
            for (int j = 0; j < L; j++) {
                uint64_t mul = (uint64_t)bi * (uint64_t)a[j] + r[i + j] + carry;
                r[i + j] = (uint32_t)mul;
                carry = mul >> 32;
            }
            r[i + L] += (uint32_t)carry; // add remaining carry
        }
        for (int i = 0; i < L * 2; i++) {
            ret[idx * (L * 2) + i] = r[i];
        }
    }
}

// unrolled version
__global__ void batch_mul_direct_kernel2(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L){
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        // Pointers to the start of the current instance
        //uint32_t * a = A + idx * L;
        uint32_t * b = B + idx * L;
        //uint32_t * r = ret + idx * (L * 2);
        uint32_t r[BATCH_MUL_DIRECT_L_MAX * 2]; // local result array in registers, max size is 128 words (512 bytes)
        uint32_t a[BATCH_MUL_DIRECT_L_MAX]; // local copy of a
        
        for (int i = 0; i < L; i++) {
            a[i] = A[idx * L + i];
        }

        // Initialize result to zero
        for (int i = 0; i < L * 2; i++) {
            r[i] = 0;
        }

        int L4 = L & -4;
        int L8 = L & -8;

        for (int i = 0; i < L8; i += 8){
            uint4 carryv1 = make_uint4(0, 0, 0, 0);
            uint4 carryv2 = make_uint4(0, 0, 0, 0);
            uint4 bv1 = make_uint4(b[i], b[i + 1], b[i + 2], b[i + 3]);
            uint4 bv2 = make_uint4(b[i + 4], b[i + 5], b[i + 6], b[i + 7]);

            for (int j = 0; j < L8; j += 8){
                uint4 av1 = make_uint4(a[j], a[j + 1], a[j + 2], a[j + 3]);
                uint4 av2 = make_uint4(a[j + 4], a[j + 5], a[j + 6], a[j + 7]);
                uint4 r0v = make_uint4(r[i + j], r[i + j + 1], r[i + j + 2], r[i + j + 3]);
                uint4 r1v = make_uint4(r[i + j + 4], r[i + j + 5], r[i + j + 6], r[i + j + 7]);
                uint4 r2v = make_uint4(r[i + j + 8], r[i + j + 9], r[i + j + 10], r[i + j + 11]);
                uint4 r3v = make_uint4(r[i + j + 12], r[i + j + 13], r[i + j + 14], r[i + j + 15]);
                uint64_t mul00 = (uint64_t)bv1.x * (uint64_t)av1.x + r0v.x + carryv1.x;
                r0v.x = (uint32_t)mul00;
                carryv1.x = mul00 >> 32;
                uint64_t mul01 = (uint64_t)bv1.x * (uint64_t)av1.y + r0v.y + carryv1.x;
                r0v.y = (uint32_t)mul01;
                carryv1.x = mul01 >> 32;
                uint64_t mul02 = (uint64_t)bv1.x * (uint64_t)av1.z + r0v.z + carryv1.x;
                r0v.z = (uint32_t)mul02;
                carryv1.x = mul02 >> 32;
                uint64_t mul03 = (uint64_t)bv1.x * (uint64_t)av1.w + r0v.w + carryv1.x;
                r0v.w = (uint32_t)mul03;
                carryv1.x = mul03 >> 32;
                uint64_t mul04 = (uint64_t)bv1.x * (uint64_t)av2.x + r1v.x + carryv1.x;
                r1v.x = (uint32_t)mul04;
                carryv1.x = mul04 >> 32;
                uint64_t mul05 = (uint64_t)bv1.x * (uint64_t)av2.y + r1v.y + carryv1.x;
                r1v.y = (uint32_t)mul05;
                carryv1.x = mul05 >> 32;
                uint64_t mul06 = (uint64_t)bv1.x * (uint64_t)av2.z + r1v.z + carryv1.x;
                r1v.z = (uint32_t)mul06;
                carryv1.x = mul06 >> 32;
                uint64_t mul07 = (uint64_t)bv1.x * (uint64_t)av2.w + r1v.w + carryv1.x;
                r1v.w = (uint32_t)mul07;
                carryv1.x = mul07 >> 32;

                uint64_t mul10 = (uint64_t)bv1.y * (uint64_t)av1.x + r0v.y + carryv1.y;
                r0v.y = (uint32_t)mul10;
                carryv1.y = mul10 >> 32;
                uint64_t mul11 = (uint64_t)bv1.y * (uint64_t)av1.y + r0v.z + carryv1.y;
                r0v.z = (uint32_t)mul11;
                carryv1.y = mul11 >> 32;
                uint64_t mul12 = (uint64_t)bv1.y * (uint64_t)av1.z + r0v.w + carryv1.y;
                r0v.w = (uint32_t)mul12;
                carryv1.y = mul12 >> 32;
                uint64_t mul13 = (uint64_t)bv1.y * (uint64_t)av1.w + r1v.x + carryv1.y;
                r1v.x = (uint32_t)mul13;
                carryv1.y = mul13 >> 32;
                uint64_t mul14 = (uint64_t)bv1.y * (uint64_t)av2.x + r1v.y + carryv1.y;
                r1v.y = (uint32_t)mul14;
                carryv1.y = mul14 >> 32;
                uint64_t mul15 = (uint64_t)bv1.y * (uint64_t)av2.y + r1v.z + carryv1.y;
                r1v.z = (uint32_t)mul15;
                carryv1.y = mul15 >> 32;
                uint64_t mul16 = (uint64_t)bv1.y * (uint64_t)av2.z + r1v.w + carryv1.y;
                r1v.w = (uint32_t)mul16;
                carryv1.y = mul16 >> 32;
                uint64_t mul17 = (uint64_t)bv1.y * (uint64_t)av2.w + r2v.x + carryv1.y;
                r2v.x = (uint32_t)mul17;
                carryv1.y = mul17 >> 32;

                uint64_t mul20 = (uint64_t)bv1.z * (uint64_t)av1.x + r0v.z + carryv1.z;
                r0v.z = (uint32_t)mul20;
                carryv1.z = mul20 >> 32;
                uint64_t mul21 = (uint64_t)bv1.z * (uint64_t)av1.y + r0v.w + carryv1.z;
                r0v.w = (uint32_t)mul21;
                carryv1.z = mul21 >> 32;
                uint64_t mul22 = (uint64_t)bv1.z * (uint64_t)av1.z + r1v.x + carryv1.z;
                r1v.x = (uint32_t)mul22;
                carryv1.z = mul22 >> 32;
                uint64_t mul23 = (uint64_t)bv1.z * (uint64_t)av1.w + r1v.y + carryv1.z;
                r1v.y = (uint32_t)mul23;
                carryv1.z = mul23 >> 32;
                uint64_t mul24 = (uint64_t)bv1.z * (uint64_t)av2.x + r1v.z + carryv1.z;
                r1v.z = (uint32_t)mul24;
                carryv1.z = mul24 >> 32;
                uint64_t mul25 = (uint64_t)bv1.z * (uint64_t)av2.y + r1v.w + carryv1.z;
                r1v.w = (uint32_t)mul25;
                carryv1.z = mul25 >> 32;
                uint64_t mul26 = (uint64_t)bv1.z * (uint64_t)av2.z + r2v.x + carryv1.z;
                r2v.x = (uint32_t)mul26;
                carryv1.z = mul26 >> 32;
                uint64_t mul27 = (uint64_t)bv1.z * (uint64_t)av2.w + r2v.y + carryv1.z;
                r2v.y = (uint32_t)mul27;
                carryv1.z = mul27 >> 32;

                uint64_t mul30 = (uint64_t)bv1.w * (uint64_t)av1.x + r0v.w + carryv1.w;
                r0v.w = (uint32_t)mul30;
                carryv1.w = mul30 >> 32;
                uint64_t mul31 = (uint64_t)bv1.w * (uint64_t)av1.y + r1v.x + carryv1.w;
                r1v.x = (uint32_t)mul31;
                carryv1.w = mul31 >> 32;
                uint64_t mul32 = (uint64_t)bv1.w * (uint64_t)av1.z + r1v.y + carryv1.w;
                r1v.y = (uint32_t)mul32;
                carryv1.w = mul32 >> 32;
                uint64_t mul33 = (uint64_t)bv1.w * (uint64_t)av1.w + r1v.z + carryv1.w;
                r1v.z = (uint32_t)mul33;
                carryv1.w = mul33 >> 32;
                uint64_t mul34 = (uint64_t)bv1.w * (uint64_t)av2.x + r1v.w + carryv1.w;
                r1v.w = (uint32_t)mul34;
                carryv1.w = mul34 >> 32;
                uint64_t mul35 = (uint64_t)bv1.w * (uint64_t)av2.y + r2v.x + carryv1.w;
                r2v.x = (uint32_t)mul35;
                carryv1.w = mul35 >> 32;
                uint64_t mul36 = (uint64_t)bv1.w * (uint64_t)av2.z + r2v.y + carryv1.w;
                r2v.y = (uint32_t)mul36;
                carryv1.w = mul36 >> 32;
                uint64_t mul37 = (uint64_t)bv1.w * (uint64_t)av2.w + r2v.z + carryv1.w;
                r2v.z = (uint32_t)mul37;
                carryv1.w = mul37 >> 32;

                uint64_t mul40 = (uint64_t)bv2.x * (uint64_t)av1.x + r1v.x + carryv2.x;
                r1v.x = (uint32_t)mul40;
                carryv2.x = mul40 >> 32;
                uint64_t mul41 = (uint64_t)bv2.x * (uint64_t)av1.y + r1v.y + carryv2.x;
                r1v.y = (uint32_t)mul41;
                carryv2.x = mul41 >> 32;
                uint64_t mul42 = (uint64_t)bv2.x * (uint64_t)av1.z + r1v.z + carryv2.x;
                r1v.z = (uint32_t)mul42;
                carryv2.x = mul42 >> 32;
                uint64_t mul43 = (uint64_t)bv2.x * (uint64_t)av1.w + r1v.w + carryv2.x;
                r1v.w = (uint32_t)mul43;
                carryv2.x = mul43 >> 32;
                uint64_t mul44 = (uint64_t)bv2.x * (uint64_t)av2.x + r2v.x + carryv2.x;
                r2v.x = (uint32_t)mul44;
                carryv2.x = mul44 >> 32;
                uint64_t mul45 = (uint64_t)bv2.x * (uint64_t)av2.y + r2v.y + carryv2.x;
                r2v.y = (uint32_t)mul45;
                carryv2.x = mul45 >> 32;
                uint64_t mul46 = (uint64_t)bv2.x * (uint64_t)av2.z + r2v.z + carryv2.x;
                r2v.z = (uint32_t)mul46;
                carryv2.x = mul46 >> 32;
                uint64_t mul47 = (uint64_t)bv2.x * (uint64_t)av2.w + r2v.w + carryv2.x;
                r2v.w = (uint32_t)mul47;
                carryv2.x = mul47 >> 32;

                uint64_t mul50 = (uint64_t)bv2.y * (uint64_t)av1.x + r1v.y + carryv2.y;
                r1v.y = (uint32_t)mul50;
                carryv2.y = mul50 >> 32;
                uint64_t mul51 = (uint64_t)bv2.y * (uint64_t)av1.y + r1v.z + carryv2.y;
                r1v.z = (uint32_t)mul51;
                carryv2.y = mul51 >> 32;
                uint64_t mul52 = (uint64_t)bv2.y * (uint64_t)av1.z + r1v.w + carryv2.y;
                r1v.w = (uint32_t)mul52;
                carryv2.y = mul52 >> 32;
                uint64_t mul53 = (uint64_t)bv2.y * (uint64_t)av1.w + r2v.x + carryv2.y;
                r2v.x = (uint32_t)mul53;
                carryv2.y = mul53 >> 32;
                uint64_t mul54 = (uint64_t)bv2.y * (uint64_t)av2.x + r2v.y + carryv2.y;
                r2v.y = (uint32_t)mul54;
                carryv2.y = mul54 >> 32;
                uint64_t mul55 = (uint64_t)bv2.y * (uint64_t)av2.y + r2v.z + carryv2.y;
                r2v.z = (uint32_t)mul55;
                carryv2.y = mul55 >> 32;
                uint64_t mul56 = (uint64_t)bv2.y * (uint64_t)av2.z + r2v.w + carryv2.y;
                r2v.w = (uint32_t)mul56;
                carryv2.y = mul56 >> 32;
                uint64_t mul57 = (uint64_t)bv2.y * (uint64_t)av2.w + r3v.x + carryv2.y;
                r3v.x = (uint32_t)mul57;
                carryv2.y = mul57 >> 32;

                uint64_t mul60 = (uint64_t)bv2.z * (uint64_t)av1.x + r1v.z + carryv2.z;
                r1v.z = (uint32_t)mul60;
                carryv2.z = mul60 >> 32;
                uint64_t mul61 = (uint64_t)bv2.z * (uint64_t)av1.y + r1v.w + carryv2.z;
                r1v.w = (uint32_t)mul61;
                carryv2.z = mul61 >> 32;
                uint64_t mul62 = (uint64_t)bv2.z * (uint64_t)av1.z + r2v.x + carryv2.z;
                r2v.x = (uint32_t)mul62;
                carryv2.z = mul62 >> 32;
                uint64_t mul63 = (uint64_t)bv2.z * (uint64_t)av1.w + r2v.y + carryv2.z;
                r2v.y = (uint32_t)mul63;
                carryv2.z = mul63 >> 32;
                uint64_t mul64 = (uint64_t)bv2.z * (uint64_t)av2.x + r2v.z + carryv2.z;
                r2v.z = (uint32_t)mul64;
                carryv2.z = mul64 >> 32;
                uint64_t mul65 = (uint64_t)bv2.z * (uint64_t)av2.y + r2v.w + carryv2.z;
                r2v.w = (uint32_t)mul65;
                carryv2.z = mul65 >> 32;
                uint64_t mul66 = (uint64_t)bv2.z * (uint64_t)av2.z + r3v.x + carryv2.z;
                r3v.x = (uint32_t)mul66;
                carryv2.z = mul66 >> 32;
                uint64_t mul67 = (uint64_t)bv2.z * (uint64_t)av2.w + r3v.y + carryv2.z;
                r3v.y = (uint32_t)mul67;
                carryv2.z = mul67 >> 32;

                uint64_t mul70 = (uint64_t)bv2.w * (uint64_t)av1.x + r1v.w + carryv2.w;
                r1v.w = (uint32_t)mul70;
                carryv2.w = mul70 >> 32;
                uint64_t mul71 = (uint64_t)bv2.w * (uint64_t)av1.y + r2v.x + carryv2.w;
                r2v.x = (uint32_t)mul71;
                carryv2.w = mul71 >> 32;
                uint64_t mul72 = (uint64_t)bv2.w * (uint64_t)av1.z + r2v.y + carryv2.w;
                r2v.y = (uint32_t)mul72;
                carryv2.w = mul72 >> 32;
                uint64_t mul73 = (uint64_t)bv2.w * (uint64_t)av1.w + r2v.z + carryv2.w;
                r2v.z = (uint32_t)mul73;
                carryv2.w = mul73 >> 32;
                uint64_t mul74 = (uint64_t)bv2.w * (uint64_t)av2.x + r2v.w + carryv2.w;
                r2v.w = (uint32_t)mul74;
                carryv2.w = mul74 >> 32;
                uint64_t mul75 = (uint64_t)bv2.w * (uint64_t)av2.y + r3v.x + carryv2.w;
                r3v.x = (uint32_t)mul75;
                carryv2.w = mul75 >> 32;
                uint64_t mul76 = (uint64_t)bv2.w * (uint64_t)av2.z + r3v.y + carryv2.w;
                r3v.y = (uint32_t)mul76;
                carryv2.w = mul76 >> 32;
                uint64_t mul77 = (uint64_t)bv2.w * (uint64_t)av2.w + r3v.z + carryv2.w;
                r3v.z = (uint32_t)mul77;
                carryv2.w = mul77 >> 32;


                r[i + j] = r0v.x;
                r[i + j + 1] = r0v.y;
                r[i + j + 2] = r0v.z;
                r[i + j + 3] = r0v.w;
                r[i + j + 4] = r1v.x;
                r[i + j + 5] = r1v.y;
                r[i + j + 6] = r1v.z;
                r[i + j + 7] = r1v.w;
                r[i + j + 8] = r2v.x;
                r[i + j + 9] = r2v.y;
                r[i + j + 10] = r2v.z;
                r[i + j + 11] = r2v.w;
                r[i + j + 12] = r3v.x;
                r[i + j + 13] = r3v.y;
                r[i + j + 14] = r3v.z;
                r[i + j + 15] = r3v.w;
            }

            for (int j = L8; j < L4; j += 4){
                uint4 av = make_uint4(a[j], a[j + 1], a[j + 2], a[j + 3]);
                uint4 r0v = make_uint4(r[i + j], r[i + j + 1], r[i + j + 2], r[i + j + 3]);
                uint4 r1v = make_uint4(r[i + j + 4], r[i + j + 5], r[i + j + 6], r[i + j + 7]);
                uint4 r2v = make_uint4(r[i + j + 8], r[i + j + 9], r[i + j + 10], r[i + j + 11]);
                uint64_t mul00 = (uint64_t)bv1.x * (uint64_t)av.x + r0v.x + carryv1.x;
                r0v.x = (uint32_t)mul00;
                carryv1.x = mul00 >> 32;
                uint64_t mul01 = (uint64_t)bv1.x * (uint64_t)av.y + r0v.y + carryv1.x;
                r0v.y = (uint32_t)mul01;
                carryv1.x = mul01 >> 32;
                uint64_t mul02 = (uint64_t)bv1.x * (uint64_t)av.z + r0v.z + carryv1.x;
                r0v.z = (uint32_t)mul02;
                carryv1.x = mul02 >> 32;
                uint64_t mul03 = (uint64_t)bv1.x * (uint64_t)av.w + r0v.w + carryv1.x;
                r0v.w = (uint32_t)mul03;
                carryv1.x = mul03 >> 32;

                uint64_t mul10 = (uint64_t)bv1.y * (uint64_t)av.x + r0v.y + carryv1.y;
                r0v.y = (uint32_t)mul10;
                carryv1.y = mul10 >> 32;
                uint64_t mul11 = (uint64_t)bv1.y * (uint64_t)av.y + r0v.z + carryv1.y;
                r0v.z = (uint32_t)mul11;
                carryv1.y = mul11 >> 32;
                uint64_t mul12 = (uint64_t)bv1.y * (uint64_t)av.z + r0v.w + carryv1.y;
                r0v.w = (uint32_t)mul12;
                carryv1.y = mul12 >> 32;
                uint64_t mul13 = (uint64_t)bv1.y * (uint64_t)av.w + r1v.x + carryv1.y;
                r1v.x = (uint32_t)mul13;
                carryv1.y = mul13 >> 32;

                uint64_t mul20 = (uint64_t)bv1.z * (uint64_t)av.x + r0v.z + carryv1.z;
                r0v.z = (uint32_t)mul20;
                carryv1.z = mul20 >> 32;
                uint64_t mul21 = (uint64_t)bv1.z * (uint64_t)av.y + r0v.w + carryv1.z;
                r0v.w = (uint32_t)mul21;
                carryv1.z = mul21 >> 32;
                uint64_t mul22 = (uint64_t)bv1.z * (uint64_t)av.z + r1v.x + carryv1.z;
                r1v.x = (uint32_t)mul22;
                carryv1.z = mul22 >> 32;
                uint64_t mul23 = (uint64_t)bv1.z * (uint64_t)av.w + r1v.y + carryv1.z;
                r1v.y = (uint32_t)mul23;
                carryv1.z = mul23 >> 32;

                uint64_t mul30 = (uint64_t)bv1.w * (uint64_t)av.x + r0v.w + carryv1.w;
                r0v.w = (uint32_t)mul30;
                carryv1.w = mul30 >> 32;
                uint64_t mul31 = (uint64_t)bv1.w * (uint64_t)av.y + r1v.x + carryv1.w;
                r1v.x = (uint32_t)mul31;
                carryv1.w = mul31 >> 32;
                uint64_t mul32 = (uint64_t)bv1.w * (uint64_t)av.z + r1v.y + carryv1.w;
                r1v.y = (uint32_t)mul32;
                carryv1.w = mul32 >> 32;
                uint64_t mul33 = (uint64_t)bv1.w * (uint64_t)av.w + r1v.z + carryv1.w;
                r1v.z = (uint32_t)mul33;
                carryv1.w = mul33 >> 32;

                uint64_t mul40 = (uint64_t)bv2.x * (uint64_t)av.x + r1v.x + carryv2.x;
                r1v.x = (uint32_t)mul40;
                carryv2.x = mul40 >> 32;
                uint64_t mul41 = (uint64_t)bv2.x * (uint64_t)av.y + r1v.y + carryv2.x;
                r1v.y = (uint32_t)mul41;
                carryv2.x = mul41 >> 32;
                uint64_t mul42 = (uint64_t)bv2.x * (uint64_t)av.z + r1v.z + carryv2.x;
                r1v.z = (uint32_t)mul42;
                carryv2.x = mul42 >> 32;
                uint64_t mul43 = (uint64_t)bv2.x * (uint64_t)av.w + r1v.w + carryv2.x;
                r1v.w = (uint32_t)mul43;
                carryv2.x = mul43 >> 32;

                uint64_t mul50 = (uint64_t)bv2.y * (uint64_t)av.x + r1v.y + carryv2.y;
                r1v.y = (uint32_t)mul50;
                carryv2.y = mul50 >> 32;
                uint64_t mul51 = (uint64_t)bv2.y * (uint64_t)av.y + r1v.z + carryv2.y;
                r1v.z = (uint32_t)mul51;
                carryv2.y = mul51 >> 32;
                uint64_t mul52 = (uint64_t)bv2.y * (uint64_t)av.z + r1v.w + carryv2.y;
                r1v.w = (uint32_t)mul52;
                carryv2.y = mul52 >> 32;
                uint64_t mul53 = (uint64_t)bv2.y * (uint64_t)av.w + r2v.x + carryv2.y;
                r2v.x = (uint32_t)mul53;
                carryv2.y = mul53 >> 32;

                uint64_t mul60 = (uint64_t)bv2.z * (uint64_t)av.x + r1v.z + carryv2.z;
                r1v.z = (uint32_t)mul60;
                carryv2.z = mul60 >> 32;
                uint64_t mul61 = (uint64_t)bv2.z * (uint64_t)av.y + r1v.w + carryv2.z;
                r1v.w = (uint32_t)mul61;
                carryv2.z = mul61 >> 32;
                uint64_t mul62 = (uint64_t)bv2.z * (uint64_t)av.z + r2v.x + carryv2.z;
                r2v.x = (uint32_t)mul62;
                carryv2.z = mul62 >> 32;
                uint64_t mul63 = (uint64_t)bv2.z * (uint64_t)av.w + r2v.y + carryv2.z;
                r2v.y = (uint32_t)mul63;
                carryv2.z = mul63 >> 32;

                uint64_t mul70 = (uint64_t)bv2.w * (uint64_t)av.x + r1v.w + carryv2.w;
                r1v.w = (uint32_t)mul70;
                carryv2.w = mul70 >> 32;
                uint64_t mul71 = (uint64_t)bv2.w * (uint64_t)av.y + r2v.x + carryv2.w;
                r2v.x = (uint32_t)mul71;
                carryv2.w = mul71 >> 32;
                uint64_t mul72 = (uint64_t)bv2.w * (uint64_t)av.z + r2v.y + carryv2.w;
                r2v.y = (uint32_t)mul72;
                carryv2.w = mul72 >> 32;
                uint64_t mul73 = (uint64_t)bv2.w * (uint64_t)av.w + r2v.z + carryv2.w;
                r2v.z = (uint32_t)mul73;
                carryv2.w = mul73 >> 32;


                r[i + j] = r0v.x;
                r[i + j + 1] = r0v.y;
                r[i + j + 2] = r0v.z;
                r[i + j + 3] = r0v.w;
                r[i + j + 4] = r1v.x;
                r[i + j + 5] = r1v.y;
                r[i + j + 6] = r1v.z;
                r[i + j + 7] = r1v.w;
                r[i + j + 8] = r2v.x;
                r[i + j + 9] = r2v.y;
                r[i + j + 10] = r2v.z;
                r[i + j + 11] = r2v.w;
            }
            // Handle remaining j's (when L is not multiple of 4)
            for (int j = L4; j < L; j++){
                uint32_t av = a[j];
                uint4 rv1 = make_uint4(r[i + j], r[i + j + 1], r[i + j + 2], r[i + j + 3]);
                uint4 rv2 = make_uint4(r[i + j + 4], r[i + j + 5], r[i + j + 6], r[i + j + 7]);

                uint64_t mul0 = (uint64_t)bv1.x * (uint64_t)av + rv1.x + carryv1.x;
                rv1.x = (uint32_t)mul0;
                carryv1.x = mul0 >> 32;

                uint64_t mul1 = (uint64_t)bv1.y * (uint64_t)av + rv1.y + carryv1.y;
                rv1.y = (uint32_t)mul1;
                carryv1.y = mul1 >> 32;

                uint64_t mul2 = (uint64_t)bv1.z * (uint64_t)av + rv1.z + carryv1.z;
                rv1.z = (uint32_t)mul2;
                carryv1.z = mul2 >> 32;

                uint64_t mul3 = (uint64_t)bv1.w * (uint64_t)av + rv1.w + carryv1.w;
                rv1.w = (uint32_t)mul3;
                carryv1.w = mul3 >> 32;

                uint64_t mul4 = (uint64_t)bv2.x * (uint64_t)av + rv2.x + carryv2.x;
                rv2.x = (uint32_t)mul4;
                carryv2.x = mul4 >> 32;

                uint64_t mul5 = (uint64_t)bv2.y * (uint64_t)av + rv2.y + carryv2.y;
                rv2.y = (uint32_t)mul5;
                carryv2.y = mul5 >> 32;

                uint64_t mul6 = (uint64_t)bv2.z * (uint64_t)av + rv2.z + carryv2.z;
                rv2.z = (uint32_t)mul6;
                carryv2.z = mul6 >> 32;

                uint64_t mul7 = (uint64_t)bv2.w * (uint64_t)av + rv2.w + carryv2.w;
                rv2.w = (uint32_t)mul7;
                carryv2.w = mul7 >> 32;

                r[i + j] = rv1.x;
                r[i + j + 1] = rv1.y;
                r[i + j + 2] = rv1.z;
                r[i + j + 3] = rv1.w;
                r[i + j + 4] = rv2.x;
                r[i + j + 5] = rv2.y;
                r[i + j + 6] = rv2.z;
                r[i + j + 7] = rv2.w;
            }
            uint64_t carry = (uint64_t)r[i + L] + carryv1.x;
            r[i + L] = (uint32_t)carry;
            carry = (carry >> 32) + r[i + L + 1] + carryv1.y;
            r[i + L + 1] = (uint32_t)carry;
            carry = (carry >> 32) + r[i + L + 2] + carryv1.z;
            r[i + L + 2] = (uint32_t)carry;
            carry = (carry >> 32) + r[i + L + 3] + carryv1.w;
            r[i + L + 3] = (uint32_t)carry;
            carry = (carry >> 32) + r[i + L + 4] + carryv2.x;
            r[i + L + 4] = (uint32_t)carry;
            carry = (carry >> 32) + r[i + L + 5] + carryv2.y;
            r[i + L + 5] = (uint32_t)carry;
            carry = (carry >> 32) + r[i + L + 6] + carryv2.z;
            r[i + L + 6] = (uint32_t)carry;
            carry = (carry >> 32) + r[i + L + 7] + carryv2.w;
            r[i + L + 7] = (uint32_t)carry;
        }


        for (int i = L8; i < L4; i += 4){
            uint4 carryv = make_uint4(0, 0, 0, 0);
            uint4 bv = make_uint4(b[i], b[i + 1], b[i + 2], b[i + 3]);

            for (int j = 0; j < L4; j += 4){
                uint4 av = make_uint4(a[j], a[j + 1], a[j + 2], a[j + 3]);
                uint4 r0v = make_uint4(r[i + j], r[i + j + 1], r[i + j + 2], r[i + j + 3]);
                uint4 r1v = make_uint4(r[i + j + 4], r[i + j + 5], r[i + j + 6], r[i + j + 7]);
                uint64_t mul00 = (uint64_t)bv.x * (uint64_t)av.x + r0v.x + carryv.x;
                r0v.x = (uint32_t)mul00;
                carryv.x = mul00 >> 32;
                uint64_t mul01 = (uint64_t)bv.x * (uint64_t)av.y + r0v.y + carryv.x;
                r0v.y = (uint32_t)mul01;
                carryv.x = mul01 >> 32;
                uint64_t mul02 = (uint64_t)bv.x * (uint64_t)av.z + r0v.z + carryv.x;
                r0v.z = (uint32_t)mul02;
                carryv.x = mul02 >> 32;
                uint64_t mul03 = (uint64_t)bv.x * (uint64_t)av.w + r0v.w + carryv.x;
                r0v.w = (uint32_t)mul03;
                carryv.x = mul03 >> 32;

                uint64_t mul10 = (uint64_t)bv.y * (uint64_t)av.x + r0v.y + carryv.y;
                r0v.y = (uint32_t)mul10;
                carryv.y = mul10 >> 32;
                uint64_t mul11 = (uint64_t)bv.y * (uint64_t)av.y + r0v.z + carryv.y;
                r0v.z = (uint32_t)mul11;
                carryv.y = mul11 >> 32;
                uint64_t mul12 = (uint64_t)bv.y * (uint64_t)av.z + r0v.w + carryv.y;
                r0v.w = (uint32_t)mul12;
                carryv.y = mul12 >> 32;
                uint64_t mul13 = (uint64_t)bv.y * (uint64_t)av.w + r1v.x + carryv.y;
                r1v.x = (uint32_t)mul13;
                carryv.y = mul13 >> 32;

                uint64_t mul20 = (uint64_t)bv.z * (uint64_t)av.x + r0v.z + carryv.z;
                r0v.z = (uint32_t)mul20;
                carryv.z = mul20 >> 32;
                uint64_t mul21 = (uint64_t)bv.z * (uint64_t)av.y + r0v.w + carryv.z;
                r0v.w = (uint32_t)mul21;
                carryv.z = mul21 >> 32;
                uint64_t mul22 = (uint64_t)bv.z * (uint64_t)av.z + r1v.x + carryv.z;
                r1v.x = (uint32_t)mul22;
                carryv.z = mul22 >> 32;
                uint64_t mul23 = (uint64_t)bv.z * (uint64_t)av.w + r1v.y + carryv.z;
                r1v.y = (uint32_t)mul23;
                carryv.z = mul23 >> 32;

                uint64_t mul30 = (uint64_t)bv.w * (uint64_t)av.x + r0v.w + carryv.w;
                r0v.w = (uint32_t)mul30;
                carryv.w = mul30 >> 32;
                uint64_t mul31 = (uint64_t)bv.w * (uint64_t)av.y + r1v.x + carryv.w;
                r1v.x = (uint32_t)mul31;
                carryv.w = mul31 >> 32;
                uint64_t mul32 = (uint64_t)bv.w * (uint64_t)av.z + r1v.y + carryv.w;
                r1v.y = (uint32_t)mul32;
                carryv.w = mul32 >> 32;
                uint64_t mul33 = (uint64_t)bv.w * (uint64_t)av.w + r1v.z + carryv.w;
                r1v.z = (uint32_t)mul33;
                carryv.w = mul33 >> 32;
                r[i + j] = r0v.x;
                r[i + j + 1] = r0v.y;
                r[i + j + 2] = r0v.z;
                r[i + j + 3] = r0v.w;
                r[i + j + 4] = r1v.x;
                r[i + j + 5] = r1v.y;
                r[i + j + 6] = r1v.z;
                r[i + j + 7] = r1v.w;
            }
            // Handle remaining j's (when L is not multiple of 4)
            for (int j = L4; j < L; j++){
                uint32_t av = a[j];
                uint4 bv = make_uint4(b[i], b[i + 1], b[i + 2], b[i + 3]);
                uint4 rv = make_uint4(r[i + j], r[i + j + 1], r[i + j + 2], r[i + j + 3]);

                uint64_t mul0 = (uint64_t)bv.x * (uint64_t)av + rv.x + carryv.x;
                rv.x = (uint32_t)mul0;
                carryv.x = mul0 >> 32;

                uint64_t mul1 = (uint64_t)bv.y * (uint64_t)av + rv.y + carryv.y;
                rv.y = (uint32_t)mul1;
                carryv.y = mul1 >> 32;

                uint64_t mul2 = (uint64_t)bv.z * (uint64_t)av + rv.z + carryv.z;
                rv.z = (uint32_t)mul2;
                carryv.z = mul2 >> 32;

                uint64_t mul3 = (uint64_t)bv.w * (uint64_t)av + rv.w + carryv.w;
                rv.w = (uint32_t)mul3;
                carryv.w = mul3 >> 32;
                r[i + j] = rv.x;
                r[i + j + 1] = rv.y;
                r[i + j + 2] = rv.z;
                r[i + j + 3] = rv.w;
            }
            uint64_t carry = (uint64_t)r[i + L] + carryv.x;
            r[i + L] = (uint32_t)carry;
            carry = (carry >> 32) + r[i + L + 1] + carryv.y;
            r[i + L + 1] = (uint32_t)carry;
            carry = (carry >> 32) + r[i + L + 2] + carryv.z;
            r[i + L + 2] = (uint32_t)carry;
            carry = (carry >> 32) + r[i + L + 3] + carryv.w;
            r[i + L + 3] = (uint32_t)carry;
        }
        for (int i = L4; i < L; i++) {
            uint32_t carry = 0;
            uint32_t bi = b[i];
            for (int j = 0; j < L; j++) {
                uint64_t mul = (uint64_t)bi * (uint64_t)a[j] + r[i + j] + carry;
                r[i + j] = (uint32_t)mul;
                carry = mul >> 32;
            }
            r[i + L] += (uint32_t)carry; // add remaining carry
        }

        for (int i = 0; i < L * 2; i++) {
            ret[idx * (L * 2) + i] = r[i];
        }
    }
}

void batch_mul_direct(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L){
    const int threads_per_block = 16;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    if (num_blocks >= 170 * 16) {
        num_blocks = 170 * 16;
    }
    batch_mul_direct_kernel2<<<num_blocks, threads_per_block>>>(A, B, ret, N, L);
}