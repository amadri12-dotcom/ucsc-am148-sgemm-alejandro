// K6 stretch goal: vectorized float4 GMEM/SMEM loads on top of K5.
//
// numba.cuda cannot express float4 / reinterpret_cast, so this kernel
// is written in CUDA C++ and launched via cupy.RawKernel. Implementing
// this is OPTIONAL. The reward is TBD — for now, treat it as a learning
// bonus.
//
// To use: implement the body below, then load and launch the kernel
// from a Python script using:
//
//     import cupy as cp, pathlib
//     code = pathlib.Path("kernel6_stretch.cu").read_text()
//     k6 = cp.RawKernel(code, "sgemm_vectorize",
//                       options=("-std=c++17", "--use_fast_math"))
//     k6((grid_x, grid_y), (256,), (dA, dB, dC, M, N, K))
//
// The launch shape matches K5 (256 threads, BM=BN=128, BK=8, TM=TN=8).
// Vectorization requires M, N, K all multiples of 4.
//
// Hints (siboehm K6):
// - Load A as float4, then SCATTER it transposed into SMEM so As is
//   stored as [BK rows x BM cols] — this makes the inner-k step's TM
//   register loads a contiguous 8-float read.
// - Load B as float4 and store linearly.
// - In the inner loop, two float4 reads cover the TM = 8 reg_a entries,
//   and two more cover the TN = 8 reg_b entries.
// - Output stores are also vectorized: two float4 stores per row of the
//   thread's 8x8 result tile.

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__
void sgemm_vectorize(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K) {
    // TODO: implement K6.
}
