#include <cuda_fp16.h>

#define WARP 32
#define MAX_PER_THREAD 8

__device__ __forceinline__ float to_f(float v)  { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float  from_f<float>(float v)  { return v; }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }

// ---- warp 内规约：shuffle，无 shared、无 __syncthreads ----
__device__ __forceinline__ float warpReduceSum(float v) {
#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---- 块内规约：warp shuffle + shared 跨 warp 汇总，结果广播回全体线程 ----
__device__ __forceinline__ float blockReduceSum(float v) {
    __shared__ float partial[WARP];
    __shared__ float bcast;
    const int lane = threadIdx.x & (WARP - 1);
    const int wid  = threadIdx.x / WARP;
    const int nwarp = (blockDim.x + WARP - 1) / WARP;

    v = warpReduceSum(v);
    if (lane == 0) partial[wid] = v;
    __syncthreads();

    v = (threadIdx.x < nwarp) ? partial[threadIdx.x] : 0.0f;
    if (wid == 0) v = warpReduceSum(v);
    if (threadIdx.x == 0) bcast = v;
    __syncthreads();
    return bcast;
}

// ============================================================
// 两遍归约版：语义与 Triton 版逐条对齐
//   s = x + residual;  mean = sum(s)/N;  d = s - mean
//   var = sum(d*d)/N;  out = d * rsqrt(var+eps) * w + b
// s 缓存在寄存器，第二遍不再读显存。
// ============================================================
template <typename T>
__global__ void fused_add_ln_2pass(
    const T* __restrict__ X, const T* __restrict__ R,
    T* __restrict__ OUT,     T* __restrict__ SUM,
    const T* __restrict__ W, const T* __restrict__ B,
    const int stride_row, const int N, const float eps, const int HAS_RESIDUAL)
{
    const long long base = (long long)blockIdx.x * stride_row;
    const T* xr = X + base;
    T* outr = OUT + base;

    float s_loc[MAX_PER_THREAD];
    float sum = 0.0f;

    int i = 0;
    for (int c = threadIdx.x; c < N; c += blockDim.x, ++i) {
        float s = to_f(xr[c]);
        if (HAS_RESIDUAL) {
            s += to_f((R + base)[c]);
            (SUM + base)[c] = from_f<T>(s);   // 残差流交还调用方，避免其重算
        }
        s_loc[i] = s;
        sum += s;
    }
    const int nloc = i;

    const float mean = blockReduceSum(sum) / (float)N;

    float sq = 0.0f;
    for (int j = 0; j < nloc; ++j) { float d = s_loc[j] - mean; sq += d * d; }
    const float var  = blockReduceSum(sq) / (float)N;
    const float rstd = rsqrtf(var + eps);

    i = 0;
    for (int c = threadIdx.x; c < N; c += blockDim.x, ++i)
        outr[c] = from_f<T>((s_loc[i] - mean) * rstd * to_f(W[c]) + to_f(B[c]));
}

// ============================================================
// 单遍归约版：E[s^2]-E[s]^2，只需一次块规约（省一次 __syncthreads 往返），
// 但相消误差更大。留作精度/性能对照。
// ============================================================
template <typename T>
__global__ void fused_add_ln_1pass(
    const T* __restrict__ X, const T* __restrict__ R,
    T* __restrict__ OUT,     T* __restrict__ SUM,
    const T* __restrict__ W, const T* __restrict__ B,
    const int stride_row, const int N, const float eps, const int HAS_RESIDUAL)
{
    const long long base = (long long)blockIdx.x * stride_row;
    const T* xr = X + base;
    T* outr = OUT + base;

    float s_loc[MAX_PER_THREAD];
    float sum = 0.0f, sqs = 0.0f;

    int i = 0;
    for (int c = threadIdx.x; c < N; c += blockDim.x, ++i) {
        float s = to_f(xr[c]);
        if (HAS_RESIDUAL) {
            s += to_f((R + base)[c]);
            (SUM + base)[c] = from_f<T>(s);
        }
        s_loc[i] = s;
        sum += s; sqs += s * s;
    }

    const float mean = blockReduceSum(sum) / (float)N;
    const float var  = blockReduceSum(sqs) / (float)N - mean * mean;
    const float rstd = rsqrtf(var + eps);

    i = 0;
    for (int c = threadIdx.x; c < N; c += blockDim.x, ++i)
        outr[c] = from_f<T>((s_loc[i] - mean) * rstd * to_f(W[c]) + to_f(B[c]));
}
