// Custom CUDA FlashAttention with block-causal masking for pi07_paligemma.
//
// This implements a fused, memory-efficient attention forward + backward that
// reconstructs the block-causal attention mask *inside the kernel* from a
// compact per-token representation, so the dense (B, S, S) mask is never
// materialized. It is the backend behind `attention_implementation="flash_cuda"`.
//
// Masking convention (matches make_att_2d_masks):
//   attend(query i, key j) == q_valid[i] && k_valid[j] && (k_blk[j] <= q_blk[i])
// where q_blk / k_blk are int32 cumulative block-ids. Cross-attention columns
// use k_blk = INT_MIN (always attended). Because the block-ids are produced by
// torch.cumsum over non-negative {0,1} attention masks, k_blk is non-decreasing
// along the key axis, which the kernel exploits for block-causal early-exit.
//
// Numerics: all reductions (QK^T, softmax, PV, gradients) accumulate in fp32
// regardless of the input dtype (fp32/fp16/bf16). The forward writes a per-row
// log-sum-exp (L) that the backward reuses, FlashAttention-2 style.
//
// Parallelization is "one warp per row" with cooperative shared-memory K/V (fwd,
// dQ) or Q/dO (dK/dV) tiles. GQA/MQA is handled by mapping query head h to kv
// head h / (H / Hkv); the dK/dV kernel loops the whole query-head group for a kv
// head, so grouped gradients accumulate without atomics.
//
// Two forward paths exist:
//   * fp32 inputs    -> the warp-per-row reference kernel below (exact, slow).
//   * fp16/bf16 inputs -> a Tensor Core (WMMA) kernel that runs both matmuls on
//     tensor cores with fp32 accumulation. This is the fast path used in
//     training/inference; it is what makes flash_cuda beat the eager backend.
// The backward uses the warp-per-row kernels for all dtypes (fp32 accumulation).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits.h>
#include <vector>

namespace {

constexpr int WARP = 32;
constexpr int WARPS_PER_BLOCK = 4;   // rows handled per thread block
constexpr int MAX_DPL = 8;           // max fp32 regs per lane => head_dim <= 256

// ---- small dtype helpers -------------------------------------------------
template <typename T> __device__ __forceinline__ float to_f(T x) { return static_cast<float>(x); }
template <typename T> __device__ __forceinline__ T from_f(float x) { return static_cast<T>(x); }

// Map a torch scalar type to its WMMA (Tensor Core) element type + float ctor.
template <typename T> struct WmmaTraits;
template <> struct WmmaTraits<at::Half> {
  using wt = __half;
  __device__ __forceinline__ static wt from_float(float x) { return __float2half(x); }
};
template <> struct WmmaTraits<at::BFloat16> {
  using wt = __nv_bfloat16;
  __device__ __forceinline__ static wt from_float(float x) { return __float2bfloat16(x); }
};

// Full butterfly reduce: every lane ends with the warp-wide sum.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Index helpers (row-major contiguous tensors).
__device__ __forceinline__ long idx_qkv(long b, long s, long h, long d, long S, long H, long D) {
  return ((b * S + s) * H + h) * D + d;
}
__device__ __forceinline__ long idx_lse(long b, long h, long s, long H, long S) {
  return (b * H + h) * S + s;
}

// =========================================================================
// Forward: O = softmax(scale * Q K^T + mask) V, plus L = logsumexp per row.
// One warp per query row. Dynamic smem holds an fp32 K/V tile + block metadata.
// =========================================================================
template <typename scalar_t>
__global__ void flash_fwd_kernel(
    const scalar_t* __restrict__ Q,  // (B, Sq, H, D)
    const scalar_t* __restrict__ K,  // (B, Sk, Hkv, D)
    const scalar_t* __restrict__ V,  // (B, Sk, Hkv, D)
    const int* __restrict__ q_blk,   // (B, Sq)
    const int* __restrict__ k_blk,   // (B, Sk)
    const bool* __restrict__ q_valid,  // (B, Sq)
    const bool* __restrict__ k_valid,  // (B, Sk)
    scalar_t* __restrict__ O,        // (B, Sq, H, D)
    float* __restrict__ L,           // (B, H, Sq)
    int B, int Sq, int Sk, int H, int Hkv, int D, float scale, int BC) {
  extern __shared__ char smem_raw[];
  float* sK = reinterpret_cast<float*>(smem_raw);     // BC * D
  float* sV = sK + (long)BC * D;                      // BC * D
  int* sKblk = reinterpret_cast<int*>(sV + (long)BC * D);  // BC
  char* sKvalid = reinterpret_cast<char*>(sKblk + BC);     // BC

  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int dpl = D / WARP;  // elems per lane

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int hk = h / (H / Hkv);
  const long qi = (long)blockIdx.x * WARPS_PER_BLOCK + warp;
  const bool row_active = (qi < Sq);

  // Load this warp's query row + its block-id into registers.
  float q_reg[MAX_DPL];
  int qb = INT_MIN;
  bool qv = false;
  if (row_active) {
    qb = q_blk[b * Sq + qi];
    qv = q_valid[b * Sq + qi];
#pragma unroll
    for (int t = 0; t < MAX_DPL; ++t) {
      if (t < dpl) q_reg[t] = to_f(Q[idx_qkv(b, qi, h, lane + t * WARP, Sq, H, D)]);
    }
  }

  // Block-wide max q_blk over active+valid rows, for uniform early-exit.
  __shared__ int s_block_max_qblk;
  if (threadIdx.x == 0) s_block_max_qblk = INT_MIN;
  __syncthreads();
  if (lane == 0 && row_active && qv) atomicMax(&s_block_max_qblk, qb);
  __syncthreads();
  const int block_max_qblk = s_block_max_qblk;

  float m = -INFINITY, l = 0.f;
  float o_reg[MAX_DPL];
#pragma unroll
  for (int t = 0; t < MAX_DPL; ++t) o_reg[t] = 0.f;

  const int n_tiles = (Sk + BC - 1) / BC;
  for (int kt = 0; kt < n_tiles; ++kt) {
    const long kc0 = (long)kt * BC;
    // Cooperative load of the K/V tile (all threads participate).
    for (int idx = threadIdx.x; idx < BC * D; idx += blockDim.x) {
      const int jj = idx / D, d = idx % D;
      const long kj = kc0 + jj;
      if (kj < Sk) {
        sK[idx] = to_f(K[idx_qkv(b, kj, hk, d, Sk, Hkv, D)]);
        sV[idx] = to_f(V[idx_qkv(b, kj, hk, d, Sk, Hkv, D)]);
      } else {
        sK[idx] = 0.f;
        sV[idx] = 0.f;
      }
    }
    for (int jj = threadIdx.x; jj < BC; jj += blockDim.x) {
      const long kj = kc0 + jj;
      sKblk[jj] = (kj < Sk) ? k_blk[b * Sk + kj] : INT_MAX;
      sKvalid[jj] = (kj < Sk) ? (k_valid[b * Sk + kj] ? 1 : 0) : 0;
    }
    __syncthreads();

    // Uniform block-causal early-exit: k_blk is non-decreasing, so the tile's
    // smallest block-id is its first key. If even the largest query block-id in
    // this block can't reach it, no later tile is reachable either.
    const bool tile_unreachable = (sKblk[0] > block_max_qblk);
    if (!tile_unreachable && row_active && qv) {
      for (int jj = 0; jj < BC; ++jj) {
        const long kj = kc0 + jj;
        if (kj >= Sk) break;
        const int kb = sKblk[jj];
        if (kb > qb) break;            // per-row causal cut (non-decreasing)
        if (!sKvalid[jj]) continue;    // padding / masked column
        float dot = 0.f;
#pragma unroll
        for (int t = 0; t < MAX_DPL; ++t)
          if (t < dpl) dot += q_reg[t] * sK[jj * D + lane + t * WARP];
        const float s = warp_sum(dot) * scale;
        const float m_new = fmaxf(m, s);
        const float corr = __expf(m - m_new);
        const float p = __expf(s - m_new);
        l = l * corr + p;
#pragma unroll
        for (int t = 0; t < MAX_DPL; ++t)
          if (t < dpl) o_reg[t] = o_reg[t] * corr + p * sV[jj * D + lane + t * WARP];
        m = m_new;
      }
    }
    __syncthreads();
    if (tile_unreachable) break;
  }

  if (row_active) {
    const float inv = (l > 0.f) ? (1.f / l) : 0.f;
#pragma unroll
    for (int t = 0; t < MAX_DPL; ++t)
      if (t < dpl) O[idx_qkv(b, qi, h, lane + t * WARP, Sq, H, D)] = from_f<scalar_t>(o_reg[t] * inv);
    if (lane == 0) L[idx_lse(b, h, qi, H, Sq)] = (l > 0.f) ? (m + logf(l)) : 0.f;
  }
}

// =========================================================================
// Backward preprocess: delta[b,h,i] = sum_d O[..d] * dO[..d]  (one warp/row).
// =========================================================================
template <typename scalar_t>
__global__ void flash_bwd_delta_kernel(
    const scalar_t* __restrict__ O, const scalar_t* __restrict__ dO,
    float* __restrict__ delta, int B, int Sq, int H, int D) {
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int dpl = D / WARP;
  const long qi = (long)blockIdx.x * WARPS_PER_BLOCK + warp;
  const int h = blockIdx.y, b = blockIdx.z;
  if (qi >= Sq) return;
  float acc = 0.f;
#pragma unroll
  for (int t = 0; t < MAX_DPL; ++t)
    if (t < dpl) {
      const long off = idx_qkv(b, qi, h, lane + t * WARP, Sq, H, D);
      acc += to_f(O[off]) * to_f(dO[off]);
    }
  acc = warp_sum(acc);
  if (lane == 0) delta[idx_lse(b, h, qi, H, Sq)] = acc;
}

// =========================================================================
// dQ kernel: one warp per query row, loop key tiles. dQ = scale * sum ds * K.
// =========================================================================
template <typename scalar_t>
__global__ void flash_bwd_dq_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, const scalar_t* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ delta,
    const int* __restrict__ q_blk, const int* __restrict__ k_blk,
    const bool* __restrict__ q_valid, const bool* __restrict__ k_valid,
    scalar_t* __restrict__ dQ, int B, int Sq, int Sk, int H, int Hkv, int D,
    float scale, int BC) {
  extern __shared__ char smem_raw[];
  float* sK = reinterpret_cast<float*>(smem_raw);
  float* sV = sK + (long)BC * D;
  int* sKblk = reinterpret_cast<int*>(sV + (long)BC * D);
  char* sKvalid = reinterpret_cast<char*>(sKblk + BC);

  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int dpl = D / WARP;
  const int b = blockIdx.z, h = blockIdx.y;
  const int hk = h / (H / Hkv);
  const long qi = (long)blockIdx.x * WARPS_PER_BLOCK + warp;
  const bool row_active = (qi < Sq);

  float q_reg[MAX_DPL], do_reg[MAX_DPL], dq_reg[MAX_DPL];
#pragma unroll
  for (int t = 0; t < MAX_DPL; ++t) { q_reg[t] = 0.f; do_reg[t] = 0.f; dq_reg[t] = 0.f; }
  int qb = INT_MIN;
  bool qv = false;
  float Li = 0.f, di = 0.f;
  if (row_active) {
    qb = q_blk[b * Sq + qi];
    qv = q_valid[b * Sq + qi];
    Li = L[idx_lse(b, h, qi, H, Sq)];
    di = delta[idx_lse(b, h, qi, H, Sq)];
#pragma unroll
    for (int t = 0; t < MAX_DPL; ++t)
      if (t < dpl) {
        q_reg[t] = to_f(Q[idx_qkv(b, qi, h, lane + t * WARP, Sq, H, D)]);
        do_reg[t] = to_f(dO[idx_qkv(b, qi, h, lane + t * WARP, Sq, H, D)]);
      }
  }

  __shared__ int s_block_max_qblk;
  if (threadIdx.x == 0) s_block_max_qblk = INT_MIN;
  __syncthreads();
  if (lane == 0 && row_active && qv) atomicMax(&s_block_max_qblk, qb);
  __syncthreads();
  const int block_max_qblk = s_block_max_qblk;

  const int n_tiles = (Sk + BC - 1) / BC;
  for (int kt = 0; kt < n_tiles; ++kt) {
    const long kc0 = (long)kt * BC;
    for (int idx = threadIdx.x; idx < BC * D; idx += blockDim.x) {
      const int jj = idx / D, d = idx % D;
      const long kj = kc0 + jj;
      if (kj < Sk) {
        sK[idx] = to_f(K[idx_qkv(b, kj, hk, d, Sk, Hkv, D)]);
        sV[idx] = to_f(V[idx_qkv(b, kj, hk, d, Sk, Hkv, D)]);
      } else { sK[idx] = 0.f; sV[idx] = 0.f; }
    }
    for (int jj = threadIdx.x; jj < BC; jj += blockDim.x) {
      const long kj = kc0 + jj;
      sKblk[jj] = (kj < Sk) ? k_blk[b * Sk + kj] : INT_MAX;
      sKvalid[jj] = (kj < Sk) ? (k_valid[b * Sk + kj] ? 1 : 0) : 0;
    }
    __syncthreads();

    const bool tile_unreachable = (sKblk[0] > block_max_qblk);
    if (!tile_unreachable && row_active && qv) {
      for (int jj = 0; jj < BC; ++jj) {
        const long kj = kc0 + jj;
        if (kj >= Sk) break;
        const int kb = sKblk[jj];
        if (kb > qb) break;
        if (!sKvalid[jj]) continue;
        float dot = 0.f, dp = 0.f;
#pragma unroll
        for (int t = 0; t < MAX_DPL; ++t)
          if (t < dpl) {
            dot += q_reg[t] * sK[jj * D + lane + t * WARP];
            dp += do_reg[t] * sV[jj * D + lane + t * WARP];
          }
        const float s = warp_sum(dot) * scale;
        const float p = __expf(s - Li);
        const float dpr = warp_sum(dp);
        const float ds = p * (dpr - di);
#pragma unroll
        for (int t = 0; t < MAX_DPL; ++t)
          if (t < dpl) dq_reg[t] += ds * sK[jj * D + lane + t * WARP];
      }
    }
    __syncthreads();
    if (tile_unreachable) break;
  }

  if (row_active) {
#pragma unroll
    for (int t = 0; t < MAX_DPL; ++t)
      if (t < dpl) dQ[idx_qkv(b, qi, h, lane + t * WARP, Sq, H, D)] = from_f<scalar_t>(dq_reg[t] * scale);
  }
}

// =========================================================================
// dK/dV kernel: one warp per key row. Loops the whole query-head group (GQA/MQA)
// and all query tiles, so grouped gradients accumulate without atomics.
//   dV = sum_i p_ij dO_i ;  dK = scale * sum_i ds_ij Q_i
// =========================================================================
template <typename scalar_t>
__global__ void flash_bwd_dkv_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, const scalar_t* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ delta,
    const int* __restrict__ q_blk, const int* __restrict__ k_blk,
    const bool* __restrict__ q_valid, const bool* __restrict__ k_valid,
    scalar_t* __restrict__ dK, scalar_t* __restrict__ dV,
    int B, int Sq, int Sk, int H, int Hkv, int D, float scale, int BQ) {
  extern __shared__ char smem_raw[];
  float* sQ = reinterpret_cast<float*>(smem_raw);  // BQ * D
  float* sdO = sQ + (long)BQ * D;                  // BQ * D
  float* sL = sdO + (long)BQ * D;                  // BQ
  float* sdelta = sL + BQ;                         // BQ
  int* sQblk = reinterpret_cast<int*>(sdelta + BQ);  // BQ
  char* sQvalid = reinterpret_cast<char*>(sQblk + BQ);  // BQ

  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int dpl = D / WARP;
  const int b = blockIdx.z, hk = blockIdx.y;
  const long kj = (long)blockIdx.x * WARPS_PER_BLOCK + warp;
  const bool row_active = (kj < Sk);
  const int group = H / Hkv;

  float k_reg[MAX_DPL], v_reg[MAX_DPL], dk_reg[MAX_DPL], dv_reg[MAX_DPL];
#pragma unroll
  for (int t = 0; t < MAX_DPL; ++t) { k_reg[t] = 0.f; v_reg[t] = 0.f; dk_reg[t] = 0.f; dv_reg[t] = 0.f; }
  int kb = INT_MAX;
  bool kv_ok = false;
  if (row_active) {
    kb = k_blk[b * Sk + kj];
    kv_ok = k_valid[b * Sk + kj];
#pragma unroll
    for (int t = 0; t < MAX_DPL; ++t)
      if (t < dpl) {
        k_reg[t] = to_f(K[idx_qkv(b, kj, hk, lane + t * WARP, Sk, Hkv, D)]);
        v_reg[t] = to_f(V[idx_qkv(b, kj, hk, lane + t * WARP, Sk, Hkv, D)]);
      }
  }

  // Block-wide min key block-id for uniform early-skip of query tiles.
  __shared__ int s_block_min_kblk;
  if (threadIdx.x == 0) s_block_min_kblk = INT_MAX;
  __syncthreads();
  if (lane == 0 && row_active && kv_ok) atomicMin(&s_block_min_kblk, kb);
  __syncthreads();
  const int block_min_kblk = s_block_min_kblk;

  const int n_qtiles = (Sq + BQ - 1) / BQ;
  for (int h = hk * group; h < (hk + 1) * group; ++h) {
    for (int qt = 0; qt < n_qtiles; ++qt) {
      const long qc0 = (long)qt * BQ;
      // q_blk is non-decreasing: tile's max block-id is its last in-range query.
      const long last_q = min((long)qc0 + BQ - 1, (long)Sq - 1);
      const int tile_max_qblk = q_blk[b * Sq + last_q];
      if (tile_max_qblk < block_min_kblk) continue;  // uniform skip (no loads)

      for (int idx = threadIdx.x; idx < BQ * D; idx += blockDim.x) {
        const int ii = idx / D, d = idx % D;
        const long qi = qc0 + ii;
        if (qi < Sq) {
          sQ[idx] = to_f(Q[idx_qkv(b, qi, h, d, Sq, H, D)]);
          sdO[idx] = to_f(dO[idx_qkv(b, qi, h, d, Sq, H, D)]);
        } else { sQ[idx] = 0.f; sdO[idx] = 0.f; }
      }
      for (int ii = threadIdx.x; ii < BQ; ii += blockDim.x) {
        const long qi = qc0 + ii;
        if (qi < Sq) {
          sL[ii] = L[idx_lse(b, h, qi, H, Sq)];
          sdelta[ii] = delta[idx_lse(b, h, qi, H, Sq)];
          sQblk[ii] = q_blk[b * Sq + qi];
          sQvalid[ii] = q_valid[b * Sq + qi] ? 1 : 0;
        } else { sL[ii] = 0.f; sdelta[ii] = 0.f; sQblk[ii] = INT_MIN; sQvalid[ii] = 0; }
      }
      __syncthreads();

      if (row_active && kv_ok) {
        for (int ii = 0; ii < BQ; ++ii) {
          const long qi = qc0 + ii;
          if (qi >= Sq) break;
          if (!sQvalid[ii]) continue;
          if (sQblk[ii] < kb) continue;  // query can't reach this key
          float dot = 0.f, dp = 0.f;
#pragma unroll
          for (int t = 0; t < MAX_DPL; ++t)
            if (t < dpl) {
              dot += k_reg[t] * sQ[ii * D + lane + t * WARP];
              dp += v_reg[t] * sdO[ii * D + lane + t * WARP];
            }
          const float s = warp_sum(dot) * scale;
          const float p = __expf(s - sL[ii]);
          const float dpr = warp_sum(dp);
          const float ds = p * (dpr - sdelta[ii]);
#pragma unroll
          for (int t = 0; t < MAX_DPL; ++t)
            if (t < dpl) {
              dv_reg[t] += p * sdO[ii * D + lane + t * WARP];
              dk_reg[t] += ds * sQ[ii * D + lane + t * WARP];
            }
        }
      }
      __syncthreads();
    }
  }

  if (row_active) {
#pragma unroll
    for (int t = 0; t < MAX_DPL; ++t)
      if (t < dpl) {
        dK[idx_qkv(b, kj, hk, lane + t * WARP, Sk, Hkv, D)] = from_f<scalar_t>(dk_reg[t] * scale);
        dV[idx_qkv(b, kj, hk, lane + t * WARP, Sk, Hkv, D)] = from_f<scalar_t>(dv_reg[t]);
      }
  }
}

// =========================================================================
// Tensor Core forward (fp16/bf16). One warp per block handles BR_W=16 query
// rows. Both matmuls (Q K^T and P V) run on WMMA tensor cores with fp32
// accumulation; the running softmax statistics and the O accumulator live in
// shared memory. Block-causal masking is applied to the score tile in shared
// memory, plus the same uniform tile-level early-exit as the reference kernel.
// =========================================================================
constexpr int WM = 16, WN = 16, WK = 16;  // WMMA tile
constexpr int BR_W = 16;                  // query rows per warp slab (== WM)
constexpr int BC_W = 16;                  // key cols per tile (== WN)
constexpr int NW = 3;                      // warps per block (slabs handled together)

// Per-block dynamic shared-memory size for the WMMA forward, in bytes.
__host__ __device__ inline size_t fwd_wmma_smem(int D) {
  // wt (Qs[NW], Ks, Vs, Ps[NW]); float (Os[NW], Ss[NW], m/l/corr[NW]); int + char meta.
  size_t wt_elems = (size_t)NW * BR_W * D + 2 * BC_W * D + (size_t)NW * BR_W * BC_W;
  size_t f_elems = (size_t)NW * BR_W * D + (size_t)NW * BR_W * BC_W + (size_t)NW * 3 * BR_W;
  size_t i_elems = (size_t)NW * BR_W + BC_W;
  size_t c_elems = (size_t)NW * BR_W + BC_W;
  return wt_elems * 2 + f_elems * 4 + i_elems * 4 + c_elems;
}

template <typename scalar_t>
__global__ void flash_fwd_wmma_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K, const scalar_t* __restrict__ V,
    const int* __restrict__ q_blk, const int* __restrict__ k_blk,
    const bool* __restrict__ q_valid, const bool* __restrict__ k_valid,
    scalar_t* __restrict__ O, float* __restrict__ L,
    int B, int Sq, int Sk, int H, int Hkv, int D, float scale) {
  using namespace nvcuda;
  using wt = typename WmmaTraits<scalar_t>::wt;
  auto f2w = WmmaTraits<scalar_t>::from_float;

  // Shared layout: per-warp slabs for Q/O/P/S/stats, block-shared K/V tile.
  extern __shared__ char smem_raw[];
  wt* Qs = reinterpret_cast<wt*>(smem_raw);              // NW * BR_W * D
  wt* Ks = Qs + (long)NW * BR_W * D;                     // BC_W * D
  wt* Vs = Ks + BC_W * D;                                // BC_W * D
  wt* Ps = Vs + BC_W * D;                                // NW * BR_W * BC_W
  float* Os = reinterpret_cast<float*>(Ps + (long)NW * BR_W * BC_W);  // NW * BR_W * D
  float* Ss = Os + (long)NW * BR_W * D;                  // NW * BR_W * BC_W
  float* sm = Ss + (long)NW * BR_W * BC_W;               // NW * BR_W
  float* sl = sm + (long)NW * BR_W;                      // NW * BR_W
  float* scorr = sl + (long)NW * BR_W;                   // NW * BR_W
  int* sQblk = reinterpret_cast<int*>(scorr + (long)NW * BR_W);  // NW * BR_W
  int* sKblk = sQblk + (long)NW * BR_W;                  // BC_W
  char* sQvalid = reinterpret_cast<char*>(sKblk + BC_W);  // NW * BR_W
  char* sKvalid = sQvalid + (long)NW * BR_W;             // BC_W

  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int b = blockIdx.z, h = blockIdx.y;
  const int hk = h / (H / Hkv);
  const long q0 = (long)blockIdx.x * (NW * BR_W) + (long)warp * BR_W;  // this warp's slab

  // Per-warp slab pointers.
  wt* qW = Qs + (long)warp * BR_W * D;
  wt* pW = Ps + (long)warp * BR_W * BC_W;
  float* oW = Os + (long)warp * BR_W * D;
  float* ssW = Ss + (long)warp * BR_W * BC_W;
  float* mW = sm + warp * BR_W;
  float* lW = sl + warp * BR_W;
  float* corrW = scorr + warp * BR_W;
  int* qblkW = sQblk + warp * BR_W;
  char* qvalidW = sQvalid + warp * BR_W;

  // Init this warp's accumulator + load its query slab.
  for (int idx = lane; idx < BR_W * D; idx += WARP) oW[idx] = 0.f;
  for (int i = lane; i < BR_W; i += WARP) { mW[i] = -INFINITY; lW[i] = 0.f; }
  for (int idx = lane; idx < BR_W * D; idx += WARP) {
    const int i = idx / D, d = idx % D;
    const long qi = q0 + i;
    qW[idx] = (qi < Sq) ? f2w(to_f(Q[idx_qkv(b, qi, h, d, Sq, H, D)])) : f2w(0.f);
  }
  for (int i = lane; i < BR_W; i += WARP) {
    const long qi = q0 + i;
    qblkW[i] = (qi < Sq) ? q_blk[b * Sq + qi] : INT_MIN;
    qvalidW[i] = (qi < Sq && q_valid[b * Sq + qi]) ? 1 : 0;
  }
  __syncthreads();

  // Block-wide max query block-id (over all NW slabs) for uniform early-exit.
  int bmax = INT_MIN;
  for (int i = 0; i < NW * BR_W; ++i)
    if (sQvalid[i]) bmax = max(bmax, sQblk[i]);

  const int n_tiles = (Sk + BC_W - 1) / BC_W;
  for (int kt = 0; kt < n_tiles; ++kt) {
    const long kc0 = (long)kt * BC_W;
    // Cooperative K/V tile load by ALL warps (overlaps global-load latency).
    for (int idx = threadIdx.x; idx < BC_W * D; idx += blockDim.x) {
      const int j = idx / D, d = idx % D;
      const long kj = kc0 + j;
      if (kj < Sk) {
        Ks[idx] = f2w(to_f(K[idx_qkv(b, kj, hk, d, Sk, Hkv, D)]));
        Vs[idx] = f2w(to_f(V[idx_qkv(b, kj, hk, d, Sk, Hkv, D)]));
      } else { Ks[idx] = f2w(0.f); Vs[idx] = f2w(0.f); }
    }
    for (int j = threadIdx.x; j < BC_W; j += blockDim.x) {
      const long kj = kc0 + j;
      sKblk[j] = (kj < Sk) ? k_blk[b * Sk + kj] : INT_MAX;
      sKvalid[j] = (kj < Sk && k_valid[b * Sk + kj]) ? 1 : 0;
    }
    __syncthreads();

    const bool unreachable = (sKblk[0] > bmax);
    if (!unreachable) {
      // S = scale * Q K^T  (Tensor Cores) -> ssW (fp32). BC_W == WN -> one n-tile.
      {
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
        wmma::fill_fragment(acc, 0.f);
        for (int kk = 0; kk < D / WK; ++kk) {
          wmma::fragment<wmma::matrix_a, WM, WN, WK, wt, wmma::row_major> af;
          wmma::fragment<wmma::matrix_b, WM, WN, WK, wt, wmma::col_major> bf;
          wmma::load_matrix_sync(af, qW + kk * WK, D);
          wmma::load_matrix_sync(bf, Ks + kk * WK, D);
          wmma::mma_sync(acc, af, bf, acc);
        }
        wmma::store_matrix_sync(ssW, acc, BC_W, wmma::mem_row_major);
      }
      __syncwarp();

      // Mask + online softmax (lane i owns query row i within this slab).
      if (lane < BR_W) {
        const int i = lane;
        if (qvalidW[i]) {
          const int qb = qblkW[i];
          float rowmax = -INFINITY;
          for (int j = 0; j < BC_W; ++j) {
            const long kj = kc0 + j;
            const bool att = (kj < Sk) && sKvalid[j] && (sKblk[j] <= qb);
            const float s = att ? (ssW[i * BC_W + j] * scale) : -INFINITY;
            ssW[i * BC_W + j] = s;
            rowmax = fmaxf(rowmax, s);
          }
          if (rowmax == -INFINITY) {
            corrW[i] = 1.f;  // nothing valid this tile; leave running stats
            for (int j = 0; j < BC_W; ++j) pW[i * BC_W + j] = f2w(0.f);
          } else {
            const float m_old = mW[i];
            const float m_new = fmaxf(m_old, rowmax);
            const float corr = __expf(m_old - m_new);  // m_old=-inf -> 0
            float rowsum = 0.f;
            for (int j = 0; j < BC_W; ++j) {
              const float s = ssW[i * BC_W + j];
              const float p = (s == -INFINITY) ? 0.f : __expf(s - m_new);
              pW[i * BC_W + j] = f2w(p);
              rowsum += p;
            }
            lW[i] = lW[i] * corr + rowsum;
            mW[i] = m_new;
            corrW[i] = corr;
          }
        } else {
          corrW[i] = 1.f;
          for (int j = 0; j < BC_W; ++j) pW[i * BC_W + j] = f2w(0.f);
        }
      }
      __syncwarp();

      // Rescale running O by the softmax correction, then O += P V (Tensor Cores).
      for (int idx = lane; idx < BR_W * D; idx += WARP) oW[idx] *= corrW[idx / D];
      __syncwarp();
      for (int dn = 0; dn < D / WN; ++dn) {
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> acco;
        wmma::load_matrix_sync(acco, oW + dn * WN, D, wmma::mem_row_major);
        wmma::fragment<wmma::matrix_a, WM, WN, WK, wt, wmma::row_major> pf;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, wt, wmma::row_major> vf;
        wmma::load_matrix_sync(pf, pW, BC_W);
        wmma::load_matrix_sync(vf, Vs + dn * WN, D);
        wmma::mma_sync(acco, pf, vf, acco);
        wmma::store_matrix_sync(oW + dn * WN, acco, D, wmma::mem_row_major);
      }
      __syncwarp();
    }
    __syncthreads();  // all warps done with shared K/V before next tile load
    if (unreachable) break;
  }

  if (lane < BR_W) {
    const int i = lane;
    const long qi = q0 + i;
    if (qi < Sq) {
      const float inv = (lW[i] > 0.f) ? (1.f / lW[i]) : 0.f;
      for (int d = 0; d < D; ++d)
        O[idx_qkv(b, qi, h, d, Sq, H, D)] = from_f<scalar_t>(oW[i * D + d] * inv);
      L[idx_lse(b, h, qi, H, Sq)] = (lW[i] > 0.f) ? (mW[i] + logf(lW[i])) : 0.f;
    }
  }
}

// ---- host launchers ------------------------------------------------------

int pick_tile(int D) {
  // Keep the two fp32 smem tiles near 64 KB (well under the 99 KB opt-in cap).
  if (D <= 64) return 128;
  if (D <= 128) return 64;
  return 32;  // D in (128, 256]
}

#define DISPATCH_FLOAT(TYPE, NAME, ...)                                  \
  [&] {                                                                  \
    switch (TYPE) {                                                      \
      case at::kFloat: { using scalar_t = float; return __VA_ARGS__(); } \
      case at::kHalf: { using scalar_t = at::Half; return __VA_ARGS__(); } \
      case at::kBFloat16: { using scalar_t = at::BFloat16; return __VA_ARGS__(); } \
      default: TORCH_CHECK(false, NAME " unsupported dtype: ", TYPE);    \
    }                                                                    \
  }()

void check_inputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be (B,S,H,D)");
  const int D = q.size(3);
  TORCH_CHECK(D % WARP == 0 && D / WARP <= MAX_DPL, "head_dim must be a multiple of 32 and <= 256, got ", D);
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q/k/v dtype mismatch");
}

std::vector<at::Tensor> flash_fwd(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor q_blk, at::Tensor k_blk,
    at::Tensor q_valid, at::Tensor k_valid, double scale) {
  check_inputs(q, k, v);
  const at::cuda::CUDAGuard guard(q.device());
  q = q.contiguous(); k = k.contiguous(); v = v.contiguous();
  q_blk = q_blk.contiguous(); k_blk = k_blk.contiguous();
  q_valid = q_valid.contiguous(); k_valid = k_valid.contiguous();

  const int B = q.size(0), Sq = q.size(1), H = q.size(2), D = q.size(3);
  const int Sk = k.size(1), Hkv = k.size(2);
  TORCH_CHECK(H % Hkv == 0, "num_heads must be divisible by num_kv_heads");

  auto O = at::empty_like(q);
  auto L = at::empty({B, H, Sq}, q.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == at::kFloat) {
    // fp32: exact warp-per-row reference kernel.
    const int BC = pick_tile(D);
    const size_t smem = (size_t)2 * BC * D * sizeof(float) + (size_t)BC * sizeof(int) + (size_t)BC;
    dim3 grid((Sq + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, B);
    dim3 block(WARPS_PER_BLOCK * WARP);
    auto kernel = flash_fwd_kernel<float>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, block, smem, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        q_blk.data_ptr<int>(), k_blk.data_ptr<int>(), q_valid.data_ptr<bool>(),
        k_valid.data_ptr<bool>(), O.data_ptr<float>(), L.data_ptr<float>(),
        B, Sq, Sk, H, Hkv, D, (float)scale, BC);
  } else {
    // fp16/bf16: Tensor Core (WMMA) kernel. NW warps/block share the K/V tile.
    const size_t smem = fwd_wmma_smem(D);
    const int rows_per_block = NW * BR_W;
    dim3 grid((Sq + rows_per_block - 1) / rows_per_block, H, B);
    dim3 block(NW * WARP);
    DISPATCH_FLOAT(q.scalar_type(), "flash_fwd_wmma", [&] {
      if constexpr (!std::is_same_v<scalar_t, float>) {
        auto kernel = flash_fwd_wmma_kernel<scalar_t>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kernel<<<grid, block, smem, stream>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            q_blk.data_ptr<int>(), k_blk.data_ptr<int>(), q_valid.data_ptr<bool>(),
            k_valid.data_ptr<bool>(), O.data_ptr<scalar_t>(), L.data_ptr<float>(),
            B, Sq, Sk, H, Hkv, D, (float)scale);
      }
    });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {O, L};
}

std::vector<at::Tensor> flash_bwd(
    at::Tensor dO, at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor O, at::Tensor L,
    at::Tensor q_blk, at::Tensor k_blk, at::Tensor q_valid, at::Tensor k_valid, double scale) {
  check_inputs(q, k, v);
  const at::cuda::CUDAGuard guard(q.device());
  dO = dO.contiguous(); q = q.contiguous(); k = k.contiguous(); v = v.contiguous();
  O = O.contiguous(); L = L.contiguous();
  q_blk = q_blk.contiguous(); k_blk = k_blk.contiguous();
  q_valid = q_valid.contiguous(); k_valid = k_valid.contiguous();

  const int B = q.size(0), Sq = q.size(1), H = q.size(2), D = q.size(3);
  const int Sk = k.size(1), Hkv = k.size(2);

  auto dQ = at::empty_like(q);
  auto dK = at::empty_like(k);
  auto dV = at::empty_like(v);
  auto delta = at::empty({B, H, Sq}, q.options().dtype(at::kFloat));

  auto stream = at::cuda::getCurrentCUDAStream();

  // delta = rowsum(O * dO)
  {
    dim3 grid((Sq + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, B);
    dim3 block(WARPS_PER_BLOCK * WARP);
    DISPATCH_FLOAT(q.scalar_type(), "flash_bwd_delta", [&] {
      flash_bwd_delta_kernel<scalar_t><<<grid, block, 0, stream>>>(
          O.data_ptr<scalar_t>(), dO.data_ptr<scalar_t>(), delta.data_ptr<float>(), B, Sq, H, D);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // dQ
  {
    const int BC = pick_tile(D);
    const size_t smem = (size_t)2 * BC * D * sizeof(float) + (size_t)BC * sizeof(int) + (size_t)BC;
    dim3 grid((Sq + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, B);
    dim3 block(WARPS_PER_BLOCK * WARP);
    DISPATCH_FLOAT(q.scalar_type(), "flash_bwd_dq", [&] {
      auto kernel = flash_bwd_dq_kernel<scalar_t>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      kernel<<<grid, block, smem, stream>>>(
          q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
          dO.data_ptr<scalar_t>(), L.data_ptr<float>(), delta.data_ptr<float>(),
          q_blk.data_ptr<int>(), k_blk.data_ptr<int>(), q_valid.data_ptr<bool>(),
          k_valid.data_ptr<bool>(), dQ.data_ptr<scalar_t>(), B, Sq, Sk, H, Hkv, D, (float)scale, BC);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // dK, dV
  {
    const int BQ = pick_tile(D);
    const size_t smem = (size_t)2 * BQ * D * sizeof(float) + (size_t)2 * BQ * sizeof(float) +
                        (size_t)BQ * sizeof(int) + (size_t)BQ;
    dim3 grid((Sk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, Hkv, B);
    dim3 block(WARPS_PER_BLOCK * WARP);
    DISPATCH_FLOAT(q.scalar_type(), "flash_bwd_dkv", [&] {
      auto kernel = flash_bwd_dkv_kernel<scalar_t>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      kernel<<<grid, block, smem, stream>>>(
          q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
          dO.data_ptr<scalar_t>(), L.data_ptr<float>(), delta.data_ptr<float>(),
          q_blk.data_ptr<int>(), k_blk.data_ptr<int>(), q_valid.data_ptr<bool>(),
          k_valid.data_ptr<bool>(), dK.data_ptr<scalar_t>(), dV.data_ptr<scalar_t>(),
          B, Sq, Sk, H, Hkv, D, (float)scale, BQ);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return {dQ, dK, dV};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_fwd", &flash_fwd, "Block-causal flash attention forward (CUDA)");
  m.def("flash_bwd", &flash_bwd, "Block-causal flash attention backward (CUDA)");
}
