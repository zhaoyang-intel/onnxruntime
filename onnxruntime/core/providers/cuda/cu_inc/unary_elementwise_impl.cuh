// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename InT, typename OutT, typename FuncT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _UnaryElementWise(
    const InT* input_data,
    OutT* output_data,
    const FuncT functor,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  InT value[NumElementsPerThread];

  CUDA_LONG id = start;
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      value[i] = input_data[id];
      id += NumThreadsPerBlock;
    }
  }

  id = start;
  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = functor(value[i]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename InT, typename OutT, typename FuncT, int NumThreadsPerBlock, int VEC>
__global__ void _UnaryElementWiseVectorized(
    const InT* input_data,
    OutT* output_data,
    const FuncT functor,
    CUDA_LONG N) {

  using LoadInT = aligned_vector<InT, VEC>;
  using LoadOutT = aligned_vector<OutT, VEC>;

  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG id = idx * VEC;

  if (id < N) {
    // vectorized load into storage
    InT src[VEC];
    LoadInT *value = reinterpret_cast<LoadInT*>(&src);
    *value = *reinterpret_cast<const LoadInT*>(&input_data[id]);

    OutT r[VEC];

    // actual computation
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
      r[i] = functor(src[i]);
    }
    // Vectorized writes for output_data
    *(reinterpret_cast<LoadOutT*>(&output_data[id])) = *reinterpret_cast<LoadOutT*>(&r[0]);
  }

}

template <typename InT, typename OutT, typename FuncT>
void UnaryElementWiseImpl(
    cudaStream_t stream,
    const InT* input_data,
    OutT* output_data,
    const FuncT& func,
    size_t count) {
  if (count == 0)  // special case where there's a dim value of 0 in the shape
    return;

  int vec_size = GridDim::maxElementsPerThread;
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  bool should_vectorized = true;
  if (std::is_same<OutT, int64_t>::value ||
      std::is_same<OutT, uint64_t>::value ||
      std::is_same<OutT, double>::value ||
      std::is_same<InT, int64_t>::value ||
      std::is_same<InT, uint64_t>::value ||
      std::is_same<InT, double>::value) {  
    should_vectorized = false;
  }

  if (N % vec_size != 0 || should_vectorized == false) {
    int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
    
    _UnaryElementWise<InT, OutT, FuncT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            input_data,
            output_data,
            func,
            N);
  } else {
    int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * vec_size));
    _UnaryElementWiseVectorized<InT, OutT, FuncT, GridDim::maxThreadsPerBlock, 4>
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            input_data,
            output_data,
            func,
            N);
  }

}

}  // namespace cuda
}  // namespace onnxruntime
