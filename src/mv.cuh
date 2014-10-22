#include "api/mv_api.cuh"

#pragma once

template<typename T, const uint_t blk>
__host__ void matvec_engine(const T * RESTRICT dA, const T * RESTRICT dx,
		T * RESTRICT dy, const uint_t nRows, const uint_t nx) {

	dim3 dim_grid((nRows + blk - 1) / blk);
	dim3 dim_block(blk);

	matvec_kernel<T, blk> <<<dim_grid, dim_block>>>(dA, dx, dy, nRows, nx);

}

template<typename T>
__host__ void matvec(const T * RESTRICT dA, const T * RESTRICT dx,
		T * RESTRICT dy, const uint_t nRows, const uint_t nx) {
	uint_t blk_size_opt = 32;

	/* Add code to decide the value of `blk_size_opt` */

	if (blk_size_opt == 32) {
		matvec_engine<T, 32>(dA, dx, dy, nRows, nx);
	} else if (blk_size_opt == 64) {
		matvec_engine<T, 64>(dA, dx, dy, nRows, nx);
	} else if (blk_size_opt == 128) {
		matvec_engine<T, 128>(dA, dx, dy, nRows, nx);
	} else if (blk_size_opt == 256) {
		matvec_engine<T, 256>(dA, dx, dy, nRows, nx);
	}

}



template<typename T, const uint_t blk>
__global__ void matvec_kernel(const T * RESTRICT  dA, const T * RESTRICT  dx,
		T * RESTRICT dy, const uint_t nRows, const uint_t nx)
{

	const uint_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	const uint_t hor_blocks = (nx + blk - 1) / blk;

#ifndef MV_USE_SHFL
	__shared__ T x_shared[blk];
#else
	T x_shfl_src, x_shfl_dest;
#endif

	register T y_val = 0.0;

	#pragma unroll
	for (uint_t m = 0; m < hor_blocks; ++m) {

		if ((m * blk + threadIdx.x) < nx){
#ifndef MV_USE_SHFL
			x_shared[threadIdx.x] = dx[threadIdx.x + m * blk];
#else
			x_shfl_src = dx[threadIdx.x + m * blk];
#endif

		} else {

#ifndef MV_USE_SHFL
			x_shared[threadIdx.x] = 0.0f;
#else
			x_shfl_src = 0.0f;
#endif
		}

		__syncthreads();

		#pragma unroll
		for (uint_t e = 0; e < blk; ++e) {
#ifndef MV_USE_SHFL
			y_val += dA[tid + (e + blk * m) * nRows] * x_shared[e];
#else
			x_shfl_dest = __shfl(x_shfl_src, e);
			y_val += dA[tid + (e + blk * m) * nRows] * x_shfl_dest;
#endif
		}

		__syncthreads();
	}

	if (tid < nRows)
		dy[tid] = y_val;

} /* End function matvec_kernel */


