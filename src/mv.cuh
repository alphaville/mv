#ifndef MV_CUH_HEADER_
#define MV_CUH_HEADER_


#include "api/mv_api.cuh"



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
	uint_t blk_size_opt = 64;

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
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ T x_shared[blk];

	T y_val = 0.0;

	#pragma unroll
	for (unsigned int m = 0; m < ((nx + blk - 1) / blk); ++m) {

		if ((m * blk + threadIdx.x) < nx)
			x_shared[threadIdx.x] = dx[threadIdx.x + m * blk];
		else
			x_shared[threadIdx.x] = 0.f;

		__syncthreads();

		#pragma unroll
		for (unsigned int e = 0; e < blk; ++e) {
			y_val += dA[tid + (e + blk * m) * nRows] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < nRows)
		dy[tid] = y_val;

} /* End function matvec_kernel */




#endif /* MV_CUH_HEADER_ */
