#ifndef MV_CUH_HEADER_
#define MV_CUH_HEADER_


#include <cuda_runtime.h>
#include "gpad_types.h"

#define 	BLOCK_SIZE		128

/* Set to __restric__ */
#define		RESTRICT


/**
 * Performs matrix-vector multiplication on the device.
 *
 * @param	dA				Address of matrix `A` on the device in column-major order
 * @param	dx				Address of vector `x` on the device
 * @param	dev_ptr_y		Address of result y = A*x on the device
 * @param	nRows			Number of rows of `A`
 * @param	nx				Size of `x` (number of columns of `A`)
 *
 * @tparam  T				Data type
 *
 */
template<typename T>
__global__ void matvec_kernel(
		const T * RESTRICT dA,
		const T * RESTRICT dx,
		T * RESTRICT dy,
		const uint_t nRows,
		const uint_t nx);

/**
 * Host-side wrapper for #matvec_kernel.
 *
 * @param	dA				Address of matrix `A` on the device.
 * 							Values in `A` are stored in column-major order.
 * @param	dx				Address of vector `x` on the device
 * @param	dev_ptr_y		Address of result y = A*x on the device.
 * @param	nRows			Number of rows of `A`
 * @param	nx				Size of `x` (number of columns of `A`)
 *
 * @tparam  T				Data type for `A` and `x`
 */
template<typename T>
__host__ void matvec(
		const T * RESTRICT dA,
		const T * RESTRICT dx,
		T * RESTRICT dy,
		const uint_t nRows,
		const uint_t nx);


/* -------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------- */



template<typename T>
__host__ void matvec(const T * RESTRICT dA, const T * RESTRICT dx,
		T * RESTRICT dy, const uint_t nRows, const uint_t nx) {
	dim3 dim_grid((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dim_block(BLOCK_SIZE);
	matvec_kernel<T><<<dim_grid, dim_block>>>(dA, dx, dy, nRows, nx);
}

template<typename T>
__global__ void matvec_kernel(const T * RESTRICT  dA, const T * RESTRICT  dx,
		T * RESTRICT dy, const uint_t nRows, const uint_t nx)
{

	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ T x_shared[BLOCK_SIZE];

	T y_val = 0.0;

	#pragma unroll
	for (unsigned int m = 0; m < ((nx + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {

		if ((m * BLOCK_SIZE + threadIdx.x) < nx)
			x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
		else
			x_shared[threadIdx.x] = 0.f;

		__syncthreads();

		#pragma unroll
		for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
			y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < nRows)
		dy[tid] = y_val;

} /* End function matvec_kernel */

#endif /* MV_CUH_HEADER_ */
