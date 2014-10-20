#ifndef MV_CUH_HEADER_
#define MV_CUH_HEADER_


#include <cuda_runtime.h>
#include "gpad_types.h"

#define 	BLOCK_SIZE 		128

/* Set to __restric__ */
#define		RESTRICT

namespace tested{
/**
 * Performs matrix-vector multiplication on the device.
 *
 * @param	dA				Address of matrix `A` on the device in column-major order
 * @param	dx				Address of vector `x` on the device
 * @param	dev_ptr_y		Address of result y = A*x
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
 * @param	dA				Address of matrix `A` on the device
 * @param	dx				Address of vector `x` on the device
 * @param	dev_ptr_y		Address of result y = A*x
 * @param	nRows			Number of rows of `A`
 * @param	nx				Size of `x` (number of columns of `A`)
 * @param	elapsed_time	Time for the kernel to complete the execution in `ms`.
 * 							If NULL is passed to this argument, the elapsed time
 * 						 	will not be computed.
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
} /* namespace tested */

/**
 * EXPERIMENTAL! TO BE TESTED!!!
 * Performs matrix-vector multiplication on the device.
 *
 * @param	dA				Address of matrix `A` on the device in row-major order
 * @param	dx				Address of vector `x` on the device
 * @param	dev_ptr_y		Address of result y = A*x
 * @param	nRows			Number of rows of `A`
 * @param	nx				Size of `x` (number of columns of `A`)
 *
 * @tparam  T				Data type
 *
 */
namespace experimental{
template<typename T>
__global__ void matvec_kernel(
		const T * RESTRICT dA,
		const T * RESTRICT dx,
		T * RESTRICT dy,
		const uint_t nRows,
		const uint_t nx);


/**
 * Host-side wrapper for #matvec_kernel_rowmajor.
 *
 * @param	dA				Address of matrix `A` on the device (row-major)
 * @param	dx				Address of vector `x` on the device
 * @param	dev_ptr_y		Address of result y = A*x
 * @param	nRows			Number of rows of `A`
 * @param	nx				Size of `x` (number of columns of `A`)
 * @param	elapsed_time	Time for the kernel to complete the execution in `ms`.
 * 							If NULL is passed to this argument, the elapsed time
 * 						 	will not be computed.
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

} /* namespace experimental */

/* -------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------- */

template<typename T>
__global__ void tested::matvec_kernel(const T * RESTRICT  dA, const T * RESTRICT  dx,
		T * RESTRICT dy, const uint_t nRows, const uint_t nx)
{

	uint_t bid = blockIdx.x;
	uint_t row = threadIdx.x;
	const uint_t block_size = blockDim.x;
	const uint_t num_hor_blocks = ((nx + block_size - 1) / block_size);
	uint_t idx_x;
	uint_t idx_Asub;
	uint_t idx_y;
	const T * Asub;
	const T * xsub;

	/* Only `x` is copied to shared memory */
	__shared__ T x_shared[BLOCK_SIZE];

	idx_y = bid * block_size;

	T * y_sub = dy + idx_y;

	T y_val = 0.0;

	for (uint_t m = 0; m < num_hor_blocks; ++m) {

		idx_Asub = block_size * (bid + m * nRows);
		idx_x = m * block_size;

		Asub = dA + idx_Asub;
		xsub = dx + idx_x;

		/*
		 * Un-comment the following line to test
		 * that inexact tiling is doen properly
		 *
		 * x_shared[row] = 100.0;
		 *
		 * */

		if (idx_x + row < nx) {
			x_shared[row] = xsub[row];
		}

		__syncthreads();

		/* If the tiling is exact */
		if ((nRows % block_size == 0 && nx % block_size == 0)
				|| (m < num_hor_blocks -1 && bid < gridDim.x - 1)
				|| (BLOCK_SIZE < nx - idx_x)) {
			#pragma unroll
			for (uint_t e = 0; e < BLOCK_SIZE; ++e) {
				y_val += Asub[row + e * nRows] * x_shared[e];
			}
		} else { /* Inexact tiling */
			#pragma unroll
			for (uint_t e = 0; e < nx - idx_x; ++e) {
				y_val += Asub[row + e * nRows] * x_shared[e];
			}
		}
		__syncthreads();
	    }

	    if (row + idx_y < nRows)
	        y_sub[row] = y_val;

} /* End function tested::matvec_kernel*/



template<typename T>
__host__ void tested::matvec(const T * RESTRICT dA, const T * RESTRICT dx,
		T * RESTRICT dy, const uint_t nRows, const uint_t nx) {
	dim3 dim_grid((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dim_block(BLOCK_SIZE);
	tested::matvec_kernel<T> <<<dim_grid, dim_block>>>(dA, dx, dy, nRows, nx);
}


template<typename T>
__host__ void experimental::matvec(const T * RESTRICT dA, const T * RESTRICT dx,
		T * RESTRICT dy, const uint_t nRows, const uint_t nx) {
	dim3 dim_grid((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dim_block(BLOCK_SIZE);
	experimental::matvec_kernel<T> <<<dim_grid, dim_block>>>(dA, dx, dy, nRows, nx);


}

/* EXPERIMENTAL! TO BE TESTED!!! */
template<typename T>
__global__ void experimental::matvec_kernel(const T * RESTRICT  dA, const T * RESTRICT  dx,
		T * RESTRICT dy, const uint_t nRows, const uint_t nx)
{

	uint_t bid = blockIdx.x;
	uint_t row = threadIdx.x;
	const uint_t block_size = blockDim.x;
	const uint_t num_hor_blocks = ((nx + block_size - 1) / block_size);
	uint_t n_star;
	uint_t idx_x;
	uint_t idx_Asub;
	uint_t idx_y;
	const T * Asub;
	const T * xsub;

	if (bid==0 && row==0) {printf("Num hor blocks : %d\n", num_hor_blocks);}
	__shared__ T x_shared[BLOCK_SIZE];

	idx_y = bid * block_size;

	T * y_sub = dy + idx_y;

	T y_val = 0.0;

	for (uint_t m = 0; m < num_hor_blocks; ++m) {

		idx_Asub = block_size * (bid * nx + m);
		idx_x = m * block_size;

		Asub = dA + idx_Asub;
		xsub = dx + idx_x;

		if (idx_x + row < nx) {
			x_shared[row] = xsub[row];
		} else {
			x_shared[row] = 0.0; /* Possibly not necessary - double-check */
		}

		__syncthreads();

		/* If the tiling is exact */
		if ((nRows % block_size == 0 && nx % block_size == 0)
				|| (m < num_hor_blocks -1 && bid < gridDim.x - 1) ) {
			#pragma unroll
			for (uint_t e = 0; e < BLOCK_SIZE; ++e) {
				y_val += Asub[row + e] * x_shared[e];
			}
		} else { /* Inexact tiling */
			n_star = min(BLOCK_SIZE, nx - idx_x);
			#pragma unroll
			for (uint_t e = 0; e < n_star; ++e) {
				y_val += Asub[row + e] * x_shared[e];
			}
		}
		__syncthreads();
	    }

	    if (row + idx_y < nRows)
	        y_sub[row] = y_val;

}

#endif /* MV_CUH_HEADER_ */
