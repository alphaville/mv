#ifndef MV_CUH_HEADER_
#define MV_CUH_HEADER_


#include <cuda_runtime.h>
#include "gpad_types.h"

#define 	BLOCK_SIZE 		16

/* Set to __restric__ */
#define		RESTRICT

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
template<typename T>
__global__ void matvec_kernel_rowmajor(
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



/* -------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------- */

template<typename T>
__global__ void matvec_kernel(const T * RESTRICT  dA, const T * RESTRICT  dx,
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

		if (idx_x + row < nx) {
			x_shared[row] = xsub[row];
			/*
			 * Double-check whether it is necessary to do:
			 * else { x_shared[row] = 0.0 }
			 */
		}

		__syncthreads();

		/* If the tiling is exact */
		if ((nRows % block_size == 0 && nx % block_size == 0)
				|| (m < num_hor_blocks -1 && bid < gridDim.x - 1) ) {
			y_val += Asub[row] * x_shared[0];
			y_val += Asub[row + nRows] * x_shared[1];
			y_val += Asub[row + 2 * nRows] * x_shared[2];
			y_val += Asub[row + 3 * nRows] * x_shared[3];
			y_val += Asub[row + 4 * nRows] * x_shared[4];
			y_val += Asub[row + 5 * nRows] * x_shared[5];
			y_val += Asub[row + 6 * nRows] * x_shared[6];
			y_val += Asub[row + 7 * nRows] * x_shared[7];
			y_val += Asub[row + 8 * nRows] * x_shared[8];
			y_val += Asub[row + 9 * nRows] * x_shared[9];
			y_val += Asub[row + 10 * nRows] * x_shared[10];
			y_val += Asub[row + 11 * nRows] * x_shared[11];
			y_val += Asub[row + 12 * nRows] * x_shared[12];
			y_val += Asub[row + 13 * nRows] * x_shared[13];
			y_val += Asub[row + 14 * nRows] * x_shared[14];
			y_val += Asub[row + 15 * nRows] * x_shared[15];
		} else { /* Inexact tiling */
			n_star = min(BLOCK_SIZE, nx - idx_x);
			#pragma unroll
			for (unsigned int e = 0; e < n_star; ++e) {
				y_val += Asub[row + e * nRows] * x_shared[e];
			}
		}
		__syncthreads();
	    }

	    if (row + idx_y < nRows)
	        y_sub[row] = y_val;

}



template<typename T>
__host__ void matvec(
		const T * RESTRICT  dA,
		const T * RESTRICT  dx,
		T * RESTRICT dy,
		const uint_t nRows,
		const uint_t nx)
{
	dim3 dim_grid( (nRows + BLOCK_SIZE -1)/ BLOCK_SIZE);
	dim3 dim_block(BLOCK_SIZE);
	matvec_kernel<T> <<<dim_grid, dim_block>>>(dA, dx, dy, nRows, nx);
}



/* EXPERIMENTAL! TO BE TESTED!!! */
template<typename T>
__global__ void matvec_kernel_rowmajor(const T * RESTRICT  dA, const T * RESTRICT  dx,
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
			y_val += Asub[row     ] * x_shared[0];
			y_val += Asub[row + 1 ] * x_shared[1];
			y_val += Asub[row + 2 ] * x_shared[2];
			y_val += Asub[row + 3 ] * x_shared[3];
			y_val += Asub[row + 4 ] * x_shared[4];
			y_val += Asub[row + 5 ] * x_shared[5];
			y_val += Asub[row + 6 ] * x_shared[6];
			y_val += Asub[row + 7 ] * x_shared[7];
			y_val += Asub[row + 8 ] * x_shared[8];
			y_val += Asub[row + 9 ] * x_shared[9];
			y_val += Asub[row + 10] * x_shared[10];
			y_val += Asub[row + 11] * x_shared[11];
			y_val += Asub[row + 12] * x_shared[12];
			y_val += Asub[row + 13] * x_shared[13];
			y_val += Asub[row + 14] * x_shared[14];
			y_val += Asub[row + 15] * x_shared[15];
		} else { /* Inexact tiling */
			n_star = min(BLOCK_SIZE, nx - idx_x);
			#pragma unroll
			for (unsigned int e = 0; e < n_star; ++e) {
				y_val += Asub[row + e] * x_shared[e];
			}
		}
		__syncthreads();
	    }

	    if (row + idx_y < nRows)
	        y_sub[row] = y_val;

}

#endif /* MV_CUH_HEADER_ */
