/*
 * mv_api.cuh
 *
 *  Created on: Oct 21, 2014
 *      Author: imt
 */

#ifndef MV_API_CUH_
#define MV_API_CUH_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "mv_types.h"

/**
 * Whether to use warp shuffle functions or not.
 * Un-comment the following line to use shuffle functions.
 */
//#define MV_USE_SHFL

#ifdef MV_USE_SHFL
#include "sm_30_intrinsics.h"
#endif


/**
 * Macro for __restrict__
 */
#define		RESTRICT __restrict__




/**
 * Performs matrix-vector multiplication on the device.
 *
 * @param	dA				Address of matrix `A` on the device in column-major order
 * @param	dx				Address of vector `x` on the device
 * @param	dy				Address of result y = A*x on the device
 * @param	nRows			Number of rows of `A`
 * @param	nx				Size of `x` (number of columns of `A`)
 *
 * @tparam  T				Data type of the matrix and vector elements. The use of
 * 							`float` is recommended.
 * @tparam  blk			    This is the block size which is necessary to statically
 * 							allocate shared memory. It was found that static allocation
 * 							offers a significant performance benefit.
 *
 * \sa ::matvec
 */
template<typename T, const uint_t blk>
__global__ void matvec_kernel(
		const T * RESTRICT  dA,
		const T * RESTRICT  dx,
		T * RESTRICT dy,
		const uint_t nRows,
		const uint_t nx);



/**
 * Host-side wrapper for #matvec_kernel.
 *
 * Calls the matrix-vector kernel with a block size that is
 * internally decided.
 *
 * @param	dA				Address of matrix `A` on the device.
 * 							Values in `A` are stored in column-major order.
 * @param	dx				Address of vector `x` on the device
 * @param	dy				Address of result y = A*x on the device.
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


/**
 * Host-side wrapper for #matvec_kernel which runs the kernel
 * with a custom block size which is given as a template parameter.
 *
 *
 * @param	dA				Address of matrix `A` on the device.
 * 							Values in `A` are stored in column-major order.
 * @param	dx				Address of vector `x` on the device
 * @param	dy				Address of result y = A*x on the device.
 * @param	nRows			Number of rows of `A`
 * @param	nx				Size of `x` (number of columns of `A`)
 *
 * @tparam  T				Data type for `A` and `x`
 * @tparam  blk			    Block size for the kernel launch
 */
template<typename T, const uint_t blk>
__host__ void matvec_engine(
		const T * RESTRICT dA,
		const T * RESTRICT dx,
		T * RESTRICT dy,
		const uint_t nRows,
		const uint_t nx);



#endif /* MV_API_CUH_ */
