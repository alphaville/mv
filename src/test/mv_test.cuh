/**
 * A Test file for the kernel defined
 * in mv.cuh
 */

#ifndef MV_TEST_HEADER_
#define MV_TEST_HEADER_

#include <curand.h>
#include "../gpad_types.h"
#include "../mv.cuh"

// Assumes col-major order
void check_correctness(float *dev_rand_data, int nrows, int ncols)
{
	float tolerance = 0.0001;
	float almost_zero = 1e-7;
	float alpha = 1.0, beta = 0.0;
	real_t * dev_y = NULL;
	real_t * hst_y = NULL;
	real_t * dev_y_cublas = NULL;
	real_t * hst_y_cublas = NULL;
	cublasHandle_t handle;
	size_t s  = nrows * sizeof(real_t);

	_CUBLAS(cublasCreate(&handle));

	hst_y_cublas = (real_t*) malloc(s);
	hst_y = (real_t*) malloc(s);
	_CUDA(cudaMalloc((void** )&dev_y, s));
	_CUDA(cudaMalloc((void** )&dev_y_cublas, s));

	tested::matvec<real_t>(dev_rand_data + ncols, dev_rand_data, dev_y, nrows, ncols);
	_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, nrows, ncols, &alpha, dev_rand_data + ncols,
				nrows, dev_rand_data, 1, &beta, dev_y_cublas, 1));

	_CUDA(cudaMemcpy(hst_y, dev_y, s, cudaMemcpyDeviceToHost));
	_CUDA(cudaMemcpy(hst_y_cublas, dev_y_cublas, s, cudaMemcpyDeviceToHost));
	for (uint_t i = 0; i < nrows; ++i) {
		if (  (fabsf(hst_y_cublas[i]) > almost_zero && fabsf(hst_y_cublas[i] - hst_y[i]) > tolerance)
				|| fabsf(hst_y[i])<almost_zero) {
			printf("\n-- Result is wrong at entry %u for %u nrows and %u columns.\n", i, nrows, ncols);
			printf("-- Relative error %f \n", fabsf(hst_y_cublas[i] - hst_y[i])/hst_y_cublas[i]);
			printf("-- CUBLAS returns y[%d] = %f \n", i, hst_y_cublas[i]);
			printf("-- Custom kernel gives y[%d] = %f \n", i, hst_y[i]);
			exit(EXIT_FAILURE);
		}
	}

	_CUBLAS(cublasDestroy(handle));
	_CUDA(cudaFree(dev_y));
	_CUDA(cudaFree(dev_y_cublas));
	if (hst_y) free(hst_y);
	if (hst_y_cublas) free(hst_y_cublas);

}


void mv_test_01(){
	curandGenerator_t gen;
	real_t *dev_rand_data = NULL; // Random data will be allocated here!
	uint_t size_tot = 35000;

	_CUDA(cudaMalloc((void** )&dev_rand_data, size_tot*sizeof(float)));

	_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 13534ULL));
	_CURAND(curandGenerateUniform(gen, dev_rand_data, size_tot));
	_CURAND(curandDestroyGenerator(gen));

	printf("Test 01");
	for (uint_t i=1; i< 100; ++i)
		check_correctness(dev_rand_data, i, 32);
	printf(" : OK!\n");

	printf("Test 02");
	for (uint_t j=5; j< 70; ++j)
		check_correctness(dev_rand_data, 64, j);
	printf(" : OK!\n");

	printf("Test 03");
	for (uint_t i=5; i< 150; ++i) {
		for (uint_t j=5; j< 150; ++j) {
			check_correctness(dev_rand_data, i, j);
		}
	}
	printf(" : OK!\n");

	if (dev_rand_data) _CUDA(cudaFree(dev_rand_data));
}


void mv_test_02(){
	// This is a test for #matvec_kernel_rowmajor...
}


void mv_test(){
	mv_test_01();
}


#endif /* MV_TEST_HEADER_ */
