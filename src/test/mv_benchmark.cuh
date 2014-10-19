#ifndef MV_BENCHMARK_CUH_HEADER_
#define MV_BENCHMARK_CUH_HEADER_

#include <curand.h>
#include "../gpad_types.h"


#define TEST_COLUMNS  		1
#define TEST_ROWS     		0

/**
 * If `TEST_WRT_` is set to `TEST_COLUMNS`, then a benchmark
 * will be performed with respect to columns (with a fixed
 * number of rows). If it is set to `TEST_ROWS`, then a benchmark will
 * run with respect to rows (fixed number of columns).
 */
#define TEST_WRT_ TEST_ROWS

#define CONSTANT_COLS 256
#define CONSTANT_ROWS 128

/**
 * In order to estimate the execution time, every
 * kernel is run `RUNS` times and the average is taken.
 */
#define RUNS 50

void do_benchmark() {
	curandGenerator_t gen;
	real_t *dev_rand_data = NULL; // Random data will be allocated here!
	real_t *dev_y = NULL;
	real_t *dev_y_cublas = NULL;
	real_t t;
	real_t t_cublas;
	const uint_t n_rows_max = 1200;
	const uint_t n_cols_max = 800;
	const uint_t ntot = n_cols_max * (1 + n_rows_max);
	const uint_t size_tot = sizeof(real_t) * ntot;

	float alpha = 1.0, beta = 0.0;
	cublasHandle_t handle;
	_CUBLAS(cublasCreate(&handle));

	start_tictoc();

	_CUDA(cudaMalloc((void** )&dev_rand_data, size_tot));
	_CUDA(cudaMalloc((void** )&dev_y, n_rows_max * sizeof(real_t)));
	_CUDA(cudaMalloc((void** )&dev_y_cublas, n_rows_max * sizeof(real_t)));

	_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
	tic();
	_CURAND(curandGenerateUniform(gen, dev_rand_data, ntot));
	t = toc();
	printf("RNG in %f ms\n", t);

	_CURAND(curandDestroyGenerator(gen));

	uint_t ncols = CONSTANT_COLS;
	uint_t nrows = CONSTANT_ROWS;
	uint_t runs = RUNS;

	cudaMemset(dev_y_cublas, 0, n_rows_max * sizeof(real_t));

	matvec<real_t>(dev_rand_data + ncols, dev_rand_data, dev_y, nrows, ncols);
	_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, nrows, ncols, &alpha, dev_rand_data + ncols,
								nrows, dev_rand_data, 1, &beta, dev_y_cublas, 1));

	FILE * pFile;
	char filename[50];
#if (TEST_WRT_ == TEST_COLUMNS)
	sprintf(filename, "times_rows%u_cols.txt", nrows);
#else
	sprintf(filename, "times_cols%u_rows.txt", ncols);
#endif

	printf("Logging to : '%s'\n", filename);
	pFile = fopen(filename, "w");
	if (pFile == NULL) {
		perror("Error opening file.");
		exit(79);
	}


#if (TEST_WRT_ == TEST_COLUMNS)
	fprintf(pFile, "0, %u, 0, 0\n", nrows);
	for (ncols = 32; ncols < n_cols_max; ncols += 3) {
#else
	fprintf(pFile, "1, %u, 0, 0\n", ncols);
	for (nrows = 32; nrows < n_rows_max; nrows += 3) {
#endif
		tic();
		for (short i = 0; i < runs; i++) {
			matvec<real_t>(dev_rand_data + ncols, dev_rand_data, dev_y, nrows,
					ncols);
		}
		t = toc() / runs;
		tic();
		for (short i = 0; i < runs; i++) {
			_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, nrows, ncols, &alpha, dev_rand_data + ncols,
							nrows, dev_rand_data, 1, &beta, dev_y_cublas, 1));
		}
		t_cublas = toc() / runs;
#if (TEST_WRT_ == TEST_COLUMNS)
		fprintf(pFile, "%u, %f, %f\n", ncols, t, t_cublas);
#else
		fprintf(pFile, "%u, %f, %f\n", nrows, t, t_cublas);
#endif
	}
	_CUBLAS(cublasDestroy(handle));

	fclose(pFile);

	if (dev_rand_data != NULL)
		_CUDA(cudaFree(dev_rand_data));
	if (dev_y!=NULL)
		_CUDA(cudaFree(dev_y));
	if (dev_y_cublas!=NULL)
			_CUDA(cudaFree(dev_y_cublas));

	stop_tictoc();
}


#endif /* MV_BENCHMARK_CUH_HEADER_ */
