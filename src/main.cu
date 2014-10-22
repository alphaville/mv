#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "helper_cuda.h"
#include "mv.cuh"
#include "cuda_timer.cuh"
#include "error_handles.cuh"
#include "api/mv_types.h"
#include "test/mv_test.cuh"
#include "test/mv_benchmark.cuh"



__global__ void ker(float ** d_X, float * d_A, float * d_B, int n){
	printf("Addresses...\n");
	printf("dX    = %p\n", d_X);
	printf("dA    = %p\n", d_A);
	printf("dB    = %p\n", d_B);
	printf("dX[0] = %p\n", d_X[0]);
	printf("dX[0] = %p\n", d_X[1]);

	float * devA  = d_X[0];
	float * devB  = d_X[1];

	printf("\nValues...\n");
	for (int i=0; i<n; i++)
		printf("A[%d] = %f\n", i, devA[i]);
	for (int i=0; i<n; i++)
			printf("B[%d] = %f\n", i, devB[i]);

}

int main(void)
{

	//mv_test();
	do_benchmark();


	printf("Bye!\n");
	return EXIT_SUCCESS;
}

