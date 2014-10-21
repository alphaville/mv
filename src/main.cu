#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "mv.cuh"
#include "cuda_timer.cuh"
#include "error_handles.cuh"
#include "api/mv_types.h"
#include "test/mv_test.cuh"
#include "test/mv_benchmark.cuh"

void deviceName(void) {
	int whichDevice;
	cudaDeviceProp prop;
	_CUDA(cudaGetDevice(&whichDevice));
	_CUDA(cudaGetDeviceProperties(&prop, whichDevice));
	printf("Device '%s'\n", prop.name);
	printf("Multiprocessor '%d'\n", prop.multiProcessorCount);
	printf("Clock '%d MHz'\n", prop.clockRate/1000);
	printf("Max threads per MP '%d'\n", prop.maxThreadsPerMultiProcessor);
}

int main(void)
{
	// deviceName();
	mv_test();
	do_benchmark();

	printf("Bye!\n");
	return EXIT_SUCCESS;
}

