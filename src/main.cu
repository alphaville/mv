#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "mv.cuh"
#include "cuda_timer.cuh"
#include "error_handles.cuh"
#include "gpad_types.h"
#include "test/mv_test.cuh"
#include "test/mv_benchmark.cuh"


int main(void)
{
	mv_test();
	//do_benchmark();
	return EXIT_SUCCESS;
}

