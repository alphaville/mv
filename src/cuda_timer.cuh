#include "api/cuda_timer_api.cuh"

static cudaEvent_t start;
static cudaEvent_t stop;
static short timer_running = 0;
static short tic_called = 0;


void start_tictoc() {
	_CUDA(cudaEventCreate(&start));
	_CUDA(cudaEventCreate(&stop));
	timer_running = 1;
}

void tic() {
	if (tic_called == 1){
		printf("WARNING: Consecutive calls to tic() are fishy...\n");
	}
	if (timer_running) {
		_CUDA(cudaEventRecord(start, 0));
		tic_called = 1;
	} else {
		printf("WARNING: tic() called without a timer running!\n");
	}
}

float toc() {
	float elapsed_time;
	if (tic_called == 0) {
		printf("WARNING: toc() called without a previous tic()!\n");
		return -1;
	}
	if (timer_running == 1) {
		_CUDA(cudaEventRecord(stop, 0));
		_CUDA(cudaEventSynchronize(stop));
		_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
		tic_called = 0;
		return elapsed_time;
	} else {
		printf("WARNING: toc() called without a timer running!\n");
		return -2;
	}

}

void stop_tictoc()
{
	if (timer_running == 1){
		_CUDA(cudaEventDestroy(start));
		_CUDA(cudaEventDestroy(stop));
		timer_running = 0;
	} else{
		printf("WARNING: stop_tictoc() called without a timer running!\n");
	}
}

