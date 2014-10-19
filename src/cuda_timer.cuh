#ifndef CUDA_TIMER_CUH_HEADER_
#define CUDA_TIMER_CUH_HEADER_


#include <cuda_runtime.h>
#include "error_handles.cuh"

static cudaEvent_t start;
static cudaEvent_t stop;
static short timer_running = 0;
static short tic_called = 0;

/**
 * Sets up the timer.
 *
 * Must be called before any invocation to
 * tic() or toc(), preferrably at the beginning of your
 * application.
 */
void start_tictoc();

/**
 * Starts the timer.
 *
 * Use `toc()` to get the elapsed time; `tic()` must
 * be called before a `toc()`.
 */
void tic();

/**
 * Returns the elapsed time between its invocation
 * and a previous invocation of `toc()`. Returns `-1`
 * and prints a warning message if `toc()` was not
 * previously called. Returns `-2` and prints and error
 * message if `start_tictoc()` has not been called.
 *
 * @return Elapsed time between `tic()` and `toc()` in milliseconds
 * with a resolution of `0.5` microseconds.
 */
float toc();

/**
 * This function should be called when the
 * time will not be being used any more. It destroys
 * the events used to time CUDA kernels. If the timer
 * is not running, this function does nothing and
 * prints a warning message.
 */
void stop_tictoc();


void start_tictoc() {
	_CUDA(cudaEventCreate(&start));
	_CUDA(cudaEventCreate(&stop));
	timer_running = 1;
}

void tic() {
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

#endif /* CUDA_TIMER_CUH_HEADER_ */
