#include "api/cuda_timer_api.cuh"

static cudaEvent_t start;
static cudaEvent_t stop;
static short timer_running = 0;
static short tic_called = 0;

// #define _TIMECHK(x) if (x>0){ printf("Timer error: %d", x); }

timerStatus_t start_tictoc() {
	if (timer_running == 0){
		_CUDA(cudaEventCreate(&start));
		_CUDA(cudaEventCreate(&stop));
		timer_running = 1;
	}
	return timer_ok;
}

timerStatus_t tic() {
	if (tic_called == 1){
#ifdef DEBUGIT
		printf("WARNING: Consecutive calls to tic() are fishy...\n");
#endif
		return timer_consequtiveTics;
	}
	if (timer_running) {
		_CUDA(cudaEventRecord(start, 0));
		tic_called = 1;
		return timer_ok;
	} else {
#ifdef DEBUGIT
		printf("WARNING: tic() called without a timer running!\n");
#endif
		return timer_notstarted;
	}
}

float toc() {
	float elapsed_time;
	if (tic_called == 0) {
#ifdef DEBUGIT
		printf("WARNING: toc() called without a previous tic()!\n");
#endif
		return -1;
	}
	if (timer_running == 1) {
		_CUDA(cudaDeviceSynchronize());
		_CUDA(cudaEventRecord(stop, 0));
		_CUDA(cudaEventSynchronize(stop));
		_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
		tic_called = 0;
		return elapsed_time;
	} else {
#ifdef DEBUGIT
		printf("WARNING: toc() called without a timer running!\n");
#endif
		return -2;
	}

}

timerStatus_t stop_tictoc()
{
	if (timer_running == 1){
		_CUDA(cudaEventDestroy(start));
		_CUDA(cudaEventDestroy(stop));
		timer_running = 0;
		return timer_ok;
	} else{
#ifdef DEBUGIT
		printf("WARNING: stop_tictoc() called without a timer running!\n");
#endif
		return timer_notstarted;
	}
}

