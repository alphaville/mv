/*
 * cuda_timer_api.cuh
 *
 *  Created on: Oct 21, 2014
 *      Author: imt
 */

#ifndef CUDA_TIMER_API_CUH_
#define CUDA_TIMER_API_CUH_

#include <cuda_runtime.h>
#include "../error_handles.cuh"

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

#endif /* CUDA_TIMER_API_CUH_ */
