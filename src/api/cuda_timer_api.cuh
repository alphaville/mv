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
 * Status of timer functions
 */
enum timerStatus {
	/**
	 * Timer function returned without problems
	 */
	timer_ok = 0,
	/**
	 * The timer was not staret. You need to call ::start_tictoc()
	 */
	timer_notstarted = 1,
	/**
	 * Successive calls to ::tic() are suspicious
	 */
	timer_consequtiveTics = 2
};

typedef enum timerStatus timerStatus_t;

/**
 * Sets up the timer.
 *
 * Must be called before any invocation to
 * ::tic() or ::toc(), preferably at the beginning of your
 * application.
 *
 * @return
 * ::timer_ok
 */
timerStatus_t start_tictoc();

/**
 * Starts the timer.
 *
 * Use `toc()` to get the elapsed time; `tic()` must
 * be called before a `toc()`.
 *
 * @return
 * ::timer_ok,
 * ::timer_consequtiveTics,
 * ::timer_notstarted
 */
timerStatus_t tic();

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
 *
 * @return
 * ::timer_ok,
 * ::timer_notstarted
 */
timerStatus_t stop_tictoc();

#endif /* CUDA_TIMER_API_CUH_ */
