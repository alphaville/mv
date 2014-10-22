#ifndef _ERROR_HANDLES_HEADER__
#define _ERROR_HANDLES_HEADER__

#include "helper_cuda.h"

#define _CUDA(x) checkCudaErrors(x)

#define _CUBLAS(x) checkCudaErrors(x)

#define _CURAND(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)


#endif /* _ERROR_HANDLES_HEADER__ */
