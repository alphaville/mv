#ifndef _ERROR_HANDLES_HEADER__
#define _ERROR_HANDLES_HEADER__

#define _CUDA(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#define _CUBLAS(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#define _CURAND(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#endif /* _ERROR_HANDLES_HEADER__ */
