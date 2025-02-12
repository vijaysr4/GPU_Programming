
#include <stdio.h>
#include <stdlib.h>

__global__ void add (int a, int b, int *res) {
  *res = a + b;
}

#define CUDA_CHECK(err) if (err != cudaSuccess) { \
  printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
  exit(EXIT_FAILURE); \
}

int main() {
  int res=10000;
  int *d_res = NULL;

  cudaMalloc((void**)&d_res, sizeof(int));

  dim3 grid(1);
  dim3 block(1);

  // Launch add() kernel on GPU
  add<<<grid,block>>>(2, 2, d_res);

  // Ensure kernel execution completes
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Before cudaMemcpy - 2 + 2 = %d\n", res);
  cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);

  printf("After cudaMemcpy - 2 + 2 = %d\n", res);

  cudaFree(d_res);
  return EXIT_SUCCESS;
}