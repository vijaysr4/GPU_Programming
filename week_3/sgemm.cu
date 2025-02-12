#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "cuda_stuff.cuh"
#include "sgemm.cuh"
#include "fmatrix.cuh"

#define THREADS_PER_BLOCK 1024
#define TILE_WIDTH 32

using namespace std;

static cublasHandle_t handle;
static int cublas_init = 0;

/* basic matrix multiplication C = alpha*A*B + beta*C on host as reference for the speedup */
void matrixMultiplication_basic_host(float alpha, fmatrix A, fmatrix B, float beta, fmatrix C) {
  float tmp = 0;
  for (int i = 0; i<A.rows; i++){
    for (int j = 0; j<B.cols; j++){
      for (int k = 0; k<A.cols; k++){
        tmp += alpha * getfm(A,i, k) * getfm(B, k, j);
      }
      getfm(C, i, j) = beta * getfm(C, i, j) + tmp;
      tmp = 0;
    }
  }
}

/* 1. Basic GPU kernel:
   Each thread computes one element of C.
   Matrices are stored in column‐major order so we use the IDX2C macro.
   
   Parameters:
     - A: pointer to matrix A (dimensions: nb_LigneA x nb_ColA)
     - B: pointer to matrix B (dimensions: nb_LigneB x nb_ColB)
     - C: pointer to matrix C (dimensions: nb_LigneA x nb_ColB)
     
   Note: For a valid multiplication, A.cols == B.rows.
*/

/* TODO : 3 different versions of matrix multiplication C = alpha*A*B + beta*C on device */
__global__
void matmul_basic_kernel(float alpha, float *A, float *B, float beta, float *C, int nb_ColA, int nb_ColB, int nb_LigneA, int nb_LigneB) {
  /* TODO */
  // Compute row and column indices for C using 2D grid
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nb_LigneA && col < nb_ColB) {
      float sum = 0.0f;
      // Loop over the shared dimension (A.cols == B.rows)
      for (int k = 0; k < nb_ColA; k++){
          // Use the IDX2C macro defined in fmatrix.cuh for column-major order
          sum += A[IDX2C(row, k, nb_LigneA)] * B[IDX2C(k, col, nb_LigneB)];
      }
      // Write the result in C (scaling by alpha and adding beta * previous C value)
      C[IDX2C(row, col, nb_LigneA)] = alpha * sum + beta * C[IDX2C(row, col, nb_LigneA)];
  }
}
void matrixMultiplication_basic(float alpha, fmatrix d_A, fmatrix d_B, float beta, fmatrix d_C) {
  // TODO - declaration of dimGrid and dimBlock

  // Determine output dimensions (C has dimensions: A.rows x B.cols)
  int rows = d_A.rows;
  int cols = d_B.cols;
  
  // Use a 2D block configuration; here we choose 16x16 threads per block.
  dim3 dimBlock(16, 16);
  dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
               (rows + dimBlock.y - 1) / dimBlock.y);
 
  matmul_basic_kernel <<< dimGrid, dimBlock >>> (alpha, d_A.data, d_B.data, beta, d_C.data, d_A.cols, d_B.cols, d_A.rows, d_B.rows);
  gpuErrchk(cudaDeviceSynchronize());
}

/**********************/
__global__
void matmul_tiled_kernel(float alpha, float *A, float *B, float beta, float *C, int nb_ColA, int nb_ColB, int nb_LigneA, int nb_LigneB){
  /* TODO */
  // Allocate shared memory tiles for A and B
  __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  float value = 0.0f;

  // Loop over all tiles required to cover A.cols (which equals B.rows)
  for (int t = 0; t < (nb_ColA + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
      // Load element of A into shared memory if within bounds; otherwise use 0.
      int a_col = t * TILE_WIDTH + threadIdx.x;
      if (row < nb_LigneA && a_col < nb_ColA)
          tile_A[threadIdx.y][threadIdx.x] = A[IDX2C(row, a_col, nb_LigneA)];
      else
          tile_A[threadIdx.y][threadIdx.x] = 0.0f;

      // Load element of B into shared memory if within bounds; otherwise use 0.
      int b_row = t * TILE_WIDTH + threadIdx.y;
      if (b_row < nb_LigneB && col < nb_ColB)
          tile_B[threadIdx.y][threadIdx.x] = B[IDX2C(b_row, col, nb_LigneB)];
      else
          tile_B[threadIdx.y][threadIdx.x] = 0.0f;

      __syncthreads();  // Wait for all threads to load their data

      // Multiply the two tiles together
      for (int k = 0; k < TILE_WIDTH; k++){
          value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
      }
      __syncthreads();  // Wait for all threads before loading new tiles
  }
  
  // Write the computed value to C if within bounds.
  if (row < nb_LigneA && col < nb_ColB)
     C[IDX2C(row, col, nb_LigneA)] = alpha * value + beta * C[IDX2C(row, col, nb_LigneA)];
}



void matrixMultiplication_tiled(float alpha, fmatrix d_A, fmatrix d_B, float beta, fmatrix d_C){
  // TODO - declaration of dimGrid and dimBlock
  int rows = d_A.rows;
  int cols = d_B.cols;

  // Configure a 2D grid using TILE_WIDTH x TILE_WIDTH threads per block.
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH,
               (rows + TILE_WIDTH - 1) / TILE_WIDTH);

  matmul_tiled_kernel <<< dimGrid, dimBlock >>> (alpha, d_A.data, d_B.data, beta, d_C.data, d_A.cols, d_B.cols, d_A.rows, d_B.rows);
  gpuErrchk(cudaDeviceSynchronize());
}

/**********************/
void matrixMultiplication_cublas(float alpha, fmatrix d_A, fmatrix d_B, float beta, fmatrix d_C){
  /* TODO */
  // Initialize cuBLAS on the first call
  if (!cublas_init){
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "CUBLAS initialization failed\n");
      exit(EXIT_FAILURE);
    }
    cublas_init = 1;
  }
  // Matrix dimensions:
  //   A is m x k, B is k x n, so C is m x n.
  int m = d_A.rows;    // number of rows of A (and C)
  int n = d_B.cols;    // number of columns of B (and C)
  int k = d_A.cols;    // number of columns of A (and rows of B)

  // Leading dimensions (since matrices are in column-major order)
  int lda = m;
  int ldb = k;
  int ldc = m;

  // Call cuBLAS's sgemm routine
  cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    m, n, k,
                                    &alpha,
                                    d_A.data, lda,
                                    d_B.data, ldb,
                                    &beta,
                                    d_C.data, ldc);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS sgemm failed\n");
      exit(EXIT_FAILURE);
  }
}



/*MAIN SGEMM*/
__host__ void gen_mat_mul(float alpha, fmatrix A, fmatrix B, float beta, fmatrix C, std::string arg){
    if (arg == "cpu"){
        matrixMultiplication_basic_host(alpha, A, B, beta, C);
    } else {
      /* kernel function*/
      if (arg == "gpu_basic"){
          matrixMultiplication_basic(alpha, A, B, beta, C);

      } else if (arg == "gpu_tiled"){
          matrixMultiplication_tiled(alpha, A, B, beta, C);

      } else if (arg == "gpu_cublas"){
         matrixMultiplication_cublas(alpha, A, B, beta, C);

      } else{
          printf("Matrix Multiplication argument is Wrong");
          exit(0);
      }
      // wait for everything to finish
      device_synchronize();
    }
}

void mat_mul(fmatrix A, fmatrix B, fmatrix C, std::string arg){
 gen_mat_mul(1.0, A, B, 0.0, C, arg);
}
