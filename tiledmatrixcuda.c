#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMulTiled(float *A, float *B, float *C, int width) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Cvalue = 0;
    for (int m = 0; m < (width / TILE_WIDTH); ++m) {
        As[ty][tx] = A[Row * width + m * TILE_WIDTH + tx];
        Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * width + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[Row * width + Col] = Cvalue;
}

int main() {
    int width = 1024; // Define the size of the matrix
    size_t size = width * width * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;


    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);


    randomInit(h_A, width * width);
    randomInit(h_B, width * width);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}


