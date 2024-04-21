#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixMulBasic(float *A, float *B, float *C, int width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < width) && (Col < width)) {
        float Cvalue = 0;
        for (int k = 0; k < width; ++k) {
            Cvalue += A[Row * width + k] * B[k * width + Col];
        }
        C[Row * width + Col] = Cvalue;
    }
}

void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

int main() {
    int width = 1024; // Size of the width of the matrix
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
    cudaMalloc((void**)&d_C, size);


    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulBasic<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
