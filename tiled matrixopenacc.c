#include <stdio.h>
#include <stdlib.h>

void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

int main() {
    int N = 1024; // Matrix size
    int TILE_SIZE = 16; // Tile size, must divide N
    size_t size = N * N * sizeof(float);
    float *A, *B, *C;

    A = (float*) malloc(size);
    B = (float*) malloc(size);
    C = (float*) malloc(size);

    randomInit(A, N*N);
    randomInit(B, N*N);

    #pragma acc data copyin(A[0:N*N], B[0:N*N]) copy(C[0:N*N])
    {
        #pragma acc kernels
        for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
            for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
                for (int i = i0; i < min(i0 + TILE_SIZE, N); ++i) {
                    for (int j = j0; j < min(j0 + TILE_SIZE, N); ++j) {
                        float sum = 0;
                        for (int k = 0; k < N; ++k) {
                            sum += A[i*N + k] * B[k*N + j];
                        }
                        C[i*N + j] = sum;
                    }
                }
            }
        }
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
