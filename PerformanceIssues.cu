// Unit-II Performance Issues: Memory model for locality, tiling for conserving memory bandwidth, handling boundary conditions, and performance considerations, simple matrix-matrix multiplication in CUDA environment

#include <stdio.h>

#define TILE_WIDTH 16

// CUDA kernel for tiled matrix multiplication
__global__ void matrixMulTiled(float *A, float *B, float *C, int numARows, int numACols, int numBCols) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Calculate row and column index of element
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0;

    // Loop over tiles needed to compute C[row][col]
    for (int t = 0; t < (numACols + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load tile from A if within bounds, else 0
        if (row < numARows && t * TILE_WIDTH + threadIdx.x < numACols)
            tileA[threadIdx.y][threadIdx.x] = A[row * numACols + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B if within bounds, else 0
        if (t * TILE_WIDTH + threadIdx.y < numACols && col < numBCols)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * numBCols + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value to C if within output bounds
    if (row < numARows && col < numBCols) {
        C[row * numBCols + col] = value;
    }
}

int main() {
    // Matrix dimensions (can be arbitrary)
    int numARows = 512;
    int numACols = 512;
    int numBRows = numACols; // must match A's cols
    int numBCols = 512;

    size_t sizeA = numARows * numACols * sizeof(float);
    size_t sizeB = numBRows * numBCols * sizeof(float);
    size_t sizeC = numARows * numBCols * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Initialize matrices
    for (int i = 0; i < numARows * numACols; i++) {
        h_A[i] = 1.0f;  // For simplicity, fill with 1.0
    }
    for (int i = 0; i < numBRows * numBCols; i++) {
        h_B[i] = 2.0f;  // Fill with 2.0
    }

    // Device memory pointers
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((numBCols + TILE_WIDTH - 1) / TILE_WIDTH, (numARows + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numARows, numACols, numBCols);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print some results for verification
    printf("C[0] = %f\n", h_C[0]);               // Should be 1.0*2.0*numACols = 1024.0 if 512 cols
    printf("C[last] = %f\n", h_C[numARows * numBCols - 1]);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
