// Unit-IV Applications: Parallel convolution pattern, parallel scan pattern, parallel histogram pattern and atomic operations, data transfer and task parallelism.

#include <stdio.h>
#include <cuda.h>

#define N 1024
#define BLOCK_SIZE 256

// --- 1D Convolution kernel ---
// Simple convolution with fixed 3-element kernel: [1, 0, -1]
__global__ void convolution1D(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < n - 1) {
        output[i] = input[i-1] * 1.0f + input[i] * 0.0f + input[i+1] * -1.0f;
    }
}

// --- Parallel prefix sum (exclusive scan) ---
// Blelloch scan simplified version
__global__ void scanPrefixSum(int *input, int *output, int n) {
    __shared__ int temp[2 * BLOCK_SIZE];
    int tid = threadIdx.x;
    int start = 2 * blockIdx.x * BLOCK_SIZE;

    // Load data into shared memory
    if (start + tid < n)
        temp[tid] = input[start + tid];
    else
        temp[tid] = 0;

    if (start + BLOCK_SIZE + tid < n)
        temp[BLOCK_SIZE + tid] = input[start + BLOCK_SIZE + tid];
    else
        temp[BLOCK_SIZE + tid] = 0;

    // Up-sweep phase
    int offset = 1;
    for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Clear last element
    if (tid == 0) {
        temp[2 * BLOCK_SIZE - 1] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d <= BLOCK_SIZE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to output array
    if (start + tid < n)
        output[start + tid] = temp[tid];
    if (start + BLOCK_SIZE + tid < n)
        output[start + BLOCK_SIZE + tid] = temp[BLOCK_SIZE + tid];
}

// --- Parallel histogram with atomic operations ---
__global__ void histogramAtomic(int *input, int *hist, int n, int numBins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = input[i] % numBins; // Simple binning by modulo
        atomicAdd(&hist[bin], 1);
    }
}

int main() {
    // --- Data for convolution ---
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 10); // Simple pattern
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    convolution1D<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Convolution output sample: ");
    for (int i = 495; i < 505; i++) printf("%.1f ", h_output[i]);
    printf("\n");

    // --- Data for scan ---
    int *h_scan_in = (int*)malloc(N * sizeof(int));
    int *h_scan_out = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_scan_in[i] = 1; // All ones for prefix sum

    int *d_scan_in, *d_scan_out;
    cudaMalloc(&d_scan_in, N * sizeof(int));
    cudaMalloc(&d_scan_out, N * sizeof(int));
    cudaMemcpy(d_scan_in, h_scan_in, N * sizeof(int), cudaMemcpyHostToDevice);

    scanPrefixSum<<<(N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE), BLOCK_SIZE>>>(d_scan_in, d_scan_out, N);
    cudaMemcpy(h_scan_out, d_scan_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Prefix sum output sample: ");
    for (int i = 490; i < 500; i++) printf("%d ", h_scan_out[i]);
    printf("\n");

    // --- Data for histogram ---
    int numBins = 10;
    int *h_hist = (int*)calloc(numBins, sizeof(int));
    int *d_hist, *d_hist_in;
    cudaMalloc(&d_hist, numBins * sizeof(int));
    cudaMalloc(&d_hist_in, N * sizeof(int));
    cudaMemcpy(d_hist_in, h_scan_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, numBins * sizeof(int));

    histogramAtomic<<<(N + BLOCK_SIZE -1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_hist_in, d_hist, N, numBins);
    cudaMemcpy(h_hist, d_hist, numBins * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Histogram bins:\n");
    for (int i = 0; i < numBins; i++) {
        printf("Bin %d: %d\n", i, h_hist[i]);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scan_in);
    cudaFree(d_scan_out);
    cudaFree(d_hist);
    cudaFree(d_hist_in);
    free(h_input);
    free(h_output);
    free(h_scan_in);
    free(h_scan_out);
    free(h_hist);

    return 0;
}
