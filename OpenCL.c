// Unit-III Introduction to OpenCL, operations such as vector addition using streams

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char* kernelSource = 
"__kernel void vectorAdd(__global const float* A, __global const float* B, __global float* C, int N) { \n"
"    int i = get_global_id(0); \n"
"    if (i < N) { \n"
"        C[i] = A[i] + B[i]; \n"
"    } \n"
"} \n";

int main() {
    int N = 1024;
    size_t dataSize = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(dataSize);
    float *h_B = (float*)malloc(dataSize);
    float *h_C = (float*)malloc(dataSize);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // 1. Get platform and device information
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // 2. Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // 3. Create command queue (stream)
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // 4. Create memory buffers on device
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, NULL);

    // 5. Copy data from host to device
    clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, dataSize, h_A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, dataSize, h_B, 0, NULL, NULL);

    // 6. Create program from kernel source
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);

    // 7. Build program
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // 8. Create kernel
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", NULL);

    // 9. Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    // 10. Execute kernel
    size_t globalSize = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // 11. Read back results
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, dataSize, h_C, 0, NULL, NULL);

    // 12. Verify results
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = 0;
            printf("Error at index %d: %f != %f + %f\n", i, h_C[i], h_A[i], h_B[i]);
            break;
        }
    }

    if (correct) {
        printf("Vector addition successful!\n");
    }

    // Cleanup
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
