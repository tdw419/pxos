// Generated from GPU primitives
// pxOS Heterogeneous Computing System

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// ========== GPU Kernels ==========

__global__ void vector_add(float* a, float* b, float* c, int n) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
bool in_bounds = (tid < n);
if (in_bounds) {
    auto val_a = a[tid];
    auto val_b = b[tid];
    auto sum = val_a + val_b;
    c[tid] = sum;
}
}



// ========== Host Code ==========

int main(int argc, char** argv) {
    printf("\npxOS Heterogeneous Computing System\n");
    printf("Example: Vector Addition on GPU\n\n");

    // Query GPU
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    printf("GPU: %s\n", props.name);
    printf("Compute Capability: %d.%d\n\n", props.major, props.minor);

    // Setup test data
    const int N = 65536;  // Number of elements
    const int size = N * sizeof(float);
    printf("Vector size: %d elements (%d bytes)\n", N, size);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int blocks = 256;
    int threads = 256;
    printf("Launching kernel: %d blocks x %d threads\n", blocks, threads);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("GPU execution time: %.3f ms\n\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify results
    printf("Verifying results...\n");
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            printf("Error at index %d: expected %.2f, got %.2f\n", i, expected, h_c[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("✓ All results correct!\n");
        // Show sample results
        printf("\nSample results:\n");
        for (int i = 0; i < 5; i++) {
            printf("  %.2f + %.2f = %.2f\n", h_a[i], h_b[i], h_c[i]);
        }
        printf("  ...\n");
    }

    // Cleanup
    free(h_a); free(h_b); free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\npxOS GPU program completed successfully.\n");
    return 0;
}
