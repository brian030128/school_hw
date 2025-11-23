#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel: Each thread calculates the value for one pixel
__global__ void mandel_kernel(float lower_x, float lower_y, 
                              float step_x, float step_y, 
                              int res_x, int res_y, 
                              int max_iterations, 
                              int *output)
{
    // Calculate global thread index (thisX, thisY)
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: ensure the thread is within the image dimensions
    if (thisX >= res_x || thisY >= res_y)
        return;

    // Calculate the complex coordinates for this pixel
    // using the formula requested in the prompt to avoid precision issues
    float c_re = lower_x + thisX * step_x;
    float c_im = lower_y + thisY * step_y;

    // Initialize z (based on the serial code provided)
    float z_re = c_re;
    float z_im = c_im;

    int i;
    for (i = 0; i < max_iterations; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    // Write the iteration count to the output buffer
    // Calculate 1D index from 2D coordinates
    int index = (thisY * res_x) + thisX;
    output[index] = i;
}

// Host front-end function
void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int *img,
             int res_x,
             int res_y,
             int max_iterations)
{
    // Calculate the step size per pixel
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    // Calculate total number of pixels and memory size
    int num_pixels = res_x * res_y;
    size_t mem_size = num_pixels * sizeof(int);

    // Allocate device memory
    int *d_img;
    cudaMalloc((void **)&d_img, mem_size);

    // Configure CUDA Kernel Grid and Block dimensions
    // Using 16x16 threads per block is a standard safe choice (256 threads total per block)
    dim3 blockSize(16, 16);
    dim3 gridSize((res_x + blockSize.x - 1) / blockSize.x,
                  (res_y + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, 
                                           res_x, res_y, max_iterations, d_img);

    // Check for kernel launch errors (optional but recommended)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Copy result from Device (GPU) to Host (CPU)
    // 'img' is the pointer to the host memory passed into this function
    cudaMemcpy(img, d_img, mem_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_img);
}