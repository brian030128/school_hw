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
    // Calculate global thread index
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (thisX >= res_x || thisY >= res_y)
        return;

    // Calculate 1D index from 2D coordinates
    int index = (thisY * res_x) + thisX;

    // Calculate the complex coordinates for this pixel
    float c_re = lower_x + thisX * step_x;
    float c_im = lower_y + thisY * step_y;

    // --- EARLY STOPPING 1: Geometric Checks ---
    // Check if point is in the Main Cardioid
    // Formula: q * (q + (x - 1/4)) < 1/4 * y^2, where q = (x - 1/4)^2 + y^2
    float x_minus_quarter = c_re - 0.25f;
    float y_sq = c_im * c_im;
    float q = x_minus_quarter * x_minus_quarter + y_sq;
    if (q * (q + x_minus_quarter) < 0.25f * y_sq)
    {
        output[index] = max_iterations;
        return;
    }

    // Check if point is in the Period-2 Bulb
    // Formula: (x + 1)^2 + y^2 < 1/16
    float x_plus_one = c_re + 1.0f;
    if (x_plus_one * x_plus_one + y_sq < 0.0625f)
    {
        output[index] = max_iterations;
        return;
    }
    // ------------------------------------------

    // Initialize z
    float z_re = c_re;
    float z_im = c_im;

    // Variables for Periodicity Checking (Cycle Detection)
    float old_re = z_re;
    float old_im = z_im;
    int period = 20; // Start checking after a few iterations
    int period_counter = 0;

    int i;
    for (i = 0; i < max_iterations; ++i)
    {
        // 1. Divergence Check (Standard)
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        // 2. Calculation
        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;

        // --- EARLY STOPPING 2: Periodicity Check ---
        // If the current Z is identical to a previous Z, we are in a loop (inside the set).
        if (z_re == old_re && z_im == old_im)
        {
            i = max_iterations; // Set to max to indicate membership
            break;
        }

        // Update the "history" value using period doubling strategy.
        // We verify against the old value for 'period' iterations, 
        // then update the old value and double the period.
        period_counter++;
        if (period_counter >= period)
        {
            old_re = z_re;
            old_im = z_im;
            period_counter = 0;
            period *= 2; // Double the check interval
        }
    }

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
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    int num_pixels = res_x * res_y;
    size_t mem_size = num_pixels * sizeof(int);

    // Allocate device memory
    int *d_img;
    cudaMalloc((void **)&d_img, mem_size);

    dim3 blockSize(8, 8);
    dim3 gridSize((res_x + blockSize.x - 1) / blockSize.x,
                  (res_y + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, 
                                           res_x, res_y, max_iterations, d_img);

    cudaMemcpy(img, d_img, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
}
