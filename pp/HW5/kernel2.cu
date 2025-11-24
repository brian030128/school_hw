#include <cstdio>
#include <cstdlib>
#include <cstring> // Added for memcpy
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel: Each thread calculates the value for one pixel
// Modified to accept 'pitch' for handling padded memory
__global__ void mandel_kernel(float lower_x, float lower_y, 
                              float step_x, float step_y, 
                              int res_x, int res_y, 
                              int max_iterations, 
                              size_t pitch, // Added pitch argument
                              int *output)
{
    // Calculate global thread index (thisX, thisY)
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: ensure the thread is within the image dimensions
    if (thisX >= res_x || thisY >= res_y)
        return;

    // Calculate the complex coordinates for this pixel
    float c_re = lower_x + thisX * step_x;
    float c_im = lower_y + thisY * step_y;

    // Initialize z
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

    // Write the iteration count to the output buffer.
    // Because we used cudaMallocPitch, the rows are padded.
    // 'pitch' is the width of a row in bytes.
    
    // 1. Cast output to char* to perform byte-wise arithmetic
    // 2. Add the offset for the current row (thisY * pitch)
    // 3. Cast back to int* to access the specific pixel in that row
    int* row_ptr = (int*)((char*)output + thisY * pitch);
    
    row_ptr[thisX] = i;
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

    // Allocate Device Memory using cudaMallocPitch
    // This ensures rows are aligned in memory for faster access
    int *d_img;
    size_t pitch; // The actual width of the row in bytes (including padding)
    size_t width_in_bytes = res_x * sizeof(int);
    
    cudaMallocPitch((void **)&d_img, &pitch, width_in_bytes, res_y);

    // Allocate Host Memory using cudaHostAlloc (Pinned Memory)
    // We allocate a temporary buffer because 'img' passed to this function 
    // might not be pinned.
    int *h_pinned;
    cudaHostAlloc((void **)&h_pinned, width_in_bytes * res_y, cudaHostAllocDefault);

    // Configure CUDA Kernel Grid and Block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((res_x + blockSize.x - 1) / blockSize.x,
                  (res_y + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    // We pass 'pitch' so the kernel knows how to calculate row offsets
    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, 
                                           res_x, res_y, max_iterations, 
                                           pitch, d_img);

    // Copy result from Device (GPU) to Host Pinned Memory
    // We use cudaMemcpy2D because the device memory is pitched (padded),
    // but the host memory is linear (packed).
    cudaMemcpy2D(h_pinned,       // Destination pointer (host)
                 width_in_bytes, // Destination pitch (width of linear row in bytes)
                 d_img,          // Source pointer (device)
                 pitch,          // Source pitch (actual width of device row in bytes)
                 width_in_bytes, // Width of matrix in bytes
                 res_y,          // Height of matrix
                 cudaMemcpyDeviceToHost);

    // Copy from our internal pinned buffer to the output buffer provided by the caller
    memcpy(img, h_pinned, width_in_bytes * res_y);

    // Free memory
    cudaFree(d_img);         // Free device memory
    cudaFreeHost(h_pinned);  // Free pinned host memory
}