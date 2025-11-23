#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel: Grid-Stride Loop implementation
// Each thread processes multiple pixels by stepping through the image
__global__ void mandel_kernel(float lower_x, float lower_y, 
                              float step_x, float step_y, 
                              int res_x, int res_y, 
                              int max_iterations, 
                              size_t pitch, 
                              int *output)
{
    // 1. Determine where this thread starts
    int startX = blockIdx.x * blockDim.x + threadIdx.x;
    int startY = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Determine the step size (stride)
    // The thread will jump this many pixels after finishing one
    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    // 3. Loop over the image (Grid-Stride)
    // If the image is larger than the grid, the loops will execute multiple times.
    for (int y = startY; y < res_y; y += strideY)
    {
        // Get the pointer to the start of this specific row
        // We use (char*) logic because 'pitch' is in bytes
        int* row_ptr = (int*)((char*)output + y * pitch);

        for (int x = startX; x < res_x; x += strideX)
        {
            // --- Standard Mandelbrot Math ---
            float c_re = lower_x + x * step_x;
            float c_im = lower_y + y * step_y;

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

            // Write result to pitched global memory
            // No need for y calculation here, we already have the row pointer
            row_ptr[x] = i;
        }
    }
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

    // 1. Allocate Device Memory (Pitched)
    int *d_img;
    size_t pitch; 
    size_t width_in_bytes = res_x * sizeof(int);
    cudaMallocPitch((void **)&d_img, &pitch, width_in_bytes, res_y);

    // 2. Allocate Host Memory (Pinned/Page-locked)
    // This allows for faster transfer speeds (PCIe bandwidth saturation)
    int *h_pinned;
    cudaHostAlloc((void **)&h_pinned, width_in_bytes * res_y, cudaHostAllocDefault);

    // 3. Configure Grid Dimensions (Fixed Size / "Group" Strategy)
    // Instead of calculating grid size based on image size, we pick a fixed 
    // amount of hardware to utilize. 
    
    // Block size: 16x16 = 256 threads per block
    dim3 blockSize(16, 16); 

    // Grid size: We will fix the grid to ensure each thread processes a "group" of pixels.
    // For example, if we define a 32x32 grid of blocks, we have ~262k threads.
    // If the image is 1600x1200 (~1.92m pixels), each thread will process 
    // approximately 7-8 pixels (1.92m / 262k).
    dim3 gridSize(32, 32);

    // Launch Kernel
    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, 
                                           res_x, res_y, max_iterations, 
                                           pitch, d_img);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 4. Copy from Device (Pitched) to Host Pinned (Linear)
    // This un-pads the memory automatically.
    cudaMemcpy2D(h_pinned,       // Dst (Host)
                 width_in_bytes, // Dst Pitch (Pack tightly)
                 d_img,          // Src (GPU)
                 pitch,          // Src Pitch (Padded)
                 width_in_bytes, // Width to copy
                 res_y,          // Height to copy
                 cudaMemcpyDeviceToHost);

    // 5. Final copy to user buffer
    // Since 'img' comes from outside, we can't force it to be pinned, 
    // so we must do this final CPU copy.
    memcpy(img, h_pinned, width_in_bytes * res_y);

    // 6. Cleanup
    cudaFree(d_img);
    cudaFreeHost(h_pinned);
}