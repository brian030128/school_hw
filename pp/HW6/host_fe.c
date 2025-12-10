#include "host_fe.h"
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

void host_fe(int filter_width,
             float *filter,
             int image_height,
             int image_width,
             float *input_image,
             float *output_image,
             cl_device_id *device,
             cl_context *context,
             cl_program *program)
{
    cl_int status;
    int filter_size = filter_width * filter_width;
    int image_size = image_height * image_width;
    
    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);
    
    // Create memory buffers
    cl_mem filter_buf = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                       filter_size * sizeof(float), filter, &status);
    cl_mem input_buf = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                      image_size * sizeof(float), input_image, &status);
    cl_mem output_buf = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, 
                                       image_size * sizeof(float), NULL, &status);
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    
    // Set kernel arguments
    int halffilter = filter_width / 2;
    status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_buf);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &filter_width);
    status |= clSetKernelArg(kernel, 4, sizeof(int), &image_width);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &image_height);
    
    // Local memory size for tile (16x16 tile + halo region)
    int tile_size = 16;
    int padded_tile = tile_size + 2 * halffilter;
    size_t local_mem_size = padded_tile * padded_tile * sizeof(float);
    status |= clSetKernelArg(kernel, 6, local_mem_size, NULL);
    
    // Execute kernel with local work size
    size_t local_work_size[2] = {(size_t)tile_size, (size_t)tile_size};
    size_t global_work_size[2] = {
        ((image_width + tile_size - 1) / tile_size) * tile_size,
        ((image_height + tile_size - 1) / tile_size) * tile_size
    };
    
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, 
                                    local_work_size, 0, NULL, NULL);
    
    // Wait for completion
    clFinish(queue);
    
    // Read results
    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, 
                                 image_size * sizeof(float), output_image, 0, NULL, NULL);
    
    // Cleanup
    clReleaseMemObject(filter_buf);
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
}