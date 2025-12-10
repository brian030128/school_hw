__kernel void convolution(__global const float *input_image,
                          __global float *output_image,
                          __global const float *filter,
                          int filter_width,
                          int image_width,
                          int image_height,
                          __local float *tile)
{
    int global_col = get_global_id(0);
    int global_row = get_global_id(1);
    
    int local_col = get_local_id(0);
    int local_row = get_local_id(1);
    
    int tile_size = get_local_size(0);
    int halffilter = filter_width / 2;
    int padded_tile = tile_size + 2 * halffilter;
    
    // Load tile data into local memory (including halo region)
    // Each work item loads multiple elements if needed
    for (int row = local_row; row < padded_tile; row += tile_size)
    {
        for (int col = local_col; col < padded_tile; col += tile_size)
        {
            int image_row = global_row - local_row + row - halffilter;
            int image_col = global_col - local_col + col - halffilter;
            
            float value = 0.0f;
            if (image_row >= 0 && image_row < image_height && 
                image_col >= 0 && image_col < image_width)
            {
                value = input_image[image_row * image_width + image_col];
            }
            tile[row * padded_tile + col] = value;
        }
    }
    
    // Synchronize to ensure all tile data is loaded
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute convolution if within bounds
    if (global_row < image_height && global_col < image_width)
    {
        float sum = 0.0f;
        
        // Convolve using local memory
        for (int k = -halffilter; k <= halffilter; k++)
        {
            for (int l = -halffilter; l <= halffilter; l++)
            {
                int tile_row = local_row + halffilter + k;
                int tile_col = local_col + halffilter + l;
                int filter_idx = (k + halffilter) * filter_width + (l + halffilter);
                
                sum += tile[tile_row * padded_tile + tile_col] * filter[filter_idx];
            }
        }
        
        output_image[global_row * image_width + global_col] = sum;
    }
}