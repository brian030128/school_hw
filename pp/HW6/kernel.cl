
__kernel void convolution(__global const float *input_image,
                          __global float *output_image,
                          __global const float *filter,
                          int filter_width,
                          int image_width,
                          int image_height)
{
    // Get global position in 2D
    int j = get_global_id(0);  // column
    int i = get_global_id(1);  // row
    
    // Boundary check
    if (i >= image_height || j >= image_width)
        return;
    
    int halffilter_size = filter_width / 2;
    float sum = 0.0f;
    
    // Apply convolution filter
    for (int k = -halffilter_size; k <= halffilter_size; k++)
    {
        for (int l = -halffilter_size; l <= halffilter_size; l++)
        {
            int row = i + k;
            int col = j + l;
            
            // Check bounds
            if (row >= 0 && row < image_height && col >= 0 && col < image_width)
            {
                int image_idx = row * image_width + col;
                int filter_idx = (k + halffilter_size) * filter_width + (l + halffilter_size);
                sum += input_image[image_idx] * filter[filter_idx];
            }
        }
    }
    
    output_image[i * image_width + j] = sum;
}