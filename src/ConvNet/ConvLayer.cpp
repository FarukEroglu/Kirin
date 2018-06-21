#include <ConvNet/ConvLayer.h>
#include <ConvNet/Filter.h>
#include <TransferFunctions.h>

#include <vector>

ConvLayer::ConvLayer(size_t layer_filter_count, size_t layer_filter_width, size_t layer_filter_height, size_t layer_filter_depth, size_t layer_stride)
{
    filter_count = layer_filter_count;

    filter_width = layer_filter_width;
    filter_height = layer_filter_height;
    filter_depth = layer_filter_depth;

    stride = layer_stride;

    for (size_t i = 0; i < layer_filter_count; i++)
    {
        Filter* filter = new Filter(layer_filter_width, layer_filter_height, layer_filter_depth);
        filters.push_back(filter);
    }
}

std::vector<std::vector<std::vector<double>>> ConvLayer::feed_forward(std::vector<std::vector<std::vector<double>>> input)
{
    std::vector<std::vector<std::vector<double>>> output_vector;

    if (input.size() == filter_depth)
    {
        output_vector.resize(filter_count);

        for (size_t i = 0; i < filter_count; i++)
        {
            output_vector[i].resize((input[0].size() - filter_height) / stride + 1);

            for (size_t j = 0; j < (input[0].size() - filter_height) / stride + 1; j++)
            {
                output_vector[i][j].resize((input[0][0].size() - filter_width) / stride + 1);

                for (size_t k = 0; k < (input[0][0].size() - filter_width) / stride + 1; k++)
                {
                    output_vector[i][j][k] = tanh_function(filters[i]->convolve(input, k, j));
                }
            }
        }
    }

    return output_vector;
}
