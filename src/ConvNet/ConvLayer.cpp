#include <ConvNet/ConvLayer.h>
#include <ConvNet/Filter.h>

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
            output_vector[i].resize((input[i].size() - filter_height) / stride);

            for (size_t j = 0; j < (input[i].size() - filter_height) / stride; j++)
            {
                output_vector[i][j].resize((input[i][j].size() - filter_width) / stride);

                for (size_t k = 0; k < (input[i][j].size() - filter_width) / stride; k++)
                {
                    output_vector[i][j][k] = std::max(0.0, filters[i]->run(input, k, j));
                }
            }
        }
    }

    return output_vector;
}
