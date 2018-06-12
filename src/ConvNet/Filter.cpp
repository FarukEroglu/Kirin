#include <ConvNet/Filter.h>
#include <Random.h>

#include <vector>

Filter::Filter(size_t filter_width, size_t filter_height, size_t filter_depth)
{
    width = filter_width;
    height = filter_height;
    depth = filter_depth;

    weights.resize(filter_depth);

    for (size_t i = 0; i < filter_depth; i++)
    {
        weights[i].resize(filter_height);

        for (size_t j = 0; j < filter_height; j++)
        {
            weights[i][j].resize(filter_width);

            for (size_t k = 0; k < filter_width; k++)
            {
                weights[i][j][k] = random_double(-1.0, 1.0);
            }
        }
    }
}

size_t Filter::get_width()
{
    return width;
}

size_t Filter::get_height()
{
    return height;
}

size_t Filter::get_depth()
{
    return depth;
}

double Filter::run(std::vector<std::vector<std::vector<double>>> input, size_t x_offset, size_t y_offset)
{
    double output_value = 0.0;

    for (size_t i = 0; i < depth; i++)
    {
        for (size_t j = 0; j < height; j++)
        {
            for (size_t k = 0; k < width; k++)
            {
                output_value += input[i][y_offset + j][x_offset + k] * weights[i][j][k];
            }
        }
    }

    return output_value;
}
