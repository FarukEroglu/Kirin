#ifndef CONVLAYER_H_INCLUDED
#define CONVLAYER_H_INCLUDED

#include <ConvNet/Filter.h>

#include <vector>

class ConvLayer
{
    public:
        ConvLayer(size_t layer_filter_count, size_t layer_filter_width, size_t layer_filter_height, size_t layer_filter_depth, size_t layer_stride);

        size_t filter_count;

        size_t get_filter_width();
        size_t get_filter_height();
        size_t get_filter_depth();

        size_t stride;

        std::vector<Filter*> filters;

        std::vector<std::vector<std::vector<double>>> feed_forward(std::vector<std::vector<std::vector<double>>> input);
    private:
        size_t filter_width;
        size_t filter_height;
        size_t filter_depth;
};

#endif // CONVLAYER_H_INCLUDED
