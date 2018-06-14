#ifndef FILTER_H_INCLUDED
#define FILTER_H_INCLUDED

#include <vector>

class Filter
{
    public:
        Filter(size_t width, size_t height, size_t depth);

        size_t get_width();
        size_t get_height();
        size_t get_depth();

        std::vector<std::vector<std::vector<double>>> weights;
        double bias;

        double convolve(std::vector<std::vector<std::vector<double>>> input, size_t x_offset, size_t y_offset);
    private:
        size_t width;
        size_t height;
        size_t depth;
};

#endif // FILTER_H_INCLUDED
