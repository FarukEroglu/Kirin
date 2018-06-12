#ifndef CONVNET_H_INCLUDED
#define CONVNET_H_INCLUDED

#include <ConvNet/ConvLayer.h>

#include <vector>

class ConvNet
{
    public:
        ConvNet(std::vector<std::vector<size_t>> network_layers, size_t network_input_channels);

        size_t get_input_channels();

        std::vector<ConvLayer*> layers;

        std::vector<std::vector<std::vector<double>>> feed_forward(std::vector<std::vector<std::vector<double>>> input);
    private:
        size_t input_channels;
};

#endif // CONVNET_H_INCLUDED
