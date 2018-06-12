#include <ConvNet/ConvNet.h>
#include <ConvNet/ConvLayer.h>

ConvNet::ConvNet(std::vector<std::vector<size_t>> network_layers, size_t network_input_channels)
{
    input_channels = network_input_channels;

    ConvLayer* input_layer = new ConvLayer(network_layers[0][0], network_layers[0][1], network_layers[0][2], network_input_channels, network_layers[0][3]);
    layers.push_back(input_layer);

    for (size_t i = 1; i < network_layers.size(); i++)
    {
        ConvLayer* layer = new ConvLayer(network_layers[i][0], network_layers[i][1], network_layers[i][2], layers[i - 1]->filter_count, network_layers[i][3]);
        layers.push_back(layer);
    }
}

size_t ConvNet::get_input_channels()
{
    return input_channels;
}

std::vector<std::vector<std::vector<double>>> ConvNet::feed_forward(std::vector<std::vector<std::vector<double>>> input)
{
    std::vector<std::vector<std::vector<double>>> output_vector = input;

    for (size_t i = 0; i < layers.size(); i++)
    {
        output_vector = layers[i]->feed_forward(output_vector);
    }

    return output_vector;
}
