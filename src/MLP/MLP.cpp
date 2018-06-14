#include <MLP/MLP.h>
#include <MLP/Layer.h>

#include <vector>

MLP::MLP(std::vector<size_t> network_layers, double learning_rate)
{
    input_count = network_layers[0];
    output_count = network_layers[network_layers.size() - 1];

    for (size_t i = 1; i < network_layers.size(); i++)
    {
        Layer* layer = new Layer(network_layers[i], network_layers[i - 1]);
        layers.push_back(layer);
    }
}

size_t MLP::get_input_count()
{
    return input_count;
}

size_t MLP::get_output_count()
{
    return output_count;
}

std::vector<double> MLP::feed_forward(std::vector<double> input)
{
    std::vector<double> output_vector;

    if (input.size() == input_count)
    {
        layers[0]->feed_forward(input);

        for (size_t i = 1; i < layers.size(); i++)
        {
            layers[i]->feed_forward(layers[i - 1]->get_outputs());
        }

        output_vector = layers[layers.size() - 1]->get_outputs();
    }

    return output_vector;
}

void MLP::propagate_backward(std::vector<double> input, std::vector<double> expected_output, double learning_rate)
{
    //TODO: Calculate gradients for output layer.

    for (size_t i = layers.size() - 2; i > (size_t) -1; i--)
    {
        layers[i]->propagate_backward(layers[i + 1]->get_gradients(), learning_rate);
    }
}
