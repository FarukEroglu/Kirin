#include <MLP/MLP.h>
#include <MLP/Layer.h>
#include <TransferFunctions.h>

#include <vector>

MLP::MLP(std::vector<size_t> network_layers)
{
    input_count = network_layers[0];
    output_count = network_layers[network_layers.size() - 1];

    inputs.resize(network_layers[0]);

    layer_count = network_layers.size() - 1;

    for (size_t i = 1; i < network_layers.size(); i++)
    {
        Layer* layer = new Layer(network_layers[i], network_layers[i - 1]);
        layers.push_back(layer);
    }

    outputs.resize(network_layers[network_layers.size() - 1]);
}

size_t MLP::get_input_count()
{
    return input_count;
}

size_t MLP::get_output_count()
{
    return output_count;
}

size_t MLP::get_layer_count()
{
    return layer_count;
}

std::vector<double> MLP::get_outputs()
{
    return outputs;
}

void MLP::feed_forward()
{
    layers[0]->inputs = inputs;
    layers[0]->feed_forward();

    for (size_t i = 1; i < layer_count; i++)
    {
        layers[i]->inputs = layers[i - 1]->get_outputs();
        layers[i]->feed_forward();
    }

    outputs = layers[layer_count - 1]->get_outputs();
}

void MLP::propagate_backward(std::vector<double> expected_outputs, double learning_rate)
{
    feed_forward();

    std::vector<double> network_outputs = get_outputs();

    std::vector<double> gradient_inputs;

    gradient_inputs.resize(output_count);

    for (size_t i = 0; i < output_count; i++)
    {
        gradient_inputs[i] = network_outputs[i] - expected_outputs[i];
    }

    layers[layer_count - 1]->gradient_inputs = gradient_inputs;
    layers[layer_count - 1]->propagate_backward(learning_rate);

    for (size_t i = layer_count - 2; i > (size_t) -1; i--)
    {
        layers[i]->gradient_inputs = layers[i + 1]->get_gradient_outputs();
        layers[i]->propagate_backward(learning_rate);
    }
}
