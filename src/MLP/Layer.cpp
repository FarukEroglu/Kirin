#include <MLP/Layer.h>
#include <MLP/Neuron.h>

#include <vector>

Layer::Layer(size_t layer_neuron_count, size_t neuron_input_count)
{
    neuron_count = layer_neuron_count;

    for (size_t i = 0; i < layer_neuron_count; i++)
    {
        Neuron* neuron = new Neuron(neuron_input_count);
        neurons.push_back(neuron);
    }

    outputs.resize(layer_neuron_count);
}

size_t Layer::get_neuron_count()
{
    return neuron_count;
}

std::vector<double> Layer::get_outputs()
{
    return outputs;
}

std::vector<double> Layer::get_gradients()
{
    return gradients;
}

void Layer::feed_forward(std::vector<double> input)
{
    for (size_t i = 0; i < neuron_count; i++)
    {
        neurons[i]->feed_forward(input);

        outputs[i] = neurons[i]->get_output();
    }
}

void Layer::propagate_backward(std::vector<double> input, double learning_rate)
{
    for (size_t i = 0; i < neuron_count; i++)
    {
        neurons[i]->propagate_backward(input, learning_rate);

        gradients[i] = neurons[i]->get_gradient();
    }
}
