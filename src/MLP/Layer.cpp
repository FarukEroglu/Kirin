#include <MLP/Layer.h>
#include <MLP/Neuron.h>

#include <vector>

Layer::Layer(size_t layer_neuron_count, size_t neuron_input_count)
{
    input_count = neuron_input_count;

    inputs.resize(neuron_input_count);

    gradient_inputs.resize(layer_neuron_count);

    neuron_count = layer_neuron_count;

    for (size_t i = 0; i < layer_neuron_count; i++)
    {
        Neuron* neuron = new Neuron(neuron_input_count);
        neurons.push_back(neuron);
    }

    outputs.resize(layer_neuron_count);

    gradient_outputs.resize(neuron_input_count);
}

size_t Layer::get_input_count()
{
    return input_count;
}

size_t Layer::get_neuron_count()
{
    return neuron_count;
}

std::vector<double> Layer::get_outputs()
{
    return outputs;
}

std::vector<double> Layer::get_gradient_outputs()
{
    return gradient_outputs;
}

void Layer::feed_forward()
{
    for (size_t i = 0; i < neuron_count; i++)
    {
        neurons[i]->inputs = inputs;
        neurons[i]->feed_forward();

        outputs[i] = neurons[i]->get_output();
    }
}

void Layer::propagate_backward(double learning_rate)
{
    std::vector<double> gradient_sums;

    gradient_sums.resize(input_count);

    for (size_t i = 0; i < neuron_count; i++)
    {
        neurons[i]->gradient_input = gradient_inputs[i];
        neurons[i]->propagate_backward(learning_rate);

        for (size_t j = 0; j < input_count; j++)
        {
            std::vector<double> neuron_gradient_outputs = neurons[i]->get_gradient_outputs();
            gradient_sums[j] += neuron_gradient_outputs[j];
        }
    }

    gradient_outputs = gradient_sums;
}
