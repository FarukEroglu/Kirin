#include <MLP/Neuron.h>
#include <TransferFunctions.h>
#include <Random.h>

#include <vector>

Neuron::Neuron(size_t neuron_input_count)
{
    input_count = neuron_input_count;

    inputs.resize(neuron_input_count);

    gradient_input = 0.0;

    for (size_t i = 0; i < neuron_input_count; i++)
    {
        double weight = random_double(-1.0, 1.0);
        input_weights.push_back(weight);
    }

    bias = random_double(-1.0, 1.0);

    output = 0.0;

    gradient_outputs.resize(neuron_input_count);
}

size_t Neuron::get_input_count()
{
    return input_count;
}

double Neuron::get_output()
{
    return output;
}

std::vector<double> Neuron::get_gradient_outputs()
{
    return gradient_outputs;
}

void Neuron::feed_forward()
{
    double weighted_sum = 0.0;

    for (size_t i = 0; i < input_count; i++)
    {
        weighted_sum += inputs[i] * input_weights[i];
    }

    output = tanh_function(weighted_sum + bias);
}

void Neuron::propagate_backward(double learning_rate)
{
    for (size_t i = 0; i < input_count; i++)
    {
        gradient_outputs[i] = gradient_input * input_weights[i];
        input_weights[i] -= gradient_input * tanh_derivative(output) * inputs[i] * learning_rate;
    }

    bias -= gradient_input * tanh_derivative(output) * learning_rate;
}
