#include <MLP/Neuron.h>
#include <Random.h>

#include <vector>

Neuron::Neuron(size_t neuron_input_count)
{
    input_count = neuron_input_count;

    for (size_t i = 0; i < neuron_input_count; i++)
    {
        double weight = random_double(-1.0, 1.0);
        input_weights.push_back(weight);
    }

    bias = random_double(-1.0, 1.0);

    output = 0.0;
}

size_t Neuron::get_input_count()
{
    return input_count;
}

double Neuron::get_output()
{
    return output;
}

double Neuron::get_gradient()
{
    return gradient;
}

void Neuron::feed_forward(std::vector<double> input)
{
    if (input.size() == input_count)
    {
        double weighted_sum = 0.0;

        for (size_t i = 0; i < input_count; i++)
        {
            weighted_sum += input[i] * input_weights[i];
        }

        output = std::max(0.0, weighted_sum + bias);
    }
}

void Neuron::propagate_backward(std::vector<double> input, double learning_rate)
{
    //TODO: Calculate weights.
}
