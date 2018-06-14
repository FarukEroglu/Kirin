#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#include <vector>

class Neuron
{
    public:
        Neuron(size_t neuron_input_count);

        size_t get_input_count();

        std::vector<double> input_weights;
        double bias;

        double get_output();

        double get_gradient();

        void feed_forward(std::vector<double> input);
        void propagate_backward(std::vector<double> input, double learning_rate);
    private:
        size_t input_count;

        double output;

        double gradient;
};

#endif // NEURON_H_INCLUDED
