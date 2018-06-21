#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED

#include <vector>

class Neuron
{
    public:
        Neuron(size_t neuron_input_count);

        size_t get_input_count();
        std::vector<double> inputs;

        double gradient_input;

        std::vector<double> input_weights;
        double bias;

        double get_output();

        std::vector<double> get_gradient_outputs();

        void feed_forward();
        void propagate_backward(double learning_rate);
    private:
        size_t input_count;

        double output;

        std::vector<double> gradient_outputs;
};

#endif // NEURON_H_INCLUDED
