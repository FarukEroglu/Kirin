#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include <MLP/Neuron.h>

#include <vector>

class Layer
{
    public:
        Layer(size_t neuron_count, size_t neuron_input_count);

        size_t get_input_count();
        std::vector<double> inputs;

        std::vector<double> gradient_inputs;

        size_t get_neuron_count();
        std::vector<Neuron*> neurons;

        std::vector<double> get_outputs();

        std::vector<double> get_gradient_outputs();

        void feed_forward();
        void propagate_backward(double learning_rate);
    private:
        size_t input_count;

        size_t neuron_count;

        std::vector<double> outputs;

        std::vector<double> gradient_outputs;
};

#endif // LAYER_H_INCLUDED
