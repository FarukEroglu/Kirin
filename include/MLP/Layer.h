#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include <MLP/Neuron.h>

#include <vector>

class Layer
{
    public:
        Layer(size_t neuron_count, size_t neuron_input_count);

        size_t get_neuron_count();

        std::vector<Neuron*> neurons;

        std::vector<double> get_outputs();

        std::vector<double> get_gradients();

        void feed_forward(std::vector<double> input);
        void propagate_backward(std::vector<double> input, double learning_rate);
    private:
        size_t neuron_count;

        std::vector<double> outputs;

        std::vector<double> gradients;
};

#endif // LAYER_H_INCLUDED
