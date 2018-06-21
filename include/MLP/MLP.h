#ifndef MLP_H_INCLUDED
#define MLP_H_INCLUDED

#include <MLP/Layer.h>

#include <vector>

class MLP
{
    public:
        MLP(std::vector<size_t> network_layers);

        size_t get_input_count();
        std::vector<double> inputs;

        size_t get_layer_count();
        std::vector<Layer*> layers;

        size_t get_output_count();
        std::vector<double> get_outputs();

        void feed_forward();
        void propagate_backward(std::vector<double> expected_outputs, double learning_rate);
    private:
        size_t input_count;

        size_t layer_count;

        size_t output_count;
        std::vector<double> outputs;
};

#endif // MLP_H_INCLUDED
