#ifndef MLP_H_INCLUDED
#define MLP_H_INCLUDED

#include <MLP/Layer.h>

#include <vector>

class MLP
{
    public:
        MLP(std::vector<size_t> network_layers, double learning_rate);

        size_t get_input_count();
        size_t get_output_count();

        std::vector<Layer*> layers;

        std::vector<double> feed_forward(std::vector<double> input);
        void propagate_backward(std::vector<double> input, std::vector<double> expected_output, double learning_rate);
    private:
        size_t input_count;
        size_t output_count;
};

#endif // MLP_H_INCLUDED
