#include <TransferFunctions.h>

#include <cmath>

double tanh_function(double input)
{
    return std::tanh(input);
}

double tanh_derivative(double input)
{
    return 1 - std::pow(std::tanh(input), 2);
}
