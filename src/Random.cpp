#include "Random.h"

#include <random>
#include <ctime>

double random_double(double min_value, double max_value)
{
    double random = (double)rand() / RAND_MAX;
    return min_value + random * (max_value - min_value);
}
