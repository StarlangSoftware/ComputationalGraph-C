//
// Created by Olcay YILDIZ on 5.01.2026.
//

#include "NeuralNetworkParameter.h"

#include <Memory/Memory.h>

Neural_network_parameter_ptr create_neural_network_parameter(int seed, Optimizer_ptr optimizer,
                                                             Initialization initialization, int epoch, double dropout) {
    Neural_network_parameter_ptr result = malloc_(sizeof(Neural_network_parameter));
    result->seed = seed;
    result->optimizer = optimizer;
    result->initialization = initialization;
    result->epoch = epoch;
    result->dropout = dropout;
    return result;
}

Neural_network_parameter_ptr create_neural_network_parameter2(int seed, Optimizer_ptr optimizer, int epoch) {
    Neural_network_parameter_ptr result = malloc_(sizeof(Neural_network_parameter));
    result->seed = seed;
    result->optimizer = optimizer;
    result->initialization = Random;
    result->epoch = epoch;
    result->dropout = 0.0;
    return result;
}

Neural_network_parameter_ptr create_neural_network_parameter3(int seed, Optimizer_ptr optimizer, int epoch,
    double dropout) {
    Neural_network_parameter_ptr result = malloc_(sizeof(Neural_network_parameter));
    result->seed = seed;
    result->optimizer = optimizer;
    result->initialization = Random;
    result->epoch = epoch;
    result->dropout = dropout;
    return result;
}
