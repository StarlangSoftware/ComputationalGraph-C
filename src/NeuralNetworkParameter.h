//
// Created by Olcay YILDIZ on 5.01.2026.
//

#ifndef COMPUTATIONALGRAPH_NEURALNETWORKPARAMETER_H
#define COMPUTATIONALGRAPH_NEURALNETWORKPARAMETER_H
#include "Optimizer/Optimizer.h"
#include "Initialization/Initialization.h"

struct neural_network_parameter {
    int seed;
    Optimizer_ptr optimizer;
    int epoch;
    Initialization initialization;
    double dropout;
};

typedef struct neural_network_parameter Neural_network_parameter;

typedef Neural_network_parameter* Neural_network_parameter_ptr;

Neural_network_parameter_ptr create_neural_network_parameter(int seed, Optimizer_ptr optimizer, Initialization initialization, int epoch, double dropout);

Neural_network_parameter_ptr create_neural_network_parameter2(int seed, Optimizer_ptr optimizer, int epoch);

Neural_network_parameter_ptr create_neural_network_parameter3(int seed, Optimizer_ptr optimizer, int epoch, double dropout);

#endif //COMPUTATIONALGRAPH_NEURALNETWORKPARAMETER_H
