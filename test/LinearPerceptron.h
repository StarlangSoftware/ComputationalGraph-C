//
// Created by Olcay YILDIZ on 24.01.2026.
//

#ifndef COMPUTATIONALGRAPH_LINEARPERCEPTRON_H
#define COMPUTATIONALGRAPH_LINEARPERCEPTRON_H
#include <ArrayList.h>
#include "../src/ComputationalGraph.h"
#include "../src/NeuralNetworkParameter.h"

void train_linear_perceptron(struct computational_graph* graph, Array_list_ptr train_set, Neural_network_parameter_ptr parameters);

void run_linear_perceptron();

#endif //COMPUTATIONALGRAPH_LINEARPERCEPTRON_H