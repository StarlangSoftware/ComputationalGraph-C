//
// Created by Olcay YILDIZ on 25.01.2026.
//

#ifndef COMPUTATIONALGRAPH_DEEPNETWORK_H
#define COMPUTATIONALGRAPH_DEEPNETWORK_H

#include <ArrayList.h>
#include "../src/NeuralNetworkParameter.h"
#include "../src/ComputationalGraph.h"

void train_deep_network(struct computational_graph* graph, Array_list_ptr train_set, Neural_network_parameter_ptr parameters);

void run_deep_network();

#endif //COMPUTATIONALGRAPH_DEEPNETWORK_H