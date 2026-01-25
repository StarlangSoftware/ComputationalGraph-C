//
// Created by Olcay YILDIZ on 24.01.2026.
//

#ifndef COMPUTATIONALGRAPH_LINEARPERCEPTRONSINGLEUNIT_H
#define COMPUTATIONALGRAPH_LINEARPERCEPTRONSINGLEUNIT_H

#include "../src/ComputationalGraph.h"
#include "../src/Node/ConcatenatedNode.h"
#include "../src/NeuralNetworkParameter.h"

Array_list_ptr get_class_labels_linear_perceptron_single_point(Computational_node_ptr output_node);

void train_linear_perceptron_single_point(struct computational_graph* graph, Array_list_ptr train_set, Neural_network_parameter_ptr parameters);

void run_linear_perceptron_single_point();

#endif //COMPUTATIONALGRAPH_LINEARPERCEPTRONSINGLEUNIT_H