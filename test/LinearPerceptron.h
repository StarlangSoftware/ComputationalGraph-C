//
// Created by Olcay YILDIZ on 24.01.2026.
//

#ifndef COMPUTATIONALGRAPH_LINEARPERCEPTRON_H
#define COMPUTATIONALGRAPH_LINEARPERCEPTRON_H
#include <ArrayList.h>
#include "../src/NeuralNetworkParameter.h"
#include "../src/ComputationalGraph.h"

void train_linear_perceptron(struct computational_graph* graph, Array_list_ptr train_set, Neural_network_parameter_ptr parameters);

Array_list_ptr get_class_labels_linear_perceptron_single_point(Computational_node_ptr output_node);

Array_list_ptr get_class_labels_linear_perceptron(Computational_node_ptr output_node);

void create_dataset(Array_list_ptr train_set, Array_list_ptr test_set);

void run_linear_perceptron();

#endif //COMPUTATIONALGRAPH_LINEARPERCEPTRON_H