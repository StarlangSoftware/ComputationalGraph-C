//
// Created by Olcay YILDIZ on 24.01.2026.
//

#include <Memory/Memory.h>
#include "LinearPerceptronSingleUnit.h"
#include "NeuralNet.h"
#include "../src/Function/SoftMax.h"
#include "../src/Node/MultiplicationNode.h"
#include "../src/Optimizer/StochasticGradientDescent.h"

Array_list_ptr get_class_labels_linear_perceptron_single_point(Computational_node_ptr output_node) {
    Array_list_ptr class_indices = create_array_list();
    array_list_add_int(class_indices, 0);
    return class_indices;
}

void train_linear_perceptron_single_point(struct computational_graph* graph, Array_list_ptr train_set, Neural_network_parameter_ptr parameters) {
    Optimizer_ptr optimizer = create_stochastic_gradient(0.1, 0.99);
    Multiplication_node_ptr input = create_multiplication_node(false, true, false);
    array_list_add(graph->input_nodes, input);
    const double initial_weights[] = {1.0, 1.0, 1.0, 1.0};
    const int weights_shape[] = {2, 2};
    Tensor_ptr weights_tensor = create_tensor(initial_weights, weights_shape, 2);
    Multiplication_node_ptr w = create_multiplication_node3(true, false, weights_tensor, false);
    Multiplication_node_ptr a = add_multiplication_edge(graph, (Computational_node_ptr)input, w, false);
    Softmax_ptr softmax = create_softmax();
    Computational_node_ptr output_node = add_edge(graph, (Computational_node_ptr) a, softmax, false);
    Tensor_ptr data_tensor = array_list_get(train_set, 0);
    Tensor_ptr input1 = create_input_tensor(data_tensor);
    free_tensor(data_tensor);
    input->node.value = input1;
    Array_list_ptr calculated_classes = forward_calculation_with_dropout(graph, false);
    free_array_list(calculated_classes, free_);
    int classes[1] = {1.0};
    back_propagation(graph, optimizer, classes);
    free_computational_node(output_node);
    free_(optimizer);
    free_(softmax);
}

void run_linear_perceptron_single_point() {
    Array_list_ptr train_set = create_array_list();
    const double data1[] = {1.0, 1.0};
    const int data_shape[] = {2};
    Tensor_ptr data_tensor = create_tensor(data1, data_shape, 1);
    array_list_add(train_set, data_tensor);
    Computational_graph_ptr graph = create_computational_graph();
    graph->train = train_linear_perceptron_single_point;
    graph->get_class_labels = get_class_labels_linear_perceptron_single_point;
    graph->train(graph, train_set, NULL);
    free_array_list(train_set, NULL);
    free_computational_graph(graph);
}