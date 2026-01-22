//
// Created by Olcay YILDIZ on 20.01.2026.
//

#include "NeuralNet.h"

#include <Memory/Memory.h>

#include "../src/ComputationalGraph.h"
#include "../src/Function/SoftMax.h"
#include "../src/Node/MultiplicationNode.h"
#include "../src/Optimizer/StochasticGradientDescent.h"

Tensor_ptr create_input_tensor(Tensor_ptr instance) {
    double* data = malloc_((instance->shape[0] - 1) * sizeof(double));
    for (int i = 0; i < instance->shape[0] - 1; i++) {
        data[i] = instance->data[i];
    }
    const int shape[2] = {1, instance->shape[0] - 1};
    return create_tensor3(data, shape, 2);
}

void train_linear_perceptron_single_point() {
    Computational_graph_ptr graph = create_computational_graph();
    Optimizer_ptr optimizer = create_stochastic_gradient(0.1, 0.99);
    Multiplication_node_ptr input = create_multiplication_node(false, true, false);
    array_list_add(graph->input_nodes, input);
    const double initial_weights[] = {1.0, 1.0, 1.0, 1.0};
    const int weights_shape[] = {2, 2};
    Tensor_ptr weights_tensor = create_tensor(initial_weights, weights_shape, 2);
    Multiplication_node_ptr w = create_multiplication_node3(true, false, weights_tensor, false);
    Multiplication_node_ptr a = add_multiplication_edge(graph, (Computational_node_ptr)input, w, false);
    add_edge(graph, (Computational_node_ptr) a, create_softmax(), false);
    const double data1[] = {1.0, 1.0};
    const int data_shape[] = {2};
    Tensor_ptr data_tensor = create_tensor(data1, data_shape, 1);
    Tensor_ptr input1 = create_input_tensor(data_tensor);
    input->node.value = input1;
    Array_list_ptr calculated_classes = forward_calculation_with_dropout(graph, false);
    int classes[1] = {data1[1]};
    back_propagation(graph, optimizer, classes);
}

int main() {
    train_linear_perceptron_single_point();
}
