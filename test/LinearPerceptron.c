//
// Created by Olcay YILDIZ on 24.01.2026.
//

#include "LinearPerceptron.h"
#include <Memory/Memory.h>
#include "../src/Function/SoftMax.h"
#include "../src/Node/MultiplicationNode.h"
#include "../src/Optimizer/StochasticGradientDescent.h"
#include "NeuralNet.h"
#include "../src/Initialization/RandomInitialization.h"

void train_linear_perceptron(struct computational_graph* graph, Array_list_ptr train_set, Neural_network_parameter_ptr parameters) {
    graph->get_class_labels = get_class_labels_classification;
    Optimizer_ptr optimizer = create_stochastic_gradient(0.1, 0.99);
    Multiplication_node_ptr input = create_multiplication_node(false, true, false);
    array_list_add(graph->input_nodes, input);
    int number_of_input_units_with_biased = 5;
    int number_of_classes = 3;
    double* initial_weights = random_initialization(number_of_input_units_with_biased, number_of_classes, 1);
    const int weights_shape[] = {number_of_input_units_with_biased, number_of_classes};
    Tensor_ptr weights_tensor = create_tensor(initial_weights, weights_shape, 2);
    free_(initial_weights);
    Multiplication_node_ptr w = create_multiplication_node3(true, false, weights_tensor, false);
    Multiplication_node_ptr a = add_multiplication_edge(graph, (Computational_node_ptr)input, w, false);
    Softmax_ptr softmax = create_softmax();
    graph->output_node = add_edge(graph, (Computational_node_ptr) a, softmax, false);
    /*Training*/
    int epoch = 10;
    for (int i = 0; i < epoch; i++) {
        for (int j = 0; j < train_set->size; j++) {
            Tensor_ptr instance = array_list_get(train_set, j);
            input->node.value = create_input_tensor(instance);
            Array_list_ptr calculated_classes = forward_calculation(graph);
            free_array_list(calculated_classes, free_);
            int index[1] = {instance->shape[0] - 1};
            int classes[1] = {(int) get_tensor_value(instance, index)};
            back_propagation(graph, optimizer, classes);
        }
    }
    free_(optimizer);
}

void run_linear_perceptron() {
    Array_list_ptr train_set = create_array_list();
    Array_list_ptr test_set = create_array_list();
    create_iris_dataset(train_set, test_set);
    Computational_graph_ptr graph = create_computational_graph();
    graph->train = train_linear_perceptron;
    graph->get_class_labels = get_class_labels_classification;
    graph->test = test_classification;
    graph->train(graph, train_set, NULL);
    Classification_performance_ptr performance = graph->test(graph, test_set);
    free_(performance);
    free_array_list(train_set, (void*) free_tensor);
    free_array_list(test_set, (void*) free_tensor);
    free_computational_graph(graph);
}