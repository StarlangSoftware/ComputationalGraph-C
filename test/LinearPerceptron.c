//
// Created by Olcay YILDIZ on 24.01.2026.
//

#include "LinearPerceptron.h"

#include <Memory/Memory.h>

#include "LinearPerceptronSingleUnit.h"
#include "../src/ComputationalGraph.h"
#include "../src/Function/SoftMax.h"
#include "../src/Node/MultiplicationNode.h"
#include "../src/Optimizer/StochasticGradientDescent.h"
#include "../src/Initialization/RandomInitialization.h"

static double iris_data[150][5] = {{5.1,3.5,1.4,0.2,0},
{4.9,3.0,1.4,0.2,0},
{4.7,3.2,1.3,0.2,0},
{4.6,3.1,1.5,0.2,0},
{5.0,3.6,1.4,0.2,0},
{5.4,3.9,1.7,0.4,0},
{4.6,3.4,1.4,0.3,0},
{5.0,3.4,1.5,0.2,0},
{4.4,2.9,1.4,0.2,0},
{4.9,3.1,1.5,0.1,0},
{5.4,3.7,1.5,0.2,0},
{4.8,3.4,1.6,0.2,0},
{4.8,3.0,1.4,0.1,0},
{4.3,3.0,1.1,0.1,0},
{5.8,4.0,1.2,0.2,0},
{5.7,4.4,1.5,0.4,0},
{5.4,3.9,1.3,0.4,0},
{5.1,3.5,1.4,0.3,0},
{5.7,3.8,1.7,0.3,0},
{5.1,3.8,1.5,0.3,0},
{5.4,3.4,1.7,0.2,0},
{5.1,3.7,1.5,0.4,0},
{4.6,3.6,1.0,0.2,0},
{5.1,3.3,1.7,0.5,0},
{4.8,3.4,1.9,0.2,0},
{5.0,3.0,1.6,0.2,0},
{5.0,3.4,1.6,0.4,0},
{5.2,3.5,1.5,0.2,0},
{5.2,3.4,1.4,0.2,0},
{4.7,3.2,1.6,0.2,0},
{4.8,3.1,1.6,0.2,0},
{5.4,3.4,1.5,0.4,0},
{5.2,4.1,1.5,0.1,0},
{5.5,4.2,1.4,0.2,0},
{4.9,3.1,1.5,0.1,0},
{5.0,3.2,1.2,0.2,0},
{5.5,3.5,1.3,0.2,0},
{4.9,3.1,1.5,0.1,0},
{4.4,3.0,1.3,0.2,0},
{5.1,3.4,1.5,0.2,0},
{5.0,3.5,1.3,0.3,0},
{4.5,2.3,1.3,0.3,0},
{4.4,3.2,1.3,0.2,0},
{5.0,3.5,1.6,0.6,0},
{5.1,3.8,1.9,0.4,0},
{4.8,3.0,1.4,0.3,0},
{5.1,3.8,1.6,0.2,0},
{4.6,3.2,1.4,0.2,0},
{5.3,3.7,1.5,0.2,0},
{5.0,3.3,1.4,0.2,0},
{7.0,3.2,4.7,1.4,1},
{6.4,3.2,4.5,1.5,1},
{6.9,3.1,4.9,1.5,1},
{5.5,2.3,4.0,1.3,1},
{6.5,2.8,4.6,1.5,1},
{5.7,2.8,4.5,1.3,1},
{6.3,3.3,4.7,1.6,1},
{4.9,2.4,3.3,1.0,1},
{6.6,2.9,4.6,1.3,1},
{5.2,2.7,3.9,1.4,1},
{5.0,2.0,3.5,1.0,1},
{5.9,3.0,4.2,1.5,1},
{6.0,2.2,4.0,1.0,1},
{6.1,2.9,4.7,1.4,1},
{5.6,2.9,3.6,1.3,1},
{6.7,3.1,4.4,1.4,1},
{5.6,3.0,4.5,1.5,1},
{5.8,2.7,4.1,1.0,1},
{6.2,2.2,4.5,1.5,1},
{5.6,2.5,3.9,1.1,1},
{5.9,3.2,4.8,1.8,1},
{6.1,2.8,4.0,1.3,1},
{6.3,2.5,4.9,1.5,1},
{6.1,2.8,4.7,1.2,1},
{6.4,2.9,4.3,1.3,1},
{6.6,3.0,4.4,1.4,1},
{6.8,2.8,4.8,1.4,1},
{6.7,3.0,5.0,1.7,1},
{6.0,2.9,4.5,1.5,1},
{5.7,2.6,3.5,1.0,1},
{5.5,2.4,3.8,1.1,1},
{5.5,2.4,3.7,1.0,1},
{5.8,2.7,3.9,1.2,1},
{6.0,2.7,5.1,1.6,1},
{5.4,3.0,4.5,1.5,1},
{6.0,3.4,4.5,1.6,1},
{6.7,3.1,4.7,1.5,1},
{6.3,2.3,4.4,1.3,1},
{5.6,3.0,4.1,1.3,1},
{5.5,2.5,4.0,1.3,1},
{5.5,2.6,4.4,1.2,1},
{6.1,3.0,4.6,1.4,1},
{5.8,2.6,4.0,1.2,1},
{5.0,2.3,3.3,1.0,1},
{5.6,2.7,4.2,1.3,1},
{5.7,3.0,4.2,1.2,1},
{5.7,2.9,4.2,1.3,1},
{6.2,2.9,4.3,1.3,1},
{5.1,2.5,3.0,1.1,1},
{5.7,2.8,4.1,1.3,1},
{6.3,3.3,6.0,2.5,2},
{5.8,2.7,5.1,1.9,2},
{7.1,3.0,5.9,2.1,2},
{6.3,2.9,5.6,1.8,2},
{6.5,3.0,5.8,2.2,2},
{7.6,3.0,6.6,2.1,2},
{4.9,2.5,4.5,1.7,2},
{7.3,2.9,6.3,1.8,2},
{6.7,2.5,5.8,1.8,2},
{7.2,3.6,6.1,2.5,2},
{6.5,3.2,5.1,2.0,2},
{6.4,2.7,5.3,1.9,2},
{6.8,3.0,5.5,2.1,2},
{5.7,2.5,5.0,2.0,2},
{5.8,2.8,5.1,2.4,2},
{6.4,3.2,5.3,2.3,2},
{6.5,3.0,5.5,1.8,2},
{7.7,3.8,6.7,2.2,2},
{7.7,2.6,6.9,2.3,2},
{6.0,2.2,5.0,1.5,2},
{6.9,3.2,5.7,2.3,2},
{5.6,2.8,4.9,2.0,2},
{7.7,2.8,6.7,2.0,2},
{6.3,2.7,4.9,1.8,2},
{6.7,3.3,5.7,2.1,2},
{7.2,3.2,6.0,1.8,2},
{6.2,2.8,4.8,1.8,2},
{6.1,3.0,4.9,1.8,2},
{6.4,2.8,5.6,2.1,2},
{7.2,3.0,5.8,1.6,2},
{7.4,2.8,6.1,1.9,2},
{7.9,3.8,6.4,2.0,2},
{6.4,2.8,5.6,2.2,2},
{6.3,2.8,5.1,1.5,2},
{6.1,2.6,5.6,1.4,2},
{7.7,3.0,6.1,2.3,2},
{6.3,3.4,5.6,2.4,2},
{6.4,3.1,5.5,1.8,2},
{6.0,3.0,4.8,1.8,2},
{6.9,3.1,5.4,2.1,2},
{6.7,3.1,5.6,2.4,2},
{6.9,3.1,5.1,2.3,2},
{5.8,2.7,5.1,1.9,2},
{6.8,3.2,5.9,2.3,2},
{6.7,3.3,5.7,2.5,2},
{6.7,3.0,5.2,2.3,2},
{6.3,2.5,5.0,1.9,2},
{6.5,3.0,5.2,2.0,2},
{6.2,3.4,5.4,2.3,2},
{5.9,3.0,5.1,1.8,2}};

void create_dataset(Array_list_ptr train_set, Array_list_ptr test_set) {
    int strides[] = {5};
    for (int i = 0; i < 150; i++) {
        Tensor_ptr input = create_tensor(iris_data[i], strides, 1);
        if (i % 5 != 0) {
            array_list_add(train_set, input);
        } else {
            array_list_add(test_set, input);
        }
    }
}

void train_linear_perceptron(struct computational_graph* graph, Array_list_ptr train_set, Neural_network_parameter_ptr parameters) {
    graph->get_class_labels = get_class_labels_linear_perceptron;
    Optimizer_ptr optimizer = create_stochastic_gradient(0.1, 0.99);
    Multiplication_node_ptr input = create_multiplication_node(false, true, false);
    array_list_add(graph->input_nodes, input);
    double* initial_weights = random_initialization(5, 3, 1);
    const int weights_shape[] = {5, 3};
    Tensor_ptr weights_tensor = create_tensor(initial_weights, weights_shape, 2);
    free_(initial_weights);
    Multiplication_node_ptr w = create_multiplication_node3(true, false, weights_tensor, false);
    Multiplication_node_ptr a = add_multiplication_edge(graph, (Computational_node_ptr)input, w, false);
    Softmax_ptr softmax = create_softmax();
    Computational_node_ptr output_node = add_edge(graph, (Computational_node_ptr) a, softmax, false);
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

Array_list_ptr get_class_labels_linear_perceptron(Computational_node_ptr output_node) {
    Array_list_ptr class_indices = create_array_list();
    Tensor_ptr output_value = output_node->value;
    int cols = output_value->shape[1];
    double max_value = -1;
    int label_index = -1;
    int indices[2] = {0, 0};
    for (int i = 0; i < cols; i++) {
        indices[1] = i;
        double value = get_tensor_value(output_value, indices);
        if (value > max_value) {
            max_value = value;
            label_index = i;
        }
    }
    array_list_add_int(class_indices, label_index);
    return class_indices;
}

Classification_performance_ptr test_linear_perceptron(struct computational_graph* graph, Array_list_ptr test_set){
    int count = 0, total = 0;
    for (int i = 0; i < test_set->size; i++) {
        Tensor_ptr instance = array_list_get(test_set, i);
        Computational_node_ptr input = array_list_get(graph->input_nodes, 0);
        set_node_value(input, create_input_tensor(instance));
        Array_list_ptr output = predict_by_computational_graph(graph);
        int class_label = array_list_get_int(output, 0);
        free_array_list(output, free_);
        int index[1] = {instance->shape[0] - 1};
        if (class_label == (int) get_tensor_value(instance, index)) {
            count++;
        }
        total++;
    }
    return create_classification_performance(count / (total + 0.0));
}

void run_linear_perceptron() {
    Array_list_ptr train_set = create_array_list();
    Array_list_ptr test_set = create_array_list();
    create_dataset(train_set, test_set);
    Computational_graph_ptr graph = create_computational_graph();
    graph->train = train_linear_perceptron;
    graph->get_class_labels = get_class_labels_linear_perceptron;
    graph->test = test_linear_perceptron;
    graph->train(graph, train_set, NULL);
    Classification_performance_ptr performance = graph->test(graph, test_set);
    free_(performance);
    free_array_list(train_set, (void*) free_tensor);
    free_array_list(test_set, (void*) free_tensor);
    free_computational_graph(graph);
}