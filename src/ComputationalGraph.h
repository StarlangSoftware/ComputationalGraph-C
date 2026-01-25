//
// Created by Olcay YILDIZ on 5.01.2026.
//

#ifndef COMPUTATIONALGRAPH_COMPUTATIONALGRAPH_H
#define COMPUTATIONALGRAPH_COMPUTATIONALGRAPH_H
#include <HashMap/HashMap.h>
#include <Performance/ClassificationPerformance.h>
#include "NeuralNetworkParameter.h"
#include "Node/ConcatenatedNode.h"
#include "Node/MultiplicationNode.h"

struct computational_graph {
    Hash_map_ptr node_map;
    Hash_map_ptr reverse_node_map;
    Array_list_ptr input_nodes;
    Computational_node_ptr output_node;
    /**
    * Trains the computational graph using the given training set and parameters.
    * @param train_set The training set.
    * @param parameters The parameters of the computational graph.
    */
    void (*train) (struct computational_graph* graph, Array_list_ptr train_set, Neural_network_parameter_ptr parameters);
    /**
     * Tests the computational graph on the given test set.
     * @param test_set The test set.
     * @return The classification performance of the computational graph on the test set.
     */
    Classification_performance_ptr (*test) (struct computational_graph* graph, Array_list_ptr test_set);
    /**
     * Retrieves the class label indexes associated with the given output node in the computational graph.
     * @param output_node The output node for which the class label indexes are to be retrieved.
     * @return A list of integers representing the class label indexes.
     */
    Array_list_ptr (*get_class_labels) (Computational_node_ptr output_node);
};

typedef struct computational_graph Computational_graph;

typedef Computational_graph* Computational_graph_ptr;

Computational_graph_ptr create_computational_graph();

void free_computational_graph(Computational_graph_ptr graph);

void* add_edge(Computational_graph_ptr graph, Computational_node_ptr first, void* second, bool is_biased);

void* add_multiplication_edge(Computational_graph_ptr graph, Computational_node_ptr first, Multiplication_node_ptr second, bool is_biased);

void* add_edge_with_hadamard(Computational_graph_ptr graph, Computational_node_ptr first, Computational_node_ptr second, bool is_biased, bool is_hadamard);

void* add_addition_edge(Computational_graph_ptr graph, Computational_node_ptr first, Computational_node_ptr second, bool is_biased);

Concatenated_node_ptr concat_edges(Computational_graph_ptr graph, Array_list_ptr nodes, int dimension);

Linked_list_ptr sort_recursive(Computational_graph_ptr graph, Computational_node_ptr node, Hash_set_ptr visited);

Linked_list_ptr topological_sort(Computational_graph_ptr graph);

void clear_recursive(Computational_graph_ptr graph, Hash_set_ptr visited, Computational_node_ptr node);

void clear_computational_graph(Computational_graph_ptr graph);

int* transpose_axes(int length);

Tensor_ptr get_biased_partial(Tensor_ptr tensor);

Tensor_ptr calculate_derivative(Computational_graph_ptr graph, Computational_node_ptr node, Computational_node_ptr child);

void calculate_r_minus_y(Computational_node_ptr output, const int* class_label_index);

void back_propagation(Computational_graph_ptr graph, Optimizer_ptr optimizer, const int* class_label_index);

void get_biased(Computational_node_ptr tensor);

Array_list_ptr predict_by_computational_graph(Computational_graph_ptr graph);

Array_list_ptr forward_calculation(Computational_graph_ptr graph);

Array_list_ptr forward_calculation_with_dropout(Computational_graph_ptr graph, bool enable_dropout);

#endif //COMPUTATIONALGRAPH_COMPUTATIONALGRAPH_H