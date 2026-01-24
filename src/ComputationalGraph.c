//
// Created by Olcay YILDIZ on 5.01.2026.
//

#include "ComputationalGraph.h"

#include <stdlib.h>
#include <Memory/Memory.h>
#include "Node/ConcatenatedNode.h"
#include <CounterHashMap.h>
#include <stdio.h>

#include "Optimizer/Adam.h"
#include "Optimizer/AdamW.h"

Computational_graph_ptr create_computational_graph() {
    Computational_graph_ptr graph = malloc_(sizeof(Computational_graph));
    graph->input_nodes = create_array_list();
    graph->train = NULL;
    graph->test = NULL;
    graph->get_class_labels = NULL;
    graph->node_map = create_hash_map((unsigned int (*)(const void *, int)) hash_function_computational_node,
                                      (int (*)(const void *, const void *)) compare_computational_node);
    graph->reverse_node_map = create_hash_map((unsigned int (*)(const void *, int)) hash_function_computational_node,
                                              (int (*)(const void *, const void *)) compare_computational_node);
    return graph;
}

void free_computational_graph(Computational_graph_ptr graph) {
    free_array_list(graph->input_nodes, NULL);
    Array_list_ptr list = key_list(graph->node_map);
    free_hash_map_of_array_list(graph->node_map, NULL);
    for (int i = 0; i < list->size; i++) {
        Computational_node_ptr node = array_list_get(list, i);
        switch (node->type) {
            case COMPUTATIONAL_NODE:
                free_computational_node(node);
                break;
            case CONCATENATED_NODE:
                free_concatenated_node((Concatenated_node_ptr)node);
                break;
            case MULTIPLICATION_NODE:
                free_multiplication_node((Multiplication_node_ptr)node);
                break;
        }
    }
    free_array_list(list, NULL);
    free_hash_map_of_array_list(graph->reverse_node_map, NULL);
    free_(graph);
}

void *add_edge(Computational_graph_ptr graph, Computational_node_ptr first, void *second,
               bool is_biased) {
    Computational_node_ptr new_node = create_computational_node2(false, is_biased, second);
    add_to_hash_map_of_array_list(graph->node_map, first, new_node);
    add_to_hash_map_of_array_list(graph->reverse_node_map, new_node, first);
    return new_node;
}

void *add_multiplication_edge(Computational_graph_ptr graph, Computational_node_ptr first,
                Multiplication_node_ptr second, bool is_biased) {
    Multiplication_node_ptr new_node = create_multiplication_node2(false, is_biased, second->is_hadamard, first);
    add_to_hash_map_of_array_list(graph->node_map, first, new_node);
    add_to_hash_map_of_array_list(graph->reverse_node_map, new_node, first);
    add_to_hash_map_of_array_list(graph->node_map, second, new_node);
    add_to_hash_map_of_array_list(graph->reverse_node_map, new_node, second);
    return new_node;
}

void *add_edge_with_hadamard(Computational_graph_ptr graph, Computational_node_ptr first,
                Computational_node_ptr second, bool is_biased, bool is_hadamard) {
    Multiplication_node_ptr new_node = create_multiplication_node2(false, is_biased, is_hadamard, first);
    add_to_hash_map_of_array_list(graph->node_map, first, new_node);
    add_to_hash_map_of_array_list(graph->reverse_node_map, new_node, first);
    add_to_hash_map_of_array_list(graph->node_map, second, new_node);
    add_to_hash_map_of_array_list(graph->reverse_node_map, new_node, second);
    return new_node;
}

void *add_addition_edge(Computational_graph_ptr graph, Computational_node_ptr first,
                        Computational_node_ptr second, bool is_biased) {
    Computational_node_ptr new_node = create_computational_node2(false, is_biased, NULL);
    add_to_hash_map_of_array_list(graph->node_map, first, new_node);
    add_to_hash_map_of_array_list(graph->reverse_node_map, new_node, first);
    add_to_hash_map_of_array_list(graph->node_map, second, new_node);
    add_to_hash_map_of_array_list(graph->reverse_node_map, new_node, second);
    return new_node;
}

Concatenated_node_ptr concat_edges(Computational_graph_ptr graph, Array_list_ptr nodes, int dimension) {
    Concatenated_node_ptr new_node = create_concatenated_node(dimension);
    for (int i = 0; i < nodes->size; i++) {
        void *node = array_list_get(nodes, i);
        add_to_hash_map_of_array_list(graph->node_map, node, new_node);
        add_to_hash_map_of_array_list(graph->reverse_node_map, new_node, node);
        add_node(new_node, node);
    }
    return new_node;
}

/**
 * Recursive helper function to perform depth-first search for topological sorting.
 * @param graph Current computational graph
 * @param node The current node being processed.
 * @param visited A set of visited nodes.
 * @return A list representing the partial topological order.
 */
Linked_list_ptr sort_recursive(Computational_graph_ptr graph, Computational_node_ptr node, Hash_set_ptr visited) {
    Linked_list_ptr queue = create_linked_list((int (*)(const void *, const void *)) compare_computational_node);
    hash_set_insert(visited, node);
    if (hash_map_contains(graph->node_map, node)) {
        Array_list_ptr list = hash_map_get(graph->node_map, node);
        for (int i = 0; i < list->size; i++) {
            Computational_node_ptr child = array_list_get(list, i);
            if (!hash_set_contains(visited, child)) {
                Linked_list_ptr result = sort_recursive(graph, child, visited);
                merge_linked_list(queue, result);
                free_(result);
            }
        }
    }
    add_last(queue, create_node(node));
    return queue;
}

/**
 * Performs topological sorting on the computational graph.
 * @return A list representing the topological order of the nodes.
 */
Linked_list_ptr topological_sort(Computational_graph_ptr graph) {
    Linked_list_ptr sorted_list = create_linked_list((int (*)(const void *, const void *)) compare_computational_node);
    Hash_set_ptr visited = create_hash_set((unsigned int (*)(const void *, int)) hash_function_computational_node,
                                           (int (*)(const void *, const void *)) compare_computational_node);
    Array_list_ptr list = key_list(graph->node_map);
    for (int i = 0; i < list->size; i++) {
        Computational_node_ptr node = array_list_get(list, i);
        if (!hash_set_contains(visited, node)) {
            Linked_list_ptr queue = sort_recursive(graph, node, visited);
            while (!is_linked_list_empty(queue)) {
                add_last(sorted_list, remove_first_node(queue));
            }
            free_linked_list(queue, NULL);
        }
    }
    free_array_list(list, NULL);
    free_hash_set(visited, NULL);
    return sorted_list;
}

/**
 * Recursive helper function to clear the values and gradients of nodes.
 */
void clear_recursive(Computational_graph_ptr graph, Hash_set_ptr visited, Computational_node_ptr node) {
    hash_set_insert(visited, node);
    if (!node->learnable) {
        if (node->value != NULL) {
            free_tensor(node->value);
        }
        node->value = NULL;
    }
    if (node->backward != NULL) {
        free_tensor(node->backward);
    }
    node->backward = NULL;
    if (hash_map_contains(graph->node_map, node)) {
        Array_list_ptr list = hash_map_get(graph->node_map, node);
        for (int i = 0; i < list->size; i++) {
            Computational_node_ptr child = array_list_get(list, i);
            if (!hash_set_contains(visited, child)) {
                clear_recursive(graph, visited, child);
            }
        }
    }
}

/**
 * Clears the values and gradients of all nodes in the graph.
 */
void clear_computational_graph(Computational_graph_ptr graph) {
    Hash_set_ptr visited = create_hash_set((unsigned int (*)(const void *, int)) hash_function_computational_node,
                                           (int (*)(const void *, const void *)) compare_computational_node);
    Array_list_ptr list = key_list(graph->node_map);
    for (int i = 0; i < list->size; i++) {
        Computational_node_ptr node = array_list_get(list, i);
        if (!hash_set_contains(visited, node)) {
            clear_recursive(graph, visited, node);
        }
    }
    free_array_list(list, NULL);
    free_hash_set(visited, NULL);
}

/**
 * Swaps last two dimensions of the Tensor.
 * @param length dimension size.
 */
int *transpose_axes(int length) {
    int *axes = malloc_(sizeof(int) * length);
    for (int i = 0; i < length - 2; i++) {
        axes[i] = i;
    }
    axes[length - 1] = length - 2;
    axes[length - 2] = length - 1;
    return axes;
}

Tensor_ptr get_biased_partial(Tensor_ptr tensor) {
    int *start_indexes = malloc_(sizeof(int) * tensor->dimensions);
    int *end_indexes = malloc_(sizeof(int) * tensor->dimensions);
    for (int i = 0; i < tensor->dimensions; i++) {
        start_indexes[i] = 0;
        if (i == tensor->dimensions - 1) {
            end_indexes[i] = tensor->shape[i] - 1;
        } else {
            end_indexes[i] = tensor->shape[i];
        }
    }
    Tensor_ptr result = partial_tensor(tensor, start_indexes, end_indexes);
    free_(start_indexes);
    free_(end_indexes);
    return result;
}

/**
 * Calculates the derivative of the child node with respect to the parent node.
 * @param graph Current computational graph
 * @param node Parent node.
 * @param child Child node.
 * @return The gradient tensor.
 */
Tensor_ptr calculate_derivative(Computational_graph_ptr graph, Computational_node_ptr node,
                                Computational_node_ptr child) {
    Array_list_ptr reverse_children = hash_map_get(graph->reverse_node_map, child);
    if (reverse_children == NULL || reverse_children->size == 0) {
        return NULL;
    }
    Tensor_ptr backward;
    bool backward_allocated = false;
    if (child->is_biased) {
        backward = get_biased_partial(child->backward);
        backward_allocated = true;
    } else {
        backward = child->backward;
    }
    if (child->function != NULL) {
        Tensor_ptr child_value;
        bool child_allocated = false;
        Function *function = child->function;
        if (child->is_biased) {
            child_value = get_biased_partial(child->value);
            child_allocated = true;
        } else {
            child_value = child->value;
        }
        Tensor_ptr result = function->derivative(function, child_value, backward);
        if (child_allocated) {
            free_tensor(child_value);
        }
        if (backward_allocated) {
            free_tensor(backward);
        }
        return result;
    }
    if (child->type == CONCATENATED_NODE) {
        int index = get_index_concatenated_node((Concatenated_node_ptr) child, node);
        int block_size = backward->shape[((Concatenated_node_ptr) child)->dimension] / reverse_children->size;
        int dimensions = block_size;
        int number_of_elements = 1;
        int *shape = malloc_(sizeof(int) * backward->dimensions);
        for (int i = 0; i < backward->dimensions; i++) {
            if (((Concatenated_node_ptr) child)->dimension > i) {
                shape[i] = backward->shape[i];
            } else {
                if (((Concatenated_node_ptr) child)->dimension < i) {
                    dimensions *= backward->shape[i];
                    shape[i] = backward->shape[i];
                } else {
                    shape[i] = block_size;
                }
            }
            number_of_elements *= shape[i];
        }
        int cur = 0;
        int i = 0;
        int pos = 0;
        double *new_values = malloc_(number_of_elements * sizeof(double));
        while (i < backward->total_elements) {
            if (cur % reverse_children->size == index) {
                for (int k = 0; k < dimensions; k++) {
                    new_values[pos++] = backward->data[i + k];
                }
            }
            cur++;
            i += dimensions;
        }
        Tensor_ptr result = create_tensor3(new_values, shape, backward->dimensions);
        if (backward_allocated) {
            free_tensor(backward);
        }
        free_(shape);
        return result;
    }
    if (child->type == MULTIPLICATION_NODE) {
        Computational_node_ptr left = array_list_get(reverse_children, 0);
        Computational_node_ptr right = array_list_get(reverse_children, 1);
        if (left == node) {
            Tensor_ptr right_value = right->value;
            if (((Multiplication_node_ptr) child)->is_hadamard) {
                return hadamard_product(right_value, backward);
            }
            int *transposed = transpose_axes(right_value->dimensions);
            Tensor_ptr right_value_transposed = transpose_tensor(right_value, transposed);
            free_(transposed);
            Tensor_ptr result = multiply_tensors(backward, right_value_transposed);
            free_tensor(right_value_transposed);
            return result;
        }
        Tensor_ptr left_value = left->value;
        if (((Multiplication_node_ptr) child)->is_hadamard) {
            return hadamard_product(left_value, backward);
        }
        if (left_value != NULL && backward != NULL) {
            int *transposed = transpose_axes(left_value->dimensions);
            Tensor_ptr left_value_transposed = transpose_tensor(left_value, transposed);
            free_(transposed);
            Tensor_ptr result = multiply_tensors(left_value_transposed, backward);
            free_tensor(left_value_transposed);
            return result;
        }
    }
    return backward;
}

/**
 * Computes the difference between the predicted and actual values (R - Y).
 * @param output The output node of the computational graph.
 * @param class_label_index A list of true class labels (index of the correct class for each sample).
 */
void calculate_r_minus_y(Computational_node_ptr output, const int *class_label_index) {
    double *values = malloc_(output->value->total_elements * sizeof(double));
    const double *output_values = output->value->data;
    int last_dimension = output->value->shape[output->value->dimensions - 1];
    for (int i = 0; i < output->value->total_elements; i++) {
        if (i % last_dimension == class_label_index[i / last_dimension]) {
            values[i] = 1 - output_values[i];
        } else {
            values[i] = -output_values[i];
        }
    }
    set_node_backward(output, create_tensor3(values, output->value->shape, output->value->dimensions));
}

/**
 * Performs backpropagation on the computational graph.
 * @param graph Current computational graph
 * @param optimizer Optimizer to be used for updating the values.
 * @param class_label_index The true class labels (as a list of integers).
 */
void back_propagation(Computational_graph_ptr graph, Optimizer_ptr optimizer, const int *class_label_index) {
    Linked_list_ptr sorted_nodes = topological_sort(graph);
    if (is_linked_list_empty(sorted_nodes)) {
        return;
    }
    Computational_node_ptr output_node = remove_first(sorted_nodes);
    calculate_r_minus_y(output_node, class_label_index);
    Computational_node_ptr tmp_node = remove_first(sorted_nodes);
    set_node_backward(tmp_node, clone_tensor(output_node->backward));
    while (!is_linked_list_empty(sorted_nodes)) {
        Computational_node_ptr node = remove_first(sorted_nodes);
        Array_list_ptr children = hash_map_get(graph->node_map, node);
        if (children != NULL) {
            for (int i = 0; i < children->size; i++) {
                Computational_node_ptr child = array_list_get(children, i);
                Tensor_ptr derivative = calculate_derivative(graph, node, child);
                if (derivative != NULL) {
                    if (node->backward == NULL) {
                        node->backward = derivative;
                    } else {
                        Tensor_ptr added = add_tensors(node->backward, derivative);
                        free_tensor(derivative);
                        set_node_backward(node, added);
                    }
                }
            }
        }
    }
    optimizer->update_values(optimizer, graph->node_map);
    clear_computational_graph(graph);
    free_linked_list(sorted_nodes, NULL);
}

/**
 * Add a bias term to the node's value by appending a column of ones.
 * @param tensor The node whose value needs to be biased.
 */
void get_biased(Computational_node_ptr tensor) {
    int last_dimension_size = tensor->value->shape[tensor->value->dimensions - 1];
    double *values = malloc_(
        (tensor->value->total_elements + (tensor->value->total_elements / last_dimension_size)) * sizeof(double));
    const double *old_values = tensor->value->data;
    int k = 0;
    for (int i = 0; i < tensor->value->total_elements; i++) {
        values[k++] = old_values[i];
        if ((i + 1) % last_dimension_size == 0) {
            values[k++] = 1.0;
        }
    }
    if (k != tensor->value->total_elements + (tensor->value->total_elements / last_dimension_size)) {
        perror("Biased tensor size does not match");
    }
    int *shape = malloc_((tensor->value->dimensions * sizeof(int)));
    for (int i = 0; i < tensor->value->dimensions; i++) {
        if (i == tensor->value->dimensions - 1) {
            shape[i] = tensor->value->shape[i] + 1;
        } else {
            shape[i] = tensor->value->shape[i];
        }
    }
    Tensor_ptr biased_value = create_tensor3(values, shape, tensor->value->dimensions);
    free_(shape);
    set_node_value(tensor, biased_value);
}

/**
 * Perform a forward pass and return predicted class indices.
 * @return A list of predicted class indices.
 */
Array_list_ptr predict_by_computational_graph(Computational_graph_ptr graph) {
    Array_list_ptr class_labels = forward_calculation(false);
    clear_computational_graph(graph);
    return class_labels;
}

/**
 * Perform a forward pass for the training phase.
 * @return A list of predicted class indices.
 */
Array_list_ptr forward_calculation(Computational_graph_ptr graph) {
    return forward_calculation_with_dropout(graph, true);
}

/**
 * Perform a forward pass through the computational graph.
 * @param graph Current computational graph
 * @param is_dropout Whether to perform dropout or not.
 * @return A list of predicted class indices.
 */
Array_list_ptr forward_calculation_with_dropout(Computational_graph_ptr graph, bool is_dropout) {
    Linked_list_ptr sorted_nodes = topological_sort(graph);
    if (is_linked_list_empty(sorted_nodes)) {
        return create_array_list();
    }
    Computational_node_ptr output_node = sorted_nodes->head->data;
    Hash_map_ptr concatenated_node_map = create_hash_map(
        (unsigned int (*)(const void *, int)) hash_function_computational_node,
        (int (*)(const void *, const void *)) compare_computational_node);
    Counter_hash_map_ptr counter_map = create_counter_hash_map(
        (unsigned int (*)(const void *, int)) hash_function_computational_node,
        (int (*)(const void *, const void *)) compare_computational_node);
    while (sorted_nodes->head->next != NULL) {
        Computational_node_ptr current_node = remove_last(sorted_nodes);
        if (current_node->is_biased) {
            get_biased(current_node);
        }
        Array_list_ptr children = hash_map_get(graph->node_map, current_node);
        if (children != NULL) {
            for (int i = 0; i < children->size; i++) {
                Computational_node_ptr child = array_list_get(children, i);
                if (child->value == NULL) {
                    if (child->function != NULL) {
                        const Function* function = child->function;
                        const Tensor* current_value = current_node->value;
                        if (function->function_type == DROPOUT) {
                            if (is_dropout) {
                                set_node_value(child, function->calculate(function, current_value));
                            } else {
                                set_node_value(child, clone_tensor(current_value));
                            }
                        } else {
                            set_node_value(child, function->calculate(function, current_value));
                        }
                    } else {
                        if (child->type == CONCATENATED_NODE) {
                            if (!hash_map_contains(concatenated_node_map, child)) {
                                Array_list_ptr reverse_list = hash_map_get(graph->reverse_node_map, child);
                                Computational_node_ptr* new_list = malloc_(reverse_list->size * sizeof(Computational_node_ptr));
                                hash_map_insert(concatenated_node_map, child, new_list);
                            }
                            Computational_node_ptr* new_list = hash_map_get(concatenated_node_map, child);
                            new_list[get_index_concatenated_node((Concatenated_node_ptr) child, current_node)] = current_node;
                            put_counter_hash_map(counter_map, child);
                            int concatenated_size = count_counter_hash_map(counter_map, child);
                            if (((Array_list_ptr)hash_map_get(graph->reverse_node_map, child))->size == concatenated_size) {
                                Computational_node_ptr* new_list2 = hash_map_get(concatenated_node_map, child);
                                Tensor_ptr current_value = new_list2[0]->value;
                                set_node_value(child, clone_tensor(current_value));
                                for (int j = 1; j < concatenated_size; j++) {
                                    Tensor_ptr concat = concat_tensor(child->value, new_list2[i]->value, ((Concatenated_node_ptr)child)->dimension);
                                    set_node_value(child, concat);
                                }
                            }
                        } else {
                            set_node_value(child, clone_tensor(current_node->value));
                        }
                    }
                } else {
                    if (child->type == MULTIPLICATION_NODE) {
                        Tensor_ptr child_value = child->value;
                        Tensor_ptr current_value = current_node->value;
                        if (((Multiplication_node_ptr)child)->is_hadamard) {
                            set_node_value(child, hadamard_product(child_value, current_value));
                        } else {
                            if (((Multiplication_node_ptr) child)->priority_node != current_node) {
                                set_node_value(child, multiply_tensors(child_value, current_value));
                            } else {
                                set_node_value(child, multiply_tensors(current_value, child_value));
                            }
                        }
                    } else {
                        Tensor_ptr result = child->value;
                        Tensor_ptr current_value = current_node->value;
                        set_node_value(child, add_tensors(result, current_value));
                    }
                }
            }
        }
    }
    free_hash_map(concatenated_node_map, free_);
    free_counter_hash_map(counter_map);
    free_linked_list(sorted_nodes, NULL);
    return graph->get_class_labels(output_node);
}