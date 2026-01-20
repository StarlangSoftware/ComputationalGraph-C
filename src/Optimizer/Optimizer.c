//
// Created by Olcay YILDIZ on 3.01.2026.
//

#include "Optimizer.h"
#include "../Node/ComputationalNode.h"
#include <Memory/Memory.h>
#include "../Node/ConcatenatedNode.h"

Optimizer_ptr create_optimizer(const double learning_rate, const double eta_decrease) {
    Optimizer_ptr result = malloc_(sizeof(Optimizer));
    set_attributes_optimizer(result, learning_rate, eta_decrease);
    result->set_gradients = NULL;
    result->optimizer = result;
    return result;
}

/**
 * Updates the learning rate of the optimizer.
 */
void set_learning_rate(Optimizer_ptr optimizer) {
    optimizer->learning_rate *= optimizer->eta_decrease;
}

/**
 * Checks if broadcasting be applied to the corresponding node.
 * @param node The node to check.
 * @return The index of the dimension where broadcasting is to be applied. -1 if broadcasting is not to be applied.
 */
int broadcast_optimizer(const Computational_node* node) {
    int* v = node->value->shape;
    int* b = node->backward->shape;
    int index = -1;
    for (int i = 0; i < node->value->dimensions; i++) {
        if (v[i] != b[i]) {
            if (v[i] == 1) {
                if (index != -1) {
                    return -1;
                }
                index = i;
            }
        }
    }
    return index;
}

/**
 * Recursive helper function to update the values of learnable nodes.
 * @param optimizer Current optimizer
 * @param visited A set of visited nodes.
 * @param node The current node being processed.
 * @param node_map A map of nodes to their children.
 */
void update_recursive(Optimizer_ptr optimizer, Hash_set_ptr visited, Computational_node_ptr node, Hash_map_ptr node_map) {
    hash_set_insert(visited, node);
    if (node->learnable) {
        int index = broadcast_optimizer(node);
        if (index != -1) {
            int v = 1, b = 1;
            for (int i = node->value->dimensions - 1; i >= index; i--) {
                v *= node->value->shape[i];
                b *= node->backward->shape[i];
            }
            double* backward_values = node->backward->data;
            double* values = malloc_(node->value->dimensions * sizeof(double));
            for (int i = 0; i < node->backward->dimensions; i--) {
                for (int j = i; j < i + b; j++) {
                    values[((j - i) % v) + v * (j / b)] += backward_values[j];
                }
                i += b - 1;
            }
            free_tensor(node->backward);
            node->backward = create_tensor(values, node->value->shape, node->value->dimensions);
        }
        optimizer->set_gradients(optimizer->optimizer, node);
        update_value(node);
    }
    if (hash_map_contains(node_map, node)) {
        Array_list_ptr node_list = hash_map_get(node_map, node);
        for (int i = 0; i < node_list->size; i++) {
            Computational_node_ptr child = array_list_get(node_list, i);
            if (!hash_set_contains(visited, child)) {
                update_recursive(optimizer, visited, child, node_map);
            }
        }
    }
}

/**
 * Updates the values of all learnable nodes in the graph.
 * @param optimizer Current optimizer
 * @param node_map A map of nodes to their children.
 */
void update_values(Optimizer_ptr optimizer, Hash_map_ptr node_map) {
    Hash_set_ptr visited = create_hash_set((unsigned int (*)(const void *, int)) hash_function_computational_node,
        (int (*)(const void *, const void *)) compare_computational_node);
    Array_list_ptr keys = key_list(node_map);
    for (int i = 0; i < keys->size; i++) {
        Computational_node_ptr node = array_list_get(keys, i);
        if (!hash_set_contains(visited, node)) {
            update_recursive(optimizer, visited, node, node_map);
        }
    }
    free_array_list(keys, NULL);
    free_hash_set(visited, NULL);
}

void set_attributes_optimizer(Optimizer *optimizer, double learning_rate, double eta_decrease) {
    optimizer->learning_rate = learning_rate;
    optimizer->eta_decrease = eta_decrease;
}
