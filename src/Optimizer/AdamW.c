//
// Created by Olcay YILDIZ on 20.01.2026.
//

#include "AdamW.h"
#include "../Node/ConcatenatedNode.h"
#include <Memory/Memory.h>

AdamW_ptr create_adamW(double learning_rate, double eta_decrease, double beta1, double beta2, double weight_decay,
                       double epsilon) {
    AdamW_ptr result = malloc_(sizeof(AdamW));
    set_attributes_adamW(result, learning_rate, eta_decrease, beta1, beta2, weight_decay, epsilon);
    result->adam.sgd.optimizer.optimizer = result;
    result->adam.sgd.optimizer.set_gradients = set_gradients_adamW;
    result->adam.sgd.optimizer.type = ADAM_W;
    return result;
}

void free_adamW(AdamW_ptr adamW) {
    free_hash_map(adamW->adam.sgd.velocity_map, free_);
    free_hash_map(adamW->adam.momentum_map, free_);
    free_(adamW);
}

/**
 * Sets the gradients for the given node using the AdamW optimization algorithm.
 * @param a Current AdamW optimizer
 * @param node The node whose gradients are to be set.
 */
void set_gradients_adamW(void *a, Computational_node_ptr node) {
    AdamW_ptr adamW = (AdamW_ptr) a;
    double* gradients = calculate_gradients_adam(a, node);
    double* values = node->value->data;
    for (int i = 0; i < node->backward->total_elements; i++) {
        gradients[i] += adamW->adam.sgd.optimizer.learning_rate * adamW->weight_decay * values[i];
    }
    update_tensor_data(node->backward, gradients);
    free_(gradients);
}

void set_attributes_adamW(AdamW *a, double learning_rate, double eta_decrease, double beta1, double beta2,
    double weight_decay, double epsilon) {
    set_attributes_adam(&(a->adam), learning_rate, eta_decrease, beta1, beta2, epsilon);
    a->weight_decay = weight_decay;
}

/**
 * Updates the values of all learnable nodes in the graph.
 * @param optimizer Current optimizer
 * @param node_map A map of nodes to their children.
 */
void update_values_adamW(Optimizer_ptr optimizer, Hash_map_ptr node_map) {
    AdamW* adam = optimizer->optimizer;
    adam->adam.current_beta1 *= adam->adam.sgd.momentum;
    adam->adam.current_beta2 *= adam->adam.beta2;
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
