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
