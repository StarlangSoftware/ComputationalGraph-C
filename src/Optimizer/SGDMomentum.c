//
// Created by Olcay YILDIZ on 3.01.2026.
//

#include "SGDMomentum.h"
#include <Memory/Memory.h>
#include "../Node/ConcatenatedNode.h"

Sgd_momentum_ptr create_sgd_momentum(const double learning_rate, const double eta_decrease, const double momentum) {
    Sgd_momentum_ptr result = malloc_(sizeof(Sgd_momentum));
    set_attributes_sgd_momentum(result, learning_rate, eta_decrease, momentum);
    result->optimizer.set_gradients = set_gradients_sgd_momentum;
    result->optimizer.optimizer = result;
    result->optimizer.type = SGD_MOMENTUM;
    return result;
}

void free_sgd_momentum(Sgd_momentum_ptr sgd_momentum) {
    free_hash_map(sgd_momentum->velocity_map, free_);
    free_(sgd_momentum);
}

/**
 * Calculates the new gradients by combining the current gradient with the previous velocity.
 * It updates the internal velocity state and modifies the node's backward tensor
 * to reflect the momentum-adjusted update step.
 *
 * @param node The node whose gradients are to be set.
 */
void set_gradients_sgd_momentum(void* sgd, Computational_node_ptr node) {
    Sgd_momentum_ptr sgd_momentum = (Sgd_momentum_ptr) sgd;
    double* new_values = malloc_(node->backward->total_elements * sizeof(double));
    for (int i = 0; i < node->backward->total_elements; i++) {
        new_values[i] = (1 - sgd_momentum->momentum) * node->backward->data[i];
    }
    if (hash_map_contains(sgd_momentum->velocity_map, node)) {
        double* list = hash_map_get(sgd_momentum->velocity_map, node);
        for (int i = 0; i < node->backward->total_elements; i++) {
            new_values[i] += list[i] * sgd_momentum->momentum;
        }
    }
    double* copy_values = malloc_(node->backward->total_elements * sizeof(double));
    for (int i = 0; i < node->backward->total_elements; i++) {
        copy_values[i] = new_values[i];
    }
    hash_map_insert(sgd_momentum->velocity_map, node, copy_values);
    for (int i = 0; i < node->backward->total_elements; i++) {
        new_values[i] *= sgd_momentum->optimizer.learning_rate;
    }
    update_tensor_data(node->backward, new_values);
    free_(new_values);
}

void set_attributes_sgd_momentum(Sgd_momentum* sgd, const double learning_rate, const double eta_decrease, const double momentum) {
    set_attributes_optimizer(&(sgd->optimizer), learning_rate, eta_decrease);
    sgd->velocity_map = create_hash_map((unsigned int (*)(const void *, int)) hash_function_computational_node,
    (int (*)(const void *, const void *)) compare_computational_node);
    sgd->momentum = momentum;
}
