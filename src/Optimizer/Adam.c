//
// Created by Olcay YILDIZ on 4.01.2026.
//

#include "Adam.h"
#include <math.h>
#include "../Node/ConcatenatedNode.h"
#include <Memory/Memory.h>

Adam_ptr create_adam(const double learning_rate, const double eta_decrease, const double beta1, const double beta2, const double epsilon) {
    Adam_ptr result = malloc_(sizeof(Adam));
    set_attributes_adam(result, learning_rate, eta_decrease, beta1, beta2, epsilon);
    result->sgd.optimizer.optimizer = result;
    result->sgd.optimizer.set_gradients = set_gradients_adam;
    return result;
}

void free_adam(Adam_ptr adam) {
    free_hash_map(adam->sgd.velocity_map, free_);
    free_hash_map(adam->momentum_map, free_);
    free_(adam);
}

/**
 * Calculates the gradient updates using the Adam optimization algorithm.
 * <p>
 * This implementation follows a multi-pass approach:
 * <ol>
 * <li><b>First Pass:</b> Calculates the weighted current gradients for both the first moment (momentum)
 * and the second moment (velocity/squared gradients).</li>
 * <li><b>Second Pass (Conditional):</b> If historical data exists, adds the decayed previous
 * momentum and velocity values to the current ones.</li>
 * <li><b>State Update:</b> Stores the raw calculated moments into the history maps.</li>
 * <li><b>Bias Correction:</b> Normalizes the moments by dividing them by <code>(1 - beta)</code>
 * to account for initialization bias.</li>
 * <li><b>Final Pass:</b> Computes the parameter update using the adaptive learning rate formula:
 * <code>(new_momentum / (sqrt(new_velocity) + epsilon)) * learningRate</code>.</li>
 * </ol>
 * </p>
 *
 * @param node The node whose gradients are to be set.
 */
double* calculate_gradients_adam(void *a, Computational_node_ptr node) {
    Adam_ptr adam = (Adam_ptr) a;
    double* new_values_momentum = malloc_(node->backward->total_elements * sizeof(double));
    double* new_values_velocity = malloc_(node->backward->total_elements * sizeof(double));
    for (int i = 0; i < node->backward->total_elements; i++) {
        new_values_momentum[i] = (1 - adam->sgd.momentum) * node->backward->data[i];
        new_values_velocity[i] = (1 - adam->beta2) * node->backward->data[i] * node->backward->data[i];
    }
    if (hash_map_contains(adam->momentum_map, node)) {
        double* momentum_list = hash_map_get(adam->momentum_map, node);
        double* velocity_list = hash_map_get(adam->sgd.velocity_map, node);
        for (int i = 0; i < node->backward->total_elements; i++) {
            new_values_velocity[i] += velocity_list[i] * adam->beta2;
            new_values_momentum[i] += momentum_list[i] * adam->sgd.momentum;
        }
    }
    double* copy_values_momentum = malloc_(node->backward->total_elements * sizeof(double));
    double* copy_values_velocity = malloc_(node->backward->total_elements * sizeof(double));
    for (int i = 0; i < node->backward->total_elements; i++) {
        copy_values_momentum[i] = new_values_momentum[i];
        copy_values_velocity[i] = new_values_velocity[i];
    }
    hash_map_insert(adam->sgd.velocity_map, node, copy_values_velocity);
    hash_map_insert(adam->momentum_map, node, copy_values_momentum);
    adam->current_beta1 *= adam->sgd.momentum;
    adam->current_beta2 *= adam->beta2;
    for (int i = 0; i < node->backward->total_elements; i++) {
        new_values_momentum[i] /= 1 - adam->current_beta1;
        new_values_velocity[i] /= 1 - adam->current_beta2;
    }
    double* new_values = malloc_(node->backward->total_elements * sizeof(double));
    for (int i = 0; i < node->backward->total_elements; i++) {
        new_values[i] = (new_values_momentum[i] / (sqrt(new_values_velocity[i]) + adam->epsilon)) * adam->sgd.optimizer.learning_rate;
    }
    free_(new_values_momentum);
    free_(new_values_velocity);
    return new_values;
}

void set_gradients_adam(void *a, Computational_node_ptr node) {
    double* new_values = calculate_gradients_adam(a, node);
    update_tensor_data(node->backward, new_values);
    free_(new_values);
}

void set_attributes_adam(Adam_ptr adam, double learning_rate, double eta_decrease, double beta1, double beta2,
    double epsilon) {
    set_attributes_sgd_momentum(&(adam->sgd), learning_rate, eta_decrease, beta1);
    adam->beta2 = beta2;
    adam->epsilon = epsilon;
    adam->current_beta1 = 1;
    adam->current_beta2 = 1;
    adam->momentum_map = create_hash_map((unsigned int (*)(const void *, int)) hash_function_computational_node,
        (int (*)(const void *, const void *)) compare_computational_node);
}
