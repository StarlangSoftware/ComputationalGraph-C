//
// Created by Olcay YILDIZ on 3.01.2026.
//

#include "StochasticGradientDescent.h"
#include <Memory/Memory.h>
#include "Optimizer.h"

Optimizer_ptr create_stochastic_gradient(double learning_rate, double eta_decrease) {
    Optimizer_ptr result = create_optimizer(learning_rate, eta_decrease);
    result->set_gradients = set_gradients_stochastic_gradient_descent;
    return result;
}

/**
 * Sets the gradients (backward values) of the node to the learning rate times the backward values.
 * @param node The node whose gradients are to be set.
 */
void set_gradients_stochastic_gradient_descent(void *sgd, Computational_node_ptr node) {
    const Optimizer* stochastic_gradient_descent = (Optimizer_ptr) sgd;
    double* values = malloc_(node->backward->total_elements * sizeof(double));
    const double* backward_values = node->backward->data;
    for (int i = 0; i < node->backward->total_elements; i++) {
        values[i] = backward_values[i] * stochastic_gradient_descent->learning_rate;
    }
    update_tensor_data(node->backward, values);
    free_(values);
}
