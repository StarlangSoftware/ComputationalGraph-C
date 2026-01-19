//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#include "Sigmoid.h"
#include <math.h>
#include "Memory/Memory.h"

Sigmoid_ptr create_sigmoid() {
    Sigmoid_ptr sigmoid = malloc_(sizeof(Sigmoid));
    sigmoid->function.function_type = SIGMOID;
    sigmoid->function.calculate = calculate_sigmoid;
    sigmoid->function.derivative = derivative_sigmoid;
    return sigmoid;
}

void free_sigmoid(Sigmoid_ptr sigmoid) {
    free_(sigmoid);
}

/**
 * Computes the Sigmoid activation for the given tensor.
 * @param function Current function
 * @param tensor The tensor whose values are to be computed.
 * @return Sigmoid(x).
 */
Tensor_ptr calculate_sigmoid(const void* function, const Tensor* tensor) {
    double* values = malloc_(tensor->total_elements * sizeof(double));
    const double* tensor_values = tensor->data;
    for (int i = 0; i < tensor->total_elements; i++) {
        values[i] = 1.0 / (1.0 + exp(-tensor_values[i]));
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

/**
 * Computes the derivative of the Sigmoid activation function.
 * @param function Current function
 * @param tensor output of the Sigmoid(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor_ptr derivative_sigmoid(const void *function, const Tensor *tensor, const Tensor *backward) {
    double* values = malloc_(tensor->total_elements * sizeof(double));
    const double* tensor_values = tensor->data;
    const double* backward_values = backward->data;
    for (int i = 0; i < tensor->total_elements; i++) {
        values[i] = backward_values[i] * tensor_values[i] * (1.0 - tensor_values[i]);
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}
