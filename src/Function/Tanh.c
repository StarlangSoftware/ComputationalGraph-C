//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#include "Tanh.h"
#include <math.h>
#include "Memory/Memory.h"

Tanh_ptr create_tanh() {
    Tanh_ptr tanh = malloc_(sizeof(Tanh));
    tanh->function.function_type = TANH;
    tanh->function.calculate = calculate_tanh;
    tanh->function.derivative = derivative_tanh;
    return tanh;
}

void free_tanh(Tanh_ptr tanh) {
    free_(tanh);
}

/**
 * Computes the Tanh activation for the given tensor.
 * @param function Current function
 * @param tensor The tensor whose values are to be computed.
 * @return Tanh(x).
 */
Tensor_ptr calculate_tanh(const void* function, const Tensor* tensor) {
    double* values = malloc_(tensor->total_elements * sizeof(double));
    const double* tensor_values = tensor->data;
    for (int i = 0; i < tensor->total_elements; i++) {
        values[i] = tanh(tensor_values[i]);
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

/**
 * Computes the derivative of the Tanh activation function.
 * @param function Current function
 * @param tensor output of the Tanh(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor_ptr derivative_tanh(const void *function, const Tensor *tensor, const Tensor *backward) {
    double* values = malloc_(tensor->total_elements * sizeof(double));
    const double* tensor_values = tensor->data;
    const double* backward_values = backward->data;
    for (int i = 0; i < tensor->total_elements; i++) {
        values[i] = backward_values[i] * (1 - tensor_values[i] * tensor_values[i]);
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}
