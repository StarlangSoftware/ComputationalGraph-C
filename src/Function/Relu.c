//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#include "Relu.h"
#include "Memory/Memory.h"

Relu_ptr create_relu() {
    Relu_ptr relu = malloc_(sizeof(Relu));
    relu->function.function_type = RELU;
    relu->function.calculate = calculate_relu;
    relu->function.derivative = derivative_relu;
    return relu;
}

void free_relu(Relu_ptr relu) {
    free_(relu);
}

/**
 * Computes the ReLU activation for the given tensor.
 * @param function Current function
 * @param matrix The tensor whose values are to be computed.
 * @return ReLU(x).
 */
Tensor_ptr calculate_relu(const void* function, const Tensor* matrix) {
    double* values = malloc_(matrix->total_elements * sizeof(double));
    const double* old_values = matrix->data;
    for (int i = 0; i < matrix->total_elements; i++) {
        if (old_values[i] > 0) {
            values[i] = old_values[i];
        } else {
            values[i] = 0;
        }
    }
    return create_tensor3(values, matrix->shape, matrix->dimensions);
}

/**
 * Computes the derivative of the ReLU activation function.
 * @param function Current function
 * @param value output of the ReLU(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor_ptr derivative_relu(const void *function, const Tensor *value, const Tensor *backward) {
    double* values = malloc_(value->total_elements * sizeof(double));
    const double* old_values = value->data;
    const double* backward_values = backward->data;
    for (int i = 0; i < value->total_elements; i++) {
        if (old_values[i] > 0) {
            values[i] = backward_values[i];
        } else {
            values[i] = 0;
        }
    }
    return create_tensor3(values, value->shape, value->dimensions);
}
