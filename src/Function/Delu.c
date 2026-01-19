//
// Created by Olcay Taner YILDIZ on 23.12.2025.
//

#include "Delu.h"
#include <math.h>
#include "Memory/Memory.h"

Delu_ptr create_delu(const double a, const double b, const double xc) {
    Delu_ptr delu = malloc_(sizeof(Delu));
    delu->a = a;
    delu->b = b;
    delu->xc = xc;
    delu->function.function_type = DELU;
    delu->function.calculate = calculate_delu;
    delu->function.derivative = derivative_delu;
    return delu;
}

Delu_ptr create_delu2() {
    return create_delu(1.0, 2.0, 1.25643);
}

void free_delu(Delu_ptr delu) {
    free_(delu);
}

/**
 * Computes the DELU activation for the given value tensor.
 * @param function Current function
 * @param matrix The tensor whose values are to be computed.
 * @return DELU(x).
 */
Tensor_ptr calculate_delu(const void* function, const Tensor* matrix) {
    double* values = malloc_(matrix->total_elements * sizeof(double));
    const double* old_values = matrix->data;
    const Delu* delu = (Delu_ptr)function;
    for (int i = 0; i < matrix->total_elements; i++) {
        if (old_values[i] > delu->xc) {
            values[i] = old_values[i];
        } else {
            values[i] = (exp(delu->a * old_values[i]) - 1) / delu->b;
        }
    }
    return create_tensor3(values, matrix->shape, matrix->dimensions);
}

/**
 * Computes the derivative of the DELU activation function.
 * @param function Current function
 * @param value output of the DELU(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor_ptr derivative_delu(const void *function, const Tensor *value, const Tensor *backward) {
    double* values = malloc_(value->total_elements * sizeof(double));
    const double* old_values = value->data;
    const double* backward_values = backward->data;
    const Delu* delu = (Delu_ptr)function;
    for (int i = 0; i < value->total_elements; i++) {
        if (old_values[i] > delu->xc) {
            values[i] = backward_values[i];
        } else {
            values[i] = backward_values[i] * ((old_values[i] * delu->b + 1) * (delu->a / delu->b));
        }
    }
    return create_tensor3(values, value->shape, value->dimensions);
}
