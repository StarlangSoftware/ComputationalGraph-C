//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#include "Elu.h"

#include <math.h>
#include "Memory/Memory.h"

Elu_ptr create_elu(const double a) {
    Elu_ptr delu = malloc_(sizeof(Elu));
    delu->a = a;
    delu->function.function_type = ELU;
    delu->function.calculate = calculate_elu;
    delu->function.derivative = derivative_elu;
    return delu;
}

Elu_ptr create_elu2() {
    return create_elu(1.0);
}

void free_elu(Elu_ptr elu) {
    free_(elu);
}

/**
 * Computes the ELU activation for the given tensor.
 * @param function Current function
 * @param matrix The tensor whose values are to be computed.
 * @return ELU(x).
 */
Tensor_ptr calculate_elu(const void* function, const Tensor* matrix) {
    double* values = malloc_(matrix->total_elements * sizeof(double));
    const double* old_values = matrix->data;
    const Elu* elu = (Elu_ptr) function;
    for (int i = 0; i < matrix->total_elements; i++) {
        if (old_values[i] < 0) {
            values[i] = elu->a * (exp(old_values[i]) - 1);
        } else {
            values[i] = old_values[i];
        }
    }
    return create_tensor3(values, matrix->shape, matrix->dimensions);
}

/**
 * Computes the derivative of the ELU activation function.
 * @param function Current function
 * @param matrix output of the ELU(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor_ptr derivative_elu(const void *function, const Tensor *matrix, const Tensor *backward) {
    double* values = malloc_(matrix->total_elements * sizeof(double));
    const double* old_values = matrix->data;
    const double* backward_values = backward->data;
    const Elu* elu = (Elu_ptr) function;
    for (int i = 0; i < matrix->total_elements; i++) {
        if (old_values[i] < 0) {
            values[i] = backward_values[i] * (elu->a + old_values[i]);
        } else {
            values[i] = backward_values[i];
        }
    }
    return create_tensor3(values, matrix->shape, matrix->dimensions);
}
