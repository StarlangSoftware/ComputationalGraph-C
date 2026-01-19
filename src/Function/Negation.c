//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#include "Negation.h"
#include "Memory/Memory.h"

Negation_ptr create_negation() {
    Negation_ptr negation = malloc_(sizeof(Negation));
    negation->function.function_type = NEGATION;
    negation->function.calculate = calculate_negation;
    negation->function.derivative = derivative_negation;
    return negation;
}

void free_negation(Negation_ptr negation) {
    free_(negation);
}

/**
 * Negates the values of the given tensor.
 * @param function Current function
 * @param matrix The tensor whose values are to be negated.
 * @return The negated tensor.
 */
Tensor_ptr calculate_negation(const void* function, const Tensor* matrix) {
    double* values = malloc_(matrix->total_elements * sizeof(double));
    const double* old_values = matrix->data;
    for (int i = 0; i < matrix->total_elements; i++) {
            values[i] = -old_values[i];
    }
    return create_tensor3(values, matrix->shape, matrix->dimensions);
}

/**
 * Calculates the derivative of the Negation function.
 * @param function Current function
 * @param matrix output of the Negation function.
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor_ptr derivative_negation(const void *function, const Tensor *matrix, const Tensor *backward) {
    double* values = malloc_(matrix->total_elements * sizeof(double));
    double* backward_values = backward->data;
    for (int i = 0; i < matrix->total_elements; i++) {
        values[i] = -backward_values[i];
    }
    return create_tensor3(values, matrix->shape, matrix->dimensions);
}
