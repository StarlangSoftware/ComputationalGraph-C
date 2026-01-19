//
// Created by Olcay YILDIZ on 23.12.2025.
//

#include "Dropout.h"
#include <stdlib.h>
#include <Memory/Memory.h>

Dropout_ptr create_dropout(double p, unsigned seed) {
    Dropout_ptr result = malloc_(sizeof(Dropout));
    result->p = p;
    result->seed = seed;
    result->function.function_type = DROPOUT;
    result->function.calculate = calculate_dropout;
    result->function.derivative = derivative_dropout;
    result->mask = NULL;
    return result;
}

void free_dropout(Dropout_ptr dropout) {
    free_(dropout);
    free_(dropout->mask);
}

/**
 * Computes the dropout values for the given value tensor.
 * @param function Current function
 * @param matrix The tensor whose values are to be computed.
 * @return Output tensor.
 */
Tensor_ptr calculate_dropout(const void *function, const Tensor *matrix) {
    Dropout_ptr dropout = (Dropout_ptr) function;
    srandom(dropout->seed);
    free_(dropout->mask);
    dropout->mask = malloc_(matrix->total_elements * sizeof(double));
    const double multiplier = 1.0 / (1 - dropout->p);
    double* values = malloc_(matrix->total_elements * sizeof(double));
    const double* old_values = matrix->data;
    for (int i = 0; i < matrix->total_elements; i++) {
        double r = (double) random() / (double) RAND_MAX;
        if (r > dropout->p) {
            dropout->mask[i] = multiplier;
            values[i] = old_values[i] * multiplier;
        } else {
            dropout->mask[i] = 0;
            values[i] = 0;
        }
    }
    return create_tensor3(values, matrix->shape, matrix->dimensions);
}

/**
 * Calculates the derivative of the dropout.
 * @param function Current function
 * @param value output of the dropout function.
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor_ptr derivative_dropout(const void *function, const Tensor *value, const Tensor *backward) {
    const Dropout* dropout = (Dropout_ptr) function;
    Tensor_ptr tmp = create_tensor(dropout->mask, value->shape, value->dimensions);
    Tensor_ptr result = hadamard_product(backward, tmp);
    free_tensor(tmp);
    return result;
}
