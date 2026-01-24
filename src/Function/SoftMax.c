//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#include "SoftMax.h"

#include <math.h>
#include "Memory/Memory.h"

Softmax_ptr create_softmax() {
    Softmax_ptr softmax = malloc_(sizeof(Softmax));
    softmax->function.function_type = SOFTMAX;
    softmax->function.calculate = calculate_softmax;
    softmax->function.derivative = derivative_softmax;
    return softmax;
}

void free_softmax(Softmax_ptr softmax) {
    free_(softmax);
}

/**
 * Computes the Softmax activation for the given tensor.
 * @param function Current function
 * @param tensor The tensor whose values are to be computed.
 * @return Softmax(x).
 */
Tensor_ptr calculate_softmax(const void* function, const Tensor* tensor) {
    double* values = malloc_(tensor->total_elements * sizeof(double));
    const double* old_values = tensor->data;
    int last_dimension_size = tensor->shape[tensor->dimensions - 1];
    double* sum_list = malloc_((tensor->total_elements / last_dimension_size) * sizeof(double));
    double sum = 0.0;
    int k = 0;
    for (int i = 0; i < tensor->total_elements; i++) {
        sum += exp(old_values[i]);
        if ((i + 1) % last_dimension_size == 0) {
            sum_list[k] = sum;
            k++;
            sum = 0.0;
        }
    }
    for (int i = 0; i < tensor->total_elements; i++) {
        values[i] = exp(old_values[i]) / sum_list[i / last_dimension_size];
    }
    free_(sum_list);
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

/**
 * Computes the derivative of the Softmax activation function.
 * @param function Current function
 * @param tensor output of the Softmax(x).
 * @param backward Backward tensor.
 * @return Gradient value of the corresponding node.
 */
Tensor_ptr derivative_softmax(const void *function, const Tensor *tensor, const Tensor *backward) {
    int last_dimension_size = tensor->shape[tensor->dimensions - 1];
    double* values = malloc_(tensor->total_elements * sizeof(double));
    const double* old_values_tensor = tensor->data;
    const double* old_values_backward = backward->data;
    double total = 0.0;
    int k = 0;
    for (int i = 0; i < tensor->total_elements; i++) {
        total += old_values_tensor[i] * old_values_backward[i];
        if ((i + 1) % last_dimension_size == 0) {
            int start_index = i / last_dimension_size;
            for (int j = 0; j < last_dimension_size; j++) {
                values[k] = old_values_backward[start_index * last_dimension_size + j] - total;
                k++;
            }
            total = 0.0;
        }
    }
    Tensor_ptr tmp = create_tensor3(values, tensor->shape, tensor->dimensions);
    Tensor_ptr result = hadamard_product(tensor, tmp);
    free_tensor(tmp);
    return result;
}
