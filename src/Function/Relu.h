//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#ifndef COMPUTATIONALGRAPH_RELU_H
#define COMPUTATIONALGRAPH_RELU_H

#include "Function.h"

struct relu {
    Function function;
};

typedef struct relu Relu;
typedef Relu *Relu_ptr;

Relu_ptr create_relu();

void free_relu(Relu_ptr relu);

Tensor_ptr calculate_relu(const void* function, const Tensor* matrix);

Tensor_ptr derivative_relu(const void* function, const Tensor* value, const Tensor* backward);

#endif //COMPUTATIONALGRAPH_RELU_H