//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#ifndef COMPUTATIONALGRAPH_SIGMOID_H
#define COMPUTATIONALGRAPH_SIGMOID_H

#include "Function.h"

struct sigmoid {
    Function function;
};

typedef struct sigmoid Sigmoid;
typedef Sigmoid *Sigmoid_ptr;

Sigmoid_ptr create_sigmoid();

void free_sigmoid(Sigmoid_ptr sigmoid);

Tensor_ptr calculate_sigmoid(const void* function, const Tensor* tensor);

Tensor_ptr derivative_sigmoid(const void* function, const Tensor* tensor, const Tensor* backward);

#endif //COMPUTATIONALGRAPH_SIGMOID_H