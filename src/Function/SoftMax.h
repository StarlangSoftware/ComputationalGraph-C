//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#ifndef COMPUTATIONALGRAPH_SOFTMAX_H
#define COMPUTATIONALGRAPH_SOFTMAX_H

#include "Function.h"

struct softmax {
    Function function;
};

typedef struct softmax Softmax;
typedef Softmax *Softmax_ptr;

Softmax_ptr create_softmax();

void free_softmax(Softmax_ptr softmax);

Tensor_ptr calculate_softmax(const void* function, const Tensor* tensor);

Tensor_ptr derivative_softmax(const void* function, const Tensor* tensor, const Tensor* backward);

#endif //COMPUTATIONALGRAPH_SOFTMAX_H