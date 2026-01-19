//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#ifndef COMPUTATIONALGRAPH_TANH_H
#define COMPUTATIONALGRAPH_TANH_H

#include "Function.h"

struct tanh {
    Function function;
};

typedef struct tanh Tanh;
typedef Tanh *Tanh_ptr;

Tanh_ptr create_tanh();

void free_tanh(Tanh_ptr tanh);

Tensor_ptr calculate_tanh(const void* function, const Tensor* tensor);

Tensor_ptr derivative_tanh(const void* function, const Tensor* tensor, const Tensor* backward);

#endif //COMPUTATIONALGRAPH_TANH_H