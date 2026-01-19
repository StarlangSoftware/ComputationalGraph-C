//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#ifndef COMPUTATIONALGRAPH_ELU_H
#define COMPUTATIONALGRAPH_ELU_H

#include "Function.h"

struct elu {
    Function function;
    double a;
};

typedef struct elu Elu;
typedef Elu *Elu_ptr;

Elu_ptr create_elu(double a);

Elu_ptr create_elu2();

void free_elu(Elu_ptr elu);

Tensor_ptr calculate_elu(const void* function, const Tensor* matrix);

Tensor_ptr derivative_elu(const void* function, const Tensor* matrix, const Tensor* backward);

#endif //COMPUTATIONALGRAPH_ELU_H