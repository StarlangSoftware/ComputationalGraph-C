//
// Created by Olcay Taner YILDIZ on 23.12.2025.
//

#ifndef COMPUTATIONALGRAPH_DELU_H
#define COMPUTATIONALGRAPH_DELU_H
#include "Function.h"

struct delu {
    Function function;
    double a;
    double b;
    double xc;
};

typedef struct delu Delu;
typedef Delu *Delu_ptr;

Delu_ptr create_delu(double a, double b, double xc);

Delu_ptr create_delu2();

void free_delu(Delu_ptr delu);

Tensor_ptr calculate_delu(const void* function, const Tensor* matrix);

Tensor_ptr derivative_delu(const void* function, const Tensor* value, const Tensor* backward);

#endif //COMPUTATIONALGRAPH_DELU_H