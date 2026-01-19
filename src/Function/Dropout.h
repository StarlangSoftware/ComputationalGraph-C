//
// Created by Olcay YILDIZ on 23.12.2025.
//

#ifndef COMPUTATIONALGRAPH_DROPOUT_H
#define COMPUTATIONALGRAPH_DROPOUT_H
#include "Function.h"

struct dropout {
    Function function;
    unsigned seed;
    double p;
    double* mask;
};

typedef struct dropout Dropout;
typedef Dropout *Dropout_ptr;

Dropout_ptr create_dropout(double p, unsigned seed);

void free_dropout(Dropout_ptr);

Tensor_ptr calculate_dropout(const void* function, const Tensor* matrix);

Tensor_ptr derivative_dropout(const void* function, const Tensor* value, const Tensor* backward);

#endif //COMPUTATIONALGRAPH_DROPOUT_H