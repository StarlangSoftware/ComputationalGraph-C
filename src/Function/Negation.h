//
// Created by Olcay Taner YILDIZ on 24.12.2025.
//

#ifndef COMPUTATIONALGRAPH_NEGATION_H
#define COMPUTATIONALGRAPH_NEGATION_H
#include "Function.h"

struct negation {
    Function function;
};

typedef struct negation Negation;
typedef Negation *Negation_ptr;

Negation_ptr create_negation();

void free_negation(Negation_ptr negation);

Tensor_ptr calculate_negation(const void* function, const Tensor* matrix);

Tensor_ptr derivative_negation(const void* function, const Tensor* matrix, const Tensor* backward);

#endif //COMPUTATIONALGRAPH_NEGATION_H