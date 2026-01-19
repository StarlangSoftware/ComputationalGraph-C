//
// Created by Olcay Taner YILDIZ on 23.12.2025.
//

#ifndef COMPUTATIONALGRAPH_FUNCTION_H
#define COMPUTATIONALGRAPH_FUNCTION_H

#include <Tensor.h>

enum function_type {
    DELU, DROPOUT, ELU, NEGATION, RELU, SIGMOID, SOFTMAX, TANH
};

typedef enum function_type Function_type;

struct function {
    Function_type function_type;
    Tensor_ptr (*calculate)(const void*, const Tensor*);
    Tensor_ptr (*derivative)(const void*, const Tensor*, const Tensor*);
};

typedef struct function Function;

#endif //COMPUTATIONALGRAPH_FUNCTION_H