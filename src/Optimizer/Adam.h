//
// Created by Olcay YILDIZ on 4.01.2026.
//

#ifndef COMPUTATIONALGRAPH_ADAM_H
#define COMPUTATIONALGRAPH_ADAM_H
#include "SGDMomentum.h"

struct Adam {
    Sgd_momentum sgd;
    Hash_map_ptr momentum_map;
    double beta2;
    double epsilon;
};

typedef struct Adam Adam;

typedef struct Adam *Adam_ptr;

Adam_ptr create_adam(double learning_rate, double eta_decrease, double beta1, double beta2, double epsilon);

void free_adam(Adam_ptr adam);

void set_gradients_adam(void* a, Computational_node_ptr node);

#endif //COMPUTATIONALGRAPH_ADAM_H
