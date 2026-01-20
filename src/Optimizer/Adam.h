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
    double current_beta1;
    double current_beta2;
};

typedef struct Adam Adam;

typedef struct Adam *Adam_ptr;

Adam_ptr create_adam(double learning_rate, double eta_decrease, double beta1, double beta2, double epsilon);

void free_adam(Adam_ptr adam);

double* calculate_gradients_adam(void *a, Computational_node_ptr node);

void set_gradients_adam(void* a, Computational_node_ptr node);

void set_attributes_adam(Adam_ptr adam, double learning_rate, double eta_decrease, double beta1, double beta2, double epsilon);

#endif //COMPUTATIONALGRAPH_ADAM_H
