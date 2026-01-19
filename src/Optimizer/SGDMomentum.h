//
// Created by Olcay YILDIZ on 3.01.2026.
//

#ifndef COMPUTATIONALGRAPH_SGDMOMENTUM_H
#define COMPUTATIONALGRAPH_SGDMOMENTUM_H
#include "Optimizer.h"

struct sgd_momentum {
    Optimizer optimizer;
    Hash_map_ptr velocity_map;
    double momentum;
};

typedef struct sgd_momentum Sgd_momentum;
typedef struct sgd_momentum *Sgd_momentum_ptr;

Sgd_momentum_ptr create_sgd_momentum(double learning_rate, double eta_decrease, double momentum);

void free_sgd_momentum(Sgd_momentum_ptr);

void set_gradients_sgd_momentum(void* sgd, Computational_node_ptr node);

#endif //COMPUTATIONALGRAPH_SGDMOMENTUM_H
