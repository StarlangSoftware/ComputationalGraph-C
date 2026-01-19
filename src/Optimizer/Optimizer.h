//
// Created by Olcay YILDIZ on 3.01.2026.
//

#ifndef COMPUTATIONALGRAPH_OPTIMIZER_H
#define COMPUTATIONALGRAPH_OPTIMIZER_H

#include "../Node/ComputationalNode.h"
#include "HashMap/HashSet.h"

struct optimizer {
    void* optimizer;
    void (*set_gradients)(void*, Computational_node_ptr);
    double learning_rate;
    double eta_decrease;
};

typedef struct optimizer Optimizer;

typedef Optimizer* Optimizer_ptr;

Optimizer_ptr create_optimizer(double learning_rate, double eta_decrease);

void set_learning_rate(Optimizer_ptr optimizer);

int broadcast_optimizer(const Computational_node* node);

void update_recursive(Optimizer_ptr optimizer, Hash_set_ptr visited, Computational_node_ptr node, Hash_map_ptr node_map);

void update_values(Optimizer_ptr optimizer, Hash_map_ptr node_map);

#endif //COMPUTATIONALGRAPH_OPTIMIZER_H
