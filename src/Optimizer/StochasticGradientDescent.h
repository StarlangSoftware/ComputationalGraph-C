//
// Created by Olcay YILDIZ on 3.01.2026.
//

#ifndef COMPUTATIONALGRAPH_STOCHASTICGRADIENTDESCENT_H
#define COMPUTATIONALGRAPH_STOCHASTICGRADIENTDESCENT_H

#include "Optimizer.h"
#include "../Node/ComputationalNode.h"

Optimizer_ptr create_stochastic_gradient(double learning_rate, double eta_decrease);

void set_gradients_stochastic_gradient_descent(void* sgd, Computational_node_ptr node);

#endif //COMPUTATIONALGRAPH_STOCHASTICGRADIENTDESCENT_H