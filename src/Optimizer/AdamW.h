//
// Created by Olcay YILDIZ on 20.01.2026.
//

#ifndef COMPUTATIONALGRAPH_ADAMW_H
#define COMPUTATIONALGRAPH_ADAMW_H

#include "Adam.h"

struct AdamW {
    Adam adam;
    double weight_decay;
};

typedef struct AdamW AdamW;

typedef AdamW *AdamW_ptr;

AdamW_ptr create_adamW(double learning_rate, double eta_decrease, double beta1, double beta2, double weight_decay, double epsilon);

void free_adamW(AdamW_ptr adamW);

void set_gradients_adamW(void* a, Computational_node_ptr node);

void set_attributes_adamW(AdamW* a, double learning_rate, double eta_decrease, double beta1, double beta2, double weight_decay, double epsilon);

#endif //COMPUTATIONALGRAPH_ADAMW_H