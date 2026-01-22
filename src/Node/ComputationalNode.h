//
// Created by Olcay Taner YILDIZ on 25.12.2025.
//

#ifndef COMPUTATIONALGRAPH_COMPUTATIONALNODE_H
#define COMPUTATIONALGRAPH_COMPUTATIONALNODE_H
#include <stdbool.h>
#include <Tensor.h>
#include "../Function/Function.h"

enum computational_node_type {
    COMPUTATIONAL_NODE, CONCATENATED_NODE, MULTIPLICATION_NODE
};

typedef enum computational_node_type Computational_node_type;

struct computational_node {
    Computational_node_type type;
    Tensor_ptr value;
    Tensor_ptr backward;
    bool learnable;
    bool is_biased;
    void* function;
};

typedef struct computational_node Computational_node;
typedef Computational_node* Computational_node_ptr;

Computational_node_ptr create_computational_node(bool learnable, bool is_biased, Function* function, Tensor_ptr value);

Computational_node_ptr create_computational_node2(bool learnable, bool is_biased, Function* function);

Computational_node_ptr create_computational_node3(bool learnable, bool is_biased);

void free_computational_node(Computational_node_ptr node);

void update_value(Computational_node_ptr node);

void set_node_backward(Computational_node_ptr node, Tensor_ptr backward);

void set_node_value(Computational_node_ptr node, Tensor_ptr value);

#endif //COMPUTATIONALGRAPH_COMPUTATIONALNODE_H