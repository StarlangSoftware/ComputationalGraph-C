//
// Created by Olcay Taner YILDIZ on 29.12.2025.
//

#ifndef COMPUTATIONALGRAPH_MULTIPLICATIONNODE_H
#define COMPUTATIONALGRAPH_MULTIPLICATIONNODE_H
#include "ComputationalNode.h"

struct multiplication_node {
    Computational_node node;
    bool is_hadamard;
    Computational_node_ptr priority_node;
};

typedef struct multiplication_node Multiplication_node;

typedef Multiplication_node* Multiplication_node_ptr;

Multiplication_node_ptr create_multiplication_node(bool learnable, bool is_biased, bool is_hadamard);

Multiplication_node_ptr create_multiplication_node2(bool learnable, bool is_biased, bool is_hadamard, Computational_node_ptr priority_node);

Multiplication_node_ptr create_multiplication_node3(bool learnable, bool is_biased, Tensor_ptr value, bool is_hadamard);

Multiplication_node_ptr create_multiplication_node4(bool learnable, Tensor_ptr value);

Multiplication_node_ptr create_multiplication_node5(Tensor_ptr value);

Multiplication_node_ptr create_multiplication_node6(bool learnable, bool is_biased);

void free_multiplication_node(Multiplication_node_ptr multiplication_node);

#endif //COMPUTATIONALGRAPH_MULTIPLICATIONNODE_H