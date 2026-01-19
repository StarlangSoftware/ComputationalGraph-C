//
// Created by Olcay Taner YILDIZ on 29.12.2025.
//

#include "MultiplicationNode.h"
#include <stdlib.h>
#include <Memory/Memory.h>

Multiplication_node_ptr create_multiplication_node(bool learnable, bool is_biased, bool is_hadamard) {
    Multiplication_node_ptr result = malloc_(sizeof(Multiplication_node));
    result->node.type = MULTIPLICATION_NODE;
    result->node.learnable = learnable;
    result->node.is_biased = is_biased;
    result->node.backward = NULL;
    result->node.value = NULL;
    result->is_hadamard = is_hadamard;
    result->priority_node = NULL;
    return result;
}

Multiplication_node_ptr create_multiplication_node2(bool learnable, bool is_biased, bool is_hadamard,
    Computational_node_ptr priority_node) {
    Multiplication_node_ptr result = create_multiplication_node(learnable, is_biased, is_hadamard);
    result->priority_node = priority_node;
    return result;
}

Multiplication_node_ptr create_multiplication_node3(bool learnable, bool is_biased, Tensor_ptr value, bool is_hadamard) {
    Multiplication_node_ptr result = create_multiplication_node(learnable, is_biased, is_hadamard);
    result->node.value = value;
    return result;
}

void free_multiplication_node(Multiplication_node_ptr multiplication_node) {
    free_(multiplication_node);
    free_tensor(multiplication_node->node.value);
    free_tensor(multiplication_node->node.backward);
}
