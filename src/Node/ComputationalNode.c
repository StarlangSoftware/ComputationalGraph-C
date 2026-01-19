//
// Created by Olcay Taner YILDIZ on 25.12.2025.
//

#include "ComputationalNode.h"
#include <Memory/Memory.h>

/**
 * Initializes a ComputationalNode.
 * @param learnable Indicates whether the node is learnable (e.g., weights)
 * @param is_biased Indicates whether the node is biased
 * @param function The function (e.g., activation like SIGMOID)
 * @param value The tensor value associated with the node (optional)
 */
Computational_node_ptr create_computational_node(bool learnable, bool is_biased, Function* function, Tensor_ptr value) {
    Computational_node_ptr result = (Computational_node_ptr)malloc_(sizeof(Computational_node));
    result->type = COMPUTATIONAL_NODE;
    result->learnable = learnable;
    result->is_biased = is_biased;
    result->function = function;
    result->value = value;
    result->backward = NULL;
    return result;
}

/**
 * Constructor overload for function type initialization
 */
Computational_node_ptr create_computational_node2(bool learnable, bool is_biased, Function* function) {
    return create_computational_node(learnable, is_biased, function, NULL);
}

/**
 * Constructor overload for operator initialization
 */
Computational_node_ptr create_computational_node3(bool learnable, bool is_biased) {
    return create_computational_node(learnable, is_biased, NULL, NULL);
}

void free_computational_node(Computational_node_ptr node) {
    free_tensor(node->value);
    free_tensor(node->backward);
    free_(node);
}

void update_value(Computational_node_ptr node) {
    Tensor_ptr result = add_tensors(node->value, node->backward);
    free_tensor(node->value);
    node->value = result;
}

void set_node_backward(Computational_node_ptr node, Tensor_ptr backward) {
    if (node->backward != NULL) {
        free_tensor(node->backward);
    }
    node->backward = backward;
}

void set_node_value(Computational_node_ptr node, Tensor_ptr value) {
    if (node->value != NULL) {
        free_tensor(node->value);
    }
    node->value = value;
}
