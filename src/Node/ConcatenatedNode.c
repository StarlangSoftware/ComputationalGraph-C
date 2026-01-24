//
// Created by Olcay Taner YILDIZ on 29.12.2025.
//

#include "ConcatenatedNode.h"

#include <stdint.h>
#include <stdlib.h>
#include <Memory/Memory.h>

Concatenated_node_ptr create_concatenated_node(int dimension) {
    Concatenated_node_ptr result = malloc_(sizeof(Concatenated_node));
    result->node.type = CONCATENATED_NODE;
    result->dimension = dimension;
    result->node.value = NULL;
    result->node.backward = NULL;
    result->node.function = NULL;
    result->node.is_biased = false;
    result->node.learnable = false;
    result->dimension = dimension;
    result->index_map = create_hash_map((unsigned int (*)(const void *, int)) hash_function_computational_node, (int (*)(const void *, const void *)) compare_computational_node);
    return result;
}

void free_concatenated_node(Concatenated_node_ptr concatenated_node) {
    free_tensor(concatenated_node->node.value);
    free_tensor(concatenated_node->node.backward);
    free_hash_map(concatenated_node->index_map, free_);
    free_(concatenated_node);
}

int compare_computational_node(const Computational_node *first, const Computational_node *second) {
    return ((uintptr_t)first) - ((uintptr_t)second);
}

int get_index_concatenated_node(const Concatenated_node* concatenated_node, const Computational_node *node) {
    return *((int*)hash_map_get(concatenated_node->index_map, node));
}

void add_node(Concatenated_node_ptr concatenated_node, const Computational_node *node) {
    int* value = malloc_(sizeof(int));
    *value = concatenated_node->index_map->count;
    hash_map_insert(concatenated_node->index_map, (void*) node, value);
}

unsigned int hash_function_computational_node(const Computational_node *node, int N) {
    return ((uintptr_t)node) % N;
}
