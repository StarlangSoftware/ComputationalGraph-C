//
// Created by Olcay Taner YILDIZ on 29.12.2025.
//

#ifndef COMPUTATIONALGRAPH_CONCATENATEDNODE_H
#define COMPUTATIONALGRAPH_CONCATENATEDNODE_H
#include <HashMap/HashMap.h>

#include "ComputationalNode.h"

struct concatenated_node {
    Computational_node node;
    Hash_map_ptr index_map;
    int dimension;
};

typedef struct concatenated_node Concatenated_node;

typedef Concatenated_node* Concatenated_node_ptr;

Concatenated_node_ptr create_concatenated_node(int dimension);

void free_concatenated_node(Concatenated_node_ptr concatenated_node);

unsigned int hash_function_computational_node(const Computational_node *node, int N);

int compare_computational_node(const Computational_node *first, const Computational_node *second);

int get_index_concatenated_node(const Concatenated_node* concatenated_node, const Computational_node *node);

void add_node(Concatenated_node_ptr concatenated_node, const Computational_node *node);

#endif //COMPUTATIONALGRAPH_CONCATENATEDNODE_H