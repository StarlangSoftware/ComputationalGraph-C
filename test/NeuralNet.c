//
// Created by Olcay YILDIZ on 20.01.2026.
//

#include "NeuralNet.h"

#include <Memory/Memory.h>

#include "../src/Node/MultiplicationNode.h"

Tensor_ptr create_input_tensor(Tensor_ptr instance) {
    double* data = malloc_((instance->shape[0]- 1) * sizeof(double));
    for (int i = 0; i < instance->shape[0] - 1; i++) {
        data[i] = instance->data[i];
    }
    const int shape[2] = {1, instance->shape[0] - 1};
    return create_tensor3(data, shape, 2);
}

void train_neural_net(Array_list_ptr train_set, Neural_network_parameter_ptr parameters) {
    Multiplication_node_ptr input = create_multiplication_node(false, true, false);
}

int main() {

}
