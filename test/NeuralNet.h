//
// Created by Olcay YILDIZ on 20.01.2026.
//

#ifndef COMPUTATIONALGRAPH_NEURALNET_H
#define COMPUTATIONALGRAPH_NEURALNET_H
#include <ArrayList.h>
#include <Tensor.h>
#include "../src/NeuralNetworkParameter.h"

Tensor_ptr create_input_tensor(Tensor_ptr instance);

void train_neural_net(Array_list_ptr train_set, Neural_network_parameter_ptr parameters);

#endif //COMPUTATIONALGRAPH_NEURALNET_H