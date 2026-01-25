//
// Created by Olcay YILDIZ on 20.01.2026.
//

#include <Memory/Memory.h>
#include "DeepNetwork.h"
#include "LinearPerceptron.h"
#include "LinearPerceptronSingleUnit.h"
#include "MultiLayerPerceptron.h"

int main() {
    start_large_memory_check();
    run_linear_perceptron_single_point();
    run_linear_perceptron();
    run_multi_layer_perceptron();
    run_deep_network();
    end_memory_check();
}
