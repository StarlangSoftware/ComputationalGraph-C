//
// Created by Olcay YILDIZ on 25.01.2026.
//

#include "NeuralNet.h"
#include <Memory/Memory.h>

Array_list_ptr get_class_labels_classification(Computational_node_ptr output_node) {
    Array_list_ptr class_indices = create_array_list();
    Tensor_ptr output_value = output_node->value;
    int cols = output_value->shape[1];
    double max_value = -1;
    int label_index = -1;
    int indices[2] = {0, 0};
    for (int i = 0; i < cols; i++) {
        indices[1] = i;
        double value = get_tensor_value(output_value, indices);
        if (value > max_value) {
            max_value = value;
            label_index = i;
        }
    }
    array_list_add_int(class_indices, label_index);
    return class_indices;
}

Classification_performance_ptr test_classification(struct computational_graph* graph, Array_list_ptr test_set){
    int count = 0, total = 0;
    for (int i = 0; i < test_set->size; i++) {
        Tensor_ptr instance = array_list_get(test_set, i);
        Computational_node_ptr input = array_list_get(graph->input_nodes, 0);
        set_node_value(input, create_input_tensor(instance));
        Array_list_ptr output = predict_by_computational_graph(graph);
        int class_label = array_list_get_int(output, 0);
        free_array_list(output, free_);
        int index[1] = {instance->shape[0] - 1};
        if (class_label == (int) get_tensor_value(instance, index)) {
            count++;
        }
        total++;
    }
    return create_classification_performance(count / (total + 0.0));
}

Tensor_ptr create_input_tensor(Tensor_ptr instance) {
    double* data = malloc_((instance->shape[0] - 1) * sizeof(double));
    for (int i = 0; i < instance->shape[0] - 1; i++) {
        data[i] = instance->data[i];
    }
    const int shape[2] = {1, instance->shape[0] - 1};
    return create_tensor3(data, shape, 2);
}

void create_iris_dataset(Array_list_ptr train_set, Array_list_ptr test_set) {
    int strides[] = {5};
    for (int i = 0; i < 150; i++) {
        Tensor_ptr input = create_tensor(iris_data[i], strides, 1);
        if (i % 5 != 0) {
            array_list_add(train_set, input);
        } else {
            array_list_add(test_set, input);
        }
    }
}
