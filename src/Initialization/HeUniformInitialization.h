//
// Created by Olcay Taner YILDIZ on 25.12.2025.
//

#ifndef COMPUTATIONALGRAPH_HEUNIFORMINITIALIZATION_H
#define COMPUTATIONALGRAPH_HEUNIFORMINITIALIZATION_H
#include <math.h>
#include <stdlib.h>
#include <Memory/Memory.h>

double* he_uniform_initialization(int row, int column, unsigned seed);

/**
 * He Uniform Initialization.
 * <p>
 * This method initializes weights using a uniform distribution, which is typically
 * optimized for layers with ReLU activation functions. It helps in maintaining
 * the variance of activations throughout the network layers.
 * </p>
 *
 * @param row    The number of rows in the matrix (typically represents the output size / number of neurons).
 * @param column The number of columns in the matrix (typically represents the input size / fan-in).
 * @param seed The seed used for generating values (allows for reproducibility).
 * @return An array of Doubles containing the initialized weight values.
 */
inline double* he_uniform_initialization(int row, int column, unsigned seed) {
    srandom(seed);
    double* data = malloc_(row * column * sizeof(double));
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            data[i * column + j] = (sqrt(6.0 / column) + sqrt(6.0 / row)) * ((double) random() / (double) RAND_MAX) - sqrt(6.0 / row);
        }
    }
    return data;
}

#endif //COMPUTATIONALGRAPH_HEUNIFORMINITIALIZATION_H
