//
// Created by Olcay Taner YILDIZ on 25.12.2025.
//

#ifndef COMPUTATIONALGRAPH_UNIFORMXAVIERINITIALIZATION_H
#define COMPUTATIONALGRAPH_UNIFORMXAVIERINITIALIZATION_H
#include <stdlib.h>
#include <tgmath.h>
#include <Memory/Memory.h>

double* uniform_xavier_initialization(int row, int column, unsigned seed);

/**
 * Xavier Uniform Initialization.
 * <p>
 * This method initializes weights using a uniform distribution within the range
 * [-limit, limit], where the limit is sqrt(6 / (fan_in + fan_out)).
 * This strategy is designed to keep the scale of the gradients roughly the same
 * in all layers and is commonly used with Sigmoid or Tanh activation functions.
 * </p>
 *
 * @param row    The number of rows in the matrix (typically represents fan-out / output size).
 * @param column The number of columns in the matrix (typically represents fan-in / input size).
 * @param seed The seed used for generating values.
 * @return An array containing the initialized weight values.
 */
inline double* uniform_xavier_initialization(int row, int column, unsigned seed) {
    srandom(seed);
    double* data = malloc_(row * column * sizeof(double));
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            data[i * column + j] = (2 * ((double) random() / (double) RAND_MAX) - 1) * sqrt(6.0 / (row + column));
        }
    }
    return data;
}

#endif //COMPUTATIONALGRAPH_UNIFORMXAVIERINITIALIZATION_H