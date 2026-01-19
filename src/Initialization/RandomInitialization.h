//
// Created by Olcay Taner YILDIZ on 25.12.2025.
//

#ifndef COMPUTATIONALGRAPH_RANDOMINITIALIZATION_H
#define COMPUTATIONALGRAPH_RANDOMINITIALIZATION_H
#include <stdlib.h>
#include <Memory/Memory.h>

double* random_initialization(int row, int column, unsigned seed);

/**
 * Random Uniform Initialization.
 * <p>
 * This method initializes the weights with small random values uniformly distributed
 * between -0.01 and 0.01. This is a basic initialization strategy used to break
 * symmetry between neurons.
 * </p>
 *
 * @param row    The number of rows in the matrix.
 * @param column The number of columns in the matrix.
 * @param seed The seed used for generating values.
 * @return An array containing the initialized weight values.
 */
inline double * random_initialization(int row, int column, unsigned seed) {
    srandom(seed);
    double* data = malloc_(row * column * sizeof(double));
    for (int i = 0; i < row * column; i++) {
        data[i] = -0.01 + 0.02 * ((double) random() / (double) RAND_MAX);
    }
    return data;
}

#endif //COMPUTATIONALGRAPH_RANDOMINITIALIZATION_H
