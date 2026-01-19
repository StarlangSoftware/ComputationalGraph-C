cmake_minimum_required(VERSION 3.23)
project(ComputationalGraph)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_FLAGS "-O3")

find_package(math_c REQUIRED)
find_package(data_structure_c REQUIRED)
find_package(classification_c REQUIRED)

add_library(ComputationalGraph
        Function/Function.h
        Function/Delu.c
        Function/Delu.h
        Function/Dropout.c
        Function/Dropout.h
        Function/Elu.c
        Function/Elu.h
        Function/Negation.c
        Function/Negation.h
        Function/Relu.c
        Function/Relu.h
        Function/Sigmoid.c
        Function/Sigmoid.h
        Function/SoftMax.c
        Function/SoftMax.h
        Function/Tanh.c
        Function/Tanh.h
        Initialization/HeUniformInitialization.h
        Initialization/RandomInitialization.h
        Initialization/UniformXavierInitialization.h
        Node/ComputationalNode.c
        Node/ComputationalNode.h
        Node/MultiplicationNode.c
        Node/MultiplicationNode.h
        Node/ConcatenatedNode.c
        Node/ConcatenatedNode.h
        Optimizer/Optimizer.c
        Optimizer/Optimizer.h
        Optimizer/SGDMomentum.c
        Optimizer/SGDMomentum.h
        Optimizer/StochasticGradientDescent.c
        Optimizer/StochasticGradientDescent.h
        Optimizer/Adam.c
        Optimizer/Adam.h
        ComputationalGraph.c
        ComputationalGraph.h)
target_link_libraries(ComputationalGraph
        math_c::math_c
        data_structure_c::data_structure_c
        classification_c::classification_c
        m)
