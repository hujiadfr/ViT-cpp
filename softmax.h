//
// Created by Jiarun on 2023/1/5.
//

#ifndef VIT_SOFTMAX_H
#define VIT_SOFTMAX_H

#include <array>

template<typename T, int DIM, int DEP>
class Softmax {
public:
    static void forward(std::array<std::array<T, DIM>, DEP> input, std::array<std::array<T, DIM>, DEP> &output) {
        for (int j = 0; j < DIM; ++j) {
            T tmp = 0;
            for (int i = 0; i < DEP; ++i) {
                tmp += input[i][j];
            }
            for (int i = 0; i < DEP; ++i) {
                output[i][j] = input[i][j] / tmp;
            }
        }
    }
};
#endif //VITS_CPP_SOFTMAX_H
