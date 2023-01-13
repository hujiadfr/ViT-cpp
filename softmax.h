//
// Created by Jiarun on 2023/1/5.
//

#ifndef VIT_SOFTMAX_H
#define VIT_SOFTMAX_H

#include <array>

template<typename T, int DIM1, int DIM2>
class Softmax {
public:
    static void forward(std::array<std::array<T, DIM1>, DIM2> &input, std::array<std::array<T, DIM1>, DIM2> &output) {
        for (int i = 0; i < DIM1; ++i) {
            T tmp = 0;
            for (int j = 0; j < DIM2; ++j) {
                tmp += exp(input[i][j]);
            }
            for (int j = 0; j < DIM2; ++j) {
                output[i][j] = exp(input[i][j]) / tmp;
            }
        }
    }
};
#endif //VITS_CPP_SOFTMAX_H
