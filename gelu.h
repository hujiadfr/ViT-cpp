//
// Created by Jiarun on 2023/1/5.
//

#ifndef VIT_GELU_H
#define VIT_GELU_H

#include <bits/stdc++.h>
#include <array>
#define PI 3.14159265358979323846
namespace transformer {
    template<typename T, int DIM>
    class Gelu {
    public:
        static void forward(std::array<T, DIM> &input, std::array<T, DIM> &output) {
            for (int i = 0; i < DIM; ++i) {
                output[i] = 0.5*input[i]*(1+tanh(sqrt(2/PI)*(input[i]+0.044715*input[i]*input[i]*input[i])));
            }
        }
    };
}

#endif //VIT_GELU_H
