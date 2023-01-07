//
// Created by Jiarun on 2023/1/5.
//

#ifndef VIT_DROPOUT_H
#define VIT_DROPOUT_H

#include <array>
namespace transformer {
    template<typename T, int DIM>
    class Dropout {
    public:
        static void forward(std::array<T, DIM> &input, std::array<T, DIM> &output, T dropout_rate) {
            for (int i = 0; i < DIM; ++i) {
                if (input[i] < dropout_rate) {
                    output[i] = 0;
                } else {
                    output[i] = input[i];
                }
            }
        }
    };

}
#endif //VIT_DROPOUT_H
