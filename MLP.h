//
// Created by Jiarun on 2023/1/5.
//

#ifndef VIT_MLP_H
#define VIT_MLP_H
#include <array>

#include "linear.h"
#include "dropout.h"
#include "gelu.h"

namespace transformer {
    template<typename T, int DIM_IN, int DIM_OUT, int DIM_HID>
    struct MLPParameter {
        LinearParameter<T, DIM_IN, DIM_HID> linear_p1;
        LinearParameter<T, DIM_HID, DIM_OUT> linear_p2;
        T dr;

        MLPParameter() {
            this->dr = 0.1;
        }

        long long count() {
            return linear_p1.count() + linear_p2.count();
        }
    };

    template<typename T, int DIM_IN, int DIM_OUT, int DIM_HID>
    class MLP {
    public:
        static void forward(std::array<T, DIM_IN> &input,
                            std::array<T, DIM_OUT> &output,
                            MLPParameter<T, DIM_IN, DIM_OUT, DIM_HID> &ff_p) {
            auto tmp = std::array<std::array<T, DIM_HID>, 3>{};
//            auto tmp2 = std::array<T, DIM_OUT>{};
            Linear<T, DIM_IN, DIM_HID>::forward(input, tmp[0], ff_p.linear_p1);
            Gelu<T, DIM_HID>::forward(tmp[0], tmp[1]);
//            Dropout<T, DIM_HID>::forward(tmp[1], tmp[2], ff_p.dr);
            Linear<T, DIM_HID, DIM_OUT>::forward(tmp[1], output, ff_p.linear_p2);
//            Dropout<T, DIM_OUT>::forward(tmp2, output, ff_p.dr);
        }
    };

}
#endif //VIT_MLP_H
