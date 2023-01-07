//
// Created by Jiarun on 2023/1/5.
//

#ifndef VIT_MLP_HEAD_H
#define VIT_MLP_HEAD_H
#include "linear.h"
#include "norm.h"
#include <array>
namespace transformer {
    template<typename T, int DIM, int N_CLASS>
    struct MLPHEADParameter {
        LayerNormParameter<T, DIM> norm1_p;
        LinearParameter<T, DIM, N_CLASS> linear_p1;
        long long count() {
            return linear_p1.count();
        }
    };

    template<typename T, int DIM, int DEP, int N_CLASS>
    class MLP_HEAD{
    public:
        static void forward(std::array<std::array<T, DIM>, DEP> &input,
                            std::array<std::array<T, N_CLASS>, DEP> &output,
                            MLPHEADParameter<T,DIM, N_CLASS> &p){
            auto tmp = std::array<std::array<std::array<T, DIM>, DEP>, 1>{};
            for (int i = 0; i < DEP; ++i){
                LayerNorm<T, DIM>::forward(input[i], tmp[0][i], p.norm1_p);
            }
            for (int i = 0; i < DEP; ++i){
                Linear<T, DIM, N_CLASS>::forward(tmp[0][i],output[i] , p.linear_p1);
            }
        }
    };
}
#endif //VIT_MLP_HEAD_H
