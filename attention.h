//
// Created by Jiarun on 2023/1/5.
//

#ifndef VIT_ATTENTION_H
#define VIT_ATTENTION_H
#include <cmath>
#include <array>

#include "linear.h"
#include "softmax.h"

namespace transformer {
    template<typename T, int DIM>
    struct MultiHeadAttentionParameter {
        LinearParameter <T, DIM, DIM> linear_q_p, linear_k_p, linear_v_p;
        LinearParameter<T, DIM, DIM> linear_p;

        long long count() {
            return linear_k_p.count() * 3 + linear_p.count();
        }
    };

    template<typename T, int DIM, int DEP, int HEAD_SIZE>
    class MultiHeadAttention {
    public:
        static void forward(std::array<std::array<T, DIM>, DEP> &q_in,
                            std::array<std::array<T, DIM>, DEP> &k_in,
                            std::array<std::array<T, DIM>, DEP> &v_in,
                            std::array<std::array<T, DIM>, DEP> &output,
                            MultiHeadAttentionParameter<T, DIM> &p) {

            auto *q_tmp = new std::array<std::array<T, DIM>, DEP>{};
            auto *k_tmp = new std::array<std::array<T, DIM>, DEP>{};
            auto *v_tmp = new std::array<std::array<T, DIM>, DEP>{};
            auto *q_tmp_split = new std::array<std::array<std::array<T, DIM/HEAD_SIZE>, DEP>, HEAD_SIZE>{};
            auto *k_tmp_split = new std::array<std::array<std::array<T, DIM/HEAD_SIZE>, DEP>, HEAD_SIZE>{};
            auto *v_tmp_split = new std::array<std::array<std::array<T, DIM/HEAD_SIZE>, DEP>, HEAD_SIZE>{};
            auto *q_k_mul = new std::array<std::array<std::array<T, DEP>, DEP>, HEAD_SIZE>{};
            auto *q_k_mul_softmax = new std::array<std::array<std::array<T, DEP>, DEP>, HEAD_SIZE>{};
            auto *q_k_v_mul = new std::array<std::array<std::array<T, DIM/HEAD_SIZE>, HEAD_SIZE>, DEP>{};
            auto *fc_tmp = new std::array<std::array<T, DIM>, DEP> {};

            MultiLinear<T, DIM, DIM, DEP>::forward(q_in, *q_tmp, p.linear_q_p);
            MultiLinear<T, DIM, DIM, DEP>::forward(k_in, *k_tmp, p.linear_k_p);
            MultiLinear<T, DIM, DIM, DEP>::forward(v_in, *v_tmp, p.linear_v_p);

            //split q,k,v
            for(int i = 0; i < DEP; i++)
                for(int j = 0; j < HEAD_SIZE; j++)
                    for(int k = 0; k < DIM/HEAD_SIZE; k++) {
                        (*q_tmp_split)[j][i][k] = (*q_tmp)[i][j*DIM/HEAD_SIZE + k];
                        (*k_tmp_split)[j][i][k] = (*k_tmp)[i][j*DIM/HEAD_SIZE + k];
                        (*v_tmp_split)[j][i][k] = (*v_tmp)[i][j*DIM/HEAD_SIZE + k];
                    }
            // mat q*k

            for(int k = 0; k < HEAD_SIZE; k++) {
                for(int m = 0; m < DEP; m++)
                    for(int n = 0; n < DEP; n++) {
                        (*q_k_mul)[k][m][n] = 0;
                        for(int i = 0; i < DIM/HEAD_SIZE; i++)
                            (*q_k_mul)[k][m][n] += (*q_tmp_split)[k][m][i] * (*k_tmp_split)[k][n][i];
                        (*q_k_mul)[k][m][n] = (*q_k_mul)[k][m][n] / sqrt((double)DIM/HEAD_SIZE);
                    }
                Softmax<T, DEP, DEP>::forward((*q_k_mul)[k], (*q_k_mul_softmax)[k]);
            }
            //mat q*k*v
            for(int k = 0; k < HEAD_SIZE; k++)
                for(int m = 0; m < DEP; m++)
                    for(int n = 0; n < DIM/HEAD_SIZE; n++) {
                        (*q_k_v_mul)[m][k][n] = 0;
                        for(int i = 0; i < DEP; i++)
                            (*q_k_v_mul)[m][k][n] += (*q_k_mul_softmax)[k][m][i] * (*v_tmp_split)[k][i][n];
                    }
            //flatten
            for(int i = 0; i < DEP; i++)
                for(int m = 0; m < HEAD_SIZE; m++)
                    for(int n = 0; n < DIM/HEAD_SIZE; n++) {
                        (*fc_tmp)[i][m*(DIM/HEAD_SIZE) + n] = (*q_k_v_mul)[i][m][n];
                    }

            MultiLinear<T, DIM, DIM, DEP>::forward(*fc_tmp, output, p.linear_p);

            //free pointers
            delete q_tmp;
            delete k_tmp;
            delete v_tmp;
            delete q_tmp_split;
            delete k_tmp_split;
            delete v_tmp_split;
            delete q_k_mul;
            delete q_k_mul_softmax;
            delete q_k_v_mul;
            delete fc_tmp;

        }
    };
}
#endif //VIT_ATTENTION_H
