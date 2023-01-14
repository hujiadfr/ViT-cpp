//
// Created by Jiarun on 2023/1/5.
//

#ifndef VIT_ENCODER_H
#define VIT_ENCODER_H

#include <array>
#include "linear.h"
#include "MLP.h"
#include "norm.h"
#include "attention.h"

namespace transformer {

    template<typename T, int DIM, int DIM_HID>
    struct EncoderLayerParameter {
        LayerNormParameter<T, DIM> norm1_p;
        MultiHeadAttentionParameter <T, DIM> attn_p;
        LayerNormParameter<T, DIM> norm2_p;
        MLPParameter <T, DIM, DIM, DIM_HID> ff_p;

        long long count() {
            return norm1_p.count() + attn_p.count() + norm2_p.count() + ff_p.count();
        }
    };

    template<typename T, int DIM, int DEP, int DIM_HID, int HEAD_SIZE>
    class EncoderLayer {
    public:
        static void forward(std::array<std::array<T, DIM>, DEP> &input,
                            std::array<std::array<T, DIM>, DEP> &output,
                            EncoderLayerParameter<T, DIM, DIM_HID> &p) {
            auto *tmp = new std::array<std::array<std::array<T, DIM>, DEP>, 4>{};
            for (int i = 0; i < DEP; ++i) {
                LayerNorm<T, DIM>::forward(input[i], (*tmp)[0][i], p.norm1_p);
            }
            MultiHeadAttention<T, DIM, DEP, HEAD_SIZE>::forward((*tmp)[0], (*tmp)[0], (*tmp)[0], (*tmp)[1], p.attn_p);
            for (int i = 0; i < DEP; ++i) {
                for (int j = 0; j < DIM; ++j) {
                    (*tmp)[1][i][j] += input[i][j];
                }
            }
            for (int i = 0; i < DEP; ++i) {
                LayerNorm<T, DIM>::forward((*tmp)[1][i], (*tmp)[2][i], p.norm2_p);
            }
            for (int i = 0; i < DEP; ++i) {
                MLP<T, DIM, DIM, DIM_HID>::forward((*tmp)[2][i], (*tmp)[3][i], p.ff_p);
            }

            for (int i = 0; i < DEP; ++i) {
                for (int j = 0; j < DIM; ++j) {
                    output[i][j] = (*tmp)[1][i][j] + (*tmp)[3][i][j];
                }
            }
            delete tmp;
        }
    };


    template<typename T, int DIM, int DIM_HID, int LAYER_CNT>
    struct EncoderParameter {
        std::array<EncoderLayerParameter<T, DIM, DIM_HID>, LAYER_CNT> layers_p;

        long long count() {
            return layers_p[0].count() * LAYER_CNT;
        }
    };

    template<typename T, int DIM, int DEP, int DIM_HID, int HEAD_SIZE, int LAYER_CNT>
    class Encoder {
    public:
        static void forward(std::array<std::array<T, DIM>, DEP> &input,
                            std::array<std::array<T, DIM>, DEP> &output,
                            EncoderParameter<T, DIM, DIM_HID, LAYER_CNT> &p) {
            auto *tmp = new std::array<std::array<std::array<T, DIM>, DEP>, LAYER_CNT-1>{};
            for (int i = 0; i < LAYER_CNT; ++i) {
                std::cout<<"Start block "<<i<<std::endl;
                if (i == 0) {
                    EncoderLayer<T, DIM, DEP, DIM_HID, HEAD_SIZE>::forward(input, (*tmp)[0], p.layers_p[i]);
                }
                else if (i == LAYER_CNT - 1) {
                    EncoderLayer<T, DIM, DEP, DIM_HID, HEAD_SIZE>::forward((*tmp)[i - 1], output, p.layers_p[i]);
                }
                else {
                    EncoderLayer<T, DIM, DEP, DIM_HID, HEAD_SIZE>::forward((*tmp)[i - 1], (*tmp)[i], p.layers_p[i]);
                }
            }
            delete tmp;
        }
    };
}

#endif //VIT_ENCODER_H
