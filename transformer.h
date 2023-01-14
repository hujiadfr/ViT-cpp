//
// Created by Jiarun on 2023/1/6.
//

#ifndef VIT_TRANSFORMER_H
#define VIT_TRANSFORMER_H
#include <array>
#include "encoder.h"
#include "patch_embed.h"
#include "MLP_head.h"

namespace transformer {
    template<typename T, int DIM, int DIM_HID, int HEAD_SIZE, int ENC_LAYER_CNT, int KERNEL_WIDTH, int N_CLASS>
    struct transformerParameter{
        Conv2dParameter<T, KERNEL_WIDTH, DIM> patch_p;
        EncoderParameter<T, DIM, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT> encoder_p;
        MLPHEADParameter<T, DIM, N_CLASS> mlp_head_p;
        long long count() {
            return patch_p.count() + encoder_p.count() + mlp_head_p.count();
        }
    };

    template<typename T, int DIM, int DEP, int DIM_HID, int HEAD_SIZE, int ENC_LAYER_CNT, int KERNEL_WIDTH, int FIG_WIDTH, int OUT_WIDTH ,int IN_CH, int N_CLASS>
    class Transformer {
    public:
        static void forward(std::array<std::array<std::array<T, FIG_WIDTH>, FIG_WIDTH>,IN_CH> &input_fig,
                            std::array<T, N_CLASS> &output,
                            std::array<T,DIM> &class_token,
                            std::array<std::array<T,DIM>, DEP+1> &position_embed,
                            transformerParameter<T, DIM, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT, KERNEL_WIDTH, N_CLASS> p) {
            auto *patches = new std::array<std::array<T, DIM>,DEP+1>{};
            auto *tmp = new std::array<std::array<T, DIM>,DEP+1>{};
        Conv2d<T, KERNEL_WIDTH, FIG_WIDTH, OUT_WIDTH, IN_CH, DIM, DEP>::forward(input_fig, *patches, class_token, position_embed, p.patch_p); //OUT_CH = DIM
//            Encoder<T, DIM, DEP+1, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT>::forward(patches, tmp, p.encoder_p); //we have class token, so dep+1
//            MLP_HEAD<T, DIM, DEP+1, N_CLASS>::forward(tmp, output, p.mlp_head_p);
        delete patches;
        delete tmp;
    }
    };
}

#endif //VIT_TRANSFORMER_H
