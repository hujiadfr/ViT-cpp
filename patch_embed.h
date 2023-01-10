//
// Created by Jiarun on 2023/1/5.
//

#ifndef VIT_PATCH_EMBED_H
#define VIT_PATCH_EMBED_H
#include <array>
#include "linear.h"
namespace transformer {
    template<typename T, int KERNEL_WIDTH, int OUT_CH> //OUT_CH = DIM
    struct Conv2dParameter{
        std::array<std::array<std::array<std::array<T, KERNEL_WIDTH>, KERNEL_WIDTH>, 3>, OUT_CH> weights; //3 is in_ch
        std::array<T, OUT_CH> bias;
        long long count(){
            long long ret = 0;
            ret += KERNEL_WIDTH * KERNEL_WIDTH * OUT_CH * 3;
            return ret;
        }
        T dr;
        Conv2dParameter() {
            this->dr = 0.1;
        }
    };
    template<typename T, int KERNEL_WIDTH, int FIG_WIDTH, int OUT_WIDTH, int IN_CH, int OUT_CH, int DEP> //outwidth = fig_width/KERNEL_WIDTH
    class Conv2d{
    public:
        static void forward(std::array<std::array<std::array<T, FIG_WIDTH>, FIG_WIDTH>, IN_CH> &input,
                            std::array<std::array<T, OUT_CH>, DEP+1> &output, //for class token add 1
                            std::array<T, OUT_CH> &class_token,
                            std::array<std::array<T, OUT_CH>, DEP+1> &position_embed,
                            Conv2dParameter<T, KERNEL_WIDTH, OUT_CH> &p) {
            auto *temp = new std::array<std::array<std::array<T, OUT_WIDTH>, OUT_WIDTH>, OUT_CH>{} ;
            auto *temp2 = new std::array<std::array<T, OUT_CH>, DEP + 1>{};
            for (int i = 0; i < OUT_WIDTH; i++ )
                for (int j = 0; j < OUT_WIDTH; j++)
                    for (int out_ch = 0; out_ch < OUT_CH; out_ch++) {
                        T tmp = 0;
                        for (int m = 0; m < KERNEL_WIDTH; m++)
                            for (int n = 0; n < KERNEL_WIDTH; n++)
                                for (int in_ch = 0; in_ch < IN_CH; in_ch ++) {
                                    if ((i*KERNEL_WIDTH+m)>=0 && (i*KERNEL_WIDTH+m)<FIG_WIDTH && (j*KERNEL_WIDTH+n)>=0 && (j*KERNEL_WIDTH+n)<FIG_WIDTH) {
                                        tmp += p.weights[out_ch][in_ch][m][n] * input[in_ch][i*KERNEL_WIDTH+m][j*KERNEL_WIDTH+n];
                                    }
                                }
                        (*temp)[out_ch][i][j] = tmp + p.bias[out_ch];
                    }
            //flatten the data
            for (int out_ch = 0; out_ch < OUT_CH; out_ch++) { //out_ch = dim
                for (int i = 0; i < OUT_WIDTH; i++)
                    for (int j = 0; j < OUT_WIDTH; j++)
                        (*temp2)[i * OUT_WIDTH + j + 1][out_ch] = (*temp)[out_ch][i][j]; //output[dep][dim]
            }
            //add class token
            for (int out_ch = 0; out_ch < OUT_CH; out_ch++) {
                (*temp2)[0][out_ch] = class_token[out_ch];
            }
            //add position embedding
            for (int i = 0; i < DEP+1; i++)
                for(int out_ch = 0; out_ch < OUT_CH; out_ch++) {
                    (*temp2)[i][out_ch] += position_embed[i][out_ch];
                }
            for (int i = 0; i < DEP+1; i++)
                for(int out_ch = 0; out_ch < OUT_CH; out_ch++) {
                    output[i][out_ch] = (*temp2)[i][out_ch];
                }
            //dropput
//            for (int i = 0; i < DEP+1; i++) {
//                Dropout<T, OUT_CH>::forward((*temp2)[i], output[i], p.dr);
//            }
            delete temp2;
            delete temp;
        }

    };
}
#endif //VIT_PATCH_EMBED_H
