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
        std::array<std::array<std::array<T, KERNEL_WIDTH>, KERNEL_WIDTH>, OUT_CH> weights;
        long long count(){
            long long ret = 0;
            ret += KERNEL_WIDTH * KERNEL_WIDTH * OUT_CH;
            return ret;
        }
    };
    template<typename T, int KERNEL_WIDTH, int FIG_WIDTH, int OUT_WIDTH, int IN_CH, int OUT_CH, int DEP> //outwidth = fig_width/KERNEL_WIDTH
    class Conv2d{
    public:
        static void forward(std::array<std::array<std::array<T, FIG_WIDTH>, FIG_WIDTH>, IN_CH> &input,
                            std::array<std::array<T, OUT_CH>, DEP+1> &output, //for class token add 1
                            std::array<T,OUT_CH> &class_token,
                            Conv2dParameter<T, KERNEL_WIDTH, OUT_CH> &p) {
            auto temp = std::array<std::array<std::array<T, OUT_WIDTH>, OUT_WIDTH>, OUT_CH>{};
            for (int i = 0; i < OUT_WIDTH; i++ )
                for (int j = 0; j < OUT_WIDTH; j++)
                    for (int out_ch = 0; out_ch < OUT_CH; out_ch++) {
                        int tmp = 0;
                        for (int m = 0; m < KERNEL_WIDTH; m++)
                            for (int n = 0; n < KERNEL_WIDTH; n++)
                                for (int in_ch = 0; in_ch < IN_CH; in_ch ++) {
                                    if ((i*KERNEL_WIDTH+m)>=0 && (i*KERNEL_WIDTH-m)<FIG_WIDTH && (j*KERNEL_WIDTH-n)>=0 && (j*KERNEL_WIDTH-n)<FIG_WIDTH) {
                                        tmp += p.weights[OUT_CH][m][n] * input[in_ch][i*KERNEL_WIDTH+m][j*KERNEL_WIDTH+n];
                                    }
                                }
                        temp[out_ch][i][j] = tmp;
                    }
            //flatten the data
            for (int out_ch = 0; out_ch < OUT_CH; out_ch++) { //out_ch = dim
                for (int i = 0; i < OUT_WIDTH; i++)
                    for (int j = 0; j < OUT_WIDTH; j++)
                output[i*OUT_WIDTH+j][out_ch] = temp[out_ch][i][j]; //output[dep][dim]
            }
            //add class token
            for (int out_ch = 0; out_ch < OUT_CH; out_ch++) {
                    output[DEP][out_ch] = class_token[out_ch];
            }
        }

    };


}
#endif //VIT_PATCH_EMBED_H
