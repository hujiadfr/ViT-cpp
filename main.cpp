#include <iostream>
#include "transformer.h"
typedef float T;
// Embedding的维度
#define DIM 768
// patches数量？
#define DEP 196
// FeedForwardNetwork中隐藏层宽度
#define DIM_HID 3072 //768*4
// MultiHeadAttention中Head的数量
#define HEAD_SIZE 8
// Encoder 的层数
#define ENC_LAYER_CNT 12

#define FIG_WIDTH 224
#define KERNEL_WIDTH 16
#define OUT_WIDTH 14 // 224/16 = 14
#define IN_CH 3
#define N_CLASS 14

int main() {
    auto *param = new transformer::transformerParameter<T, DIM, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT, KERNEL_WIDTH, N_CLASS>();
    std::cout << "parameters count: " << param->count() << std::endl;
    auto *input_fig = new std::array<std::array<std::array<T, FIG_WIDTH>, FIG_WIDTH>,IN_CH>{};
    auto *output = new std::array<T, N_CLASS>{};
    auto *class_token = new std::array<T,DIM>{};
    auto *position_embed = new std::array<std::array<T,DIM>, DEP+1>{};

    /*Read Parameters*/
    //class token
    float x;
    std::ifstream srcFile("./parameter/token.txt", std::ios::in);
    if (!srcFile) { //打开失败
        std::cout << "error opening source file." << std::endl;
        return 0;
    }
    int i = 0;
    while (srcFile >> x) {
        (*class_token)[i] = x;
        i++;
    }
    for(i = 0; i < DIM; i++ )
        std::cout<<(*class_token)[i]<<"\n";


    transformer::Transformer<T, DIM, DEP, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT, KERNEL_WIDTH, FIG_WIDTH, OUT_WIDTH ,IN_CH, N_CLASS>::forward(*input_fig,
                                                                                                                                          *output,
                                                                                                                                          *class_token,
                                                                                                                                          *position_embed,
                                                                                                                                          *param);
    return 0;
}
