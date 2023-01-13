#include <iostream>
#include "transformer.h"
#include "patch_embed.h"
#include "Encoder.h"
#include "read_parameter.h"


typedef double T;
// Embedding的维度
#define DIM 768
// patches数量？
#define DEP 576
// FeedForwardNetwork中隐藏层宽度
#define DIM_HID 3072 //768*4
// MultiHeadAttention中Head的数量
#define HEAD_SIZE 12
// Encoder 的层数
#define ENC_LAYER_CNT 1

#define FIG_WIDTH 384
#define KERNEL_WIDTH 16
#define OUT_WIDTH 24 // 384/16 = 24
#define IN_CH 3
#define N_CLASS 14


int main() {
    auto *param = new transformer::transformerParameter<T, DIM, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT, KERNEL_WIDTH, N_CLASS>{};
    std::cout << "parameters count: " << param->count() << std::endl;
    auto *input_fig = new std::array<std::array<std::array<T, FIG_WIDTH>, FIG_WIDTH>,IN_CH>{};
    auto *output = new std::array<T, N_CLASS>{};
    auto *class_token = new std::array<T,DIM>{};
    auto *position_embed = new std::array<std::array<T,DIM>, DEP+1>{};
    auto *patch_p = new transformer::Conv2dParameter<T, KERNEL_WIDTH, DIM>{};
    auto *encode_p = new transformer::EncoderParameter<T, DIM, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT>{};


    /*******Read Parameters********/
    //class token
    T x;
    std::ifstream token_File("./parameter/token.txt", std::ios::in);
    if (!token_File) { //打开失败
        std::cout << "error opening source file." << std::endl;
        return 0;
    }
    int i = 0;
    while (token_File >> x) {
        (*class_token)[i] = x;
        i++;
    }
    token_File.close();
    std::cout<<"token read end"<<std::endl;

    //weights of embed
    std::ifstream weight_File("./parameter/e_weight.txt", std::ios::in);
    if (!weight_File) { //打开失败
        std::cout << "error opening source file." << std::endl;
        return 0;
    }
    i = 0;
    int j = 0;
    int dim = 0;
    int k = 0;
    for(dim = 0; dim < DIM; dim ++) {
        for (k = 0; k < 3; k++)
            for (i = 0; i < 16; i++)
                for (j = 0; j < 16; j++) {
                    weight_File >> x;
                    (param)->patch_p.weights[dim][k][i][j] = x;
                    patch_p->weights[dim][k][i][j] = x;
                }

    }
    weight_File.close();
    std::cout<<"weight read end"<<std::endl;


    //bias of embed
    std::ifstream bias_File("./parameter/e_bias.txt", std::ios::in);
    if (!bias_File) { //打开失败
        std::cout << "error opening source file." << std::endl;
        return 0;
    }
    i = 0;
    while (bias_File >> x) {
        (param)->patch_p.bias[i] = x;
        patch_p->bias[i] = x;
        i++;
    }
    bias_File.close();

    //position_embed
    std::ifstream emb_File("./parameter/e_pos.txt", std::ios::in);
    if (!emb_File) { //打开失败
        std::cout << "error opening source file." << std::endl;
        return 0;
    }
    for(i = 0; i < DEP+1; i++)
        for(j = 0; j < DIM; j++) {
            emb_File >> x;
            (*position_embed)[i][j] = x;
        }
    emb_File.close();

    /*****READ FIGURE*****/
    std::ifstream fig_File("./parameter/img.txt", std::ios::in);
    for(k = 0; k < 3; k++)
        for(i = 0; i < FIG_WIDTH; i++)
            for(j = 0; j < FIG_WIDTH; j++) {
                fig_File >> x;
                (*input_fig)[k][i][j] = x;
            }
    fig_File.close();
    read_block_parameter<T, DIM, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT>(*encode_p);
//    transformer::Transformer<T, DIM, DEP, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT, KERNEL_WIDTH, FIG_WIDTH, OUT_WIDTH ,IN_CH, N_CLASS>::forward(*input_fig,
//                                                                                                                                          *output,
//                                                                                                                                          *class_token,
//                                                                                                                                          *position_embed,
//                                                                                                                                          *param);
    auto *cov_output = new std::array<std::array<T, DIM>, DEP+1>{};
    transformer::Conv2d<T,  KERNEL_WIDTH,  FIG_WIDTH,  OUT_WIDTH,  IN_CH,  DIM,  DEP>::forward(*input_fig,
                                                                                               *cov_output,
                                                                                               *class_token,
                                                                                               *position_embed,
                                                                                               *patch_p);
    std::cout<<"Patch Embedding End"<<std::endl;
//    std::ofstream outFile("emb_out.txt", std::ios::out);
//    for(i = 0; i < DEP + 1; i++) {
//        for( j = 0; j < DIM; j++) {
//            outFile << (*cov_output)[i][j] << " ";
//        }
//        outFile << "\n";
//    }


    auto *tmp = new std::array<std::array<T, DIM>,DEP+1>{};
    transformer::Encoder<T, DIM, DEP+1, DIM_HID, HEAD_SIZE, ENC_LAYER_CNT>::forward(*cov_output, *tmp, *encode_p);
    delete tmp;
    delete cov_output;
    delete input_fig;
    delete output;
    delete class_token;
    delete position_embed;
    delete patch_p;
    delete encode_p;
    return 0;
}
