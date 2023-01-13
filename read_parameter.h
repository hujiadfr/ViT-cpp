//
// Created by Jiarun on 2023/1/8.
//

#ifndef VIT_READ_PARAMETER_H
#define VIT_READ_PARAMETER_H
// now only for encode_parameter
#include "Encoder.h"
template<typename T, int DIM, int DIM_HID, int HEAD_SIZE, int LAYER_CNT>
void read_block_parameter(transformer::EncoderParameter<T, DIM, DIM_HID, HEAD_SIZE, LAYER_CNT> &encode_parameter) {
    std::ifstream norm1_bias_File;
    std::ifstream norm1_weight_File;
    std::ifstream norm2_bias_File;
    std::ifstream norm2_weight_File;
    std::ifstream proj_k_bias_File;
    std::ifstream proj_q_bias_File;
    std::ifstream proj_v_bias_File;
    std::ifstream proj_k_weight_File;
    std::ifstream proj_q_weight_File;
    std::ifstream proj_v_weight_File;
    std::ifstream proj_weight_File;
    std::ifstream proj_bias_File;
    std::ifstream pwff_fc1_weight_File;
    std::ifstream pwff_fc1_bias_File;
    std::ifstream pwff_fc2_weight_File;
    std::ifstream pwff_fc2_bias_File;
    std::string path;
    std::string path2;
    std::string path3;
    for (int block = 0; block < LAYER_CNT; block ++) {
        std::cout << "read block " << block << " Parameters Start";
        /*****norm1_bias*****/
        path = "./parameter/block";
        path.append(std::to_string(block));
        path.append("/norm1_bias.txt");
        norm1_bias_File.open(path, std::ios::in);
        if (!norm1_bias_File) { //打开失败
            std::cout << "error opening source file." << std::endl;
            return;
        }
        for (int i = 0; i < DIM; i++) {
            norm1_bias_File >> encode_parameter.layers_p[block].norm1_p.bias[i];
        }
        norm1_bias_File.close();
        /*****norm1_weights*****/
        path = "./parameter/block";
        path.append(std::to_string(block));
        path.append("/norm1_weight.txt");
        norm1_weight_File.open(path, std::ios::in);
        if (!norm1_weight_File) {
            std::cout << "error opening source file." << std::endl;
            return;
        }
        for (int i = 0; i < DIM; i++) {
            norm1_weight_File >> encode_parameter.layers_p[block].norm1_p.weights[i];
        }
        norm1_weight_File.close();

        /*****norm2_bias*****/
        path = "./parameter/block";
        path.append(std::to_string(block));
        path.append("/norm2_bias.txt");
        norm2_bias_File.open(path, std::ios::in);
        if (!norm2_bias_File) { //打开失败
            std::cout << "error opening source file." << std::endl;
            return;
        }
        for (int i = 0; i < DIM; i++) {
            norm2_bias_File >> encode_parameter.layers_p[block].norm2_p.bias[i];
        }
        norm2_bias_File.close();
        /*****norm2_weights*****/
        path = "./parameter/block";
        path.append(std::to_string(block));
        path.append("/norm2_weight.txt");
        norm2_weight_File.open(path, std::ios::in);
        if (!norm2_weight_File) {
            std::cout << "error opening source file." << std::endl;
            return;
        }
        for (int i = 0; i < DIM; i++) {
            norm2_weight_File >> encode_parameter.layers_p[block].norm2_p.weights[i];
        }
        norm2_weight_File.close();

        /******read Multi Attention bias********/
        path = "./parameter/block";
        path.append(std::to_string(block));
        path.append("/proj_k_bias.txt");
        path2 = "./parameter/block";
        path2.append(std::to_string(block));
        path2.append("/proj_q_bias.txt");
        path3 = "./parameter/block";
        path3.append(std::to_string(block));
        path3.append("/proj_v_bias.txt");
        proj_k_bias_File.open(path, std::ios::in);
        proj_q_bias_File.open(path2, std::ios::in);
        proj_v_bias_File.open(path3, std::ios::in);
        if (!proj_k_bias_File || !proj_q_bias_File || !proj_v_bias_File) {
            std::cout << "error opening source file." << std::endl;
            return;
        }
        for (int i = 0; i < DIM; i++) {
            proj_k_bias_File >> encode_parameter.layers_p[block].attn_p.linear_k_p.bias[i];
            proj_q_bias_File >> encode_parameter.layers_p[block].attn_p.linear_q_p.bias[i];
            proj_v_bias_File >> encode_parameter.layers_p[block].attn_p.linear_v_p.bias[i];
        }
        proj_k_bias_File.close();
        proj_q_bias_File.close();
        proj_v_bias_File.close();


        /******Multi Attention weights*******/
        path = "./parameter/block";
        path.append(std::to_string(block));
        path.append("/proj_k_weight.txt");
        path2 = "./parameter/block";
        path2.append(std::to_string(block));
        path2.append("/proj_q_weight.txt");
        path3 = "./parameter/block";
        path3.append(std::to_string(block));
        path3.append("/proj_v_weight.txt");
        proj_k_weight_File.open(path, std::ios::in);
        proj_q_weight_File.open(path2, std::ios::in);
        proj_v_weight_File.open(path3, std::ios::in);
        if (!proj_k_weight_File || !proj_q_weight_File || !proj_v_weight_File) {
            std::cout << "error opening source file." << std::endl;
            return;
        }
        for (int i = 0; i < DIM; i++)
            for (int j = 0; j < DIM; j++) {
                proj_k_weight_File >> encode_parameter.layers_p[block].attn_p.linear_k_p.weights[j][i];
                proj_q_weight_File >> encode_parameter.layers_p[block].attn_p.linear_q_p.weights[j][i];
                proj_v_weight_File >> encode_parameter.layers_p[block].attn_p.linear_v_p.weights[j][i];
            }
        proj_k_weight_File.close();
        proj_q_weight_File.close();
        proj_v_weight_File.close();


        /*****Read FC weights and bias*****/
        path = "./parameter/block";
        path.append(std::to_string(block));
        path.append("/project_bias.txt");
        path2 = "./parameter/block";
        path2.append(std::to_string(block));
        path2.append("/project_weight.txt");
        proj_bias_File.open(path);
        proj_weight_File.open(path2);
        if (!proj_bias_File || !proj_weight_File) {
            std::cout << "error opening source file." << std::endl;
            return;
        }
        for (int i = 0; i < DIM; i++) {
            proj_bias_File >> encode_parameter.layers_p[block].attn_p.linear_p.bias[i];
            for (int j = 0; j < DIM; j++) {
                proj_weight_File >> encode_parameter.layers_p[block].attn_p.linear_p.weights[j][i];
            }
        }
        proj_bias_File.close();
        proj_weight_File.close();

        /*****Read MLP Parameters*****/
        path = "./parameter/block";
        path.append(std::to_string(block));
        path.append("/pwff_fc1_weight.txt");
        path2 = "./parameter/block";
        path2.append(std::to_string(block));
        path2.append("/pwff_fc2_weight.txt");
        pwff_fc1_weight_File.open(path);
        pwff_fc2_weight_File.open(path2);

        path = "./parameter/block";
        path.append(std::to_string(block));
        path.append("/pwff_fc1_bias.txt");
        path2 = "./parameter/block";
        path2.append(std::to_string(block));
        path2.append("/pwff_fc2_bias.txt");
        pwff_fc1_bias_File.open(path);
        pwff_fc2_bias_File.open(path2);

        for (int i = 0; i < DIM_HID; i++) {
            pwff_fc1_bias_File >> encode_parameter.layers_p[block].ff_p.linear_p1.bias[i];
            for (int j = 0; j < DIM; j++) {
                pwff_fc1_weight_File >> encode_parameter.layers_p[block].ff_p.linear_p1.weights[j][i];
            }
        }
        for (int i = 0; i < DIM; i++) {
            pwff_fc2_bias_File >> encode_parameter.layers_p[block].ff_p.linear_p2.bias[i];
            for (int j = 0; j < DIM_HID; j++) {
                pwff_fc2_weight_File >> encode_parameter.layers_p[block].ff_p.linear_p2.weights[j][i];
            }
        }
        pwff_fc2_weight_File.close();
        pwff_fc1_weight_File.close();
        pwff_fc1_bias_File.close();
        pwff_fc2_bias_File.close();
    }
    std::cout<<"Read end"<<std::endl;
};
#endif //VIT_READ_PARAMETER_H
