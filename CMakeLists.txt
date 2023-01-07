cmake_minimum_required(VERSION 3.24)
project(ViT)

set(CMAKE_CXX_STANDARD 17)

add_executable(ViT main.cpp norm.h linear.h dropout.h softmax.h gelu.h MLP.h Encoder.h attention.h MLP_head.h patch_embed.h patch_embed.h transformer.h)
