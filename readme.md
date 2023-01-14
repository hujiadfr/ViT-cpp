
# VIT-CPP



| key           | Value | description                 |
|---------------|-------|-----------------------------|
| FIG_WIDTH     | 224   | 图像宽度                        |
| KERNEL_WIDTH  | 16    | 卷积和大小，for ViT B/16          |
| DIM           | 768   | Embedding的维度 (16x16x3)      |
| DEP           | 196   | 图像patch 数量 ((224/16)^2=196) |
| DIM_HID       | 3072  | MLP_HEAD中隐藏层宽度 (768x4)      |
| HEAD_SIZE     | 12    | MultiHeadAttention中Head的数量  |
| ENC_LAYER_CNT | 12    | Encoder 的层数                 |
| N_CLASS       | 1000  | 类别总数                        |
请将encode block的预训练参数解压到parameter文件夹下, 保持block+数字的文件夹名字
https://zjuintl-my.sharepoint.com/:u:/g/personal/jiarun_19_intl_zju_edu_cn/ERdxaZSizMZOiB3vWGCqZPIB4ktltAS66PPfnouynf3FSQ?e=Eu9iYv
## MLP Head

<img src="./Image/image-20230107170417917-1673082668405-10.png" alt="image-20230107170417917" style="zoom:50%;" />



## Encoder

<img src="./Image/image-20230107170452520-1673082665045-8.png" alt="image-20230107170452520" style="zoom: 50%;" />

## Attention

<img src="./Image/image-20230107170554683-1673082684531-12.png" alt="image-20230107170554683" style="zoom:50%;" />





## Patch-Embedding

<img src="./Image/image-20230107170704035-1673082654959-6.png" alt="image-20230107170704035" style="zoom: 50%;" />


## MLP Head

<img src="./Image/image-20230107170753557-1673082690096-14.png" alt="image-20230107170753557" style="zoom:50%;" />

```

