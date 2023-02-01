
# VIT-CPP



| key           | Value | description                 |
|---------------|-------|-----------------------------|
| FIG_WIDTH     | 384   | Resolution of input image                        |
| KERNEL_WIDTH  | 16    | The convolution kernel size of Patch embedding          |
| DIM           | 768   | The dimension of embedding patch (16x16x3)      |
| DEP           | 576   | The number of patches ((384/16)^2=576) |
| DIM_HID       | 3072  | The dimension of hidden layer in MLP_HEAD module (768x4)      |
| HEAD_SIZE     | 12    | The number of heads in MultiHeadAttention module |
| ENC_LAYER_CNT | 12    | The number of Encoder blocks                 |
| N_CLASS       | 1000  | The dimension of final prediction                        |
Please extract the pre-trained encoder block paramters to 'parameter' folder. 
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

