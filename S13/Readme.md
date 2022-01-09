## S13 - ViT Training


## Team - Weights_Heist
### Team Members - 

| Name        | mail           |
|:-------------|:--------------|
|Gopinath Venkatesan|gops75[at]gmail[dot]com|
|Muhsin Mohammed|askmuhsin[at]gmail[dot]com|
|Subramanya R|subrananjun[at]gmail[dot]com| 


## ViT Training - Cats and Dogs dataset

[Notebook Link](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S13/assignment13_vit_cats_dogs.ipynb)

In this assignment we train a ViT model with a custom dataset to detect cats and dogs. The dataset is from Kaggle competition and the notebook is from this [blog post](https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/). 

### Dataset

We can download the [dataset](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) from Kaggle site. It will have two zip files train.zip and test.zip. We will have to unzip these and extract the files to a data folder.

Training data file names will have prefix indicating the image class (cat or dog). We build training labels 1 for dog and 0 for cat. We also split the training data into train and validation sets using 80:20 ratio.

### Model

The model is imported from vit-pytorch library. We use Linformer library for the transformer block.

Following code blcok captures the parameters we use to define the model with embedding size of 128 and patch size of 32.

~~~
efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)
~~~


~~~
model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)
~~~

### Training

We use cross-entropy loss function, adam optimizer with learning rate of 3e-5 and StepLR scheduler.

Below is the training log output.

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 1 - loss : 0.5879 - acc: 0.6773 - val_loss : 0.5870 - val_acc: 0.6853

100%|██████████| 79/79 [01:21<00:00,  1.04s/it]
Epoch : 2 - loss : 0.5884 - acc: 0.6826 - val_loss : 0.5892 - val_acc: 0.6721

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 3 - loss : 0.5890 - acc: 0.6793 - val_loss : 0.5849 - val_acc: 0.6821

100%|██████████| 79/79 [01:22<00:00,  1.05s/it]
Epoch : 4 - loss : 0.5832 - acc: 0.6867 - val_loss : 0.5871 - val_acc: 0.6794

100%|██████████| 79/79 [01:22<00:00,  1.05s/it]
Epoch : 5 - loss : 0.5847 - acc: 0.6847 - val_loss : 0.5855 - val_acc: 0.6808

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 6 - loss : 0.5810 - acc: 0.6899 - val_loss : 0.5911 - val_acc: 0.6801

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 7 - loss : 0.5821 - acc: 0.6856 - val_loss : 0.5820 - val_acc: 0.6945

100%|██████████| 79/79 [01:20<00:00,  1.02s/it]
Epoch : 8 - loss : 0.5805 - acc: 0.6865 - val_loss : 0.5738 - val_acc: 0.6965

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 9 - loss : 0.5835 - acc: 0.6885 - val_loss : 0.5810 - val_acc: 0.6898

100%|██████████| 79/79 [01:21<00:00,  1.04s/it]
Epoch : 10 - loss : 0.5760 - acc: 0.6907 - val_loss : 0.5808 - val_acc: 0.6938

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 11 - loss : 0.5769 - acc: 0.6911 - val_loss : 0.5804 - val_acc: 0.6874

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 12 - loss : 0.5778 - acc: 0.6920 - val_loss : 0.5775 - val_acc: 0.6934

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 13 - loss : 0.5733 - acc: 0.6937 - val_loss : 0.5760 - val_acc: 0.6908

100%|██████████| 79/79 [01:20<00:00,  1.02s/it]
Epoch : 14 - loss : 0.5764 - acc: 0.6939 - val_loss : 0.5780 - val_acc: 0.6923

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 15 - loss : 0.5718 - acc: 0.6974 - val_loss : 0.5759 - val_acc: 0.6906

100%|██████████| 79/79 [01:21<00:00,  1.04s/it]
Epoch : 16 - loss : 0.5704 - acc: 0.6934 - val_loss : 0.5931 - val_acc: 0.6808

100%|██████████| 79/79 [01:21<00:00,  1.04s/it]
Epoch : 17 - loss : 0.5743 - acc: 0.6947 - val_loss : 0.5742 - val_acc: 0.6963

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 18 - loss : 0.5734 - acc: 0.6938 - val_loss : 0.5750 - val_acc: 0.7027

100%|██████████| 79/79 [01:22<00:00,  1.05s/it]
Epoch : 19 - loss : 0.5750 - acc: 0.6944 - val_loss : 0.5701 - val_acc: 0.6924

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 20 - loss : 0.5768 - acc: 0.6909 - val_loss : 0.5734 - val_acc: 0.6912

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 21 - loss : 0.5690 - acc: 0.7003 - val_loss : 0.5734 - val_acc: 0.6868

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 22 - loss : 0.5717 - acc: 0.6940 - val_loss : 0.5717 - val_acc: 0.6955

100%|██████████| 79/79 [01:21<00:00,  1.04s/it]
Epoch : 23 - loss : 0.5679 - acc: 0.6978 - val_loss : 0.5650 - val_acc: 0.7058

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 24 - loss : 0.5652 - acc: 0.7016 - val_loss : 0.5655 - val_acc: 0.7016

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 25 - loss : 0.5652 - acc: 0.7022 - val_loss : 0.5627 - val_acc: 0.7026

100%|██████████| 79/79 [01:23<00:00,  1.05s/it]
Epoch : 26 - loss : 0.5658 - acc: 0.7004 - val_loss : 0.5646 - val_acc: 0.7033

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 27 - loss : 0.5651 - acc: 0.7009 - val_loss : 0.5628 - val_acc: 0.7031

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 28 - loss : 0.5639 - acc: 0.7047 - val_loss : 0.5668 - val_acc: 0.7050

100%|██████████| 79/79 [01:21<00:00,  1.04s/it]
Epoch : 29 - loss : 0.5664 - acc: 0.6981 - val_loss : 0.5713 - val_acc: 0.6967

100%|██████████| 79/79 [01:22<00:00,  1.05s/it]
Epoch : 30 - loss : 0.5604 - acc: 0.7055 - val_loss : 0.5648 - val_acc: 0.6998

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 31 - loss : 0.5651 - acc: 0.6985 - val_loss : 0.5618 - val_acc: 0.7079

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 32 - loss : 0.5635 - acc: 0.7050 - val_loss : 0.5665 - val_acc: 0.7016

100%|██████████| 79/79 [01:22<00:00,  1.05s/it]
Epoch : 33 - loss : 0.5644 - acc: 0.7028 - val_loss : 0.5630 - val_acc: 0.7029

100%|██████████| 79/79 [01:22<00:00,  1.05s/it]
Epoch : 34 - loss : 0.5652 - acc: 0.7040 - val_loss : 0.5662 - val_acc: 0.7043

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 35 - loss : 0.5637 - acc: 0.7044 - val_loss : 0.5651 - val_acc: 0.7027

100%|██████████| 79/79 [01:21<00:00,  1.04s/it]
Epoch : 36 - loss : 0.5647 - acc: 0.7049 - val_loss : 0.5639 - val_acc: 0.7066

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 37 - loss : 0.5564 - acc: 0.7110 - val_loss : 0.5778 - val_acc: 0.6938

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 38 - loss : 0.5613 - acc: 0.7010 - val_loss : 0.5690 - val_acc: 0.7029

100%|██████████| 79/79 [01:20<00:00,  1.03s/it]
Epoch : 39 - loss : 0.5592 - acc: 0.7094 - val_loss : 0.5633 - val_acc: 0.7052

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 40 - loss : 0.5594 - acc: 0.7107 - val_loss : 0.5535 - val_acc: 0.7123

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 41 - loss : 0.5534 - acc: 0.7121 - val_loss : 0.5555 - val_acc: 0.7098

100%|██████████| 79/79 [01:20<00:00,  1.02s/it]
Epoch : 42 - loss : 0.5586 - acc: 0.7085 - val_loss : 0.5499 - val_acc: 0.7150

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 43 - loss : 0.5541 - acc: 0.7108 - val_loss : 0.5553 - val_acc: 0.7125

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 44 - loss : 0.5493 - acc: 0.7148 - val_loss : 0.5616 - val_acc: 0.7048

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 45 - loss : 0.5546 - acc: 0.7095 - val_loss : 0.5578 - val_acc: 0.7126

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 46 - loss : 0.5583 - acc: 0.7081 - val_loss : 0.5488 - val_acc: 0.7173

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 47 - loss : 0.5617 - acc: 0.7069 - val_loss : 0.5553 - val_acc: 0.7102

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 48 - loss : 0.5575 - acc: 0.7034 - val_loss : 0.5625 - val_acc: 0.7021

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 49 - loss : 0.5523 - acc: 0.7161 - val_loss : 0.5618 - val_acc: 0.7112

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 50 - loss : 0.5512 - acc: 0.7143 - val_loss : 0.5617 - val_acc: 0.7043

100%|██████████| 79/79 [01:22<00:00,  1.05s/it]
Epoch : 51 - loss : 0.5585 - acc: 0.7055 - val_loss : 0.5570 - val_acc: 0.7113

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 52 - loss : 0.5452 - acc: 0.7170 - val_loss : 0.5591 - val_acc: 0.7084

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 53 - loss : 0.5492 - acc: 0.7171 - val_loss : 0.5629 - val_acc: 0.7126

100%|██████████| 79/79 [01:20<00:00,  1.02s/it]
Epoch : 54 - loss : 0.5502 - acc: 0.7160 - val_loss : 0.5611 - val_acc: 0.7034

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 55 - loss : 0.5444 - acc: 0.7176 - val_loss : 0.5509 - val_acc: 0.7132

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 56 - loss : 0.5525 - acc: 0.7134 - val_loss : 0.5446 - val_acc: 0.7227

100%|██████████| 79/79 [01:20<00:00,  1.02s/it]
Epoch : 57 - loss : 0.5459 - acc: 0.7174 - val_loss : 0.5486 - val_acc: 0.7197

100%|██████████| 79/79 [01:20<00:00,  1.02s/it]
Epoch : 58 - loss : 0.5443 - acc: 0.7241 - val_loss : 0.5491 - val_acc: 0.7142

100%|██████████| 79/79 [01:20<00:00,  1.02s/it]
Epoch : 59 - loss : 0.5503 - acc: 0.7140 - val_loss : 0.5645 - val_acc: 0.7124

100%|██████████| 79/79 [01:20<00:00,  1.02s/it]
Epoch : 60 - loss : 0.5511 - acc: 0.7157 - val_loss : 0.5549 - val_acc: 0.7198

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 61 - loss : 0.5459 - acc: 0.7188 - val_loss : 0.5573 - val_acc: 0.7132

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 62 - loss : 0.5501 - acc: 0.7115 - val_loss : 0.5530 - val_acc: 0.7139

100%|██████████| 79/79 [01:23<00:00,  1.05s/it]
Epoch : 63 - loss : 0.5465 - acc: 0.7143 - val_loss : 0.5492 - val_acc: 0.7140

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 64 - loss : 0.5447 - acc: 0.7222 - val_loss : 0.5511 - val_acc: 0.7094

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 65 - loss : 0.5485 - acc: 0.7193 - val_loss : 0.5568 - val_acc: 0.7173

100%|██████████| 79/79 [01:22<00:00,  1.05s/it]
Epoch : 66 - loss : 0.5442 - acc: 0.7157 - val_loss : 0.5462 - val_acc: 0.7234

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 67 - loss : 0.5479 - acc: 0.7186 - val_loss : 0.5529 - val_acc: 0.7182

100%|██████████| 79/79 [01:22<00:00,  1.05s/it]
Epoch : 68 - loss : 0.5447 - acc: 0.7186 - val_loss : 0.5425 - val_acc: 0.7214

100%|██████████| 79/79 [01:22<00:00,  1.04s/it]
Epoch : 69 - loss : 0.5382 - acc: 0.7265 - val_loss : 0.5623 - val_acc: 0.7047

100%|██████████| 79/79 [01:22<00:00,  1.05s/it]
Epoch : 70 - loss : 0.5458 - acc: 0.7207 - val_loss : 0.5541 - val_acc: 0.7153

100%|██████████| 79/79 [01:21<00:00,  1.04s/it]
Epoch : 71 - loss : 0.5448 - acc: 0.7196 - val_loss : 0.5478 - val_acc: 0.7212

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 72 - loss : 0.5439 - acc: 0.7182 - val_loss : 0.5437 - val_acc: 0.7253

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 73 - loss : 0.5437 - acc: 0.7176 - val_loss : 0.5567 - val_acc: 0.7117

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 74 - loss : 0.5409 - acc: 0.7190 - val_loss : 0.5449 - val_acc: 0.7243

100%|██████████| 79/79 [01:21<00:00,  1.03s/it]
Epoch : 75 - loss : 0.5399 - acc: 0.7201 - val_loss : 0.5457 - val_acc: 0.7178
