## S10 - OBJECT LOCALISATION


## Team - Weights_Heist
### Team Members - 

| Name        | mail           |
|:-------------|:--------------|
|Gopinath Venkatesan|gops75[at]gmail[dot]com|
|Muhsin Mohammed|askmuhsin[at]gmail[dot]com|
|Rashu Tyagi|rashutyagi116[at]gmail[dot]com| 
|Subramanya R|subrananjun[at]gmail[dot]com| 

## Train ResNet18 with Tiny-Imagenet-200 dataset(70/30 split) - Target 50%+ Validation Accuracy

### Data Preparation

Tiny-Imagenet-200 dataset has 200 classes with 500 training images for each class and 50 validation image per class. 
We need to prepare the dataset to have split of 70/30 for each class and put them separate train and test directories.
We do this by running the [python script](https://github.com/eva7wandb/Weights_Heist_Flow/blob/main/prepare_tiny-imagenet-200.py). 
If needed, we could zip the processed folder and save it for future reloading (saves processing time when reconnecting in colab).

### Data Augmentation

We do the following albumentations augmentations to the images. 

   - padding of 4 pixels on all sides (p=1)
   - random crop of 64x64 after padding (p=1)
   - Horizontal flip (p=0.25)
   - Rotate (p=0.25)
   - RGB Shift of 20 (p=0.25)
   - Cut of 1 hole

Here is the code snippet:

       transforms_list.extend([
            A.PadIfNeeded(min_height=72, min_width=72, p=1.0),
            A.RandomCrop(height=64, width=64, p=1.0),
            A.HorizontalFlip(p=0.25),
            A.Rotate(limit=15, p=0.25),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),
            A.CoarseDropout(max_holes=1, max_height=32, max_width=32, min_height=8,
                            min_width=8, fill_value=mean*255.0, p=0.5),
        ])

### Training Logs
[INFO] Begin training for 40 epochs.
TRAIN Epoch:0 Loss:4.9952 Batch:150 Acc:2.74: 100%|██████████| 151/151 [05:16<00:00,  2.09s/it]

TEST         Loss:4.8815         Acc:4.78         [1576 / 33000]

TRAIN Epoch:1 Loss:4.4532 Batch:150 Acc:6.55: 100%|██████████| 151/151 [04:55<00:00,  1.96s/it]

TEST         Loss:4.4146         Acc:8.90         [2936 / 33000]

TRAIN Epoch:2 Loss:3.6964 Batch:150 Acc:11.72: 100%|██████████| 151/151 [05:23<00:00,  2.14s/it]

TEST         Loss:4.0434         Acc:14.81         [4886 / 33000]

TRAIN Epoch:3 Loss:3.8201 Batch:150 Acc:17.03: 100%|██████████| 151/151 [05:14<00:00,  2.08s/it]
TEST         Loss:3.7531         Acc:20.52         [6772 / 33000]
TRAIN Epoch:4 Loss:3.611 Batch:150 Acc:22.48: 100%|██████████| 151/151 [05:10<00:00,  2.06s/it] 
TEST         Loss:3.5475         Acc:24.45         [8070 / 33000]
TRAIN Epoch:5 Loss:3.2511 Batch:150 Acc:27.14: 100%|██████████| 151/151 [05:29<00:00,  2.18s/it]
TEST         Loss:3.4673         Acc:26.07         [8604 / 33000]
TRAIN Epoch:6 Loss:3.2491 Batch:150 Acc:32.08: 100%|██████████| 151/151 [04:58<00:00,  1.98s/it]
TEST         Loss:3.2962         Acc:30.44         [10046 / 33000]
TRAIN Epoch:7 Loss:2.7027 Batch:150 Acc:36.20: 100%|██████████| 151/151 [05:25<00:00,  2.16s/it]
TEST         Loss:3.3030         Acc:30.40         [10032 / 33000]
TRAIN Epoch:8 Loss:2.643 Batch:150 Acc:40.32: 100%|██████████| 151/151 [05:18<00:00,  2.11s/it] 
TEST         Loss:3.2156         Acc:33.50         [11054 / 33000]
TRAIN Epoch:9 Loss:2.5807 Batch:150 Acc:44.01: 100%|██████████| 151/151 [05:04<00:00,  2.02s/it]
TEST         Loss:2.9368         Acc:38.44         [12685 / 33000]
TRAIN Epoch:10 Loss:2.5135 Batch:150 Acc:47.41: 100%|██████████| 151/151 [05:27<00:00,  2.17s/it]
TEST         Loss:2.8717         Acc:40.43         [13341 / 33000]
TRAIN Epoch:11 Loss:2.4404 Batch:150 Acc:50.93: 100%|██████████| 151/151 [05:01<00:00,  1.99s/it]
TEST         Loss:2.7180         Acc:44.15         [14571 / 33000]
TRAIN Epoch:12 Loss:2.1137 Batch:150 Acc:53.87: 100%|██████████| 151/151 [05:21<00:00,  2.13s/it]
TEST         Loss:2.9123         Acc:40.81         [13467 / 33000]
TRAIN Epoch:13 Loss:2.2871 Batch:150 Acc:56.67: 100%|██████████| 151/151 [05:22<00:00,  2.14s/it]
TEST         Loss:2.8027         Acc:43.30         [14289 / 33000]
TRAIN Epoch:14 Loss:1.9516 Batch:150 Acc:59.64: 100%|██████████| 151/151 [04:59<00:00,  1.99s/it]
TEST         Loss:2.7266         Acc:44.69         [14747 / 33000]
TRAIN Epoch:15 Loss:2.009 Batch:150 Acc:62.46: 100%|██████████| 151/151 [05:26<00:00,  2.16s/it] 
TEST         Loss:2.6933         Acc:46.62         [15383 / 33000]
TRAIN Epoch:16 Loss:1.8804 Batch:150 Acc:65.34: 100%|██████████| 151/151 [05:06<00:00,  2.03s/it]
TEST         Loss:2.7272         Acc:46.02         [15187 / 33000]
TRAIN Epoch:17 Loss:1.7632 Batch:150 Acc:68.08: 100%|██████████| 151/151 [05:17<00:00,  2.10s/it]
TEST         Loss:2.7781         Acc:46.14         [15227 / 33000]
TRAIN Epoch:18 Loss:1.6659 Batch:150 Acc:70.90: 100%|██████████| 151/151 [05:24<00:00,  2.15s/it]
TEST         Loss:2.6958         Acc:48.24         [15918 / 33000]
TRAIN Epoch:19 Loss:1.4148 Batch:150 Acc:73.80: 100%|██████████| 151/151 [04:56<00:00,  1.96s/it]
TEST         Loss:2.8164         Acc:46.88         [15471 / 33000]
TRAIN Epoch:20 Loss:1.4251 Batch:150 Acc:76.42: 100%|██████████| 151/151 [05:27<00:00,  2.17s/it]
TEST         Loss:2.7530         Acc:47.65         [15724 / 33000]
TRAIN Epoch:21 Loss:1.4945 Batch:150 Acc:79.16: 100%|██████████| 151/151 [05:10<00:00,  2.06s/it]
TEST         Loss:2.8030         Acc:47.33         [15618 / 33000]
TRAIN Epoch:22 Loss:1.2799 Batch:150 Acc:81.60: 100%|██████████| 151/151 [05:10<00:00,  2.06s/it]
TEST         Loss:2.7790         Acc:48.43         [15982 / 33000]
TRAIN Epoch:23 Loss:1.3351 Batch:150 Acc:83.99: 100%|██████████| 151/151 [05:24<00:00,  2.15s/it]
TEST         Loss:2.7362         Acc:49.42         [16309 / 33000]
TRAIN Epoch:24 Loss:1.2798 Batch:150 Acc:86.11: 100%|██████████| 151/151 [04:58<00:00,  1.97s/it]
TEST         Loss:2.7845         Acc:49.44         [16315 / 33000]
TRAIN Epoch:25 Loss:1.0848 Batch:150 Acc:87.99: 100%|██████████| 151/151 [05:25<00:00,  2.16s/it]
TEST         Loss:2.7818         Acc:50.31         [16602 / 33000]
TRAIN Epoch:26 Loss:1.0384 Batch:150 Acc:89.71: 100%|██████████| 151/151 [05:17<00:00,  2.10s/it]
TEST         Loss:2.7323         Acc:50.62         [16705 / 33000]
TRAIN Epoch:27 Loss:0.9941 Batch:150 Acc:91.05: 100%|██████████| 151/151 [05:06<00:00,  2.03s/it]
TEST         Loss:2.7031         Acc:50.47         [16655 / 33000]
TRAIN Epoch:28 Loss:1.0151 Batch:150 Acc:92.39: 100%|██████████| 151/151 [05:25<00:00,  2.16s/it]
TEST         Loss:2.7201         Acc:50.54         [16678 / 33000]
TRAIN Epoch:29 Loss:0.9009 Batch:150 Acc:93.51: 100%|██████████| 151/151 [05:01<00:00,  2.00s/it]
TEST         Loss:2.6668         Acc:51.12         [16871 / 33000]
TRAIN Epoch:30 Loss:0.8384 Batch:150 Acc:94.24: 100%|██████████| 151/151 [05:22<00:00,  2.14s/it]
TEST         Loss:2.6623         Acc:51.14         [16876 / 33000]
TRAIN Epoch:31 Loss:0.8512 Batch:150 Acc:95.09: 100%|██████████| 151/151 [05:21<00:00,  2.13s/it]
TEST         Loss:2.6349         Acc:51.64         [17040 / 33000]
TRAIN Epoch:32 Loss:0.8769 Batch:150 Acc:95.63: 100%|██████████| 151/151 [05:01<00:00,  2.00s/it]
TEST         Loss:2.6126         Acc:52.02         [17168 / 33000]
TRAIN Epoch:33 Loss:0.773 Batch:70 Acc:96.37:  47%|████▋     | 71/151 [02:35<02:53,  2.17s/it] 

