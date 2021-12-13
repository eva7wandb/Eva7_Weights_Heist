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


