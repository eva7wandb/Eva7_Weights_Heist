## S11 - OBJECT LOCALISATION -- PART 2 -- YOLOV3 on Custom Dataset


## Team - Weights_Heist
### Team Members - 

| Name        | mail           |
|:-------------|:--------------|
|Gopinath Venkatesan|gops75[at]gmail[dot]com|
|Muhsin Mohammed|askmuhsin[at]gmail[dot]com|
|Subramanya R|subrananjun[at]gmail[dot]com| 


### Custom Dataset

Already available [Customer dataset](https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view?usp=sharing) has images for four classes.

- Hard Hat
- Mask
- Vest
- Boots

We have added 25 images of each class to this dataset of 3000+ images that were collected by previous batch of EAV students. 


### Training

We use the [Repo](https://github.com/theschoolofai/YoloV3) to train the model on our custom dataset with following changes. 

- Replace the folder data/customdata in the cloned folder with our customdata folder that contains our 4 classes data
- Update the yolov3-custom.cfg configuration file with our changes. Our specific file [cfg] [https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S11/yoloV3_custom_dataset/artifacts/yolov3-custom.cfg] 
- Our training notebook [Link](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S11/yoloV3_custom_dataset/Assignment11_YoloV3Sample.ipynb)


### Traing log

![Training Log](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S11/yoloV3_custom_dataset/training_log.png)


### Video Inference Output

[![Video Output](https://youtu.be/_FQJxBh8qI8/0.jpg)](https://youtu.be/_FQJxBh8qI8)


### Sixteen (4 per class) Images Inference

![Images](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S11/yoloV3_custom_dataset/16_inferred_pics.png)

