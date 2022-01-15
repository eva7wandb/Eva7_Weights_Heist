## S14 - DETR


## Team - Weights_Heist
### Team Members - 

| Name        | mail           |
|:-------------|:--------------|
|Gopinath Venkatesan|gops75[at]gmail[dot]com|
|Muhsin Mohammed|askmuhsin[at]gmail[dot]com|
|Subramanya R|subrananjun[at]gmail[dot]com| 

## Custom Dataset -- Balloons
More details on the dataset here -- [link](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)
This dataset contains 2 classes (balloons and background). And contains in total 61 image in training, and 13 for test.

You can also dowload the dataset -- 
```
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
```

The 👆 data is not in the COCO format. The annotation files need to be converted to COCO like, and also the directory stucture need to be modified as follows -- 
```
.
|-- annotations
|   |-- custom_train.json
|   `-- custom_val.json
|-- train
`-- val
```

This can be done using this repo -- [link](https://github.com/woctezuma/VIA2COCO) (the instructions to do it is available in the repo).

### Sample images -- 

![image](https://user-images.githubusercontent.com/8600096/149612881-a9ac8010-9bb8-4451-8897-21882cee3d60.png)

