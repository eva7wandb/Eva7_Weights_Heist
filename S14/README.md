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

Since the dataset is relatively small we have included the whole dataset with the above transformations here -- [path](https://github.com/eva7wandb/Eva7_Weights_Heist/tree/main/S14/coco_dataset/balloon_coco)

### Sample images -- 

![image](https://user-images.githubusercontent.com/8600096/149612881-a9ac8010-9bb8-4451-8897-21882cee3d60.png)



## Train DETR on custom Dataset
In order to train the download the following repo -- 
```bash
git clone https://github.com/woctezuma/detr.git
cd ./detr/
git checkout finetune 
```
once the above repo is cloned, ensure -- 
- the dataset is prepared as in the previous step.
- the DETR model is downloaded.
    - The model can be downloaded from this repo -- [link](https://github.com/facebookresearch/detr#model-zoo)
    - once the model is downloaded, remove the class weights and save it. the **DETR_pre_finetune_setup.ipynb**  notebook has the steps to do it.
- train the model by running the following command, from the detr repo.
```bash
python main.py \
  --dataset_file "custom" \
  --coco_path "../coco_dataset/balloon_coco/" \
  --output_dir "../data/output/" \
  --resume "../data/detr-r50_no-class-head.pth" \
  --num_classes 1 \
  --epochs 10
  ```

## Evaluation result
The eval notebook **DETR_model_evaluation.ipynb**, contains loss graph, and inference on test image.

The training logs can be found in dir -- `./test_data/log.txt`

![image](https://user-images.githubusercontent.com/8600096/149626278-6768012d-8014-4ef6-9584-b6c72251b8ab.png)
![image](https://user-images.githubusercontent.com/8600096/149626281-3a425fde-4283-4c53-954e-25496b641a40.png)
![image](https://user-images.githubusercontent.com/8600096/149626287-d25fb760-c116-4769-b681-76d0a3ec54dd.png)

### Test image -- 
![image](https://user-images.githubusercontent.com/8600096/149626299-e9adc593-be37-44e6-9e79-c717c0b417ad.png)


