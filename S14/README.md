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
    - once the model is downloaded, remove the class weights and save it. the [**DETR_pre_finetune_setup.ipynb**](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S14/DETR_pre_finetune_setup.ipynb)  notebook has the steps to do it.
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
The eval notebook [**DETR_model_evaluation.ipynb**](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S14/DETR_model_evaluation.ipynb), contains loss graph, and inference on test image.




### Test images -- 
![image](https://user-images.githubusercontent.com/8600096/149630782-672914da-2bd6-4d94-b875-e6fef9943072.png)
![image](https://user-images.githubusercontent.com/8600096/149630788-466bc100-1c53-4c47-8ffb-d62ac6f4202a.png)


### Training logs 
![image](https://user-images.githubusercontent.com/8600096/149630741-8bb2f13e-e0ef-483b-af04-e42087433727.png)
![image](https://user-images.githubusercontent.com/8600096/149630756-0e18130b-5a90-4f50-866c-46fcf70e45fc.png)
![image](https://user-images.githubusercontent.com/8600096/149630760-516743b3-9747-4c26-8d2d-1519bd616c61.png)


Full training logs can be found in dir -- `./test_data/log.txt`
logs for last epoch from last epoch)
```bash
Epoch: [99] Total time: 0:00:11 (0.3914 s / it)
Averaged stats: lr: 0.000100  class_error: 0.00  loss: 2.4144 (2.9051)  loss_ce: 0.0039 (0.0433)  loss_bbox: 0.1126 (0.1316)  loss_giou: 0.2125 (0.2782)  loss_ce_0: 0.0943 (0.1143)  loss_bbox_0: 0.1426 (0.1576)  loss_giou_0: 0.2357 (0.3111)  loss_ce_1: 0.0291 (0.0758)  loss_bbox_1: 0.1282 (0.1396)  loss_giou_1: 0.2237 (0.2763)  loss_ce_2: 0.0112 (0.0536)  loss_bbox_2: 0.1254 (0.1372)  loss_giou_2: 0.2140 (0.2793)  loss_ce_3: 0.0056 (0.0483)  loss_bbox_3: 0.1221 (0.1327)  loss_giou_3: 0.2082 (0.2777)  loss_ce_4: 0.0041 (0.0445)  loss_bbox_4: 0.1124 (0.1283)  loss_giou_4: 0.2059 (0.2756)  loss_ce_unscaled: 0.0039 (0.0433)  class_error_unscaled: 0.0000 (2.7778)  loss_bbox_unscaled: 0.0225 (0.0263)  loss_giou_unscaled: 0.1062 (0.1391)  cardinality_error_unscaled: 0.0000 (0.9500)  loss_ce_0_unscaled: 0.0943 (0.1143)  loss_bbox_0_unscaled: 0.0285 (0.0315)  loss_giou_0_unscaled: 0.1178 (0.1556)  cardinality_error_0_unscaled: 2.5000 (4.5500)  loss_ce_1_unscaled: 0.0291 (0.0758)  loss_bbox_1_unscaled: 0.0256 (0.0279)  loss_giou_1_unscaled: 0.1119 (0.1382)  cardinality_error_1_unscaled: 1.0000 (2.4000)  loss_ce_2_unscaled: 0.0112 (0.0536)  loss_bbox_2_unscaled: 0.0251 (0.0274)  loss_giou_2_unscaled: 0.1070 (0.1397)  cardinality_error_2_unscaled: 0.0000 (1.1833)  loss_ce_3_unscaled: 0.0056 (0.0483)  loss_bbox_3_unscaled: 0.0244 (0.0265)  loss_giou_3_unscaled: 0.1041 (0.1388)  cardinality_error_3_unscaled: 0.0000 (1.0000)  loss_ce_4_unscaled: 0.0041 (0.0445)  loss_bbox_4_unscaled: 0.0225 (0.0257)  loss_giou_4_unscaled: 0.1030 (0.1378)  cardinality_error_4_unscaled: 0.0000 (0.8833)
Test:  [0/7]  eta: 0:00:08  class_error: 0.00  loss: 4.9160 (4.9160)  loss_ce: 0.0237 (0.0237)  loss_bbox: 0.1925 (0.1925)  loss_giou: 0.4607 (0.4607)  loss_ce_0: 0.2197 (0.2197)  loss_bbox_0: 0.2768 (0.2768)  loss_giou_0: 0.6076 (0.6076)  loss_ce_1: 0.0921 (0.0921)  loss_bbox_1: 0.2601 (0.2601)  loss_giou_1: 0.6029 (0.6029)  loss_ce_2: 0.0706 (0.0706)  loss_bbox_2: 0.1921 (0.1921)  loss_giou_2: 0.5071 (0.5071)  loss_ce_3: 0.0329 (0.0329)  loss_bbox_3: 0.1979 (0.1979)  loss_giou_3: 0.4959 (0.4959)  loss_ce_4: 0.0266 (0.0266)  loss_bbox_4: 0.1870 (0.1870)  loss_giou_4: 0.4700 (0.4700)  loss_ce_unscaled: 0.0237 (0.0237)  class_error_unscaled: 0.0000 (0.0000)  loss_bbox_unscaled: 0.0385 (0.0385)  loss_giou_unscaled: 0.2303 (0.2303)  cardinality_error_unscaled: 0.5000 (0.5000)  loss_ce_0_unscaled: 0.2197 (0.2197)  loss_bbox_0_unscaled: 0.0554 (0.0554)  loss_giou_0_unscaled: 0.3038 (0.3038)  cardinality_error_0_unscaled: 0.5000 (0.5000)  loss_ce_1_unscaled: 0.0921 (0.0921)  loss_bbox_1_unscaled: 0.0520 (0.0520)  loss_giou_1_unscaled: 0.3015 (0.3015)  cardinality_error_1_unscaled: 1.0000 (1.0000)  loss_ce_2_unscaled: 0.0706 (0.0706)  loss_bbox_2_unscaled: 0.0384 (0.0384)  loss_giou_2_unscaled: 0.2535 (0.2535)  cardinality_error_2_unscaled: 1.0000 (1.0000)  loss_ce_3_unscaled: 0.0329 (0.0329)  loss_bbox_3_unscaled: 0.0396 (0.0396)  loss_giou_3_unscaled: 0.2479 (0.2479)  cardinality_error_3_unscaled: 0.5000 (0.5000)  loss_ce_4_unscaled: 0.0266 (0.0266)  loss_bbox_4_unscaled: 0.0374 (0.0374)  loss_giou_4_unscaled: 0.2350 (0.2350)  cardinality_error_4_unscaled: 0.5000 (0.5000)  time: 1.2069  data: 0.9947  max mem: 5063
Test:  [6/7]  eta: 0:00:00  class_error: 0.00  loss: 4.9160 (5.7922)  loss_ce: 0.0558 (0.1551)  loss_bbox: 0.1611 (0.2166)  loss_giou: 0.4607 (0.5139)  loss_ce_0: 0.2174 (0.3093)  loss_bbox_0: 0.2597 (0.2527)  loss_giou_0: 0.6076 (0.6227)  loss_ce_1: 0.0958 (0.2193)  loss_bbox_1: 0.2038 (0.2607)  loss_giou_1: 0.5757 (0.5539)  loss_ce_2: 0.0863 (0.1872)  loss_bbox_2: 0.1455 (0.2279)  loss_giou_2: 0.5071 (0.4987)  loss_ce_3: 0.0770 (0.1566)  loss_bbox_3: 0.1549 (0.2245)  loss_giou_3: 0.4959 (0.5044)  loss_ce_4: 0.0667 (0.1564)  loss_bbox_4: 0.1538 (0.2161)  loss_giou_4: 0.4700 (0.5162)  loss_ce_unscaled: 0.0558 (0.1551)  class_error_unscaled: 9.0909 (14.2919)  loss_bbox_unscaled: 0.0322 (0.0433)  loss_giou_unscaled: 0.2303 (0.2569)  cardinality_error_unscaled: 0.5000 (0.7143)  loss_ce_0_unscaled: 0.2174 (0.3093)  loss_bbox_0_unscaled: 0.0519 (0.0505)  loss_giou_0_unscaled: 0.3038 (0.3113)  cardinality_error_0_unscaled: 2.0000 (2.0000)  loss_ce_1_unscaled: 0.0958 (0.2193)  loss_bbox_1_unscaled: 0.0408 (0.0521)  loss_giou_1_unscaled: 0.2879 (0.2770)  cardinality_error_1_unscaled: 1.0000 (1.0714)  loss_ce_2_unscaled: 0.0863 (0.1872)  loss_bbox_2_unscaled: 0.0291 (0.0456)  loss_giou_2_unscaled: 0.2535 (0.2494)  cardinality_error_2_unscaled: 1.0000 (0.9286)  loss_ce_3_unscaled: 0.0770 (0.1566)  loss_bbox_3_unscaled: 0.0310 (0.0449)  loss_giou_3_unscaled: 0.2479 (0.2522)  cardinality_error_3_unscaled: 0.5000 (0.7143)  loss_ce_4_unscaled: 0.0667 (0.1564)  loss_bbox_4_unscaled: 0.0308 (0.0432)  loss_giou_4_unscaled: 0.2350 (0.2581)  cardinality_error_4_unscaled: 0.5000 (0.7143)  time: 0.3790  data: 0.1622  max mem: 5063
Test: Total time: 0:00:02 (0.3915 s / it)
Averaged stats: class_error: 0.00  loss: 4.9160 (5.7922)  loss_ce: 0.0558 (0.1551)  loss_bbox: 0.1611 (0.2166)  loss_giou: 0.4607 (0.5139)  loss_ce_0: 0.2174 (0.3093)  loss_bbox_0: 0.2597 (0.2527)  loss_giou_0: 0.6076 (0.6227)  loss_ce_1: 0.0958 (0.2193)  loss_bbox_1: 0.2038 (0.2607)  loss_giou_1: 0.5757 (0.5539)  loss_ce_2: 0.0863 (0.1872)  loss_bbox_2: 0.1455 (0.2279)  loss_giou_2: 0.5071 (0.4987)  loss_ce_3: 0.0770 (0.1566)  loss_bbox_3: 0.1549 (0.2245)  loss_giou_3: 0.4959 (0.5044)  loss_ce_4: 0.0667 (0.1564)  loss_bbox_4: 0.1538 (0.2161)  loss_giou_4: 0.4700 (0.5162)  loss_ce_unscaled: 0.0558 (0.1551)  class_error_unscaled: 9.0909 (14.2919)  loss_bbox_unscaled: 0.0322 (0.0433)  loss_giou_unscaled: 0.2303 (0.2569)  cardinality_error_unscaled: 0.5000 (0.7143)  loss_ce_0_unscaled: 0.2174 (0.3093)  loss_bbox_0_unscaled: 0.0519 (0.0505)  loss_giou_0_unscaled: 0.3038 (0.3113)  cardinality_error_0_unscaled: 2.0000 (2.0000)  loss_ce_1_unscaled: 0.0958 (0.2193)  loss_bbox_1_unscaled: 0.0408 (0.0521)  loss_giou_1_unscaled: 0.2879 (0.2770)  cardinality_error_1_unscaled: 1.0000 (1.0714)  loss_ce_2_unscaled: 0.0863 (0.1872)  loss_bbox_2_unscaled: 0.0291 (0.0456)  loss_giou_2_unscaled: 0.2535 (0.2494)  cardinality_error_2_unscaled: 1.0000 (0.9286)  loss_ce_3_unscaled: 0.0770 (0.1566)  loss_bbox_3_unscaled: 0.0310 (0.0449)  loss_giou_3_unscaled: 0.2479 (0.2522)  cardinality_error_3_unscaled: 0.5000 (0.7143)  loss_ce_4_unscaled: 0.0667 (0.1564)  loss_bbox_4_unscaled: 0.0308 (0.0432)  loss_giou_4_unscaled: 0.2350 (0.2581)  cardinality_error_4_unscaled: 0.5000 (0.7143)
Accumulating evaluation results...
DONE (t=0.03s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.473
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.790
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.486
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.394
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.689
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.160
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.404
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.730
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Training time 0:42:45
```

