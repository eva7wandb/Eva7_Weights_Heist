# S5 - Regularizations

## Team - Weights_Heist
## Team Members - 

| Name        | mail           |
| ------------- |:-------------:|
|Gopinath Venkatesan|gops75[at]gmail[dot]com|
|Muhsin Mohammed|askmuhsin[at]gmail[dot]com|
|Rashu Tyagi|rashutyagi116[at]gmail[dot]com| 
|Subramanya R|subrananjun[at]gmail[dot]com| 

In this set of experiments we have tried out different normalization techniques on CNN model to classify on MNIST dataset.

# Final Notebook with all experiments --> [Notebook](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S6/run_experiments.ipynb)
# Parametrized model class
The model class is parametrized, and can be optionally set or unset the following layers --
- `dropout_value` -> float; setting to 0 will turn off dropout. and 0<dropout_value<1 will set drop out to that value.
- `batch_norm` -> bool; will apply batch normalization
- `layer_norm` -> bool; will apply layer normalization
- `group_norm` -> bool; will apply group normalization
- `group_norm_groups` -> int; to set the number of groups for group norm; only when group_norm is set to True

### example usage of the model
By default batch norm is on and drop out is set at 0.01
```python
from models.model import Net

model = Net()
```

To set **Group Normalization** only (_and set Group norm groups to 2_)
```python
model = Net(
    batch_norm=False,
    layer_norm=False,
    group_norm=True,
    group_norm_groups=2,
)
```

To set **Layer Normalization** only
```python
model = Net(
    batch_norm=False,
    layer_norm=True,
    group_norm=False,
)
```

To set **Batch Normalization** only and no dropout
```python
model = Net(
    batch_norm=True,
    layer_norm=False,
    group_norm=False,
    dropout_value=0,
)
```

---

# Training Graphs
<img src="https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S6/resources/test_acc_4models.png" width="500" height="400" />
<img src="https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S6/resources/test_acc_3models.png" width="500" height="400" />
<img src="https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S6/resources/test_loss_4models.png" width="500" height="400" />
<img src="https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S6/resources/test_loss_3models.png" width="500" height="400" />

---

# Misclassified Images
- Model 0. (BN only)
<img src="https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S6/resources/model_0_mis.png" width="800" height="400" />
- Model 1. (GN only)
<img src="https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S6/resources/model_1_mis.png" width="800" height="400" />
- Model 2. (LN only)
<img src="https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S6/resources/model_2_mis.png" width="800" height="400" />
- Model 3. (BN + L1)
<img src="https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S6/resources/model_3_mis.png" width="800" height="400" />

---

# Analysis
- the first version of the model only has Batch Normalization.
- out all the different Normalization tried, Layer Normalization yielded the best results. It also had a robust accuracy maintained after the 10th epoch.
- adding L1 brought down the performance to about as good as a random model. We tried different approaches like removing LR scheduler, lower LR rate, and also higher lambdas. nothing seemed to work. 

--- 

# Normailzations implementation on Sheet --> [sheet](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S6/updated_Normalizations.xlsx)
This spread sheet demonstrates the working of Batch Normalization, Group Normalization, and Layer Normalization.
![image](https://user-images.githubusercontent.com/8600096/139729172-9010cae7-d71d-4200-bd29-508aa153db09.png)
![image](https://user-images.githubusercontent.com/8600096/139729056-f2b6ef12-48be-443f-90aa-0bf10ec7e70c.png)
![image](https://user-images.githubusercontent.com/8600096/139729084-dc601498-5fa8-4cab-bd38-807e1224c3ba.png)



