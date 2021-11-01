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

