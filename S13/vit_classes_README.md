# ViT Classes Documentation 
Notebook with ViT code here -- [colab link](https://colab.research.google.com/github/eva7wandb/Eva7_Weights_Heist/blob/main/S13/S13_Class_Notes.ipynb)


## ViT Class Structure
The ViT implementation in the above notebook in total has 12 classes. (including the ViTConfig class).    
The nested structure in the diagram below indicates how some classes are wrapping other classes within it. (beyond wrapping, there are also transformations in most of these classes).     

![Vit Class Structure](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S13/resources/vit_class_structure.png)


## Descriptions

### ViTConfig
The ViTConfig class contains the hyperparameters for the model. Like number of attention heads, drop out values for different layers, activation choice, layernorm eps, image size, image channels, patch size, size of hidden layers, and so on. The default values used are as follows. 👇
```python
{'attention_probs_dropout_prob': 0.0,
 'hidden_act': 'gelu',
 'hidden_dropout_prob': 0.0,
 'hidden_size': 768,
 'image_size': 224,
 'initializer_range': 0.02,
 'intermediate_size': 3072,
 'layer_norm_eps': 1e-12,
 'num_attention_heads': 12,
 'num_channels': 3,
 'num_hidden_layers': 12,
 'patch_size': 16}
```

### PatchEmbeddings
This class coverts the input image to patches. The hyperparameter choices at this stage are the input image size (and channels), patch size, and embedding dimensions. 

### ViTEmbeddings

### ViTSelfAttention
### ViTSelfOutput
### ViTAttention

### ViTIntermediate
### ViTOutput

### ViTLayer

### ViTEncoder

### ViTPooler

### ViTModel


