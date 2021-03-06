# Class Descriptions in ViT Model definition


## Block

![Block Flow](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S12/resources/block_flow.JPG)


In this class, we use normalization with attention and normalization with MLP layers to generate the output and update the weights.
First we apply layer norm and the result is fed into the Attention function. Attention function will  update the weights and then we add the input to the results of attention (skip connection) to derive the output.
Next we take the output from attention plus input and pass it thru another layer norm before passing it thru the feed forward MLP layers. The result of this is added with the input (another skip connection ) to arrrive at the final output of the block.
There is also a utility function 'load_from' that takes weights and number of blocks as parameters to help load the weights and biases of the model attention and FF(or MLP) layers Query, Key, value and outputs attributes.

## Embeddings


This class helps to build patch and position embeddings given the image size and patch size or grid size as input. All of the embeddings will be of the same hidden size (passed as config parameter) so that they can be easily added or concatenated. 
During initilization, we calculate the number of patches based on the patch_size or grid_size received as input parameters. Patch embeddings are derived from a conv2d layer with kernal of size equal to patch_size. 
In forward function, the positial embedding is concatenated to the patch_embeddings after it is flattened and transposed. Finally a drop_out is applied to generate the final embeddings.


## MLP


![MLP Flow](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S12/resources/mlp_flow.JPG)

In this class we apply linear transformations with 'gelu' activation function with drop out. Then another linear transformation with just the dropout and there is no activation function used. 
Xavier_uniform initialization function initializes the weights of the two linear layers and the biases are initialized with norm function.


## Attention

![Attention flow](https://github.com/eva7wandb/Eva7_Weights_Heist/blob/main/S12/resources/attention_flow.JPG)

This class helps to define and process multi-head attention layers. 
Input is broken down into Query, Key and Value and they are passed thru a liner layer. They are transposed and permuated to get them to be of same shape. Attention probabilities are calculated by applying softmax on the result of matix multiplication of query and key layers. Dropout is also applied on attention probabilities. 
Context layer is calculated by matrix multiplication of attention probabilities with values . To get the attention output, linear layer with dropout is applied on context layer output.


## Encoder

This class applies all the layers in multi-head attention transformer blocks to generate the encoder embeddings along with attention weights. 
In the init function it builds the layers as specified in the configuration input layer size. In forward function it applies the layer blocks and the layer normalization to generate the output
