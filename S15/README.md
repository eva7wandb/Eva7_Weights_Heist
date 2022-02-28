# Training Resnet34 model over CIFAR100 dataset on AWS Sagemaker Notebook Instance

The task is to focus on end-to-end development of deep learning application for image
classification using AWS Sagemaker Instances.

This process consists of three phases :-

  1. Model Development (algorithms / architecture --> model)
  2. Training the model (Train model using train data based on loss or other metrics)
  3. Deployment of trained model

Each of these 3 phases can have a different instance on Sagemaker which implies different m/c can be used for these tasks. Usually, model development can be done on a generic CPU instance (minimal performance) such as ml.t2.medium. The training process needs a computation intensive m/c instance, usually high performing multiple GPU instances (like P or G instances). For the deployment, the application target audience determines the instance choice.

The team tried to get access to the spot requests on GPU instances, specifically, P3 instances so that a multi-GPU (4 GPU) can be applied for the training phase, but did not
get approval from the AWS customer service. Hence it is decided to use the following instances for the end-to-end development of image classifier.

  1. ml.t2.medium for notebook model development (coding part)
  2. ml.c5.9xlarge for training procedure
  3. m.c4.8xlarge for deployment purpose

### Steps to create end-to-end model using Sagemaker

  1. Create and start Sagemaker Notebook Instance: While creating the notebook instance, 
     one can also get to choose associated role and S3 bucket for the instance. After creating the role, add policies governing access to role, with regard to S3 bucket used for storage as well as policies governing API access. Finally, check if the required policies got attached to the role we wanted.

  2. In the Sagemaker notebook instance, one can add required files and jupyter notebook
     and choose the kernel (the virtual environment) to execute these files.

  3. The model training instance and the model deployment instance are called from the
     jupyter notebook mentioned in the above step.

  4. We can cross check the Sagemaker menu for relevant files and endpoints. Once 
     finished, if not using for inferencing work, delete those endpoints.

Some of the hyperparameters used in the repo files while training are, epochs = 4,
batch_size = 128, learning_rate = 2e-4. For the test data, we set the batch_size = 4 as it will be easy to cross-check during inferencing.

Things to note here is we had to set GPU=0 as we are using the CPU instances only even for the training of the model.

Brief on the model: Resnet34 architecture is used here. A pre-trained model is fetched from the torchvision.models module, and since it was pre-trained from Imagenet model, we need to change the number of output features to suit the number of classes in CIFAR100 dataset i.e., we have change the output features (classes) from 1000 to 100. This is done using the function `create_model()` included in the script `cifar100_pl.py`. This script `cifar100_pl.py` contains `CIFAR100DataModule` as well as `LitResnet34` classes defined. The `CIFAR100DataModule` class defines the dataloader for loading train, validation and test data loaders. The `LitResnet34` class defines the Resnet model, train, validation and test steps, save model options, optimizer and the trainer method from pytorch lightning module. The args used for the train function is also defined in the `LitResnet34` class.


