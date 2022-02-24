import sagemaker
import uuid

sagemaker_session = sagemaker.Session()
print("SageMaker version: " + sagemaker.__version__)

bucket = sagemaker_session.default_bucket()
prefix = "sagemaker/eva7-s15-cifar100"

role = sagemaker.get_execution_role()

checkpoint_suffix = str(uuid.uuid4())[:8]
checkpoint_s3_path = "s3://{}/checkpoint-{}".format(bucket, checkpoint_suffix)

print("Checkpointing Path: {}".format(checkpoint_s3_path))

print(role)

import os
import subprocess

instance_type = "local"

if subprocess.call("nvidia-smi") == 0:
    ## Set type to GPU if one is present
    instance_type = "local_gpu"

print("Instance Type = " + instance_type)

!pip install pytorch-lightning --quiet

import numpy as np
import torchvision, torch
import matplotlib.pyplot as plt
import pickle

import pytorch_lightning as pl
from pytorch_lightning import seed_everything  # Global Seeding

from utils_cifar100 import CIFAR100DataModule, unpickle, imshow

early_stop_callback = EarlyStopping(
    monitor="val_loss", patience=10, verbose=False, mode="min"
)

MODEL_CKPT_PATH = "./resnet34_model/"
MODEL_CKPT = "resnet34-{epoch:02d}-{val_loss:.2f}"

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=MODEL_CKPT_PATH,
    filename=MODEL_CKPT,
    save_top_k=3,
    mode="min",
)

# Init our data pipeline
dm = CIFAR100DataModule(batch_size=64)

# To access the x_dataloader we need to call prepare_data and setup.
dm.prepare_data()
dm.setup()

trainloader = dm.train_dataloader()
testloader = dm.test_dataloader()

metaData = unpickle("./data/cifar-100-python/meta")

label_names = metaData["fine_label_names"]
print(len(label_names))

# get some random training images
trainiter = iter(trainloader)
img_train, lbl_train = trainiter.next()

# show images
imshow(torchvision.utils.make_grid(img_train))
labels_list = lbl_train.tolist()
print("Showing label names of first 8 train images")
for i in range(8):
    print(labels_list[i], label_names[labels_list[i]])

# get some random test images
testiter = iter(testloader)
img_test, lbl_test = testiter.next()

# show images
imshow(torchvision.utils.make_grid(img_test))
labels_list = lbl_test.tolist()
print("Showing label names of first 8 test images")
for i in range(8):
    print(labels_list[i], label_names[labels_list[i]])


inputs = sagemaker_session.upload_data(
    path="data", bucket=bucket, key_prefix="data/cifar100"
)

from sagemaker.pytorch import PyTorch

use_spot_instances = True
max_run = 600
max_wait = 1200 if use_spot_instances else None

hyperparameters = {"batch_size": 64, "checkpoint-path": checkpoint_s3_path}

checkpoint_local_path = "/opt/ml/checkpoints"

cifar100_estimator = PyTorch(
    entry_point="source_dir/cifar100.py",
    role=role,
    framework_version="1.7.1",
    py_version="py3",
    hyperparameters=hyperparameters,
    instance_count=1,
    instance_type="ml.p3.8xlarge",
    base_job_name="cifar100-Feb24-v0-spot",
    checkpoints_s3_uri=checkpoint_s3_path,
    checkpoint_local_path=checkpoint_local_path,
    debugger_hook_config=False,
    use_spot_instances=use_spot_instances,
    max_run=max_run,
    max_wait=max_wait,
)

cifar100_estimator.fit(inputs)

from sagemaker.pytorch import PyTorchModel

predictor = cifar100_estimator.deploy(
    initial_instance_count=1, instance_type="ml.m4.xlarge"
)

# get some test images
dataiter = iter(testloader)
images, labels = dataiter.next()
print(images.size())

labels_list = labels.tolist()

# print images
imshow(torchvision.utils.make_grid(images))
print("Ground Truth: ", " ".join("%4s" % label_names[labels_list[j]] for j in range(4)))

outputs = predictor.predict(images.numpy())

_, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)

print("Predicted: ", " ".join("%4s" % label_names[predicted[j]] for j in range(4)))


# predictor.delete_endpoint()  #Very Important !!!
