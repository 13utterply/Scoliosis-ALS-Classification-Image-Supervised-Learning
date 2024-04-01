# Scoliosis-ALS-Classification-Image-Supervised-Learning
This project demonstrates how to apply pre-trained deep learning models to classify 7 types of scoliosis based on the Augmented Lehnert-Schroth (ALS) classification.

![ALS7types_](https://github.com/13utterply/Scoliosis-ALS-Classification-Image-Supervised-Learning/assets/151118115/7144a71c-28c0-4d34-ad1d-43db68ca7bb5)

## Preparation
### System requirement
The network was implemented based on the Keras-TensorFlow framework in Python and the Jupyter Notebook platform.
the implementation included CUDA 11.2, cuDNN8.1, TensorFlow 2.10.0 and Python 3.8.18.
These experiments were performed on a system with an Intel Core i7-12700H 2.70 GHz CPU, 64 GB of memory, and NVIDIA GeForce RTX 3080 Ti.

## Dataset
The datasets were derived from 3 private sources and 2 open sources. 
The open-source datasets provided by
1. Fraiwan, Mohammad; Audat, Ziad; Manasreh, Tereq (2022), "A dataset of scoliosis, spondylolisthesis, and normal vertebrae X-ray images", Mendeley Data, V1, doi: 10.17632/xkt857dsxk.1
2. Challenge dataset download at https://aasce19.github.io/
    - Training data release: "Dataset 16: 609 spinal anterior-posterior x-ray images" download at http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets
    - test data release: download at https://aasce19.github.io/
  
## Import libraries on Jupyter Notebook.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import InceptionV3, Xception, ResNet50, ResNet101

import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, classification_report, precision_score, recall_score, f1_score
import itertools
import os
import shutil
import random
import glob
import warnings
from datetime import datetime
import cv2
from PIL import Image

from tensorflow.python.keras.utils.data_utils import Sequence
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline
from tqdm import tqdm
from time import sleep
from tpot import TPOTClassifier
from tpot import TPOTRegressor
import pandas as pd

## Parameter of Deep learning models
Best fine-tuned the custom classification head, which included Global average pooling, Dense layer of 1024, 512, 256, and 128 units, respectively, serving as fully connected layers.
PReLU was used as an activation function. L2 regulation of 0.01 was applied to the 1024, 512, 256 layers, while a dropout of 0.5 was applied to the last layer.
The activation function for predictions used the sigmoid function with 7 outputs. The complied model used the Adam optimizer with a learning rate of 0.0001. 
The Model was trained with parameters of 10 epochs, a bach size of 32.


#Load Best fine-tuned each pre-trained models (file format ... .h5)
1. InceptionV3: 
2. Xception: 
3. ResNet50: 
4. ResNet101:

Example image for test models
1. 3CH: 
2. 3CTL: 
3. 3CN: 
4. 3CL: 
5. 4C: 
6. 4CL: 
7. 4CTL:



