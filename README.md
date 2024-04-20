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
1. Fraiwan, Mohammad; Audat, Ziad; Manasreh, Tereq (2022), "A dataset of scoliosis, spondylolisthesis, and normal vertebrae X-ray images", Mendeley Data, V1, doi: 10.17632/xkt857dsxk.1,
   download at https://data.mendeley.com/datasets/xkt857dsxk/1 
3. Challenge dataset download at https://aasce19.github.io/
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
input  image size: 299x299 mode: RGB

Download files:
1. For testing model.ipynb
2. Example images folder
#Load Best models (file format ... .h5)
3. InceptionV3 Model 
4. Xception Model
5. ResNett50 Model
6. ResNet101 Model





