import tensorflow as tf
import os
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
import tensorflow_addons as tfa
import gc
from tensorflow.keras.backend import clear_session
from tqdm import tqdm
from TripletLoss import *
import numpy as np


data_dir = os.listdir('/usr/app/src/dataset/post-processed')

batch_size = 256    
img_height = 112
img_width = 112


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  labels='inferred',
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  labels='inferred',
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

tf.keras.applications.resnet50.preprocess_input( train_ds, data_format=None)
tf.keras.applications.resnet50.preprocess_input( val_ds, data_format=None)