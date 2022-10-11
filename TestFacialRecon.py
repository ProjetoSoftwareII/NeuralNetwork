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
import numpy as np
def distancia_euclidiana(embeding1,embeding2):
  sub = embeding1.numpy() - embeding2.numpy()
  square = sub**2
  summatory = np.sum(square)
  return np.sqrt(summatory)


T = 5.5


TP = 0
FN = 0

TN = 0
FP = 0

count=0
average_distance_same_sub = 0
average_distance_dif_sub = 0
for i in range(len(test_labels)):
  if len(test_images[i])<=1:
    continue
  img1 = test_images[i][0]
  img2 = test_images[i][1]

  img1 = tf.expand_dims(img1,0)
  emb1 = model(img1)
  img2 = tf.expand_dims(img2,0)
  emb2 = model(img2)

  distance = distancia_euclidiana(emb1,emb2)
  if distance < T:
    TP+=1
  else:                 #calcular com a mesma pessoa
    FN+=1
  average_distance_same_sub+=distance

  if i<len(test_labels)-1:
    img3 = test_images[i+1][0]
  else:
    img3 = test_images[1][0]
  
  img3 = tf.expand_dims(img3,0)
  emb3 = model(img3)

  distance = distancia_euclidiana(emb1,emb3)
  if distance >= T:
    TN+=1
  else:                 #calcular com a pessoa diferente
    FP+=1
  average_distance_dif_sub+=distance
  count+=1

average_distance_same_sub = average_distance_same_sub/count
average_distance_dif_sub = average_distance_dif_sub/count

print('True positives ',TP)
print('False negative ',FN)
print('True Negative ',TN)
print('False positive ',FP)

print('distancia media mesma pessoa',average_distance_same_sub)
print('distancia media pessoa diferente',average_distance_dif_sub)