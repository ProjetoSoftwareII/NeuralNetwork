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
#from numpy import uint8
def load_image(path):
  image = tf.io.read_file(path)
  image = tf.io.decode_image(image)
  image = tf.image.convert_image_dtype(image,dtype=tf.uint8)
  return image


#
# Seção onde é feito o load das imagens, possivelmente precisa ser modificado o diretorio
#

images = []
labels = []

dirs = os.listdir('/usr/app/src/dataset/post-processed')
#dirs = dirs[:100]


for dir_name in tqdm(dirs):
  dir_path = '/usr/app/src/dataset/post-processed'+'/'+dir_name
  labels.append(dir_name)
  images.append([])
  for image_name in os.listdir(dir_path):
    image = load_image(dir_path+'/'+image_name)

    images[labels.index(dir_name)].append(image)

train_percent = 0.7
val_percent = 0.2
test_percent = 0.1

train_end = int(len(labels)*train_percent)

test_end = int(len(labels)*test_percent) + train_end

val_end = int(len(labels)*val_percent) + test_end


train_images = images[:train_end]
train_labels = labels[:train_end]
#print(train_images[0][0].shape)


test_images = images[train_end : test_end]
test_labels = labels[train_end : test_end]



val_images = images[test_end : val_end]
val_labels = labels[test_end : val_end]

#
# Termino da seção de input de imagens
#
def distancia_euclidiana(embeding1,embeding2):
  sub = embeding1.numpy() - embeding2.numpy()
  square = sub**2
  summatory = np.sum(square)
  return np.sqrt(summatory)


teste=True



if not teste:
  

#
#Define a arquitetura da rede neural
#
  target_shape = (112,112)
  base_cnn = resnet.ResNet50(
      weights='imagenet', input_shape= target_shape + (3,), include_top=False
  )
  
  #base_cnn.layers.trainable = False

  flatten = layers.Flatten()(base_cnn.output)
  dense1 = layers.Dense(512, activation="relu")(flatten)
  dense1 = layers.BatchNormalization()(dense1)
  dense2 = layers.Dense(256, activation="relu")(dense1)
  dense2 = layers.BatchNormalization()(dense2)
  output = layers.Dense(256)(dense2)



  model = Model(base_cnn.input, output, name="layer")
  
  for layer in model.layers[:-3]:
    layer.trainable = False


  model.compile()


  learning_rate = 2e-5
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  tolerance = 4

  #
  #   Termino da arquitetura
  #

  #
  #   Funções de treino e validação durante o treino
  #

  #se o treinamento nao correr bem, voltar a separar o dataset dentro da função de treino e validação, caso o erro required broadcastable shapes [Op:AddV2] voltar
  # ver a exception Disable protected access

  def train_step(x,y):
    with tf.GradientTape() as tape:
      outputs = model(x)
      loss = batch_hard_triplet_loss(y,outputs,0.2,squared=True) #calcula loss
      grads = tape.gradient(loss,model.trainable_weights) #calcula gradiente
      optimizer.apply_gradients(zip(grads,model.trainable_weights)) #aplica os pesos
    return loss

  def val_step(x,y):
    outputs = model(x)
    loss = batch_hard_triplet_loss(y,outputs,0.2,squared=True) #calcula loss
    return loss

  #
  # termino   Funções de treino e validação durante o treino
  #

  ########### separa o dataset de novo
  x_train = []
  y_train = []
  for i,label in enumerate(train_labels):
    for image in train_images[i]:
      x_train.append(image)
      y_train.append(label)

  x_val = []
  y_val = []
  for i,label in enumerate(val_labels):
    for image in val_images[i]:
      x_val.append(image)
      y_val.append(label)

  x_train = tf.stack(x_train)
  x_val = tf.stack(x_val)
  ######### termino da separação

  #
  # Treino da rede neural
  # 
  epochs = 10000

  batch_size = 256

  progress_bar = tqdm(epochs)
  tolerance_count = 0
  last_loss_val=0
  for epoch in tqdm(range(epochs)):

    loss_train = 0
    loss_val =0
    for i in range(0, len(x_train), batch_size):
      batch_x = x_train[i:i+batch_size:]
      batch_y = y_train[i:i+batch_size:]
      loss_train += train_step(batch_x,batch_y)

    for i in range(0, len(x_val), batch_size):
      batch1_x = x_val[i:i+batch_size:]
      batch1_y = y_val[i:i+batch_size:]
      loss_val += val_step(batch1_x,batch1_y)
    #loss_val = val_step(x_val,y_val)

    loss_train = loss_train.numpy()
    loss_val = loss_val.numpy()

    progress_bar.set_postfix({'loss_train':loss_train,'loss_val':loss_val})
    if epoch%5==0:
      model.save('/usr/app/src/dataset/treino/'+'lt: '+str(loss_train)+' lv: '+str(loss_val)+' epoch:'+str(epoch)+'.h5')
    if epoch != 1:
      if (last_loss_val - loss_val) < 10:
        tolerance_count+=1
      else:
        tolerance_count=0
        model.save('/usr/app/src/dataset/treino/rede_treinada.h5')

    if tolerance_count == tolerance:
      break
    last_loss_val = loss_val
  #
  # Fim do treino da rede neural
  #  
else:
  model = tf.keras.models.load_model('/usr/app/src/dataset/treino/rede_treinada.h5')
  
  T = 3.63


  TP = 0
  FN = 0

  TN = 0
  FP = 0

  count=0
  average_distance_same_sub = 0
  average_distance_dif_sub = 0
  for i in tqdm(range(len(test_labels))):
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