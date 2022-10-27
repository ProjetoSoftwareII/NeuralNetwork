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


#
#Define a arquitetura da rede neural
#
target_shape = (112,112)
base_cnn = resnet.ResNet50(
    weights='imagenet', input_shape= target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)



model = Model(base_cnn.input, output, name="layer")


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

def train_step(x,y):
  losses = 0
  for i in range(0, len(x), 256):
    batch_x = x[i:i+256:]
    batch_y = y[i:i+256:]
    with tf.GradientTape() as tape:
      outputs += model(batch_x)
  loss = batch_hard_triplet_loss(y,outputs,0.2,squared=True) #calcula loss
  grads = tape.gradient(loss,model.trainable_weights) #calcula gradiente
  optimizer.apply_gradients(zip(grads,model.trainable_weights)) #aplica os pesos
  return losses

def val_step(x,y):
  for i in range(0, len(x), 256):
    batch_x = x[i:i+256:]
    batch_y = y[i:i+256:]
    outputs += model(batch_x)
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
epochs = 100

progress_bar = tqdm(epochs)
tolerance_count = 0
last_loss_val=0
for epoch in tqdm(range(epochs)):


  '''for i in range(0, len(x_train), 256):
    batch_x = x_train[i:i+256:]
    batch_y = y_train[i:i+256:]'''
  loss_train = train_step(x_train,y_train)

  '''for i in range(0, len(x_val), 256):
    batch1_x = x_val[i:i+256:]
    batch1_y = y_val[i:i+256:]'''
  loss_val = val_step(x_val,y_val)
  #loss_val = val_step(x_val,y_val)

  loss_train = loss_train.numpy()
  loss_val = loss_val.numpy()

  progress_bar.set_postfix({'loss_train':loss_train,'loss_val':loss_val})
  if epoch%5==0:
    model.save('/usr/app/src/dataset/'+'lt: '+str(loss_train)+' lv: '+str(loss_val)+' epoch:'+str(epoch)+'.h5')
  if epoch != 1:
    if (last_loss_val - loss_val) < 0.001:
      tolerance_count+=1
    else:
      tolerance_count=0
      model.save('/usr/app/src/dataset/rede_treinada.h5')

  if tolerance_count == tolerance:
    break
  last_loss_val = loss_val
#
# Fim do treino da rede neural
#