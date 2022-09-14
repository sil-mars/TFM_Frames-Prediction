import os
import argparse
import numpy as np
import tensorflow as tf
import h5py
from PIL import Image
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv3D, ConvLSTM2D, Dropout
from tensorflow.keras import Input
from google.colab.patches import cv2_imshow
 
# PARSER
 
class Args:
  mode = 'test'
  input = '/dataset'
  save = '/save'
  lr = 0.001
  epochs = 100
 
pars=Args()

def my_generator(f):
    keys = list(f.keys())
    np.random.shuffle(keys)
    for k in keys:
        yield list([f[k][0]]),list([f[k][1]])
  
# LOAD DATASET
if pars.mode == 'continue':
  f = h5py.File(pars.input + '/' + 'ConvLSTM_train.hdf5','r')
else:
  f = h5py.File(pars.input + '/' + 'ConvLSTM_' + pars.mode + '.hdf5','r')
 
samples = len(list(f.keys()))
images_per_sample = len(list(f[list(f.keys())[0]][0]))
img_width = len(list(f[list(f.keys())[0]][0][0]))
img_height = len(list(f[list(f.keys())[0]][0][0][0]))
 
if pars.mode == 'train' or pars.mode =='continue':
 
    data = tf.data.Dataset.from_generator(lambda: my_generator(f), output_types=(tf.float32,tf.float32),
                                       output_shapes=(tf.TensorShape([1,images_per_sample,img_width,img_height,3]),
                                                      tf.TensorShape([1,images_per_sample,img_width,img_height,3])))
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=(pars.save + '/ConvLSTM/cp/' + 'cp.ckpt'),
                                        save_weights_only=True,
                                        verbose=1,
                                        save_freq=samples*20)
 
    # MODEL
    model = Sequential()
    model.add(Input(shape=(None, img_width, img_height, 3)))
    model.add(ConvLSTM2D(filters=20,   
                        kernel_size=(5, 5),
                        padding="same", 
                        return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=20,
                        kernel_size=(5, 5),
                        padding="same", 
                        return_sequences=True))
    model.add(BatchNormalization())
    """model.add(ConvLSTM2D(filters=20,
                        kernel_size=(5, 5),
                        padding="same",
                        return_sequences=True))
    model.add(BatchNormalization())"""
    model.add(Conv3D(filters=3,
                    kernel_size=(10, 10, 10),
                    activation="tanh",
                    padding="same",
          data_format='channels_last'))
 
    model.summary()
 
    optimizer= tf.keras.optimizers.Adadelta(learning_rate=pars.lr)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer)
    if pars.mode == 'continue':
        model.load_weights(pars.save + '/ConvLSTM/cp/' + 'cp.ckpt')
    model.fit(data,
                epochs=pars.epochs,
                verbose=2, callbacks=[cp])          
    model.save(pars.save + '/ConvLSTM/' + 'full_model')
 
elif pars.mode == 'test':
    data = []
    keys = list(f.keys())
    for k in keys:
        data.append(list(f[k][0]))
 
    data = np.array(data)[::,:10,::,::,::]  
    imageRGB = cv2.cvtColor(data[0][9], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(imageRGB.astype(np.uint8)*255)
    img.save(pars.save + '/ConvLSTM/0.png')

    model = tf.keras.models.load_model(pars.save + '/ConvLSTM/' + 'full_model')

    for j in range(10): 
        new = model.predict(data[::,::,::,::,::])
        new = (new * 255).astype(np.uint8)

        new = new[::, -1, ::, ::, ::]
        data = np.concatenate((data,new[np.newaxis, ::, ::, ::, ::]), axis=1)
        imageRGB = cv2.cvtColor(new[0], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(imageRGB)
        img.save(pars.save + '/ConvLSTM/' + str(j+1) + '.jpg',format='JPEG',subsampling=0,quality=100)
