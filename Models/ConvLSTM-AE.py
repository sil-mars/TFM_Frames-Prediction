import os
import argparse
import numpy as np
import tensorflow as tf
import h5py
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.layers import BatchNormalization, Conv3D, ConvLSTM2D, Conv3DTranspose
from tensorflow.keras import Input
from google.colab.patches import cv2_imshow
 
# PARSER
 
class Args:
  mode = 'test'
  input = '/content'
  save = '/content/drive/MyDrive/TFM/dataset'
  lr = 0.001
  epochs = 200

pars=Args()
def my_generator(f):
    keys = list(f.keys())
    np.random.shuffle(keys)
    for k in keys:
        images = f[k][0]
        yield [images],[images]
 
# LOAD DATASET
 
f = h5py.File(pars.input + '/' + 'ConvLSTM-AE_' + pars.mode + '.hdf5','r')
samples = len(list(f.keys()))
images_per_sample = len(list(f[list(f.keys())[0]][0]))
img_width = len(list(f[list(f.keys())[0]][0][0]))
img_height = len(list(f[list(f.keys())[0]][0][0][0]))
 
if pars.mode == 'train':
 
    data = tf.data.Dataset.from_generator(lambda: my_generator(f), output_types=(tf.float32,tf.float32),
                                                          output_shapes=(tf.TensorShape([1,images_per_sample,img_width,img_height,3]),
                                                                                tf.TensorShape([1,images_per_sample,img_width,img_height,3])))
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=(pars.save + '/ConvLSTM-AE/cp/' + 'cp.ckpt'),
                                        save_weights_only=True,
                                        verbose=1,
                                        save_freq=samples*20)
 
    # -----  MODEL  -----
    model = Sequential()
    #model.add(Input(shape=(img_width, img_height,images_per_sample,3)))
    # CODIF
    model.add(Conv3D(filters=256,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(400,400,10,3)))
    model.add(Conv3D(filters=128,kernel_size=(5,5,1),strides=(3,3,1),padding='valid'))
    #model.add(Conv3D(filters=64,kernel_size=(1,2,2),strides=(1,2,2),padding='valid',activation='relu'))
    # TEMPORAL AE
    model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))
    model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True,))  
    model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))
    # DECODIF
    #model.add(Conv3DTranspose(filters=128,kernel_size=(1,3,3),strides=(1,2,2),padding='valid',activation='relu'))
    model.add(Conv3DTranspose(filters=256,kernel_size=(5,5,1),strides=(3,3,1),padding='valid'))
    model.add(Conv3DTranspose(filters=3,kernel_size=(12,12,1),strides=(4,4,1),padding='valid'))
    model.summary()
 
    optimizer= tf.keras.optimizers.Adam(learning_rate=pars.lr)
    # 'mean_square_error'
    # 'binary_crossentropy' no tiene sentido xq relu da valores [0,inf)
    #model.compile(loss='mean_squared_error',
                  #optimizer=optimizer)
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
 
    model.fit(data,
                epochs=pars.epochs,
                verbose=2, callbacks=[cp])          
    model.save(pars.save + '/ConvLSTM-AE/' + 'full_model')
 
elif pars.mode == 'test':
    data = [f[list(f.keys())[0]][0]]
    data = np.array(data)[::,::,::,:10,::]
    cv2.imwrite(pars.save + '/ConvLSTM-AE/0.jpg',data[0,::, ::, 9, ::]*255)

    model = tf.keras.models.load_model(pars.save + '/ConvLSTM-AE/' + 'full_model')
    
    for j in range(10): 
        new = model.predict(data[::,::,::,::,::])
        new = (new * 255).astype(np.uint8)
        new = new[::, ::, ::, -1, ::]
        data = np.concatenate((data[::,::,::,1:,::], new[::, ::, ::, np.newaxis,::]), axis=3)
        cv2.imwrite(pars.save + '/ConvLSTM-AE/' + str(j+1) + '.jpg', new[0])
