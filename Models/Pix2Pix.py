import os
import argparse
import numpy as np
import tensorflow as tf
import h5py
import random
from PIL import Image
import cv2
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import concatenate, LeakyReLU, ReLU, Dropout,ZeroPadding2D
from tensorflow.keras import Input
from google.colab.patches import cv2_imshow 

# PARSER
 
class Args:
  mode = 'test'
  input = '/dataset'
  save = '/save'
  lr = 0.001
  epochs = 100
  batch = 5
  LAMBDA = 100
 
pars=Args()


def my_generator(f):
    keys = list(f.keys())
    np.random.shuffle(keys)
    for k in keys:
        yield f[k][0], f[k][1]

def conv(filters, kernel, strides, batch=True):
    model = tf.keras.Sequential()
    model.add(Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding='same', use_bias=False))
    if batch:
        model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    return model

def deconv(filters, kernel, strides, drop=True):
    model = tf.keras.Sequential()
    model.add(Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides, padding='same', use_bias=False))
    model.add(BatchNormalization())
    if drop:
        model.add(Dropout(0.5))
    model.add(ReLU())
    return model

def gan_gen(filters, kernel, strides):
    
    inp = Input(shape=img_shape)
    #  CONVOLUTION
    conv_lay = [
            conv(filters,kernel,strides,False),
            conv(filters*2,kernel,strides),
            conv(filters*4,kernel,strides),
            conv(filters*8,kernel,strides),
            conv(filters*8,kernel,strides),
            conv(filters*8,kernel,strides),
            conv(filters*8,kernel,strides),
            conv(filters*8,kernel,strides)
            ]

    #  DECONVOLUTION
    deconv_lay = [
            deconv(filters*8, kernel, strides),
            deconv(filters*8, kernel, strides),
            deconv(filters*8, kernel, strides),
            deconv(filters*8, kernel, strides, False),
            deconv(filters*4, kernel, strides, False),
            deconv(filters*2, kernel, strides, False),
            deconv(filters, kernel, strides, False)    
            ]
    #  OUTPUT    
    out = Conv2DTranspose(3, kernel_size=kernel, strides=strides, padding='same', activation='tanh')

    
    skips, res = [], inp
    for c in conv_lay:
        res = c(res)
        skips.append(res)
    skips = reversed(skips[:-1])

    for d, skip in zip(deconv_lay, skips):
        res = d(res)
        res = concatenate([res, skip])
    res = out(res)

    return tf.keras.Model(inputs=inp, outputs=res)

def gan_dis(filters, kernel, strides):
    #  IMAGE CONCATENATION
    img1 = Input(img_shape) # origin
    img2 = Input(img_shape) # target
    images = concatenate([img1,img2])

    #  CONVOLUTION
    d1 = conv(filters,kernel,strides, False)(images)
    d2 = conv(filters*2, kernel,strides)(d1)
    d3 = conv(filters*4, kernel,strides)(d2)

    #  OUTPUT
    d4 = ZeroPadding2D()(d3)
    d5 = Conv2D(filters*8,kernel_size=kernel, strides=1, padding='same')(d4)
    d6 = BatchNormalization()(d5)
    d7 = LeakyReLU()(d6)
    d8 = ZeroPadding2D()(d7)
    out = Conv2D(1,kernel_size=kernel, strides=1, padding='same')(d8)
    return tf.keras.Model(inputs=[img1,img2], outputs=out)
    
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (pars.LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

@tf.function
def train(image, label):
    with tf.GradientTape() as gen, tf.GradientTape() as dis:
        g_out = generator(image, training=True)
        d_out = discriminator([image, label], training=True)
        d_out2 = discriminator([image, g_out], training=True)

        g_total, g_loss, g_l1 = generator_loss(d_out2, g_out, label)
        d_loss = discriminator_loss(d_out, d_out2)

    g_gradients = gen.gradient(g_total,
                                          generator.trainable_variables)
    d_gradients = dis.gradient(d_loss,
                                               discriminator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients,
                                          generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_gradients,
                                              discriminator.trainable_variables))
    
    return g_total, d_loss


# LOAD DATASET
 
f = h5py.File(pars.input + '/' + 'GAN_' + pars.mode + '.hdf5','r')

samples = len(list(f.keys()))
img_width = len(list(f[list(f.keys())[0]][0]))
img_height = len(list(f[list(f.keys())[0]][0][0]))
img_channels = len(list(f[list(f.keys())[0]][0][0][0]))
img_shape = (img_width, img_height, img_channels)

d_optimizer = tf.keras.optimizers.Adam(learning_rate=pars.lr, beta_1=0.5)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=pars.lr, beta_1=0.5)
discriminator = gan_dis(64,4,1)
generator = gan_gen(64,4,2)
cp = tf.train.Checkpoint(generator_optimizer=g_optimizer,
                        discriminator_optimizer=d_optimizer,
                        generator=generator,
                        discriminator=discriminator)


if pars.mode == 'train':
    data = tf.data.Dataset.from_generator(lambda: my_generator(f), output_types=(tf.float32,tf.float32),
                                       output_shapes=(tf.TensorShape([img_width,img_height,img_channels]),
                                                      tf.TensorShape([img_width,img_height,img_channels])))
    data = data.batch(pars.batch)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for e in range(pars.epochs):  
        print("Epoch: ", e+1)
        g_total, d_total = 0, 0
        start = time.time()
        for n, (img, label) in data.enumerate():
            gl, dl = train(img, label)
            g_total += gl
            d_total += dl
        if e % samples*20 == 0:
            cp.save(pars.save + '/Pix2Pix/cp/')
        g_total = g_total/(samples/pars.batch)
        d_total = g_total/(samples/pars.batch)
        print('Gen loss: ', np.asarray(g_total), 'Disc loss: ', np.asarray(d_total))
        print ('Time taken for epoch {} is {} sec\n'.format(e + 1,
                                                        time.time()-start))
    cp.save(pars.save + '/Pix2Pix/cp/')
 
elif pars.mode == 'test':
    cp.restore(tf.train.latest_checkpoint(pars.save + '/Pix2Pix/cp'))
    keys = sorted(list(f.keys()), key=len)
    data = [f[n][0] for n in keys]

    data = np.array(data)

    new = generator(data)
    new = np.array(new)
    new = ((new + 1)* 2  * 255).astype(np.uint8)
    imageRGB = cv2.cvtColor(new[-1,:,:,:], cv2.COLOR_BGR2RGB)
    cv2.imshow(imageRGB)
    cv2.waitKey()
    img = Image.fromarray(imageRGB)
    img.save(pars.save + '/Pix2Pix/Prediction.jpg',format='JPEG',subsampling=0,quality=100)

    #real = cv2.cvtColor(f[keys[9]][1]*255, cv2.COLOR_BGR2RGB)
    #real = Image.fromarray(real)
    #real.save(pars.save + '/Pix2Pix/Real.jpg',format='JPEG',subsampling=0,quality=100)
  
