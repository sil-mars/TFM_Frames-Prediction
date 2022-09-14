# TFM_Frames-Prediction

https://user-images.githubusercontent.com/68307425/190269443-4d2ba2ad-6d8a-4103-81bd-e3a31ce047d1.mp4

## ATTENTION!

This code is adapted to google colab. Changes will be necessary to run it in local. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------


The project is based on the idea of predicting the next frames in a video of a bacterial colony. This is useful in order to predict how a colony grows. Especially in cases of several types of bacteria, where the relationship between them may be competitive or cooperative. 

There are 4 files:

  - Preprocessing: input frames from video are cropped and lowered in number to reduce noise and computation time, respectively. Afterwards, the images are normalized.

  - ConvLSTM: two stacked Conv-LSTM layers.

  - ConvLSTM-AE: autoencoder with 3 Conv-LSTM layers in between.

  - Pix2Pix: GAN for image transformation 1 to 1. 

---------------------------------------------------------------------------------------------------------------------------------------------------------------

## ATENCION !

Este código está adaptado a google colab. Será necesario hacer cambios para ejecutarlo en local.

--------------------------------------------------------------------------------------------------------------------------------------------------------------

Este proyecto esta basado en la idea de predecir los siguientes fotogramas en un video de una colonia de bacterias. Esto es util de cara a predecir como crecerá una colonia bacteriana. Especialmente en aquellos casos en los que hay varios tipos de bacterias en la colonia, y donde la relacion entre ellos puede ser competitiva o cooperativa.

Hay 4 archivos:

  - Preprocesamiento: los frames de entrada de un video son recortados y reducidos en número para reducir el ruido y el tiempo computacional respectivamente.  Posteriormente, las imagenes son normalizadas.

  - ConvLSTM: dos capas Conv-LSTM apiladas.

  - ConvLSTM-AE: un autoencoder con dos capas Conv-LSTM entre medias.

  - Pix2Pix: GAN para la transformación de imagenes 1 a 1.
