# TFM_Frames-Prediction

## ATTENTION!

This code is adapted to google colab. Changes will be necessary to run it in local. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------


The project is based on the idea of predicting the next frames in a video of bacteria colonies. This is useful in order to predict how a colony grows. Especially in cases of several types of bacteria, where the relationship between them may be competitive or cooperative. 


https://user-images.githubusercontent.com/68307425/190269443-4d2ba2ad-6d8a-4103-81bd-e3a31ce047d1.mp4



There are 4 files:

  - Preprocessing: input frames from video are cropped and lowered in number to reduce noise and computation time, respectively. Afterwards, the images are normalized.

  - ConvLSTM: two stacked Conv-LSTM layers.

  - ConvLSTM-AE: autoencoder with 3 Conv-LSTM layers in between.

  - Pix2Pix: GAN for image transformation 1 to 1. 
