#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:52:15 2020

@author: shubham
"""

import numpy as np
import os
from tensorflow.keras.callbacks import  ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pandas as pd
physical_devices =tf.config.experimental.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


METRICS = [
      tf.keras.metrics.Accuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')]

batch_size = 4

train_df = pd.read_csv('/home/gpu3/shubham/refuge/code/localization/complete_localisation.csv')
val_df = pd.read_csv('/home/gpu3/shubham/refuge/code/localization/val_localisation.csv')
#img_list = train_df['ImgName'].to_list()

#train_datagen = ImageDataGenerator(rescale = 1. / 255.,
#                    preprocessing_function=preprocess_input)
#
train_datagen = ImageDataGenerator(rotation_range=5,
                    width_shift_range=5,
                    height_shift_range=5,
                    fill_mode='nearest',
                    rescale = 1. / 255.,
                    preprocessing_function=preprocess_input)

##preprocessing_function=preprocess
test_datagen = ImageDataGenerator(rescale = 1. / 255.,
                                  preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="/home/gpu3/shubham/refuge/Tsegmentation/images/complete",
        x_col="ImgName",
        y_col=["Fovea_X","Fovea_Y"],
        batch_size=batch_size,
        shuffle=True,
        class_mode="raw",
        target_size=(600,600))

val_generator=test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory="/home/gpu3/shubham/refuge/Tsegmentation/images/val",
        x_col="ImgName",
        y_col=["Fovea_X","Fovea_Y"],
        batch_size=batch_size,
        shuffle=True,
        class_mode="raw",
        target_size=(600,600))


import tensorflow
import efficientnet.tfkeras as efn
base_model = efn.EfficientNetB5(input_shape=(600,600,3), weights='imagenet', include_top=False,classes=1)
x = tensorflow.keras.layers.GlobalAveragePooling2D()(base_model.output)
#x = tensorflow.keras.layers.Dense(128, activation='relu')(x)
#x = tensorflow.keras.layers.Dense(16, activation='relu')(x)
output = tensorflow.keras.layers.Dense(2, activation='sigmoid')(x)
model = tensorflow.keras.models.Model(inputs=[base_model.input], outputs=[output])                 
model.summary()


#
#Net, preprocess_input = Classifiers.get('xception')
#base_model = Net(input_shape=(512, 512, 3),classes=2,weights= "imagenet", include_top=False)
#x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
#output = tf.keras.layers.Dense(2, activation='sigmoid')(x)
#model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])                 
#model.summary()

model.load_weights('refuge_localization_efficientnetb7.h5')

model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.0001), loss = 'MSE', metrics = METRICS)


checkpoint = ModelCheckpoint("refuge_localization_efficientnetb7_final.h5", 
                             monitor='val_loss', verbose=1, save_best_only=True)

history=model.fit_generator(train_generator,
                            int((960*3)/batch_size),100,
                            validation_data = val_generator, 
                            validation_steps =240//batch_size, 
                            verbose=1,callbacks=[checkpoint])
#lis=[]
#
##gt = model.evaluate_generator(val_generator,steps = 240//batch_size)
imagex_val = sorted(os.listdir('/home/gpu3/shubham/refuge/refuge_val'))
import cv2
from skimage import io
from tqdm import tqdm
i=imagex_val[0]
lis=[]
for i in tqdm(imagex_val):
    img = io.imread('/home/gpu3/shubham/refuge/refuge_val/'+i)
    dims = img.shape
    img = preprocess_input(img)
    img = img / 255.
    img = cv2.resize(img,(600,600))
    img = np.reshape(img,(1,)+img.shape)
    pr = model.predict(img)
    pr =np.squeeze(pr)
    x = np.round(pr[0]*dims[0],2)
    y = np.round(pr[1]*dims[1],2)
    prl = [i,x ,y]
    lis.append(prl)

r = pd.DataFrame(lis,columns=['ImageName','Fovea_X','Fovea_Y'])
r.to_csv('localization_e7.csv',index=False)
#x= pr[0][0]
#y = pr[0][1]
#dims = img.shape
#x = x * dims[0]
#y = y * dims[1]
#x
#rows = val_df['ImgName'][:] 
#x1 =lis[0][0][1]*dims[0]
#y1 = 0.494623 *dims[1]
#
#lst_gt=sorted(lst_gt)
#img_name = val_df['ImgName'].to_list()
#x = val_df["Fovea_X"].to_list()
#y = val_df["Fovea_Y"].to_list()
#lst_gt=[]
#for i in range(len(img_name)):
#    img = io.imread('/home/gpu3/shubham/refuge/Tsegmentation//images/val/'+img_name[i])
##    img = img / 255.
#    lt = [img_name[i],x[i],y[i]]
#    lst_gt.append(lt)
dist = []
#not_correct =[]
#
dist = np.average(dist)
##i=0    
from scipy.spatial import distance
#
for i in tqdm(range(400)):
#    w, h ,_ = img.shape    
    xp = xf[i]
    yp = yf[i]
    xg = x_b[i]
    yg = y_b[i]    

    dst = distance.euclidean((xp,yp), (xg,yg))
    dist.append(dst)
#    
