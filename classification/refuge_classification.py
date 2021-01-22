#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:52:15 2020

@author: shubham
"""

import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.keras.callbacks import  ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from tensorflow.keras.applications.imagenet_utils import preprocess_input
physical_devices =tf.config.experimental.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')]

batch_size = 8

#def preprocess(img):
#    img = preprocess(img)
#    return img

train_datagen = ImageDataGenerator(rotation_range=5,
                    width_shift_range=3,
                    height_shift_range=3,
                    horizontal_flip=True,
                    vertical_flip=True,
                    brightness_range=[0.2,1.0],
                    fill_mode='nearest',
                    rescale=1./255.,
                    preprocessing_function=preprocess_input)
#preprocessing_function=preprocess
#train_datagen = ImageDataGenerator(rescale=1./255.,
#                                  preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(rescale=1./255.,
                                  preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    '/home/gpu3/shubham/refuge/Tclassification/complete',
    target_size=(331, 331),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    '/home/gpu3/shubham/refuge/Tclassification/val',
    target_size=(331, 331),
    batch_size=batch_size,
    class_mode='binary')


import tensorflow
import efficientnet.tfkeras as efn
base_model = efn.EfficientNetB7(input_shape=(600,600,3), weights='imagenet', include_top=False,classes=1)
x = tensorflow.keras.layers.GlobalAveragePooling2D()(base_model.output)
#x = tensorflow.keras.layers.Dense(128, activation='relu')(x)
#x = tensorflow.keras.layers.Dense(16, activation='relu')(x)
output = tensorflow.keras.layers.Dense(1, activation='sigmoid', name = 'dense_final')(x)
model = tensorflow.keras.models.Model(inputs=[base_model.input], outputs=[output])                 
model.summary()

#seresnext101	
#senet154
#xception 299x299
Net, _ = Classifiers.get('inceptionresnetv2')
base_model = Net(input_shape=(224,224, 3),classes=1,weights= "imagenet", include_top=False)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])                 
base_model.summary()

model.load_weights('refuge_classify_irv2_final_1.h5')
model.load_weights('refuge_classify_efficientnetb7_final_1.h5')

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss = 'binary_crossentropy', metrics = METRICS)


checkpoint = ModelCheckpoint("refuge_classify_nasnetlarge.h5", 
                             monitor='val_loss', verbose=1, save_best_only=True)

history=model.fit_generator(train_generator,
                            int((1200*6)/batch_size),500,
                            validation_data = validation_generator, 
                            validation_steps =240//batch_size, 
                            verbose=1,callbacks=[checkpoint])


#rr = model.evaluate_generator(validation_generator,120)

#
import pandas as pd
import scipy
import cv2
from skimage import io
from tqdm import tqdm
import os
import numpy as np

lis = sorted(os.listdir('/home/gpu3/shubham/refuge/refuge_test'))
i=lis[4]
e=[]
for i in tqdm(lis):
    img = cv2.imread('/home/gpu3/shubham/refuge/refuge_test/'+i)
    im = cv2.resize(img,(224,224))
    im = preprocess_input(im)
    im = im/255.
    im = np.reshape(im,(1,)+im.shape)
    d = model.predict(im)
    d = d[0][0]    
    e.append([i,d])    

x = pd.DataFrame(e,columns=['FileName','Glaucoma Risk'])
x.to_csv('./final_csvs/classification_results_test_efficienetb7.csv',index=False)
#

suba= pd.read_csv('./final_csvs/classification_results_nasnetlarge.csv').set_index('FileName')
subb= pd.read_csv('./final_csvs/classification_results_irv2_1.csv').set_index('FileName')
subc= pd.read_csv('./final_csvs/classification_results_eff5_1.csv').set_index('FileName')
subd= pd.read_csv('./final_csvs/classification_results_eff6_1.csv').set_index('FileName')
sube= pd.read_csv('./final_csvs/classification_results_eff7_1.csv').set_index('FileName')
#ya = y = pd.concat([suba,subb,subc,subd,sube]).groupby(level=0).mean()
#
sub1= pd.read_csv('./final_csvs/classification_results_nasnetlarge.csv').set_index('FileName')
sub2= pd.read_csv('./final_csvs/classification_results_irv2_2.csv').set_index('FileName')
sub3= pd.read_csv('./final_csvs/classification_results_eff5_2.csv').set_index('FileName')
sub4= pd.read_csv('./final_csvs/classification_results_eff6_2.csv').set_index('FileName')
sub5= pd.read_csv('./final_csvs/classification_results_eff7_2.csv').set_index('FileName')
sub6= pd.read_csv('./final_csvs/classification_results_xception.csv').set_index('FileName')
#sub7= pd.read_csv('./final_csvs/classification_results_senet.csv').set_index('FileName') 
y = pd.concat([sub1,sub2,sub3,sub4,sub5,sub6]).groupby(level=0).mean()
y.to_csv('classification_results_all_ensemble_all_no_senet.csv')
#
#y1 = pd.concat([y,ya]).groupby(level=0).mean()
#
#
#w
#from scipy import ndimage
#
#
##rotation angle in degree
#
#e=[]
#for i in tqdm(lis):
#    img = cv2.imread('/home/gpu3/shubham/refuge/refuge_val/'+i)
#    im = cv2.resize(img,(224,224))
#    im = preprocess_input(im)
#    im = im/255.
##    im1 = np.reshape(im,(1,)+im.shape)
##    d1 = model.predict(im1)
##    d1 = d1[0][0]    
##    
##    im2 = cv2.flip(im, 1)
##    im2 = np.reshape(im2,(1,)+im.shape)
##    d2 = model.predict(im2)
##    d2 = d2[0][0]
#    
#    im3 = ndimage.rotate(im, 5,reshape=False)
#    im3 = np.reshape(im3,(1,)+im.shape)
#    d3 = model.predict(im3)
#    d3 = d3[0][0]
#    
##    d = np.a([d1,d2,d3])
#    e.append([i,d3])