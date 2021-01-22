#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:24:05 2020

@author: gpu3
"""
import numpy as np
from skimage import io 
from tqdm import tqdm
import cv2
from glob import glob
import pandas as pd 
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm

model = sm.Unet('efficientnetb0', input_shape=(512,512,3), classes=1, activation='sigmoid',
                encoder_weights='imagenet')
#model.load_weights('refuge_localization_segmentation_unet_b0_256_final.h5')
model.load_weights('refuge_localization_segmentation_unet_512_final.h5')

df = pd.read_csv('/home/gpu3/shubham/refuge/submission/TeamTiger_sub2/fovea_location_results.csv')
train_images_path = '/home/gpu3/shubham/refuge/refuge_test/'
val_images_path = '/home/gpu3/shubham/refuge/Tsegmentation/images/val/'
preprocess_input = sm.get_preprocessing('efficientnetb6')
xf=[]
yf=[]
img_list = df['ImageName'].to_list()
x_cor = df['Fovea_X'].to_list()
y_cor = df['Fovea_Y'].to_list()
count=0
train_count=0
val_count=0
i=11
not_r=[]
for i in tqdm(range(len(img_list))):
    img_arr = io.imread(train_images_path+img_list[i])
    h,w,_ = img_arr.shape
    mask = np.zeros([h,w])
    x_m = int(x_cor[i])
    y_m = int(y_cor[i])
#    patch_img = img_arr[y_m-128:y_m+128,x_m-128:x_m+128,:] 
#    patch_img1 = img_arr[y_m-128:y_m+128,x_m-128:x_m+128,:]
    patch_img1 = img_arr[y_m-256:y_m+256,x_m-256:x_m+256,:]
#    patch_img2 = cv2.flip(patch_img1, 0)
#    patch_img3 = cv2.flip(patch_img1, 1)
#    patch_img2 = img_arr[y_m-64:y_m+192,x_m-64:x_m+192,:]
#    patch_img3 = img_arr[y_m-32:y_m+224,x_m-32:x_m+224,:]
#    patch_img4 = img_arr[y_m-96:y_m+160,x_m-96:x_m+160,:]
#    patch_img5 = img_arr[y_m-16:y_m+240,x_m-16:x_m+240,:]
#    patch_img = preprocess_input(patch_img1)
#    pimg = np.reshape(patch_img,(1,)+patch_img.shape)
#    d = model.predict(pimg)
#    d = np.squeeze(d,axis=0)
#    d = np.squeeze(d,axis=2)
#    d[d>=0.5]=255
#    d[d<0.5]=0
#    d = np.uint8(d)
#    
    if np.any(d1=255):
        patch_img1 = preprocess_input(patch_img1)
        pimg1 = np.reshape(patch_img1,(1,)+patch_img1.shape)
        d1 = model.predict(pimg1)
        d1 = np.squeeze(d1,axis=0)
        d1 = np.squeeze(d1,axis=2)
        d1[d1>=0.5]=255
        d1[d1<0.5]=0
        d1 = np.uint8(d1)
        contours, hierarchy = cv2.findContours(d1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        x_final = x +x_m-256+(w/2)
        y_final = y +y_m -256 +(h/2)
        xf.append(x_final)
        yf.append(y_final)
    elif np.any(d2=255):     
        patch_img2 = preprocess_input(patch_img2)
        pimg2 = np.reshape(patch_img2,(1,)+patch_img2.shape)
        d2 = model.predict(pimg2)
        d2 = np.squeeze(d2,axis=0)
        d2 = np.squeeze(d2,axis=2)
        d2[d2>=0.5]=255
        d2[d2<0.5]=0
        d2 = np.uint8(d2)
        contours, hierarchy = cv2.findContours(d2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        x_final = x +x_m-256+(w/2)
        y_final = y +y_m -256 +(h/2)
        xf.append(x_final)
        yf.append(y_final)
    elif np.any(d3=255):    
        patch_img3 = preprocess_input(patch_img3)
        pimg3 = np.reshape(patch_img3,(1,)+patch_img3.shape)
        d3 = model.predict(pimg3)
        d3 = np.squeeze(d3,axis=0)
        d3 = np.squeeze(d3,axis=2)
        d3[d3>=0.5]=255
        d3[d3<0.5]=0
        d3 = np.uint8(d3)
        contours, hierarchy = cv2.findContours(d3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        x_final = x +x_m-256+(w/2)
        y_final = y +y_m -256 +(h/2)
        xf.append(x_final)
        yf.append(y_final)
    else:
        count = count + 1
        not_r.append(img_list[i])
        xf.append(x_m)
        yf.append(y_m)
        
    try:
        contours, hierarchy = cv2.findContours(d1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        x_final = x +x_m-256+(w/2)
        y_final = y +y_m -256 +(h/2)
        xf.append(x_final)
        yf.append(y_final)
    except:
        count = count + 1
        not_r.append(img_list[i])
        xf.append(x_m)
        yf.append(y_m)

final =[]
for i in tqdm(range(len(img_list))):
    final.append([img_list[i],xf[i],yf[i]])
io.imshow()
x= pd.DataFrame(final,columns=['ImageName','Fovea_X','Fovea_Y'])
x.to_csv('patch_localation_eff0_augmentation_5.csv',index=False)

r = cv2.rectangle(d,(x,y),(x+w,y+h),color = (255, 0, 0),thickness = 2)

import matplotlib.pyplot as plt
plt.imshow(patch_img2)
plt.imshow(d1,alpha=0.2)
