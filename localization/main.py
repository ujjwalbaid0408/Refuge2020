import tensorflow as tf
physical_devices =tf.config.experimental.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
from iddlite_data import *
#from models import deeplab
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import os
#from unet_model import unet
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm


#data_dirt = '/home/ujjwal/Iddlite/dataset/train'
#data_dirv = '/home/ujjwal/Iddlite/dataset/validation'
data_dir = '/home/gpu3/shubham/refuge/Tlocalization/'

smooth=1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f)+ K.sum(y_pred_f) + smooth)

data_gen_args = dict(horizontal_flip=True,
                     vertical_flip=True,
                    fill_mode='nearest')
    
#data_gen_args = dict()
data_gen_args_val = dict(horizontal_flip=True,
                     vertical_flip=True,
                    fill_mode='nearest')
  
batch_size = 16

myGene = trainGenerator(batch_size, '/home/gpu3/shubham/refuge/Tlocalization/train',
                        'image','mask',data_gen_args,save_to_dir = None)

myGeneval = trainGeneratorval(batch_size,'/home/gpu3/shubham/refuge/Tlocalization/val',
                              'image','mask',data_gen_args_val,save_to_dir = None)

#model = deeplab()

model = sm.Unet('efficientnetb0', input_shape=(512,512,3), classes=1, activation='sigmoid',
                encoder_weights='imagenet')
#class_weight = {0: 1.,
#                1: 10.}
#model =unet()
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), 
              loss=sm.losses.binary_focal_jaccard_loss,
              metrics=[sm.metrics.iou_score,dice_coef, 'accuracy'])
model.load_weights('refuge_localization_segmentation_unet_512.h5')
#model.load_weights('idd_lite/iddlite_unet_val_loss_r34_0.4090val_iou_score0.7006.hdf5')
#class_weights ={0:1.,255:160.}
model_checkpoint = ModelCheckpoint('refuge_localization_segmentation_unet_512_final.h5', 
                                   monitor='val_loss',verbose=1, save_best_only=True) 
#model.fit(myGene,steps_per_epoch=int((960*5)//batch_size),epochs=300,
#          callbacks=[model_checkpoint], verbose=1, shuffle=True)
#model.fit_generator(myGene, steps_per_epoch=20970, epochs=100, callbacks=[model_checkpoint])

model.fit(myGene, 
          steps_per_epoch=int((6000*3)//batch_size),
          epochs=300,
          callbacks=[model_checkpoint],
          validation_data=myGeneval, 
          validation_steps=int((1200*3)//batch_size),
          shuffle=True)
          #,class_weight=class_weight)
#model.fit(myGene,validation_data = myGeneval, epochs=100, callbacks=[model_checkpoint], steps_per_epoch=int(8286//batch_size))
#model.fit(myGene,validation_data = myGeneval, epochs=100, callbacks=[model_checkpoint],steps_per_epoch=int(8286//batch_size),validation_steps=int(204//batch_size))
