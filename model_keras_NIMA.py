import os

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

image_size = 224

base_model = MobileNet((image_size, image_size, 3), alpha=1, include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.summary()
print(model)