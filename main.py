#!/usr/bin/env python
# coding: utf-8

import keras
import numpy as np
import os
import pickle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import librosa
from dataset import BreathDataGenerator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

BATCH_SIZE = 32
N_CLASSES = 4
LR = 0.0005
N_EPOCHS = 50
INPUT_SIZE = (40, 216, 1)


train_generator = BreathDataGenerator(
        'dataset/training_dataset',
        batch_size=BATCH_SIZE,
        dim=INPUT_SIZE,
        shuffle=True)
N_TRAIN_SAMPLES = len(train_generator.wavs)
# train_generator.__getitem__(0)
print("Train samples: {}".format(N_TRAIN_SAMPLES))
# exit(1)
validation_generator = BreathDataGenerator(
        'dataset/testing_dataset',
        batch_size=BATCH_SIZE,
        dim=INPUT_SIZE,
        shuffle=False)
N_TEST_SAMPLES = len(validation_generator.wavs)
print("Test samples: {}".format(N_TEST_SAMPLES))

# import keras.applications
from keras.applications.mobilenet import MobileNet
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model 
from resnet import ResnetBuilder
from model import SimpleCNN

# model = MobileNet(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), include_top=True, classes=N_CLASSES, weights=None)
# model = MobileNetV2(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), include_top=True, classes=N_CLASSES, weights=None)
# model = ResnetBuilder.build_resnet_18(input_shape=INPUT_SIZE, num_outputs=N_CLASSES)
model = SimpleCNN.build(input_shape=INPUT_SIZE, classes=N_CLASSES)
model.summary()

# Training
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta

model_file = "models/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tbCallBack = keras.callbacks.TensorBoard(log_dir='logs/tensorboard', write_graph=True, write_images=True)

callbacks_list = [checkpoint, tbCallBack]
#model.load_weights("models/weights-improvement-00-1.08.hdf5")
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=N_TRAIN_SAMPLES // BATCH_SIZE,
        initial_epoch=0,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=N_TEST_SAMPLES // BATCH_SIZE,
        callbacks=callbacks_list,
        max_queue_size=20,
        workers=10,
        use_multiprocessing=True,
        )





