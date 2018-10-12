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
from sklearn.metrics import classification_report, confusion_matrix

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

BATCH_SIZE = 32
LIST_LABELS = ['normal', 'deep', 'rush']
N_CLASSES = len(LIST_LABELS)
LR = 0.0005
N_EPOCHS = 5
INPUT_SIZE = (40, 216, 1)


train_generator = BreathDataGenerator(
        'dataset/training_dataset',
        list_labels=LIST_LABELS,
        batch_size=BATCH_SIZE,
        dim=INPUT_SIZE,
        shuffle=True)
N_TRAIN_SAMPLES = len(train_generator.wavs)
# train_generator.__getitem__(0)
print("Train samples: {}".format(N_TRAIN_SAMPLES))
# exit(1)
validation_generator = BreathDataGenerator(
        'dataset/testing_dataset',
        list_labels=LIST_LABELS,
        batch_size=BATCH_SIZE,
        dim=INPUT_SIZE,
        shuffle=False)
N_VALID_SAMPLES = len(validation_generator.wavs)
print("Validation samples: {}".format(N_VALID_SAMPLES))

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
# model.load_weights("models/weights-improvement-02-0.39.hdf5")
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=N_TRAIN_SAMPLES // BATCH_SIZE,
        initial_epoch=0,
        epochs=N_EPOCHS,
        validation_data=validation_generator,
        validation_steps=N_VALID_SAMPLES // BATCH_SIZE,
        callbacks=callbacks_list,
        max_queue_size=6,
        workers=3,
        use_multiprocessing=True,
        )

test_generator = BreathDataGenerator(
        'dataset/testing_dataset',
        list_labels=LIST_LABELS,
        batch_size=1,
        dim=INPUT_SIZE,
        shuffle=False)
N_TEST_SAMPLES = len(test_generator.wavs)
print("Test samples: {}".format(N_TEST_SAMPLES))
#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_generator, N_TEST_SAMPLES)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.labels, y_pred))
print('Classification Report')
print(classification_report(test_generator.labels, y_pred, target_names=LIST_LABELS))




