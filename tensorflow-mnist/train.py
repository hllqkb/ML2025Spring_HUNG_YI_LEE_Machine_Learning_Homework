from __future__ import absolute_import, division, print_function,unicode_literals
import numpy as np
import tensorflow as tf
from model import MyModel
import matplotlib.pyplot as plt
mnist=tf.keras.datasets.mnist
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制使用 CPU
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
# imgs,labs=x_test[:10],y_test[:10]
# plot_images=np.hstack(imgs)
# plt.imshow(plot_images,cmap='gray')
# plt.show()
# add a channel dimension to the images
x_train=x_train[...,tf.newaxis]
x_test=x_test[...,tf.newaxis]
# create data generator

train_ds=tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)

test_ds=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)
model=MyModel()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# define the callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'), # if val_loss doesn't improve for 3 epochs, stop training
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', 
                                     save_best_only=True,
                                     monitor='val_accuracy') # save the best model based on val_accuracy
]

# train the model
model.fit(train_ds, epochs=20, validation_data=test_ds, callbacks=callbacks)

# evaluate the model on the test set
model.evaluate(test_ds)
