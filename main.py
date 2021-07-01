
import os
import json

import tensorflow as tf
import mnist
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configs
per_worker_batch_size = 64
epochs = 2

tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
current_worker = tf_config['task']['index']
epochs_per_worker = epochs // num_workers

# Distribute details
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Datasets
train_df, validate_df = mnist.mnist_df()
train_generator, validation_generator = mnist.mnist_generators(train_df, validate_df, per_worker_batch_size)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

# Model
with strategy.scope():
  multi_worker_model = mnist.build_and_compile_cnn_model()

# Training
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

multi_worker_model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//per_worker_batch_size,
    steps_per_epoch=total_train//per_worker_batch_size,
    initial_epoch=(epochs_per_worker * current_worker),
    callbacks=[earlystop, learning_rate_reduction]
)
