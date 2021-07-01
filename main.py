
import os
import json

import tensorflow as tf
import mnist
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Configs
per_worker_batch_size = 15
epochs = 4
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
current_worker = tf_config['task']['index']
# epochs_per_worker = epochs // num_workers

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
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

history = multi_worker_model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//per_worker_batch_size,
    steps_per_epoch=total_train//per_worker_batch_size,
    # initial_epoch=(epochs_per_worker * current_worker),
    callbacks=[earlystop, learning_rate_reduction]
)

multi_worker_model.save_weights('result/model.' + str(current_worker))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

fig = plt.figure()
plt.plot(range(10))
fig.savefig('result/work' + str(current_worker) + '.png', dpi=fig.dpi)
