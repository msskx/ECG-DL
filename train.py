import numpy as np

from Function.evaluate import model_evaluate
from Data.Dataloader import DataLoader
from model.CNN import CNN
from tensorflow import keras

data = DataLoader()
X_train, y_train, X_test, y_test = data.getdata01()

verbose, epochs, batch_size = 1, 15, 64
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], 2

num_classes = len(np.unique(y_train))

input_shape = (X_train.shape[1], X_train.shape[2])
model = CNN.cnn(input_shape)

print(model.summary())

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
model.fit(X_train, y_train, batch_size=batch_size, epochs=1000, validation_split=0.1, callbacks=callbacks)

model_evaluate(X_test, y_test, model)
