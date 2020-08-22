from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import callbacks
from keras.metrics import BinaryCrossentropy, Accuracy,BinaryAccuracy, AUC
import matplotlib.pyplot as plt

columns = ['Feature%02d'%i for i in range(0, 30)]

X = pd.read_csv('C:/Users/V/Desktop/MADE/1/train.csv', header=None)#skipinitialspace=True
y = pd.read_csv('C:/Users/V/Desktop/MADE/1/train-target.csv', header=None, dtype='int')
target = pd.read_csv('C:/Users/V/Desktop/MADE/1/test.csv', header=None)

X.columns=columns
target.columns = columns

X['target'] = y
for index, row in X.iterrows():
    if row['target'] == 1:
        X.at[index, 'Feature16'] = row['Feature16'] - 1.2442
        X.at[index,'Feature09'] = row['Feature09'] - 2.525
X = X.drop(['target'], axis=1)

# X.columns=columns
# target.columns = columns
drop_cols = [
            'Feature15','Feature22', 'Feature17',
            'Feature11',
            'Feature01', 'Feature03', 'Feature04', 'Feature18', 'Feature20', 'Feature24','Feature25','Feature27',
            ]
X = X.drop(drop_cols, axis=1)
target = target.drop(drop_cols, axis=1)

X_sc = MinMaxScaler(feature_range=(0,1)).fit(X)
X_scaled = X_sc.transform(X)
X_target = X_sc.transform(target)

y_sc = MinMaxScaler(feature_range=(0,1)).fit(y)
y_scaled = y_sc.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=.25)
y_train = np.reshape(y_train, -1)
y_test = np.reshape(y_test,-1)

learning_rate = 0.01
filepath_loss = 'C:/Users/V/Desktop/MADE/1/bin_weight.hdf5'
batch_size = 750
epochs = 500
input_layer = 18
layer_1 = 32
layer_2 = 16
layer_3 = 32
layer_4 = 128
layer_5 = 256
output_layer = 1
activation = 'softmax' #sigmoid
activation_output = 'sigmoid'

# define model
model = Sequential()
#1st layer
model.add(Dense(layer_1, input_dim=input_layer)) #kernel_initializer='normal'/'uniform'
model.add(BatchNormalization())
model.add(Activation(activation))
#2nd layer
# model.add(Dense(layer_2))
# model.add(BatchNormalization())
# model.add(Activation(activation))
#3rd layer
# model.add(Dense(layer_3))
# model.add(BatchNormalization())
# model.add(Activation(activation))
#
# model.add(Dense(layer_4))
# model.add(BatchNormalization())
# model.add(Activation(activation))
#
# model.add(Dense(layer_5))
# model.add(BatchNormalization())
# model.add(Activation(activation_output))
#output layer
model.add(Dense(output_layer))
model.add((Activation(activation_output)))

cb_loss = callbacks.ModelCheckpoint(filepath=filepath_loss,
                                    monitor='val_loss',
                                    verbose=0,
                                    mode='min',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    period=1
                                    )

optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', AUC()]) #binary_crossentropy, accuracy

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    use_multiprocessing = True,
                    epochs=epochs,
                    callbacks=[cb_loss],
                    verbose=2)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score)

model.load_weights('C:/Users/V/Desktop/MADE/1/bin_weight.hdf5')
predict = model.predict_proba(X_target)
predict = pd.DataFrame(predict)
predict.to_csv('C:/Users/V/Desktop/MADE/1/predict.csv', index=False, header=False)
print(predict.shape)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.figure(figsize=(epochs/100,3))
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()