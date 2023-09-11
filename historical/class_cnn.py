import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import swarm.code.utils.params as params
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from timeit import default_timer as timer
from swarm.code.utils.elapse import elapse_time

start = timer()
CPUs = os.cpu_count()
GPUs = len(tf.config.list_physical_devices('GPU'))
print('\n*** RESOURCES ***')
print(f"Tensorflow Version: {tf.__version__}")
print(f"GPUs available: {GPUs}")
print(f"CPUs available: {CPUs}\n") 

print('\n*** PARAMETERS ***')
hparams = params.get_hparams()
if not os.path.exists(hparams.model_dir):
    os.mkdir(hparams.model_dir)
params.save_hparams(hparams)


## IMPORT DATASET
# training and test data
data=np.load(hparams.data_path)
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# CNN accepts any shape input
time_steps=x_train.shape[1]
num_features=x_train.shape[2]
print('\n*** DATA ***')
print('xtrain shape:',x_train.shape)
print('xtest shape:',x_test.shape)
print('ytrain shape:',y_train.shape)
print('ytest shape:',y_test.shape)
print('num features:',num_features)
print('xtrain sample (1st instance, 1st time step, all features)\n',x_train[0,:num_features])
print('ytrain sample (first instance)',y_train[0])


## DETERMINE DATA CLASSES AND RUNS TO INDEX TEST RESULTS AT END
print('\n*** FIND INDEX FOR TEST RESULTS ***')
n_data_classes=len(np.unique(y_test))
n_runs=y_test.shape[0]
lpc=n_runs//n_data_classes
c1idx=0
c2idx=lpc
c3idx=lpc*2
c4idx=lpc*3
print('num data classes:',n_data_classes)
print('num test runs:',n_runs)
print(f'start indices: c1 {c1idx}, c2 {c2idx}, c3 {c3idx}, c4 {c4idx}')


## REDUCE OBSERVATION WINDOW
print('\n*** REDUCED OBSERVATION WINDOW ***')
window=hparams.window
# if window = -1 or invalid window (too large>min_time): use entire observation window (time_steps) for all runs
if window==-1 or window>time_steps:
    window=time_steps
x_train=x_train[:,:window]
x_test=x_test[:,:window]
print('time steps available:',time_steps)
print('window used:',window)
print('xtrain shape:',x_train.shape)
print('xtest shape:',x_test.shape,'\n')


## DEFINE MODEL
def cnn_model(input_shape):
    pool_size=2

    inputs = keras.Input(shape=(input_shape))

    conv1 = keras.layers.Conv1D(activation="relu", filters=64, kernel_size=3, padding="same")(inputs)
    conv1 = keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_size, padding='same')(conv1)

    conv2 = keras.layers.Conv1D(activation="relu", filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_size, padding='same')(conv2)

    conv3 = keras.layers.Conv1D(activation="relu", filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_size, padding='same')(conv3)

    flat = keras.layers.Flatten()(conv3)
    drop = keras.layers.Dropout(hparams.dropout)(flat)

    dense = keras.layers.Dense(100, activation='relu')(drop)
    drop = keras.layers.Dropout(hparams.dropout)(dense)

    outputs = keras.layers.Dense(n_classes, activation="softmax")(drop)

    return keras.Model(inputs=inputs, outputs=outputs, name='CNN')

def cnn_model_v1(input_shape):
    inputs = keras.Input(shape=(input_shape))

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(inputs)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    outputs = keras.layers.Dense(n_classes, activation="softmax")(gap)

    return keras.Model(inputs=inputs, outputs=outputs, name='CNN')


## BUILD MODEL
n_classes = len(np.unique(y_train)) #number of classes = number of unique  labels
cnn_input_shape = x_train.shape[1:]

model = cnn_model(cnn_input_shape)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"]
)

model.summary()


## CALLBACKS
ckpt_path=hparams.model_dir + "checkpoint.h5"

modelcheckpoint=tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

earlystopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=50, #15 or 50
    verbose=1,
    mode='min',
    restore_best_weights=True
)


## TRAIN MODEL
model_history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=hparams.num_epochs,
    batch_size=hparams.batch_size,
    verbose=0,
    callbacks=[earlystopping, modelcheckpoint]
)
# print("history.history\n", model_history.history)
# print("history.history.keys()\n", model_history.history.keys())


## SAVE ENTIRE MODEL
model.save(hparams.model_dir + 'model.keras')


## TRAINING CURVE
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
mvl=min(val_loss)
print(f"Minimum Val Loss: {mvl}") 

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig(hparams.model_dir + "loss_vs_epoch.png")


## EVALUATE MODEL
print('\n*** TEST DATA RESULTS (5 per class) ***')
pred=model.predict(x_test, verbose=0) #predicted label probabilities for test data
y_pred=np.argmax(pred,axis=1).reshape((-1,1))  #predicted label for test data

# np.set_printoptions(precision=2) #show only 2 decimal places (does not change actual numbers)
# print('\nprediction & label:\n',np.hstack((pred,y_test))) #probability comparison

c1res=np.hstack((y_pred[c1idx:c1idx+5],y_test[c1idx:c1idx+5]))
c2res=np.hstack((y_pred[c2idx:c2idx+5],y_test[c2idx:c2idx+5]))
c3res=np.hstack((y_pred[c3idx:c3idx+5],y_test[c3idx:c3idx+5]))
c4res=np.hstack((y_pred[c4idx:c4idx+5],y_test[c4idx:c4idx+5]))
# samp_res=np.vstack((c1res,c2res,c3res,c4res))
print(f'LABELS\nPredicted vs. True\n\nGreedy\n {c1res}\n\nGreedyPRO\n {c2res}\n\nAuction\n {c3res}\n\nAuctionPRO\n {c4res}\n') #label comparison

eval=model.evaluate(x_test, y_test, verbose=0) #loss and accuracy
print('model.metrics_names:\n',model.metrics_names) #print evaluation metrics names (loss and accuracy)
print(eval) #print evaluation metrics numbers


## CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.savefig(hparams.model_dir + "conf_matrix.png")


## PRINT ELAPSE TIME
elapse_time(start)