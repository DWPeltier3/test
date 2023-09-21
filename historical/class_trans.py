import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import swarm.code.utils.params as params
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from timeit import default_timer as timer
import math

start = timer()
CPUs = os.cpu_count()
GPUs = len(tf.config.list_physical_devices('GPU'))
print(f"\nTensorflow Version: {tf.__version__}")
print(f"GPUs available: {GPUs}")
print(f"CPUs available: {CPUs}\n") 

hparams = params.get_hparams()
if not os.path.exists(hparams.model_dir):
    os.mkdir(hparams.model_dir)
params.save_hparams(hparams)


## IMPORT DATASET
data=np.load(hparams.data_path)
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

print('\nxtrain shape:',x_train.shape)
print('xtest shape:',x_test.shape)
print('ytrain shape:',y_train.shape)
print('ytest shape:',y_test.shape)

print('\nx train sample',x_train[0,0])
print('y train sample',y_train[0])


## REDUCE OBSERVATION WINDOW
print('\nREDUCED OBSERVATION WINDOW')
min_time=x_train.shape[1]
window=hparams.window
print('Min Run Time:',min_time)
print('Window:',window)
# -1 or invalid window (too large>min_time): uses entire observation window (min_time for all runs)
if window!=-1 and window<min_time:
    x_train=x_train[:,:window,:]
    x_test=x_test[:,:window,:]
print('xtrain shape:',x_train.shape)
print('xtest shape:',x_test.shape,'\n')


## DEFINE MODEL
n_classes = len(np.unique(y_train)) #number of classes = number of unique  labels

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)  #channel_last? https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling1D
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


## BUILD MODEL
input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy", # 2+ classes, labels=integers, predictions=#classes floats (NOT one hot encoded, else CCE)
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"], #number of times prediction equals label, divided by count
)

model.summary()


## CALLBACKS
modelcheckpoint=tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(hparams.model_dir, "checkpoint{epoch:02d}-{val_loss:.2f}.h5"),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    save_freq=10*hparams.batch_size #once every 10 epochs; else 'epoch'
)

earlystopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=15,
    verbose=1,
    mode='min',
    baseline=None,
    restore_best_weights=True
)


## TRAIN MODEL
model_history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=hparams.num_epochs,
    batch_size=hparams.batch_size,
    callbacks=[earlystopping] #, modelcheckpoint] #may need to update TF version to save model??
)


## TRAINING CURVE
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
mvl=min(val_loss)
print(f"\nMinimum Val Loss: {mvl}\n") #added \n for spacing

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
# plt.show()
plt.savefig(hparams.model_dir + "/loss_vs_epoch.png")


## EVALUATE MODEL
pred=model.predict(x_test, verbose=0) #predicted label probabilities for test data
y_pred=np.argmax(pred,axis=1).reshape((-1,1))  #predicted label for test data
np.set_printoptions(precision=2) #show only 2 decimal places (does not change actual numbers)

print('\nprediction & label:\n',np.hstack((pred,y_test))) #probability comparison
print('\npredicted label & actual label:\n',np.hstack((y_pred,y_test))) #label comparison

eval=model.evaluate(x_test, y_test, verbose=0) #loss and accuracy
print('\nevaluation:\n',eval)

print('\nmodel.metrics_names:',model.metrics_names) #print evaluation metrics (loss and accuracy)


## CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.savefig(hparams.model_dir + "/conf_matrix.png")


## PRINT ELAPSE TIME
end = timer()
elapse=end-start
hours=0
minutes=0
seconds=0
remainder=0
if elapse>3600:
    hours=math.trunc(elapse/3600)
    remainder=elapse%3600
if elapse>60:
    if remainder>60:
        minutes=math.trunc(remainder/60)
        seconds=remainder%60
        seconds=math.trunc(seconds)
    else:
        minutes=math.trunc(elapse/60)
        seconds=elapse%60
        seconds=math.trunc(seconds)
if elapse<60:
    seconds=math.trunc(elapse)

print(f"Elapse Time: {hours}h, {minutes}m, {seconds}s")