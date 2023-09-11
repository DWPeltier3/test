import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import swarm.code.utils.params as params
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,multilabel_confusion_matrix,hamming_loss,classification_report
from timeit import default_timer as timer
from swarm.code.utils.elapse import elapse_time

start = timer()
CPUs = os.cpu_count()
GPUs = len(tf.config.list_physical_devices('GPU'))
print('\n*** RESOURCES ***')
print(f"Tensorflow Version: {tf.__version__}")
print(f"GPUs available: {GPUs}")
print(f"CPUs available: {CPUs}") 

print('\n*** PARAMETERS ***')
hparams = params.get_hparams()
if not os.path.exists(hparams.model_dir):
    os.mkdir(hparams.model_dir)
params.save_hparams(hparams)


## IMPORT TEST DATASET
data=np.load(hparams.data_path)
x_test = data['x_test']
y_test = data['y_test']

# flatten timeseries data into 1 column per run for fully connected model
time_steps=x_test.shape[1]
num_features=x_test.shape[2]
num_inputs=time_steps*num_features
x_test=np.reshape(x_test,(len(x_test),num_inputs))
print('\n*** DATA ***')
print('xtest shape:',x_test.shape)
print('ytest shape:',y_test.shape)
print('num features:',num_features)
print('xtest sample (1st instance, 1st time step, all features)\n',x_test[0,:num_features])
print('ytest sample (first instance)',y_test[0])

## DETERMINE DATA CLASSES AND RUNS TO INDEX TEST RESULTS AT END
print('\n*** FIND INDEX FOR TEST RESULTS ***')
n_data_classes=len(np.unique(y_test))
n_runs=y_test.shape[0]
lpc=n_runs//n_data_classes #integer division (returns integer for indexing)
c1idx=0
c2idx=lpc
c3idx=lpc*2
c4idx=lpc*3
print('num data classes:',n_data_classes)
print('num test runs:',n_runs)
print(f'start indices: c1 {c1idx}, c2 {c2idx}, c3 {c3idx}, c4 {c4idx}')


## REPLACE LABELS WITH MULTILABEL
# add column for second label
y_test=np.insert(y_test,1,0,axis=1)
# update labels: columns are classes
# 1st col=Comms(Auction), 2nd col=ProNav
y_test[y_test[:,0]==0]=[0,0] # Greedy
y_test[y_test[:,0]==1]=[0,1] # Greedy+
y_test[y_test[:,0]==2]=[1,0] # Auction
y_test[y_test[:,0]==3]=[1,1] # Auction+
print('\n*** MULTILABEL ***')
print('ytest shape:',y_test.shape)
print('ytest (first instance)',y_test[0,:],'\n')


## LOAD MODEL
model_filepath='/home/donald.peltier/swarm/model/swarm_cfc_2023-08-25_13-20-33_TRAIN_ML_10v10_r4800_d2_w20/model.keras'
model = tf.keras.saving.load_model(model_filepath)
model.summary()


## REDUCE OBSERVATION WINDOW
print('\n*** REDUCED OBSERVATION WINDOW ***')
input_shape=model.input_shape[-1]
window=hparams.window
print('time steps available:',time_steps)
print('window used:',window)
# truncate data to window
x_test=x_test[:,:input_shape]
print('num inputs (window*features):',window*num_features)
print('input shape:',input_shape)
print('xtest shape:',x_test.shape,'\n')


## EVALUATE MODEL
print('\n*** TEST DATA RESULTS (5 per class) ***')
pred=model.predict(x_test, verbose=0) #predicted label probabilities for test data
y_pred=pred.round()  #predicted label for test data

# np.set_printoptions(precision=2) #show only 2 decimal places (does not change actual numbers)
# print('\nprediction & label:\n',np.hstack((pred,y_test))) #probability comparison
c1res=np.hstack((y_pred[c1idx:c1idx+5],y_test[c1idx:c1idx+5]))
c2res=np.hstack((y_pred[c2idx:c2idx+5],y_test[c2idx:c2idx+5]))
c3res=np.hstack((y_pred[c3idx:c3idx+5],y_test[c3idx:c3idx+5]))
c4res=np.hstack((y_pred[c4idx:c4idx+5],y_test[c4idx:c4idx+5]))
# samp_res=np.vstack((c1res,c2res,c3res,c4res))
print(f'LABELS\nPredicted vs. True\n\nGreedy\n {c1res}\n\nGreedyPRO\n {c2res}\n\nAuction\n {c3res}\n\nAuctionPRO\n {c4res}\n') #label comparison

eval=model.evaluate(x_test, y_test, verbose=0) #loss and accuracy
# print('\nevaluation:\n',eval) #print evaluation metrics numbers
print('model.metrics_names:\n',model.metrics_names) #print evaluation metrics names (loss and accuracy)
print(eval) #print evaluation metrics numbers


## CONFUSION MATRIX & CLASSIFICATION REPORT
print('\n*** CONFUSION MATRIX ***\n[TN FP]\n[FN TP]')
cm = multilabel_confusion_matrix(y_test, y_pred)
print('\nLabel0 (Comms)\n',cm[0])
print('\nLabel1 (ProNav)\n',cm[1])

print('\nHamming Loss:',hamming_loss(y_test, y_pred),'\n')

target_names = ['Comms', 'ProNav']
print(classification_report(y_test, y_pred, target_names=target_names))


## PRINT ELAPSE TIME
elapse_time(start)