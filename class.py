import numpy as np
import tensorflow as tf
from timeit import default_timer as timer

from utils.elapse import elapse_time
from utils.resources import print_resources
import utils.params as params
from utils.datapipeline import import_dataset
from utils.model import get_model
from utils.compiler import get_loss, get_optimizer, get_metric
from utils.callback import callback_list
from utils.results import print_results
from utils.trainplot import train_plot

## INTRO
start = timer() # start timer to calculate run time
GPUs=print_resources() # computation resources available
hparams = params.get_hparams() # parse BASH run-time hyperparameters (used throughout script below)
params.save_hparams(hparams) #create model folder and save hyperparameters list .txt


## IMPORT DATASET
x_train, y_train, x_test, y_test, num_classes, cs_idx, input_shape, output_shape = import_dataset(hparams)


## BUILD & COMPILE MODEL
if hparams.mode == 'train':
    loss_weights=None #  single output head
    if hparams.output_type == 'mh': # multihead output
        loss_weights={'output_class':0.2,'output_attr':0.8}
        print(f"Loss Weights: {loss_weights}\n")
    if GPUs>1:
        print(f"GPUs availale: {GPUs}, MULTI GPU TRAINING")
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = get_model(hparams, input_shape, output_shape)
            model.compile(
                loss=get_loss(hparams),
                optimizer=get_optimizer(hparams),
                metrics=get_metric(hparams),
                loss_weights=loss_weights)
    else: # single gpu
        model = get_model(hparams, input_shape, output_shape)
        model.compile(
            loss=get_loss(hparams),
            optimizer=get_optimizer(hparams),
            metrics=get_metric(hparams),
            loss_weights=loss_weights)

elif hparams.mode == 'predict':
    model = tf.keras.models.load_model(hparams.trained_model)


## VISUALIZE MODEL
model.summary()
# make GRAPHVIZ plot of model
tf.keras.utils.plot_model(model, hparams.model_dir + "graphviz.png", show_shapes=True)


## TRAIN MODEL
if hparams.mode == 'train':
    model_history = model.fit(
        x_train,
        y_train,
        validation_split=hparams.train_val_split,
        epochs=hparams.num_epochs,
        batch_size=hparams.batch_size,
        verbose=0,
        callbacks=callback_list(hparams)
    )
    ## PRINT HISTORY KEYS
    print("\nhistory.history.keys()\n", model_history.history.keys()) # this is what you want (only key names)
    ## PRINT TRAINING ELAPSE TIME
    elapse_time(start)
    ## SAVE ENTIRE MODEL AFTER TRAINING
    model.save(filepath=hparams.model_dir+"model.keras", save_format="keras") #saves entire model: weights and layout
    ## TRAINING CURVE: LOSS vs. EPOCH
    train_plot(hparams, model_history)


## TEST DATA PREDICTIONS
if hparams.output_type != 'mh':
    pred=model.predict(x_test, verbose=0) #predicted label probabilities for test data
else: # multihead
    pred_class, pred_attr=model.predict(x_test, verbose=0) # multihead outputs 2 predictions (class and attribute)

if hparams.output_type == 'mc' and hparams.output_length == 'vec':
    y_pred=np.argmax(pred,axis=1).reshape((-1,1))  #predicted class label for test data
elif hparams.output_type == 'mc' and hparams.output_length == 'seq':
    y_pred=np.argmax(pred,axis=2)  #predicted class label for test data
elif hparams.output_type == 'ml':
    y_pred=pred.round()  #predicted attribute label for test data
elif hparams.output_type == 'mh':
    y_pred_attr=pred_attr.round()
    if hparams.output_length == 'vec':
        y_pred_class=np.argmax(pred_class,axis=1).reshape((-1,1))
    else:
        y_pred_class=np.argmax(pred_class,axis=2)
    y_pred=[y_pred_class,y_pred_attr]
        

## TEST DATA PREDICTION SAMPLES
# np.set_printoptions(precision=2) #show only 2 decimal places for probability comparision (does not change actual numbers)
# print('\nprediction & label:\n',np.hstack((pred,y_test))) #probability comparison
num_results=2
print(f'\n*** TEST DATA RESULTS COMPARISON ({num_results} per class) ***')
print('    LABELS\nTrue vs. Predicted')
class_names = ['Greedy', 'Greedy+', 'Auction', 'Auction+']
if hparams.output_type != 'mh':
    for c in range(len(cs_idx)): # print each class name and corresponding prediction samples
        print(f"\n{class_names[c]}")
        # print(f"first true label: \n{y_test[cs_idx[c]]}")
        print(np.concatenate((y_test[cs_idx[c]:cs_idx[c]+num_results],
                            y_pred[cs_idx[c]:cs_idx[c]+num_results]), axis=-1))
else: # multihead output
    for c in range(len(cs_idx)): # print each class name and corresponding prediction samples
        print(f"\n{class_names[c]}")
        print('class')
        print(np.concatenate((y_test[0][cs_idx[c]:cs_idx[c]+num_results],
                            y_pred_class[cs_idx[c]:cs_idx[c]+num_results]), axis=-1))
        print('attribute')
        print(np.concatenate((y_test[1][cs_idx[c]:cs_idx[c]+num_results],
                            y_pred_attr[cs_idx[c]:cs_idx[c]+num_results]), axis=-1))


## EVALUATE MODEL
eval=model.evaluate(x_test, y_test, verbose=0) #loss and accuracy
print('model.metrics_names:\n',model.metrics_names) #print evaluation metrics names (loss and accuracy)
print(eval) #print evaluation metrics numbers


## RESULTS
print_results(hparams, y_test, y_pred)


## PRINT ELAPSE TIME
elapse_time(start)