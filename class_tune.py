import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
import keras_tuner

from utils.elapse import elapse_time
from utils.resources import print_resources
import utils.params as params
from utils.datapipeline import import_data, get_dataset
from utils.model_tune import get_model
from utils.compiler import get_loss, get_optimizer, get_metric
from utils.callback import callback_list
from utils.results import print_results
from utils.trainplot import train_plot

## INTRO
start = timer() # start timer to calculate run time
GPUs=print_resources() # computation resources available
hparams = params.get_hparams() # parse BASH run-time hyperparameters (used throughout script below)
params.save_hparams(hparams) #create model folder and save hyperparameters list .txt


## IMPORT DATA
x_train, y_train, x_test, y_test, num_classes, cs_idx, input_shape, output_shape = import_data(hparams)
## CREATE DATASET OBJECTS (to allow multi-GPU training)
train_dataset, val_dataset, test_dataset = get_dataset(hparams, x_train, y_train, x_test, y_test)

loss_weights=None #  single output head
if hparams.output_type == 'mh': # multihead output
    loss_weights={'output_class':0.2,'output_attr':0.8}
    # print(f"Loss Weights: {loss_weights}\n")


## HYPERPARAMETER TUNING
def build_model(hp):
    
    # define model type and tuneable hyperparameters (store in "hparams" to build model)
    model_type=hparams.model_type
    if model_type == 'fc':
        mlp_units=[]
        for i in range(hp.Int("num_units", min_value=1, max_value=6, step=1)):
            mlp_units.append(hp.Int(f"units_{i}", min_value=10, max_value=100, step=10))
        hparams.mlp_units=mlp_units
        hparams.dropout=hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)

    elif model_type == 'cn':
        filters=[]
        kernels=[]
        for i in range(hp.Int("num_filters", min_value=1, max_value=6, step=1)):
            filters.append(hp.Int(f"filter_{i}", min_value=32, max_value=256, step=32))
            kernels.append(hp.Int(f"kernel_{i}", min_value=3, max_value=7, step=2))
        hparams.filters=filters
        hparams.kernels=kernels
        hparams.pool_size=hp.Int("pool_size", min_value=2, max_value=5, step=1)
        hparams.dropout=hp.Float("dropout", min_value=0, max_value=0.5, step=0.1)

    elif model_type == 'fcn':
        filters=[]
        kernels=[]
        for i in range(hp.Int("num_filters", min_value=1, max_value=6, step=1)):
            filters.append(hp.Int(f"filter_{i}", min_value=32, max_value=256, step=32))
            kernels.append(hp.Int(f"kernel_{i}", min_value=3, max_value=7, step=2))
        hparams.filters=filters
        hparams.kernels=kernels

    elif model_type == 'res':
        hparams.num_res_layers=hp.Int("num_res_layers", min_value=1, max_value=6, step=1)
        filters=[]
        kernels=[]
        for i in range(hp.Int("num_filters", min_value=3, max_value=9, step=1)):
            filters.append(hp.Int(f"filter_{i}", min_value=32, max_value=256, step=32))
            kernels.append(hp.Int(f"kernel_{i}", min_value=3, max_value=7, step=2))
        hparams.filters=filters
        hparams.kernels=kernels

    elif model_type == 'lstm':
        units=[]
        for i in range(hp.Int("num_units", min_value=1, max_value=6, step=1)):
            units.append(hp.Int(f"units_{i}", min_value=10, max_value=150, step=10))
        hparams.units=units
        hparams.dropout=hp.Float("dropout", min_value=0.0, max_value=0.05, step=0.05)
    
    # window = hp.Int("window", min_value=10, max_value=58, step=4, default=20)
    # hparams.window=window
    # activation = hp.Choice("activation", ["relu", "tanh"])
    # dropout = hp.Boolean("dropout")
    # lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    # call existing model-building code with the hyperparameter values
    model = get_model(hparams, input_shape, output_shape)
    model.compile(
        loss=get_loss(hparams),
        optimizer=get_optimizer(hparams),
        metrics=get_metric(hparams),
        loss_weights=loss_weights)
    
    return model

## DEFINE SEARCH SPACE
# tuner types: RandomSearch( , BayesianOptimization( , Hyperband
tuner=hparams.tune_type
if tuner=="r": # random search
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_loss",
        max_trials=200,
        executions_per_trial=1,
        ## USE FOR NEW TUNE
        overwrite=True,
        directory=hparams.model_dir,
        ## USE TO CONTINUE PREVIOUS TUNE
        # overwrite=False,
        # directory='/home/donald.peltier/swarm/model/swarm_class09-13_18-01-12',
        project_name="tune")
elif tuner=="h": # hyperband search
    tuner = keras_tuner.Hyperband(
        hypermodel=build_model,
        objective="val_loss",
        max_epochs=hparams.tune_epochs,
        ## USE FOR NEW TUNE
        overwrite=True,
        directory=hparams.model_dir,
        ## USE TO CONTINUE PREVIOUS TUNE
        # overwrite=False,
        # directory="/home/donald.peltier/swarm/model/swarm_class09-18_16-26-50_HTuneFCml_NoValues",
        project_name="tune")

print('\n*** SEARCH SPACE SUMMARY ***')
tuner.search_space_summary() # print search space summary


## START HYPERTUNING
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=hparams.patience)
tuner.search(train_dataset,
             validation_data=val_dataset,
             epochs=hparams.tune_epochs,
             verbose=0,
             callbacks=[stop_early]
             )


## SEARCH RESULTS
print('\n*** RESULTS SUMMARY (TOP 3) ***')
tuner.results_summary(3)


# GET BEST HYPER-PARAMETERS FOUND
print('\n*** BEST H-PARAMS ***')
print(tuner.get_best_hyperparameters()[0].values,'\n')
## PRINT TUNING ELAPSE TIME
elapse_time(start)


# BUILD BEST MODEL
model = build_model(tuner.get_best_hyperparameters()[0])
## VISUALIZE MODEL
model.summary()
# MAKE GRAPHVIZ MODEL DIAGRAM
tf.keras.utils.plot_model(model, hparams.model_dir + "graphviz.png", show_shapes=True)


## TRAIN BEST MODEL
if hparams.mode == 'train':

    # # USING DATA
    # model_history = model.fit(
    #     x_train,
    #     y_train,
    #     validation_split=hparams.train_val_split,
    #     epochs=hparams.num_epochs,
    #     batch_size=hparams.batch_size,
    #     verbose=0,
    #     callbacks=callback_list(hparams)
    #     )

    # USING DATASET
    model_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=hparams.num_epochs,
        verbose=0,
        callbacks=callback_list(hparams)
        )
    ## PRINT HISTORY KEYS
    print("\nhistory.history.keys()\n", model_history.history.keys()) # this is what you want (only key names)
    ## PRINT TRAINING ELAPSE TIME
    elapse_time(start)
    ## SAVE ENTIRE MODEL AFTER TRAINING
    model.save(filepath=hparams.model_dir+"model.keras", save_format="keras") #saves entire model: weights and layout
    ## TRAINING CURVE: ACCURACY/LOSS vs. EPOCH
    train_plot(hparams, model_history)


## TEST DATA PREDICTIONS
if hparams.output_type != 'mh':
    pred=model.predict(x_test, verbose=0) #predicted label probabilities for test data
else: # multihead
    pred_class, pred_attr=model.predict(x_test, verbose=0) # multihead outputs 2 predictions (class and attribute)
# check for errors in requested output_length
output_length=hparams.output_length
if hparams.model_type!='lstm' and hparams.model_type!='tr': # only 'lstm' and 'tr' can output sequences
    output_length='vec'
# convert from probability to label
if hparams.output_type == 'mc' and output_length == 'vec':
    y_pred=np.argmax(pred,axis=1).reshape((-1,1))  #predicted class label for test data
elif hparams.output_type == 'mc' and output_length == 'seq':
    y_pred=np.argmax(pred,axis=2)  #predicted class label for test data
elif hparams.output_type == 'ml':
    y_pred=pred.round()  #predicted attribute label for test data
elif hparams.output_type == 'mh':
    y_pred_attr=pred_attr.round()
    if output_length == 'vec':
        y_pred_class=np.argmax(pred_class,axis=1).reshape((-1,1))
    else:
        y_pred_class=np.argmax(pred_class,axis=2)
    y_pred=[y_pred_class,y_pred_attr]
        

## TEST DATA PREDICTION SAMPLES
# np.set_printoptions(precision=2) #show only 2 decimal places for probability comparision (does not change actual numbers)
# print('\nprediction & label:\n',np.hstack((pred,y_test))) #probability comparison
num_results=2 # nunber of examples to view
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