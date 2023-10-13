from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,multilabel_confusion_matrix,hamming_loss,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def print_train_plot(hparams, model_history):
    ## TRAINING CURVE: Loss & Accuracy vs. EPOCH
    epoch = np.array(model_history.epoch)
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    if hparams.output_type == 'mc':
        acc=model_history.history["sparse_categorical_accuracy"]
        val_acc = model_history.history['val_sparse_categorical_accuracy']
    elif hparams.output_type == 'ml':
        acc=model_history.history["binary_accuracy"]
        val_acc = model_history.history['val_binary_accuracy']
    
    mvl=min(val_loss)
    print(f"Minimum Val Loss: {mvl}") 

    plt.figure()
    # must shift TRAINING values 1/2 epoch left b/c validaiton values computed AFTER each epoch, while training values are averaged during epoch
    if hparams.output_type != 'mh':
        plt.plot(epoch-0.5, loss, 'r', label='Training Loss')
        plt.plot(epoch, val_loss, 'b', label='Validation Loss')
        plt.plot(epoch-0.5, acc, 'tab:orange', label='Training Accuracy')
        plt.plot(epoch, val_acc, 'g', label='Validation Accuracy')
    else:
        # color names found at https://matplotlib.org/stable/gallery/color/named_colors.html
        plt.plot(epoch-0.5, model_history.history['loss'], 'r', label='Training Loss')
        plt.plot(epoch-0.5, model_history.history['output_class_loss'], "mistyrose", label='Class Trng Loss')
        plt.plot(epoch-0.5, model_history.history['output_attr_loss'], "lightcoral", label='Attribute Trng Loss')
        
        plt.plot(epoch, model_history.history['val_loss'], 'b', label='Validation loss')
        plt.plot(epoch, model_history.history['val_output_class_loss'], "lightsteelblue", label='Class Val Loss')
        plt.plot(epoch, model_history.history['val_output_attr_loss'], "cornflowerblue", label='Attribute Val Loss')

        plt.plot(epoch-0.5, model_history.history['output_class_sparse_categorical_accuracy'], "bisque", label='Class Trng Accuracy')
        plt.plot(epoch-0.5, model_history.history['output_attr_binary_accuracy'], "darkorange", label='Attribute Trng Accuracy')

        plt.plot(epoch, model_history.history['val_output_class_sparse_categorical_accuracy'], "lightgreen", label='Class Val Accuracy')
        plt.plot(epoch, model_history.history['val_output_attr_binary_accuracy'], "green", label='Attribute Val Accuracy')

    plt.title('Training & Validation Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(hparams.model_dir + "Training_LossAccuracy_vs_Epoch.png")

def print_cm(hparams, y_test, y_pred, class_names=None, attribute_names=None):
    # multiclass
    if hparams.output_type == 'mc':
        if hparams.output_length == 'seq':
            y_test=y_test.reshape((-1,1)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred=y_pred.reshape((-1,1)) #flatten both predictions and labels into one column vector each
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot()
        plt.savefig(hparams.model_dir + "conf_matrix_MC.png")
    # multilabel
    elif hparams.output_type == 'ml':
        if hparams.output_length == 'seq':
            y_test=y_test.reshape((-1,2)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred=y_pred.reshape((-1,2)) #flatten both predictions and labels into 2 column vectors each
        
        # Generate confusion matrix FIGURE & PRINT to log
        num_labels = len(y_test[0])
        cm = multilabel_confusion_matrix(y_test, y_pred)
        print('\n*** CONFUSION MATRIX ***\n[TN FP]\n[FN TP]')
        for label_index in range(num_labels):
            disp = ConfusionMatrixDisplay(cm[label_index], display_labels=['No', 'Yes'])
            disp.plot()
            plt.title(f'{attribute_names[label_index]} Confusion Matrix')
            plt.savefig(hparams.model_dir + f"conf_matrix_ML_{attribute_names[label_index]}.png")
            print(f'\nLabel{label_index} {attribute_names[label_index]}\n',cm[label_index]) 
        print('\nHamming Loss:',hamming_loss(y_test, y_pred),'\n')
        print(classification_report(y_test, y_pred, target_names=attribute_names))
    # multihead
    elif hparams.output_type == 'mh':
        # multiclass results
        if hparams.output_length == 'seq':
            y_test[0]=y_test[0].reshape((-1,1)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred[0]=y_pred[0].reshape((-1,1)) #flatten predictions and labels into column vectors
        cm = confusion_matrix(y_test[0], y_pred[0])
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot()
        plt.savefig(hparams.model_dir + "conf_matrix_MC.png")
        # multilabel results
        if hparams.output_length == 'seq':
            y_test[1]=y_test[1].reshape((-1,2)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred[1]=y_pred[1].reshape((-1,2)) #flatten predictions and labels into 2 column vectors
        # Generate confusion matrix FIGURE & PRINT to log
        num_labels = len(y_test[1][0])
        cm = multilabel_confusion_matrix(y_test[1], y_pred[1])
        print('\n*** CONFUSION MATRIX ***\n[TN FP]\n[FN TP]')
        for label_index in range(num_labels):
            disp = ConfusionMatrixDisplay(cm[label_index], display_labels=['No', 'Yes'])
            disp.plot()
            plt.title(f'{attribute_names[label_index]} Confusion Matrix')
            plt.savefig(hparams.model_dir + f"conf_matrix_ML_{attribute_names[label_index]}.png")
            print(f'\nLabel{label_index} {attribute_names[label_index]}\n',cm[label_index])
        print('\nHamming Loss:',hamming_loss(y_test[1], y_pred[1]),'\n')
        print(classification_report(y_test[1], y_pred[1], target_names=attribute_names))


def print_cam(hparams, model, x_train):
    ''' This function  prints the "Class Activation Map" along with model inputs to help 
    visualize input importance in making a prediction.  This is only applicable for models
    that have a Global Average Pooling layer (ex: Fully Convolutional Network)'''
    
    # select one training sample to analyze
    sample = x_train[20]
    # Get the class activation map for that sample
    last_cov_layer=-5 if hparams.output_type == 'mh' else -3 # multihead v2 has 2 extra layers at end
    heatmap = get_cam(model, sample, model.layers[last_cov_layer].name)
    ## Visualize Class Activation Map
    # ALL FEATURES: Plot the heatmap values along with the time series data for all features (all agents) in that sample
    plt.figure(figsize=(10, 8))
    plt.plot(sample, c='black')
    plt.plot(heatmap, label='CAM [importance]', c='red', lw=5, linestyle='dashed')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Value [normalized]')
    plt.legend()
    plt.title(f'Class Activation Map vs. All Input Features')
    plt.savefig(hparams.model_dir + "CAM_all.png")
    # ONE AGENT: Plot the heatmap values along with the time series features for one agent in that sample
    num_features=x_train.shape[2]
    num_agents=num_features//4
    plt.figure(figsize=(10, 8))
    agent_idx=1
    plt.plot(sample[:, agent_idx], label='Px')
    plt.plot(sample[:, agent_idx+num_agents], label='Py')
    plt.plot(sample[:, agent_idx+2*num_agents], label='Vx')
    plt.plot(sample[:, agent_idx+3*num_agents], label='Vy')
    plt.plot(heatmap, label='CAM [importance]', c='red', lw=5, linestyle='dashed')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Value [normalized]')
    plt.legend()
    plt.title(f"Class Activation Map vs. One Agent's Input Features")
    plt.savefig(hparams.model_dir + "CAM_one.png")

def get_cam(model, sample, last_conv_layer_name):
    # This function requires the trained FCN model, input sample, and the name of the last convolutional layer
    # Get the model of the intermediate layers
    cam_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output[0]]
        )
    # Get the last conv_layer outputs and full model predictions
    with tf.GradientTape() as tape:
        conv_outputs, predictions = cam_model(np.array([sample]))
        predicted_class = tf.argmax(predictions[0]) # predicted class
        predicted_class_val = predictions[:, predicted_class] # predicted class probability
    # Get the gradients and pooled gradients
    grads = tape.gradient(predicted_class_val, conv_outputs) # gradients between predicted class probability WRT CONV outputs maps
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1)) # average ("pool") feature gradients (gives single importance value for each map)
    # Multiply pooled gradients (importance) with the conv layer output, then average across all feature maps, to get the 2D heatmap
    # Heatmap highlights areas that most influence models prediction
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    # Normalize the heatmap (between 0 and largest feature value)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = heatmap * np.max(sample)
    return heatmap[0]

def print_tsne(hparams, features, labels, class_names, title, perplexity):
    tsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(features) # set perplexity = 50-100
    scaler = MinMaxScaler() #scale between 0 and 1
    tsne = scaler.fit_transform(tsne.reshape(-1, tsne.shape[-1])).reshape(tsne.shape) # fit amoungst features, then back to original shape
    # print(f'features.shape {features.shape}')
    # print(f'labels.shape {labels.shape}')
    # print(f'tsne.shape {tsne.shape}')
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    colors = ['red', 'blue', 'green', 'brown']#, 'yellow']
    plt.figure()
    plt.title('tSNE Dimensionality Reduction: '+title)
    for idx, c in enumerate(colors):
        # if label == color index, plot in that color
        indices = [i for i, l in enumerate(labels) if idx == l]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        plt.scatter(current_tx, current_ty, c=c, label=class_names[idx], alpha=0.5, marker=".")
    plt.legend(loc='best')
    plt.savefig(hparams.model_dir + "tSNE_"+title+".png")