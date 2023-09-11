import numpy as np
import matplotlib.pyplot as plt

def train_plot(hparams, model_history):
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