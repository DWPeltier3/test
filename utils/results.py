from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,multilabel_confusion_matrix,hamming_loss,classification_report
import matplotlib.pyplot as plt

def print_results(hparams, y_test, y_pred, class_names, attribute_names):

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
            plt.savefig(hparams.model_dir + f"conf_matrix_{attribute_names[label_index]}.png")
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
            plt.savefig(hparams.model_dir + f"conf_matrix_{attribute_names[label_index]}.png")
            print(f'\nLabel{label_index} {attribute_names[label_index]}\n',cm[label_index])
        print('\nHamming Loss:',hamming_loss(y_test[1], y_pred[1]),'\n')
        print(classification_report(y_test[1], y_pred[1], target_names=attribute_names))