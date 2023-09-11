from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,multilabel_confusion_matrix,hamming_loss,classification_report
import matplotlib.pyplot as plt

def print_results(hparams, y_test, y_pred):
    
    # multiclass
    if hparams.output_type == 'mc':
        if hparams.output_length == 'seq':
            y_test=y_test.reshape((-1,1)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred=y_pred.reshape((-1,1)) #flatten both predictions and labels into one column vector each
        cm = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(cm).plot()
        plt.savefig(hparams.model_dir + "conf_matrix.png")
    
    # multioutput
    elif hparams.output_type == 'ml':
        if hparams.output_length == 'seq':
            y_test=y_test.reshape((-1,2)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred=y_pred.reshape((-1,2)) #flatten both predictions and labels into 2 column vectors each
        print('\n*** CONFUSION MATRIX ***\n[TN FP]\n[FN TP]')
        cm = multilabel_confusion_matrix(y_test, y_pred)
        print('\nLabel0 (Comms)\n',cm[0])
        print('\nLabel1 (ProNav)\n',cm[1])

        print('\nHamming Loss:',hamming_loss(y_test, y_pred),'\n')

        target_names = ['Comms', 'ProNav']
        print(classification_report(y_test, y_pred, target_names=target_names))

    # multihead
    elif hparams.output_type == 'mh':
        # multiclass results
        if hparams.output_length == 'seq':
            y_test[0]=y_test[0].reshape((-1,1)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred[0]=y_pred[0].reshape((-1,1)) #flatten predictions and labels into column vectors
        cm = confusion_matrix(y_test[0], y_pred[0])
        cm_display = ConfusionMatrixDisplay(cm).plot()
        plt.savefig(hparams.model_dir + "conf_matrix.png")

        # multilabel results
        if hparams.output_length == 'seq':
            y_test[1]=y_test[1].reshape((-1,2)) #finds 'pseudo' CM for sequence output (every time step is prediction)
            y_pred[1]=y_pred[1].reshape((-1,2)) #flatten predictions and labels into 2 column vectors
        print('\n*** CONFUSION MATRIX ***\n[TN FP]\n[FN TP]')
        cm = multilabel_confusion_matrix(y_test[1], y_pred[1])
        print('\nLabel0 (Comms)\n',cm[0])
        print('\nLabel1 (ProNav)\n',cm[1])

        print('\nHamming Loss:',hamming_loss(y_test[1], y_pred[1]),'\n')

        target_names = ['Comms', 'ProNav']
        print(classification_report(y_test[1], y_pred[1], target_names=target_names))