import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
import math

## IMPORT DATA
def import_data(hparams):
    
    data_path=hparams.data_path
    model_type=hparams.model_type # fc=fully connect, cn=CNN, fcn=FCN, res=ResNet, lstm=LSTM, tr=transformer
    window=hparams.window
    output_type=hparams.output_type # mc=multiclass, ml=multilabel, mh=multihead
    output_length=hparams.output_length # vec=vector, seq=sequence
    
    ## LOAD DATA
    data=np.load(data_path)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    ## CHARACTERIZE DATA
    time_steps=x_train.shape[1]
    num_features=x_train.shape[2]
    
    ## VISUALIZE DATA
    print('\n*** DATA ***')
    print('xtrain shape:',x_train.shape)
    print('xtest shape:',x_test.shape)
    print('ytrain shape:',y_train.shape)
    print('ytest shape:',y_test.shape)
    print('num features:',num_features)
    print('xtrain sample (1st instance, 1st time step, all features)\n',x_train[0,:num_features])
    print('ytrain sample (first instance)',y_train[0])

    ## REDUCE OBSERVATION WINDOW, IF REQUIRED
    if window==-1 or window>time_steps: # if window = -1 (use entire window) or invalid window (too large>min_time): uses entire observation window (min_time) for all runs
        window=time_steps
    input_length=window
    x_train=x_train[:,:input_length]
    x_test=x_test[:,:input_length]
    print('\n*** REDUCED OBSERVATION WINDOW ***')
    print('time steps available:',time_steps)
    print('window used:',window)
    print('input length:',input_length)
    print('xtrain shape:',x_train.shape)
    print('xtest shape:',x_test.shape)

    ## TIME SERIES PLOTS (VISUALIZE PATTERNS)
    # Plots all agents' features vs. time window for one class
    # Select sample
    # sample_idx = np.random.randint(0, x_train.shape[0]) # if want a random sample
    sample_idx = 0 # first sample
    sample_data = x_train[sample_idx] # Get data for that sample
    # Plot positions and velocities over time for each agent
    num_agents=num_features//4
    num_subplot=math.ceil(math.sqrt(num_agents))
    plt.figure(figsize=(20,20))
    for agent_idx in range(num_agents):
        plt.subplot(num_subplot,num_subplot, agent_idx + 1)
        plt.plot(sample_data[:, agent_idx], label='Px')
        plt.plot(sample_data[:, agent_idx+num_agents], label='Py')
        plt.plot(sample_data[:, agent_idx+2*num_agents], label='Vx')
        plt.plot(sample_data[:, agent_idx+3*num_agents], label='Vy')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value [normalized]')
        plt.legend()
        plt.title(f'Agent {agent_idx + 1}')
    plt.savefig(hparams.model_dir + "Agent_feature_plots.png")

    # Plots one agents' features vs. time window for all classes
    # Find the unique classes and their first index
    unique_classes, unique_indices = np.unique(y_train, return_index=True)
    num_classes = len(unique_classes)
    num_agents=num_features//4
    agent_idx=0

    # Create a subplot for each class to visualize features for Agent 1
    class_names = ['Greedy', 'Greedy+', 'Auction', 'Auction+']
    plt.figure(figsize=(20, 5 * num_classes))  # Adjust the figure size
    for i, idx in enumerate(unique_indices):
        sample_data = x_train[idx]  # Get data for that sample
        plt.subplot(num_classes, 1, i + 1)
        plt.plot(sample_data[:, agent_idx], label='Px')  # Position X for Agent
        plt.plot(sample_data[:, agent_idx+num_agents], label='Py')  # Position Y for Agent
        plt.plot(sample_data[:, agent_idx+2*num_agents], label='Vx')  # Velocity X for Agent
        plt.plot(sample_data[:, agent_idx+3*num_agents], label='Vy')  # Velocity Y for Agent
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value [normalized]')
        plt.legend()
        plt.title(f'{class_names[i]}')
    plt.savefig(hparams.model_dir + "Agent_feature_plots_per_class.png")



    # ## PCA
    # class_names = ['Greedy', 'Greedy+', 'Auction', 'Auction+']
    # num_classes=len(np.unique(y_train))
    # # # remove a class == value
    # # value = 2
    # # indices_to_remove = np.where(y_train == value)[0]
    # # x_train = np.delete(x_train, indices_to_remove, axis=0)
    # # y_train = np.delete(y_train, indices_to_remove, axis=0)
    # x_pca = x_train.reshape(-1, num_features)  # Reshape data to be 2D: (num_samples * num_timesteps, num_features)
    # # must have label for every timestep (same as "sequence output")
    # train_temp=np.zeros((y_train.shape[0], window), dtype=np.int8)
    # for c in range(num_classes):
    #     train_temp[y_train[:,0]==c]=c
    # y_pca=train_temp.ravel() # reshape labels to be 1D: (num_samples * num_timesteps,)
    # print('\n*** DATA for PCA ***')
    # print('x_pca shape:',x_pca.shape)
    # print('y_pca shape:',y_pca.shape)
    # # Perform PCA
    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(x_pca)
    # # Normalize labels to map to the colormap
    # norm = plt.Normalize(y_pca.min(), y_pca.max())
    # # Create a custom legend
    # unique_labels = np.unique(y_pca)
    # handles = [Patch(color=plt.cm.jet(norm(label)), label=f"{class_names[label]}") for label in unique_labels]
    # # 2D Scatter plot of the first two principal components
    # plt.figure(figsize=(10, 5))
    # scatter=plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_pca, cmap='jet', norm=norm, alpha=0.5, marker=".")
    # plt.legend(handles=handles, title="Classes")
    # plt.title('2D Principle Component Analysis of Input Data')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.savefig(hparams.model_dir + "PCA_2D.png")
    # # 3D Scatter plot of the first three principal components
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=y_pca, cmap='jet', norm=norm, alpha=0.5, marker=".")
    # plt.legend(handles=handles, title="Classes")
    # ax.set_title('3D Principle Component Analysis of Input Data')
    # plt.savefig(hparams.model_dir + "PCA_3D.png")

    ## TEST SET: DETERMINE NUM DATA CLASSES AND NUM RUNS (HELPS SHOW PORTION OF TEST RESULTS AT END)
    num_classes=len(np.unique(y_test)) # before restructuring labels (as required, for multilabel)
    num_runs=y_test.shape[0]
    rpc=num_runs//num_classes # runs per class
    cs_idx=[] # class start index
    for c in range(num_classes):
        cs_idx.append(c*rpc)
    print('\n*** START INDEX FOR EACH TEST CLASS ***')
    print('num test classes:',num_classes)
    print('num test runs:',num_runs)
    print(f'test set class start indices: {cs_idx}')

    ## IF MULTILABEL (and output length=vector), RESTRUCTURE LABELS
    # check for errors in requested output_length
    if model_type!='lstm' and model_type!='tr': # only 'lstm' and 'tr' can output sequences
        output_length='vec'
    if (output_type=='ml' or output_type=='mh') and output_length!='seq':
        if output_type == 'mh': # multihead must preserve multiclass labels
            y_train_class=y_train
            y_test_class=y_test
        # add column for second label
        y_train=np.insert(y_train,1,0,axis=1)
        y_test=np.insert(y_test,1,0,axis=1)
        # update labels: columns are classes
        # 1st col=Comms(Auction), 2nd col=ProNav
        y_train[y_train[:,0]==0]=[0,0] # Greedy
        y_train[y_train[:,0]==1]=[0,1] # Greedy+
        y_train[y_train[:,0]==2]=[1,0] # Auction
        y_train[y_train[:,0]==3]=[1,1] # Auction+
        y_test[y_test[:,0]==0]=[0,0] # same as above
        y_test[y_test[:,0]==1]=[0,1]
        y_test[y_test[:,0]==2]=[1,0]
        y_test[y_test[:,0]==3]=[1,1]
        print('\n*** MULTILABEL ***')
        print('ytrain shape:',y_train.shape)
        print('ytest shape:',y_test.shape)
        print('ytrain sample (first instance)',y_train[0,:])
        print('ytest sample (first instance)',y_test[0,:])
        if output_type == 'mh':
            print('\n*** MULTIHEAD MULTICLASS ***')
            print('ytrain_class shape:',y_train_class.shape)
            print('ytest_class shape:',y_test_class.shape)
            print('ytrain_class sample (first instance)',y_train_class[0,:])
            print('ytest_class sample (first instance)',y_test_class[0,:])
            y_train=[y_train_class,y_train] # combine class & attribute labels
            y_test=[y_test_class,y_test]
            print('\n*** MULTIHEAD COMBINED LABELS ***')
            print(f'ytrain shape: class {y_train[0].shape} and attr {y_train[1].shape}')
            print(f'ytest shape: class {y_test[0].shape} and attr {y_test[1].shape}')

    ## IF SEQUENCE OUTPUT, RESTRUCTURE LABELS
    if output_length == 'seq':
        if output_type=='mc' or output_type=='mh': # multiclass
            train_temp=np.zeros((y_train.shape[0], window), dtype=np.int8)
            test_temp=np.zeros((y_test.shape[0], window), dtype=np.int8)
            for c in range(num_classes):
                train_temp[y_train[:,0]==c]=c
                test_temp[y_test[:,0]==c]=c
            if output_type == 'mh':
                y_train_class=train_temp
                y_test_class=test_temp
        if output_type=='ml' or output_type=='mh': # multilabel
            num_attributes=2
            train_temp=np.zeros((y_train.shape[0], window, num_attributes), dtype=np.int8)
            test_temp=np.zeros((y_test.shape[0], window, num_attributes), dtype=np.int8)
            train_temp[y_train[:,0]==0]=[0,0] # Greedy
            train_temp[y_train[:,0]==1]=[0,1] # Greedy+
            train_temp[y_train[:,0]==2]=[1,0] # Auction
            train_temp[y_train[:,0]==3]=[1,1] # Auction+
            test_temp[y_test[:,0]==0]=[0,0] # same as above
            test_temp[y_test[:,0]==1]=[0,1]
            test_temp[y_test[:,0]==2]=[1,0]
            test_temp[y_test[:,0]==3]=[1,1]
        y_train=train_temp
        y_test=test_temp
        print('\n*** SEQUENCE OUTPUTS ***')
        print('ytrain shape:',y_train.shape)
        print('ytest shape:',y_test.shape)
        print('ytrain sample (first instance)\n',y_train[0,:])
        print('ytest sample (first instance)\n',y_test[0,:])
        if output_type == 'mh':
            print('\n*** MULTIHEAD MULTICLASS ***')
            print('ytrain_class shape:',y_train_class.shape)
            print('ytest_class shape:',y_test_class.shape)
            print('ytrain_class sample (first instance)',y_train_class[0,:])
            print('ytest_class sample (first instance)',y_test_class[0,:])
            y_train=[y_train_class,y_train] # combine class & attribute labels
            y_test=[y_test_class,y_test]
            print('\n*** MULTIHEAD COMBINED LABELS ***')
            print(f'ytrain shape: class {y_train[0].shape} and attr {y_train[1].shape}')
            print(f'ytest shape: class {y_test[0].shape} and attr {y_test[1].shape}')

    ## RESHAPE DATA IF "FULLY CONNECTED" NN
    if  model_type == 'fc':
        # flatten timeseries data into 1 column per run for fully connected model
        x_train=x_train.reshape(x_train.shape[0],-1) # Reshape to (batch, time*feature)
        x_test=x_test.reshape(x_test.shape[0],-1) # Reshape to (batch, time*feature)
        print('\n*** FULLY CONNECTED DATA ***')
        print('xtrain shape:',x_train.shape)
        print('xtest shape:',x_test.shape)

    ## SIZE MODEL INPUTS AND OUTPUTS
    # inputs
    input_shape=x_train.shape[1:] # time steps and features
    # outputs
    if output_type == 'mc': # multiclass
        output_shape=len(np.unique(y_train)) # number of unique labels (auto flattens to find unique labels)
    elif output_type == 'ml': # multilabel
        output_shape=y_train.shape[-1] # number of binary labels; [-1]=last dimension (works for vector & sequence)
    elif output_type == 'mh': # multihead
        output_shape=[len(np.unique(y_train[0])),y_train[1].shape[-1]]
    print('\n*** SIZE MODEL INPUTS & OUTPUTS ***')
    print('input shape: ', input_shape)
    print('output shape: ', output_shape)

    return x_train, y_train, x_test, y_test, num_classes, cs_idx, input_shape, output_shape



## CREATE DATASET
def get_dataset(hparams, x_train, y_train, x_test, y_test):

    batch_size = hparams.batch_size
    validation_split=hparams.train_val_split
    num_val_samples = round(x_train.shape[0]*validation_split)
    print(f'Val Split {validation_split} and # Val Samples {num_val_samples}')

    # Reserve "num_val_samples" for validation
    x_val = x_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]

    if hparams.output_type!='mh':
        y_val = y_train[-num_val_samples:]
        y_train = y_train[:-num_val_samples]
        train_dataset=tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        val_dataset=tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        test_dataset=tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        
        ## SUBCLASS TRANSFORMER (Encoder/Decoder) ONLY
        # if hparams.model_type == 'tr': #transformer needs dataset form (input,label),label
        #     print("*** TRANSFORMER DATASET ***")
            # train_dataset=make_batches(train_dataset)
            # val_dataset=make_batches(val_dataset)
            # test_dataset=make_batches(test_dataset)

            # example=train_dataset.take(3)
            # for element in example:
            #     print(f'train element {element}')
            # for element in example.as_numpy_iterator():
            #     print(f'train element as np_it {element}')


    else: # multihead classifier (2 outputs)
        y_val=[]
        for head in range(len(y_train)):
            y_val.append(y_train[head][-num_val_samples:])
            y_train[head] = y_train[head][:-num_val_samples]
        train_dataset=tf.data.Dataset.from_tensor_slices((x_train, {'output_class':y_train[0], 'output_attr':y_train[1]})).batch(batch_size)
        val_dataset=tf.data.Dataset.from_tensor_slices((x_val, {'output_class':y_val[0], 'output_attr':y_val[1]})).batch(batch_size)
        test_dataset=tf.data.Dataset.from_tensor_slices((x_test, {'output_class':y_test[0], 'output_attr':y_test[1]})).batch(batch_size)
    
    # print('xtrain shape:',x_train.shape)
    # print('x-val shape:',x_val.shape)
    # print('ytrain[0] shape:',y_train[0].shape)
    # print('y-val[0] shape:',y_val[0].shape)
    # print('ytrain[1] shape:',y_train[1].shape)
    # print('y-val[1] shape:',y_val[1].shape)
    # print('ytrain sample (first instance)',y_train[0])
    # print('y-val[0] sample ',y_val[0][:5])
    # print('y-val[1] sample ',y_val[1][:5,:])

    # removes "auto sharding" warning
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)
    test_dataset = test_dataset.with_options(options)

    # autotune prefetching
    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(autotune)
    val_dataset = val_dataset.prefetch(autotune)

    # cache and shuffle
    train_dataset = train_dataset.cache().shuffle(train_dataset.cardinality())
    val_dataset = val_dataset.cache().shuffle(val_dataset.cardinality())

    return (train_dataset, val_dataset, test_dataset)

#SUBCLASS TRANS(E/D) needs dataset in form (input,label),label
def make_batches(ds):
  return (ds.map(prepare_batch, tf.data.AUTOTUNE))
def prepare_batch(data, label):
    label_inputs=label
    label_inputs[:,0]=-1 # replace first label with random start token
    return (data, label_inputs), label