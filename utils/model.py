from tensorflow import keras

def get_model(hparams, input_shape, output_shape):

    model_type=hparams.model_type

    if model_type == 'fc':
        model=fc_model(hparams, input_shape, output_shape)
    elif model_type == 'cn':
        model=cnn_model(hparams, input_shape, output_shape)
    elif model_type == 'fcn':
        model=fcn_model(hparams, input_shape, output_shape)
    elif model_type == 'res':
        model=resnet_model(hparams, input_shape, output_shape)
    elif model_type == 'lstm':
        model=lstm_model(hparams, input_shape, output_shape)
    
    return model


## FULLY CONNECTED (MULTILAYER PERCEPTRON)
def fc_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    mlp_units=[100,12] # each entry becomes a dense layer with corresponding # neurons (# entries = # hidden layers)
    dropout=hparams.dropout
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    kernel_initializer=hparams.kernel_initializer
    if hparams.output_type == 'mc': # multiclass
        out_activation="softmax"
    elif hparams.output_type == 'ml': # multilabel
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation="relu", kernel_regularizer=kernel_regularizer,
                               kernel_initializer=kernel_initializer)(x)
        x = keras.layers.Dropout(dropout)(x)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(x)
    else: # multihead classifier (2 outputs)
        ## VERSION 1
        # output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(x)
        # output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(x)
        # outputs=[output_class, output_attr]
        ## VERSION 2
        output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(x)
        concat=keras.layers.Concatenate()([x,output_attr]) #use attribute output to try and improve class output
        output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(concat)
        outputs=[output_class, output_attr]

    return keras.Model(inputs=inputs, outputs=outputs, name="Fully_Connected_" + hparams.output_type)


## CONVOLUTIONAL
def cnn_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    filters=[64,64,64] # each entry becomes a Conv1D layer (# entries = # conv layers)
    kernels=[3,3,3] # corresponding kernel size for Conv1d layer above
    pool_size=2 # max pooling window
    stride=2
    padding="same" # "same" keeps output size = input size with padding
    dropout=hparams.dropout
    kernel_initializer=hparams.kernel_initializer
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    if hparams.output_type == 'mc':
        out_activation="softmax"
    elif hparams.output_type == 'ml':
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for (filter, kernel) in zip(filters, kernels):
         x = keras.layers.Conv1D(activation="relu", filters=filter, kernel_size=kernel,
                                 padding=padding, kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=kernel_initializer)(x)
         x = keras.layers.MaxPooling1D(pool_size=pool_size, strides=stride, padding=padding)(x)
    flat = keras.layers.Flatten()(x) #(conv3)
    drop = keras.layers.Dropout(dropout)(flat)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(drop)
    else: # multihead classifier (2 outputs)
        ## VERSION 1
        # output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(drop)
        # output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(drop)
        # outputs=[output_class, output_attr]
        ## VERSION 2
        output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(drop)
        concat=keras.layers.Concatenate()([x,output_attr]) #use attribute output to try and improve class output
        output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(concat)
        outputs=[output_class, output_attr]

    return keras.Model(inputs=inputs, outputs=outputs, name='CNN_' + hparams.output_type)


## FULLY CONVOLUTIONAL
def fcn_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    # filters=[128,256,128] # each entry becomes a Conv1D layer (# entries = # conv layers)
    filters=[64,128,256] # each entry becomes a Conv1D layer (# entries = # conv layers)
    kernels=[8,5,3] # corresponding kernel size for Conv1d layer above
    padding="same" # "same" keeps output size = input size with padding
    kernel_initializer=hparams.kernel_initializer
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    if hparams.output_type == 'mc':
        out_activation="softmax"
    elif hparams.output_type == 'ml':
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for (filter, kernel) in zip(filters, kernels):
         x = keras.layers.Conv1D(activation="relu", filters=filter, kernel_size=kernel,
                                 padding=padding, kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=kernel_initializer)(x)
         x = keras.layers.BatchNormalization()(x)
         x = keras.layers.ReLU()(x)
    gap = keras.layers.GlobalAveragePooling1D()(x)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(gap)
    else: # multihead classifier (2 outputs)
        ## VERSION 1
        # output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(gap)
        # output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(gap)
        # outputs=[output_class, output_attr]
        ## VERSION 2
        output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(gap)
        concat=keras.layers.Concatenate()([x,output_attr]) #use attribute output to try and improve class output
        output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(concat)
        outputs=[output_class, output_attr]

    return keras.Model(inputs=inputs, outputs=outputs, name='FCN_' + hparams.output_type)


## RESNET (with CONSTANT filter/kernel in each residual unit layer)
def resnet_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    num_res_layers=3 # number of layers in each residual unit (RU)
    filters=[64,64,64,128,128,128,256,256,256] # each entry becomes an RU (# entries = # RU)
    kernels=[7,5,3,7,5,3,7,5,3] # corresponding kernel size for each RU above
    if hparams.output_type == 'mc':
        out_activation="softmax"
    elif hparams.output_type == 'ml':
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for (filter, kernel) in zip(filters, kernels):
         x=res_unit(hparams, num_res_layers, filter, kernel, x)
    gap = keras.layers.GlobalAveragePooling1D()(x)
    if hparams.output_type!='mh':
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(gap)
    else: # multihead classifier (2 outputs)
        ## VERSION 1
        # output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(gap)
        # output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(gap)
        # outputs=[output_class, output_attr]
        ## VERSION 2
        output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(gap)
        concat=keras.layers.Concatenate()([x,output_attr]) #use attribute output to try and improve class output
        output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(concat)
        outputs=[output_class, output_attr]

    return keras.Model(inputs=inputs, outputs=outputs, name='ResNet_' + hparams.output_type)

def res_unit(
        hparams,
        num_res_layers,
        filter,
        kernel,
        input
        ):
    ## PARAMETERS
    padding="same"
    kernel_initializer=hparams.kernel_initializer
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    ## RESIDUAL UNIT
    x = input
    skip = keras.layers.Conv1D(activation="relu", filters=filter, kernel_size=1,
                                 padding=padding, kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=kernel_initializer)(input)
    count=1
    for _ in range(num_res_layers):
         x = keras.layers.Conv1D(activation="relu", filters=filter, kernel_size=kernel,
                                 padding=padding, kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=kernel_initializer)(x)
         x = keras.layers.BatchNormalization()(x)
         if count==num_res_layers: #on last filter add original input (skip formated=Conv1D, kernel=1) before Relu
            print("KERNEL (1D): ",kernel)
            print("skip: ",skip.shape)
            print("x original: ",x.shape)
            x=x+skip
            print("x = x + skip: ",x.shape,'\n')
         x = keras.layers.ReLU()(x)
         count+=1
    output = x

    return output



def lstm_model(
        hparams,
        input_shape,
        output_shape,
        ):
    ## PARAMETERS
    units=[40,20] # each entry becomes an LSTM layer
    kernel_initializer=hparams.kernel_initializer
    dropout=hparams.dropout
    if hparams.kernel_regularizer == "none":
        kernel_regularizer=None
    else:
        kernel_regularizer=hparams.kernel_regularizer
    if hparams.output_type == 'mc':
        out_activation="softmax"
    elif hparams.output_type == 'ml':
        out_activation="sigmoid"
    elif hparams.output_type == 'mh': # multihead = mc & ml
        out_activation=["softmax","sigmoid"]
    ## MODEL
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    count=1
    return_sequences=True # must return sequences when linking LSTM layers together
    for unit in units:
        if count == len(units): # on last LSTM unit, determine output: sequence or vector
            if hparams.output_length == "seq":
                return_sequences=True # sequence output
            else:
                return_sequences=False # vector output
        x = keras.layers.LSTM(unit,return_sequences=return_sequences, dropout=dropout,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer)(x)
        count+=1
    if hparams.output_type!='mh':
        # time distributed (dense) layer improves by 1%, but messes up "vector" output size
        # outputs = keras.layers.TimeDistributed(keras.layers.Dense(output_shape, activation=out_activation))(x)
        outputs = keras.layers.Dense(output_shape, activation=out_activation)(x)
    else: # multihead classifier (2 outputs)
        ## VERSION 1
        # output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(x)
        # output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(x)
        # outputs=[output_class, output_attr]
        ## VERSION 2
        output_attr=keras.layers.Dense(output_shape[1], activation=out_activation[1], name='output_attr')(x)
        concat=keras.layers.Concatenate()([x,output_attr])
        output_class=keras.layers.Dense(output_shape[0], activation=out_activation[0], name='output_class')(concat)
        outputs=[output_class, output_attr]

    return keras.Model(inputs=inputs, outputs=outputs, name='LSTM_' + hparams.output_type)