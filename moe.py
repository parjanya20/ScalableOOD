import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def train_mixture_model(X_tr, w_tr, y_tr, epochs=10, batch_size=32, hidden_layers=[128], 
                       validation_split=0.1, patience=3, random_state=42):
    num_components = w_tr.shape[1]
    num_features = X_tr.shape[1]
    num_classes = len(np.unique(y_tr))
    
    if num_classes == 2:
        activation = 'sigmoid'
        y_tr = y_tr.reshape(-1, 1)
    else:
        activation = 'softmax'
        y_tr = to_categorical(y_tr)
        
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_tr, y_tr, w_tr, test_size=validation_split, random_state=random_state)
    
    inputs = Input(shape=(num_features,), name='input_layer')
    weights_input = Input(shape=(num_components,), name='weights_input')
    
    x = inputs
    for units in hidden_layers:
        x = Dense(units, activation='relu')(x)
    
    components = []
    for i in range(num_components):
        if num_classes == 2:
            output = Dense(1, activation='sigmoid')(x)
        else:
            output = Dense(num_classes, activation='softmax')(x)
        components.append(output)
    
    if num_classes == 2:
        combined_output = tf.reduce_sum([components[i] * weights_input[:, i:i+1] 
                                       for i in range(num_components)], axis=0)
    else:
        combined_output = tf.reduce_sum([components[i] * tf.expand_dims(weights_input[:, i], -1) 
                                       for i in range(num_components)], axis=0)
    
    model = Model(inputs=[inputs, weights_input], outputs=combined_output)
    
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, 
                                 verbose=1, mode='min', restore_best_weights=True)
    
    model.fit([X_train, w_train], y_train, 
              validation_data=([X_val, w_val], y_val),
              epochs=epochs, batch_size=batch_size, 
              verbose=0, callbacks=[early_stopping])
    
    return model


