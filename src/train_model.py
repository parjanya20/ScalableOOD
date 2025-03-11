import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.initializers import Constant


class ClipConstraint(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, weights):
        w = tf.clip_by_value(weights, self.min_value, self.max_value)
        return w / tf.reduce_sum(w, axis=1, keepdims=True)
    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

class VarianceRegularizer(Regularizer):    
    def __init__(self, factor=0.01):
        self.factor = factor
    
    def __call__(self, x):
        variances = tf.math.reduce_variance(x, axis=1)
        max_variance = tf.reduce_max(variances)
        return self.factor * max_variance
    
    def get_config(self):
        return {'factor': self.factor}

    

def get_trained_model(X, s, n_z, model_name, learning_rate=1e-4, epochs=20, verbose=0, var_reg=0.3, validation_split=0.1, batch_size=1024):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dense(s.shape[1], activation='linear'),
        BatchNormalization(),
        Dense(n_z, activation='softmax', name='target_layer'),
        Dense(s.shape[1], activation='linear', use_bias=False, kernel_initializer=Constant(1/n_z), kernel_constraint=ClipConstraint(0, 1), kernel_regularizer=VarianceRegularizer(factor=var_reg)),
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model_checkpoint_path = model_name
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_best_only=True,
        monitor='loss',
        mode='min',
        verbose=verbose
    )
    print(model.get_weights()[-1])
    history = model.fit(
        X,
        s,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[model_checkpoint_callback]
    )
    best_model = load_model(model_checkpoint_path, custom_objects={'ClipConstraint': ClipConstraint, 'VarianceRegularizer': VarianceRegularizer})
    
    val_loss = min(history.history['val_loss']) if 'val_loss' in history.history else None
    
    return best_model, val_loss


def get_latent_model(X, s, max_n_z, model_name_prefix, learning_rate=1e-4, epochs=20, verbose=0, var_reg=0.3, validation_split=0.1, batch_size=1024):
    best_val_loss = float('inf')
    best_model = None
    
    for n_z in range(1, max_n_z + 1):
        model_name = f"{model_name_prefix}_nz{n_z}"
        
        current_model, current_val_loss = get_trained_model(
            X, s, n_z, model_name, 
            learning_rate=learning_rate,
            epochs=epochs,
            verbose=verbose,
            var_reg=var_reg,
            validation_split=validation_split,
            batch_size=batch_size
        )
        
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model = current_model
        else:
            break
    
    return best_model
