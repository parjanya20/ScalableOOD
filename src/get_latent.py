import numpy as np
from tensorflow.keras.models import Model

def pzx(X,best_model,arg_max=True):
    softmax_output_model = Model(inputs=best_model.input, outputs=best_model.layers[-2].output)
    p = softmax_output_model.predict(X)
    if(arg_max):
        p = np.argmax(p,axis=1)
    return p

def pzxs(X,s,best_model,arg_max=True):
    softmax_output_model = Model(inputs=best_model.input, outputs=best_model.layers[-2].output)
    p = softmax_output_model.predict(X)
    psz_matrix = best_model.get_weights()[-1]
    p2 = p*psz_matrix[:,s.astype(int)].T
    p2 = p2/np.reshape(np.sum(p2,axis=1),(np.sum(p2,axis=1).shape[0],1))

    if(arg_max):
        p2 = np.argmax(p2,axis=1)
    return p2
