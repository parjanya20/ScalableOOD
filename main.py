from src import get_latent, latent_shift, moe, train_model
import numpy as np
def run(X, y, s1, e1, t, verbose = 0, epochs = 5, var_reg=0.3):
    #learn the encoder-decoder model to get P(Z|X)
    model = train_model.get_latent_model(X, s1, 10, 'model', learning_rate=1e-4, epochs=epochs, verbose=verbose, var_reg=var_reg, validation_split=0.1, batch_size=1024)
    
    #Get the latent predictions for train and test
    p_tr = get_latent.pzx(X[t==0], model, False)


    #Learn the moe model on train
    model_mix = moe.train_mixture_model(X[t==0], p_tr, y[t==0])

    #Now use tes set and get latent predictions on test
    p_te = get_latent.pzx(X[t==1], model, False)
    p_tr_hard = get_latent.pzx(X[t==0], model, True)
    p_te_hard = get_latent.pzx(X[t==1], model, True)

    #Calculate shift
    z_tr = latent_shift.sample_y(p_tr)

    pz_te, pz_tr = latent_shift.estimate_test_pz(z_tr, p_tr_hard, p_te_hard)
    #Get calibrated predictions for test
    p_te1 = latent_shift.recalibrate(p_te, pz_te, pz_tr)

    #Evaluate the model
    _, acc = model_mix.evaluate([X[t==1], p_te1], y[t==1])

    #Get predictions
    y_te = model_mix.predict([X[t==1], p_te1])
    return y_te, acc
