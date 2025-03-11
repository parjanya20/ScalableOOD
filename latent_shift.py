import numpy as np 


def estimate_test_pz(z_tr_true,z_tr_pred,z_te_pred):
    n_classes = len(np.unique(z_tr_true))
    pz_z_hat = np.zeros((n_classes,n_classes))
    qz_hat = np.zeros(n_classes)
    for i in range(n_classes):
        for j in range(n_classes):
            pz_z_hat[i,j] = np.mean((z_tr_true == j)&(z_tr_pred == i))
    for i in range(n_classes):
        qz_hat[i] = np.mean(z_te_pred == i)
    pz = np.sum(pz_z_hat,axis=0)
    qz = np.dot(np.diag(pz),np.dot(np.linalg.inv(pz_z_hat+1e-6*np.eye(n_classes)),qz_hat))
    qz = (qz - np.min(qz)+1e-6)
    qz = qz / np.sum(qz)
    return qz, pz

def recalibrate(p_zx,qz,pz):
    qz = qz/pz
    return ((p_zx*qz).T/np.sum(p_zx*qz,axis=1)).T

def sample_y(p_y_given_x):
    num_samples = p_y_given_x.shape[0]
    y_values = np.arange(p_y_given_x.shape[1])
    sampled_y = np.array([np.random.choice(y_values, p=probs) for probs in p_y_given_x])
    return sampled_y