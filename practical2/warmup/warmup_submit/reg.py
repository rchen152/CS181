import StringIO
import csv
import numpy as np

# calculate Phi
def get_trans(basis_fns, data):
    fst_fn = basis_fns[0]
    trans = [map(fst_fn, data)]
    for fn in (basis_fns[1:]):
        trans = np.concatenate((trans,[map(fn, data[:,0])]),axis = 0)
    return (trans, np.transpose(trans))
    
# do MLE
def freq_reg(basis_fns, train_data, test_data, LAMBDA = 0):
    # get the weights
    num_pts = len(train_data[:,0])  
    trans, phi = get_trans(basis_fns, train_data)
    temp_pseudo = LAMBDA * np.identity(len(basis_fns))+np.dot(trans, phi)
    pseudo_inv = np.dot(np.linalg.inv(temp_pseudo),trans)
    coeffs = np.dot(pseudo_inv, train_data[:,1])

    # make a prediction for a single data point
    def product(x):
        out = 0.
        for i in range(len(basis_fns)):
            out += coeffs[i]*basis_fns[i](x)
        return out

    # get the standard deviation
    var_vect = train_data[:,1] - np.dot(phi, np.transpose(coeffs))
    mle_variance = np.sum(var_vect * var_vect,axis = 0)/num_pts
    mle_std = np.sqrt(mle_variance)

    # compute predictions +/- one standard deviation
    preds = []
    for i in range(test_data.shape[0]):
        pred = product(test_data[i])
        preds.append((pred, pred + mle_std, pred - mle_std))
    return preds

# do Bayesian linear regression
def bayes_reg(basis_fns, train_data, test_data, BETA = 0.05, ALPHA = 1.):
    # calculate updated mean and variance for coefficients
    basis_dim = len(basis_fns)
    init_mean = np.zeros((basis_dim))
    init_var = ALPHA * np.identity(basis_dim)
    trans, phi = get_trans(basis_fns, train_data)
    coeffs_updated_var = np.linalg.inv(np.linalg.inv(init_var) +
                                       BETA * (np.dot(trans,phi)))
    mean_factor = np.dot(np.linalg.inv(coeffs_updated_var),
                         init_mean) + BETA * (np.dot(trans,train_data[:,1]))
    coeffs_updated_mean = np.dot(coeffs_updated_var,mean_factor)

    # get the predictive distribution
    def pred_dist(x):
        pred_list = map(lambda t: t(x), basis_fns)
        pred_vect = np.array(pred_list)
        var = (np.dot(np.dot(np.transpose(pred_vect),coeffs_updated_var),
                      pred_vect))+ 1/BETA
        mean = np.dot(np.transpose(coeffs_updated_mean),pred_vect)
        return [mean,var]

    # make predictions
    preds = []
    for i in range(test_data.shape[0]):
        pred = pred_dist(test_data[i])
        preds.append((pred[0], pred[0]+pred[1], pred[0]-pred[1]))
    return preds
