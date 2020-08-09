"""
Code developed by:

Ioana Ciucă, 3rd year PhD student (MSSL/UCL)

using the brilliant resources of Thomas Wiecki and Payel Das.

Date of writting the data: approx Sept 2018
Current date: Febr 2020
"""

import numpy as np
from scipy import stats
import pymc3 as pm
import theano
import theano.tensor as tt
import pandas as pd
import os
from pymc3 import floatX as floatX_pymc3
from pymc3.theanof import set_tt_rng, MRG_RandomStreams
from warnings import filterwarnings
import pickle
from collections import defaultdict

filterwarnings('ignore')
layer_names = ['w_in_1_grp', 'w_1_in_2_grp', 'w_2_out_grp', 'b_in_1_grp', 'b_1_in_2_grp', 'b_2_out_grp']
np.random.seed(1234)
floatX = theano.config.floatX

"""
The parameters for the pymc NUTS method:

ntune = the tuning parameter
nchains = no of chains
ncores = np of cores
nsamples = how many samples to draw from the posterior
"""
ninit = 200000
nchains_hier = 4
ncores_hier = 4
nsamples_hier = 100
nchains = 4
ncores = 1
nsamples = 500

n_hidden_1 = 16
n_hidden_2 = 16

cache_file_hier = './HBNN_output/hier_logAge.trace'
cache_file_bnn = './HBNN_output/bnn_logAge.trace'
cache_file_samples = './HBNN_output/HBNN_samples_logAge.csv'

def get_data(input_data):

    # Assuming your data is in a fits table you want to make it into a pandas dataframe (maybe to use if for Keras)
    # Assumes here you've done most of the data handling in a different code
    train_dataset = pd.read_csv(input_data)
    #print("The length of the data sample is:", len(train_dataset))

    loggTrain = np.array(train_dataset['LOGG_NORM']).astype(floatX)
    teffTrain = np.array(train_dataset['TEFF_NORM']).astype(floatX)
    alphamTrain = np.array(train_dataset['ALPHA_M_NORM']).astype(floatX)
    mhTrain = np.array(train_dataset['M_H_NORM']).astype(floatX)
    cfeTrain = np.array(train_dataset['C_FE_NORM']).astype(floatX)
    nfeTrain = np.array(train_dataset['N_FE_NORM']).astype(floatX)
    gmagTrain = np.array(train_dataset['G_NORM']).astype(floatX)
    bpmagTrain = np.array(train_dataset['BP_NORM']).astype(floatX)
    rpmagTrain = np.array(train_dataset['RP_NORM']).astype(floatX)
    jmagTrain = np.array(train_dataset['J_NORM']).astype(floatX)
    hmagTrain = np.array(train_dataset['H_NORM']).astype(floatX)
    kmagTrain = np.array(train_dataset['K_NORM']).astype(floatX)

    loggTrain_err = np.array(train_dataset['LOGG_ERR_NORM']).astype(floatX)
    teffTrain_err = np.array(train_dataset['TEFF_ERR_NORM']).astype(floatX)
    alphamTrain_err = np.array(train_dataset['ALPHA_M_ERR_NORM']).astype(floatX)
    mhTrain_err = np.array(train_dataset['M_H_ERR_NORM']).astype(floatX)
    cfeTrain_err = np.array(train_dataset['C_FE_ERR_NORM']).astype(floatX)
    nfeTrain_err = np.array(train_dataset['N_FE_ERR_NORM']).astype(floatX)
    gmagTrain_err = np.array(train_dataset['G_ERR_NORM']).astype(floatX)
    bpmagTrain_err = np.array(train_dataset['BP_ERR_NORM']).astype(floatX)
    rpmagTrain_err = np.array(train_dataset['RP_ERR_NORM']).astype(floatX)
    jmagTrain_err= np.array(train_dataset['J_ERR_NORM']).astype(floatX)
    hmagTrain_err = np.array(train_dataset['H_ERR_NORM']).astype(floatX)
    kmagTrain_err = np.array(train_dataset['K_ERR_NORM']).astype(floatX)

    logAgeTrain = np.array(train_dataset['logAge']).astype(floatX)
    logAgeTrain_err =  np.array(train_dataset['logAgeErr']).astype(floatX)

    inputsTrain = np.column_stack((loggTrain, teffTrain, alphamTrain, mhTrain, cfeTrain, nfeTrain, \
                              gmagTrain, bpmagTrain, rpmagTrain, jmagTrain, hmagTrain, kmagTrain))
    errInputsTrain = np.column_stack((loggTrain_err, teffTrain_err, alphamTrain_err, mhTrain_err, cfeTrain_err, nfeTrain_err, \
                                  gmagTrain_err, bpmagTrain_err, rpmagTrain_err, jmagTrain_err, \
                                  hmagTrain_err, kmagTrain_err))

    targetsTrain    = np.array(logAgeTrain)
    errTargetsTrain = np.array(logAgeTrain_err)

    # Make the arrays a floatX array
    inputsTrain = inputsTrain.astype(floatX)
    errInputsTrain = errInputsTrain.astype(floatX)

    targetsTrain = targetsTrain.astype(floatX)
    errTargetsTrain = errTargetsTrain.astype(floatX)

    # First rule of a machine learning project: standardize the data. It has been standardized in dataAugmentation.py
    # Data Processing.ipynb -> Data Augmentation.ipynb -> HBNN_train_Model
    return inputsTrain, errInputsTrain, targetsTrain, errTargetsTrain

datA = './HBNN_train_data/AllTrainedNormAugShuffled_A.csv'
datB = './HBNN_train_data/AllTrainedNormAugShuffled_B.csv'
datC = './HBNN_train_data/AllTrainedNormAugShuffled_C.csv'
datD = './HBNN_train_data/AllTrainedNormAugShuffled_D.csv'
datE = './HBNN_train_data/AllTrainedNormAugShuffled_E.csv'

# Only use two small datasets for testing purposes
inputsTrain_A, errInputsTrain_A, targetsTrain_A, errTargetsTrain_A = get_data(datA)
inputsTrain_B, errInputsTrain_B, targetsTrain_B, errTargetsTrain_B = get_data(datB)
inputsTrain_C, errInputsTrain_C, targetsTrain_C, errTargetsTrain_C = get_data(datC)
inputsTrain_D, errInputsTrain_D, targetsTrain_D, errTargetsTrain_D = get_data(datD)
inputsTrain_E, errInputsTrain_E, targetsTrain_E, errTargetsTrain_E = get_data(datE)

# Only use two small datasets for testing purposes
ntrain = len(inputsTrain_A)
print("Number of training samples: ", ntrain)

XsTrain = np.stack([inputsTrain_A, inputsTrain_B, inputsTrain_C, inputsTrain_D])
YsTrain = np.stack([targetsTrain_A, targetsTrain_B, targetsTrain_C, targetsTrain_D])

errXsTrain = np.stack([errInputsTrain_A, errInputsTrain_B, errInputsTrain_C, errInputsTrain_D])
errYsTrain = np.stack([errTargetsTrain_A, errTargetsTrain_B, errTargetsTrain_C, errTargetsTrain_D])

XsTrain = theano.shared(XsTrain)
YsTrain = theano.shared(YsTrain)
errXsTrain = theano.shared(errXsTrain)
errYsTrain = theano.shared(errYsTrain)

n_grps = XsTrain.shape[0].eval()
n_data = XsTrain.shape[2].eval()
ntargets = 1

def hierarchical_NN(annInput, errAnnInput, annTarget, errAnnTarget):

    # Initialize random weights between each layer
    init_1 = floatX_pymc3(np.random.randn(n_data, n_hidden_1))
    init_1_in_2 = floatX_pymc3(np.random.randn(n_hidden_1, n_hidden_2))
    init_out = floatX_pymc3(np.random.randn(n_hidden_2))

    # Initialize random biases between each layer
    bias_init_1  = np.random.randn(n_hidden_1)
    bias_init_2 = np.random.randn(n_hidden_2)
    bias_out = np.random.randn(ntargets)

    with pm.Model() as neural_network:

        #LAYER 1
        # Group mean distribution for input to hidden layer
        Xs_true = pm.Normal('xtrue', mu=0., sd=20., shape = annInput.shape.eval(), testval=annInput.eval())

        weights_in_1_grp = pm.Normal('w_in_1_grp', mu=0., sd=1.,
                                 shape=(n_data, n_hidden_1),
                                 testval=init_1)
        # Group standard-deviation
        weights_in_1_grp_sd = pm.HalfNormal('w_in_1_grp_sd', sd=1.)

        #biases in 1 now
        biases_in_1_grp = pm.Normal('b_in_1_grp', mu=0., sd=1., shape=(n_hidden_1), testval = bias_init_1)
        biases_in_1_grp_sd = pm.HalfNormal('b_in_1_grp_sd', sd=1.)

        #LAYER 2
        weights_1_in_2_grp = pm.Normal('w_1_in_2_grp', mu=0., sd=1.,
                                      shape=(n_hidden_1, n_hidden_2),
                                      testval=init_1_in_2)

        weights_1_in_2_grp_sd = pm.HalfNormal('w_1_in_2_grp_sd', sd=1.)

        biases_1_in_2_grp = pm.Normal('b_1_in_2_grp', mu=0., sd=1.,
                                     shape=(n_hidden_2),
                                     testval=bias_init_2)
        biases_1_in_2_grp_sd = pm.HalfNormal('b_1_in_2_grp_sd', sd=1.)

        # Group mean distribution from hidden layer to output
        weights_2_out_grp = pm.Normal('w_2_out_grp', mu=0., sd=1.,
                                  shape=(n_hidden_2,),
                                  testval=init_out)

        weights_2_out_grp_sd = pm.HalfNormal('w_2_out_grp_sd', sd=1.)

        biases_2_out_grp = pm.Normal('b_2_out_grp', mu=0., sd=1., shape=(ntargets,), testval=bias_out)
        biases_2_out_grp_sd = pm.HalfNormal('b_2_out_grp_sd', sd=1.)

        # Separate weights for each different model, just add a 3rd dimension
        # of weights
        weights_in_1_raw = pm.Normal('w_in_1', shape=(n_grps, n_data, n_hidden_1))

        # Non-centered specification of hierarchical model
        weights_in_1 = weights_in_1_raw * weights_in_1_grp_sd + weights_in_1_grp

        biases_in_1_raw = pm.Normal('b_in_1', shape = (n_grps, 1, n_hidden_1,))
        biases_in_1 = biases_in_1_raw * biases_in_1_grp_sd + biases_in_1_grp

        #LAYER 2 BROADCASTING
        weights_1_in_2_raw = pm.Normal('w_1_in_2', shape=(n_grps, n_hidden_1, n_hidden_2))
        weights_1_in_2 = weights_1_in_2_raw * weights_1_in_2_grp_sd + weights_1_in_2_grp

        biases_1_in_2_raw = pm.Normal('b_1_in_2', shape=(n_grps, 1, n_hidden_2,)) # may not be right
        biases_1_in_2 = biases_1_in_2_raw * biases_1_in_2_grp_sd + biases_1_in_2_grp

        # OUTLAYER
        weights_2_out_raw = pm.Normal('w_2_out',
                                      shape=(n_grps, n_hidden_2))
        weights_2_out = weights_2_out_raw * weights_2_out_grp_sd + weights_2_out_grp

        biases_2_out_raw = pm.Normal('b_2_out', shape=(n_grps, ntargets))
        biases_2_out = biases_2_out_raw * biases_2_out_grp_sd + biases_2_out_grp

        # Build neural-network using relu activation function
        # tt.batched_dot just calls .dot along an axis
        act_1 = tt.nnet.relu(tt.batched_dot(Xs_true, weights_in_1) + biases_in_1)
        act_2 = tt.nnet.relu(tt.batched_dot(act_1, weights_1_in_2) + biases_1_in_2)

        # Debug 1: get rid of he biases_in_2; see if it works
        # Debug 2: will use relu
        # Debug 3: look into the uncertainties
        # Debug 4: look at changing the prior on the biases
        act_out = tt.batched_dot(act_2, weights_2_out) + biases_2_out

        pred = pm.Deterministic('pred', act_out)
        likelihood_x = pm.Normal('x', mu=Xs_true, sd=errAnnInput, observed=annInput)

        out = pm.Normal('out', mu=pred, observed=annTarget, sd = errAnnTarget)

    return neural_network

neural_network = hierarchical_NN(XsTrain, errXsTrain, YsTrain, errYsTrain)

print("Starting the training of the hierarchical BNN...")

if not os.path.exists(cache_file_hier):

    with neural_network:
        #fit model
        trace_hier = pm.sample(draws=nsamples_hier, init='advi+adapt_diag', n_init=ninit, tune=ninit//2, chains=nchains_hier, cores=ncores_hier,
                     nuts_kwargs={'target_accept': 0.90}, discard_tuned_samples=True, compute_convergence_checks=True,
                     progressbar=False)
    pm.save_trace(trace_hier, directory=cache_file_hier)
else:
    trace_hier = pm.load_trace(cache_file_hier, model=neural_network)

print("Done...")

if not os.path.exists(cache_file_samples):

    samples_tmp = defaultdict(list)
    samples = {}

    for layer_name in layer_names:
        for mu, sd in zip(trace_hier.get_values(layer_name, burn=nsamples_hier//2, combine=True),
                          trace_hier.get_values(layer_name+'_sd', burn=nsamples_hier//2, combine=True)):
            for _ in range(50): # not sure why the `size` kwarg doesn't work
                samples_tmp[layer_name].append(stats.norm(mu, sd).rvs())
        samples[layer_name] = np.asarray(samples_tmp[layer_name])
        with open(cache_file_samples, 'wb') as f:
            pickle.dump(samples, f)

else:
    with open(cache_file_samples, 'rb') as f:
        samples = pickle.load(f)

def flat_bnn(annInput, errAnnInput, annTarget, errAnnTarget):

    annInput = theano.shared(annInput)
    errAnnInput = theano.shared(errAnnInput)
    annTarget = theano.shared(annTarget)
    errAnnTarget = theano.shared(errAnnTarget)

    n_samples = samples['w_in_1_grp'].shape[0]

    prior_1_mu = samples['w_in_1_grp'].mean(axis=0)
    bias_1_mu = samples['b_in_1_grp'].mean(axis=0)
    prior_1_cov = np.cov(samples['w_in_1_grp'].reshape((n_samples, -1)).T)
    bias_1_cov = np.cov(samples['b_in_1_grp'].reshape((n_samples, -1)).T)

    prior_1_in_2_mu = samples['w_1_in_2_grp'].mean(axis=0)
    bias_1_in_2_mu = samples['b_1_in_2_grp'].mean(axis=0)
    prior_1_in_2_cov = np.cov(samples['w_1_in_2_grp'].reshape((n_samples, -1)).T)
    bias_1_in_2_cov = np.cov(samples['b_1_in_2_grp'].reshape((n_samples, -1)).T)

    prior_out_mu = samples['w_2_out_grp'].mean(axis=0)
    bias_out_mu = samples['b_2_out_grp'].mean(axis=0)
    prior_out_cov = np.cov(samples['w_2_out_grp'].reshape((n_samples, -1)).T)
    bias_out_cov = np.cov(samples['b_2_out_grp'].reshape((n_samples, -1)).T)

    with pm.Model() as flat_neural_network:

        Xs_true = pm.Normal('xtrue', mu=0., sd=20., shape=annInput.shape.eval(), testval=annInput.eval())

        # In order to model the correlation structure between the 2D weights,
        # we flatten them first. Now here we have to reshape to give them their
        # original 2D shape.
        weights_in_1 = (pm.MvNormal('w_in_1', prior_1_mu.flatten(),
                                   cov=prior_1_cov,
                                   shape=prior_1_cov.shape[0]).reshape((n_data, n_hidden_1)))
        print(weights_in_1.tag.test_value.shape)

        bias_in_1 = (pm.MvNormal('bias_in_1', bias_1_mu.flatten(),
                                 cov=bias_1_cov,
                                 shape=bias_1_cov.shape[0]).reshape((n_hidden_1,)))
        print(weights_in_1.tag.test_value.shape)

        # Layer 2
        weights_1_in_2 = (pm.MvNormal('w_1_in_2', prior_1_in_2_mu.flatten(),
                                   cov=prior_1_in_2_cov,
                                   shape=prior_1_in_2_cov.shape[0]).reshape((n_hidden_1, n_hidden_2)))

        print(weights_1_in_2.tag.test_value.shape)

        bias_1_in_2 = (pm.MvNormal('bias_1_in_2', bias_1_in_2_mu.flatten(),
                                   cov=bias_1_in_2_cov,
                                   shape=bias_1_in_2_cov.shape[0]).reshape((n_hidden_2,)))

        print(bias_1_in_2.tag.test_value.shape)

        # Weights from hidden layer to output
        weights_2_out = (pm.MvNormal('w_2_out', prior_out_mu.flatten(),
                                    cov=prior_out_cov,
                                    shape=prior_out_cov.shape[0]).reshape((n_hidden_2,)))

        bias_2_out = (pm.Normal('bias_2_out', mu=bias_out_mu.flatten(), sd=bias_out_cov, shape=(ntargets)).reshape((ntargets,)))

        # Build neural-network using relu (informed by hyperas on 2 layers) activation function
        act_1 =  tt.nnet.relu(tt.dot(Xs_true, weights_in_1) + bias_in_1)
        act_2 = tt.nnet.relu(tt.dot(act_1, weights_1_in_2) + bias_1_in_2)

        act_out = tt.dot(act_2, weights_2_out) + bias_2_out
        pred = pm.Deterministic('pred', act_out)
        likelihood_x = pm.Normal('x', mu=Xs_true, sd=errAnnInput, observed=annInput)

        out = pm.Normal('out', pred, observed=annTarget, sd=errAnnTarget)

    return flat_neural_network

XsTrainBNN =  inputsTrain_E
YsTrainBNN =  targetsTrain_E
errXsTrainBNN = errInputsTrain_E
errYsTrainBNN =  errTargetsTrain_E

bnn_neural_network = flat_bnn(XsTrainBNN, errXsTrainBNN, YsTrainBNN, errYsTrainBNN)

print("Starting the training of the final bnn....")

if not os.path.exists(cache_file_bnn):

    with bnn_neural_network:
        #fit model
        trace_bnn = pm.sample(draws=nsamples, init='advi+adapt_diag', n_init=ninit, tune=ninit//2, chains=nchains, cores=ncores,
                     nuts_kwargs={'target_accept': 0.90}, discard_tuned_samples=True, compute_convergence_checks=True,
                     progressbar=False)
    pm.save_trace(trace_bnn, directory=cache_file_bnn)
else:
    trace_bnn = pm.load_trace(cache_file_bnn, model=bnn_neural_network)

print("Done...")
