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
from collections import defaultdict
import pickle

filterwarnings('ignore')

np.random.seed(1234)
floatX = theano.config.floatX

ninit = 200000
nchains = 4
ncores = 4
nsamples = 500

cache_file_bnn = './BNN_output/BNN_logAge_AllTrainedNormAandS_modelA.trace'

def get_data(input_data):

    # Assuming your data is in a fits table you want to make it into a pandas dataframe (maybe to use if for Keras)
    # Assumes here you've done most of the data handling in a different code
    train_dataset = pd.read_csv(input_data)
    print("The length of the data sample is:", len(train_dataset))

    # Here is when you choose the sample size for your data etc

    #train_stats = train_dataset.describe()
    #train_head = train_dataset.head(2)
    #print("The stats of the training set", train_stats)
    #print("The head of the training set", train_head)

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
    # Note: might need the mu and standard deviation but for now I think we are fine

    return inputsTrain, errInputsTrain, targetsTrain, errTargetsTrain

data = './HBNN_train_data/AllTrainedNormAugShuffled.csv'
# Only use two small datasets for testing purposes
inputsTrain, errInputsTrain, targetsTrain, errTargetsTrain = get_data(data)

# Only use two small datasets for testing purposes
ntrain = len(inputsTrain)
print("Number of training samples: ", ntrain)

XsTrain = inputsTrain
YsTrain = targetsTrain

errXsTrain = errInputsTrain
errYsTrain = errTargetsTrain

ntargets = 1
n_hidden = 16

def construct_nn(annInput, errAnnInput, annTarget, errAnnTarget):

    # Initialize random weights between each layer
    init_1 = np.random.randn(XsTrain.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)

    bias_init_1  = np.random.randn(n_hidden)
    bias_init_2 = np.random.randn(n_hidden)
    bias_out = np.random.randn(ntargets)

    with pm.Model() as neural_network:

        Xs_true = pm.Normal('xtrue', mu=0., sd=20., shape = annInput.shape.eval(), testval=annInput.eval())
        weights_in_1 = pm.Normal('w_in_1', 0., sd=1.,
                                 shape=(XsTrain.shape[1], n_hidden),
                                 testval=init_1)
        bias_in_1 = pm.Normal('b_in_1', 0., sd=1., shape=(n_hidden), testval = bias_init_1)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', mu=0., sd=1.,
                                shape=(n_hidden, n_hidden),
                                testval=init_2)
        bias_1_2 = pm.Normal('b_1_2', mu=0., sd=1.,
                                shape=(n_hidden,),
                                testval=bias_init_2)

        # Weights from hidden layer to output
        weights_2_out = pm.Normal('w_2_out', mu=0., sd=1.,
                                  shape=(n_hidden,),
                                  testval=init_out)
        bias_2_out = pm.Normal('b_2_out', mu=0., sd=1., shape=(ntargets), testval=bias_out)

        # Build neural-networtt.nnet.relu activation function
        act_1 = tt.nnet.relu(pm.math.dot(Xs_true, weights_in_1) + bias_in_1)
        act_2 = tt.nnet.relu(pm.math.dot(act_1, weights_1_2) + bias_1_2)
        act_out = pm.math.dot(act_2, weights_2_out) +  bias_2_out

        likelihood_x = pm.Normal('x', mu=Xs_true, sd=errAnnInput, observed=annInput)

        # Binary classification -> Bernoulli likelihood
        out = pm.Normal('out', act_out, observed=annTarget, sd=errAnnTarget,
                         total_size=YsTrain.shape[0])

    return neural_network

annInput = theano.shared(XsTrain)
annTarget = theano.shared(YsTrain)
errAnnInput = theano.shared(errXsTrain)
errAnnTarget = theano.shared(errYsTrain)

neural_network = construct_nn(annInput, errAnnInput, annTarget, errAnnTarget)

print("Starting the training of the BNN...")

if not os.path.exists(cache_file_bnn):

    with neural_network:
        #fit model
        trace = pm.sample(draws=nsamples, init='advi+adapt_diag', n_init=ninit, tune=ninit//2, chains=nchains, cores=ncores,
                     nuts_kwargs={'target_accept': 0.90}, discard_tuned_samples=True, compute_convergence_checks=True,
                     progressbar=False)
    pm.save_trace(trace, directory=cache_file_bnn)
else:
    trace = pm.load_trace(cache_file_bnn, model=neural_network)

print("Done...")
