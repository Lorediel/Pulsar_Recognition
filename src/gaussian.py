#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:31:09 2022

@author: lorenzodamico
"""

from random import shuffle
import numpy
import minDCF as dcf
import utils
import scipy.special

"""
K fold for gaussian classifiers
"""
def kfold(D, L, k, classifier, priors):
    
    #numpy.random.seed(seed)
    #this is for a randomization of the sample order for the kfold approach.
    #idx = numpy.random.permutation(D.shape[1])
    idx = range(D.shape[1])

    splits = numpy.array_split(idx, k)

    total_samples_tested = D.shape[1]
    total_correct_predictions = 0
    total_scores = []
    for iteration in range(k):
        trainIdxList = splits[:iteration] + splits[iteration+1:]
        trainIdx = numpy.concatenate(trainIdxList, axis=0)
        DTest = D[:, splits[iteration]]
        DTrain = D[:, trainIdx]
        LTest = L[splits[iteration]]
        LTrain = L[trainIdx]
        params = classifier(DTrain, LTrain)
        llrs, pred = compute_posterior_probabilities_GAU(params, priors, DTest, LTest)
        total_scores += list(llrs)
        total_correct_predictions += (pred == LTest).sum()
    accuracy = total_correct_predictions / total_samples_tested
    err_rate = 1 - accuracy
    return numpy.array(total_scores)
"""
Computes the logpdf value for every feature in X
"""
def logpdf_GAU_ND(X,mu,C):
    P = numpy.linalg.inv(C)
    const = -0.5 * X.shape[0] * numpy.log(2*numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(C)[1]
    Y=[]
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const + -0.5 * numpy.dot( (x-mu).T, numpy.dot(P, (x-mu)))
        Y.append(res)
        
    return numpy.array(Y).ravel()

def loglikelihood(X,mu,C):
    Y = logpdf_GAU_ND(X, mu, C)
    return sum(Y)

"""
Computes the parameters for each class
"""
def get_parameters_per_class(DTrain, LTrain, num_classes):
    
    parameters_for_class = {}
    
    for label in range(num_classes):
        mu = utils.compute_empirical_mean(DTrain[:, LTrain == label])
        C = utils.compute_empirical_covariance(DTrain[:, LTrain == label])
        parameters_for_class[label] = [mu,C]

    return parameters_for_class

def compute_posterior_probabilities_GAU(parameters, classPriors, DTest, LTest):

    logSJoint = numpy.zeros((2, DTest.shape[1]))
    for label in [0,1]:
        mu, C = parameters[label]
        logSJoint[label, :] = logpdf_GAU_ND(DTest, mu, C).ravel() + numpy.log(classPriors[label])

    for label in [0,1]:
        mu, C = parameters[label]
        logSJoint[label, :] = logpdf_GAU_ND(DTest, mu, C).ravel() + numpy.log(classPriors[label])
        
    logSMarginal = utils.vrow(scipy.special.logsumexp(logSJoint, axis = 0))

    #post1 = SJoint / SMarginal
    logPost = logSJoint -logSMarginal
    llrs = logPost[1,:] - logPost[0,:]
    # 1 if llrs[i] > 0, 0 else
    lpred = (llrs>0)
    
    return llrs, lpred

# CLASSIFIERS
def multivariate_gaussian_classifier(DTrain, LTrain):
    p = get_parameters_per_class(DTrain, LTrain,2 )
    return p

def naive_bayes_gaussian_classifier(DTrain, LTrain):
    p = get_parameters_per_class(DTrain, LTrain,2 ) 
    for i in range(2):
        C = p[i][1]
        p[i][1] = C*numpy.eye(C.shape[0])
    return p

def tied_covariance_GAU(DTrain, LTrain):
    p = get_parameters_per_class(DTrain, LTrain,2 )
    #We can also calculate it like we did in the LDA with the within-class covariance
    C = 0
    for label in range(2):
        Nc = (LTrain == label).sum()
        C += Nc * p[label][1]
    
    C = C / DTrain.shape[1]
    for label in range(2):
        p[label][1] = C
    
    return p

def tied_naive_bayes_classifier(DTrain, LTrain):
    p = get_parameters_per_class(DTrain, LTrain,2 )
    #We can also calculate it like we did in the LDA with the within-class covariance
    C = 0
    for label in range(2):
        Nc = (LTrain == label).sum()
        C += Nc * p[label][1]
    
    C = C / DTrain.shape[1]
    for label in range(2):
        p[label][1] = C*numpy.eye(C.shape[0])
        
    return p

def compute_minDCF_for_classifiers(D, L, filename):
    applications = [0.5, 0.1, 0.9]
    raw_min_dcfs = []
    for prior in applications:
        #application priors

        priors = [1-prior, prior]  
        D, L = utils.shuffle_dataset(D, L)
        scores_full_cov = kfold(D, L, 10, multivariate_gaussian_classifier, priors)
        scores_naive = kfold(D, L, 10, naive_bayes_gaussian_classifier, priors)
        scores_tied = kfold(D, L, 10, tied_covariance_GAU, priors)
        scores_tied_naive = kfold(D, L, 10, tied_naive_bayes_classifier, priors)

        minDCF_full_cov = dcf.compute_min_DCF(scores_full_cov, L, prior, 1, 1)
        minDCF_naive = dcf.compute_min_DCF(scores_naive, L, prior, 1, 1)
        minDCF_tied = dcf.compute_min_DCF(scores_tied, L, prior, 1, 1)
        minDCF_tied_naive = dcf.compute_min_DCF(scores_tied_naive, L, prior, 1, 1)  

        raw_min_dcfs.append([minDCF_full_cov, minDCF_naive, minDCF_tied, minDCF_tied_naive])
    return raw_min_dcfs

def print_results_tabular(res):
    for prior_data in res:
        for minDCF in prior_data:
            print(round(minDCF, 3), end = ' ')
        print()

def print_all_results(DDict, L):
    for k in data_dict:
        print(k)
        min_dcf = compute_minDCF_for_classifiers(DDict[k], L, k)
        print_results_tabular(min_dcf)

# Model building

def train(D, L, classifierName, filename=None):
    if classifierName == "full_cov":
        p = multivariate_gaussian_classifier(D, L)
    elif classifierName == "diagonal":
        p = naive_bayes_gaussian_classifier(D, L)
    elif classifierName == "tied":
        p = tied_covariance_GAU(D, L)
    elif classifierName == "tied_diagonal":
        p = tied_naive_bayes_classifier(D, L)
    if filename:
            numpy.save("../model_parameters/" + classifierName + "/" + filename, p)
    return p

"""
TRAIN MODELS AND SAVE THE DATA
"""
def train_models():
    D = numpy.load("../npy_data/data/znorm_data.npy")
    L = numpy.load("../npy_data/data/labels.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    D_PCA_5 = numpy.load("../npy_data/data/raw_data_PCA_5.npy")
    D_PCA_6 = numpy.load("../npy_data/data/raw_data_PCA_6.npy")
    D_PCA_7 = numpy.load("../npy_data/data/raw_data_PCA_7.npy")
    Y_PCA_5 = numpy.load("../npy_data/data/gaussianized_data_PCA_5.npy")
    Y_PCA_6 = numpy.load("../npy_data/data/gaussianized_data_PCA_6.npy")
    Y_PCA_7 = numpy.load("../npy_data/data/gaussianized_data_PCA_7.npy")
    classifiers= ["full_cov", "diagonal", "tied", "tied_diagonal"]
    for c in classifiers:
        train(D, L,c, "raw_data")
        train(Y, L,c, "gaussianized_data")
        train(D_PCA_5, L,c, "raw_PCA_5")
        train(D_PCA_6, L,c, "raw_PCA_6")
        train(D_PCA_7, L,c, "raw_PCA_7")
        train(Y_PCA_5, L,c, "gaussianized_PCA_5")
        train(Y_PCA_6, L,c, "gaussianized_PCA_6")
        train(Y_PCA_7, L,c, "gaussianized_PCA_7")

def minDCF_final(D, L, dataset_type):
    print(dataset_type)
    classifiers= ["full_cov", "diagonal", "tied", "tied_diagonal"]
    applications = [0.5, 0.1, 0.9]
    for classifier in classifiers:
        f = "../model_parameters/" + classifier + "/" + dataset_type + ".npy"
        p = numpy.load(f, allow_pickle=True)
        p = p.item()
        dcfs = []
        for prior in applications:
            priors = (1-prior, prior)
            scores, lPred = compute_posterior_probabilities_GAU(p, priors, D, L)
            minDCF = dcf.compute_min_DCF(scores, L, prior, 1, 1)
            dcfs.append(minDCF)
        for d in dcfs:
            print(round(d,3), end = " ")
        print()

if __name__ == "__main__":
    # minDCF kfold
    D = numpy.load("../npy_data/data/znorm_data.npy")
    L = numpy.load("../npy_data/data/labels.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    D_PCA_5 = numpy.load("../npy_data/data/raw_data_PCA_5.npy")
    D_PCA_6 = numpy.load("../npy_data/data/raw_data_PCA_6.npy")
    D_PCA_7 = numpy.load("../npy_data/data/raw_data_PCA_7.npy")
    Y_PCA_5 = numpy.load("../npy_data/data/gaussianized_data_PCA_5.npy")
    Y_PCA_6 = numpy.load("../npy_data/data/gaussianized_data_PCA_6.npy")
    Y_PCA_7 = numpy.load("../npy_data/data/gaussianized_data_PCA_7.npy")


    data_dict = {
        'raw_data': D,
        'gaussianized_data': Y,
        'raw_PCA_5': D_PCA_5,
        'raw_PCA_6': D_PCA_6,
        'raw_PCA_7': D_PCA_7,
        'gaussianized_PCA_5': Y_PCA_5,
        'gaussianized_PCA_6': Y_PCA_6,
        'gaussianized_PCA_7': Y_PCA_7,
        }

    print_all_results(data_dict, L)
    train_models()
