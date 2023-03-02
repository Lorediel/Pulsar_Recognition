#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 09:58:20 2022

@author: lorenzodamico
"""

import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import seaborn
from matplotlib.lines import Line2D

feature_names = {
    0: "Mean of the integrated profile",
    1: "Standard deviation of the integrated profile",
    2: "Excess kurtosis of the integrated profile",
    3: "Skewness of the integrated profile",
    4: "Mean of the DM-SNR curve",
    5: "Standard deviation of the DM-SNR curve",
    6: "Excess kurtosis of the DM-SNR curve",
    7: "Skewness of the DM-SNR curve"
    }

label_names = {
    0: "Negative",
    1: "Positive"
    }

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def compute_empirical_mean(D):
    return vcol(D.mean(1))


def compute_empirical_covariance(D):
    # mean by columns
    mu = compute_empirical_mean(D)
    # center the data exploiting broadcasting
    DC = D - mu
    # compute C
    C = numpy.dot(DC, DC.T)
    C = C / D.shape[1]
    return C



# To load the dataset
def load(fileName):
    listOfVecs = []
    labelList = []
    with open(fileName, 'r') as f:
        for line in f:
            try:
                splittedLine = line.split(",")
                features = splittedLine[:8]
                label = splittedLine[-1].strip()
                oneDimVec = numpy.array([float(i) for i in features]).reshape(len(features),1)
                listOfVecs.append(oneDimVec)
                labelList.append(label)
            except:
                pass
    return numpy.hstack(listOfVecs), numpy.array(labelList, dtype=int)


def plot_hist_for_feature(D,L, fileString = None):
    DC0 = D[:, L==0]
    DC1 = D[:, L==1]
    s = "hist"
    if (fileString != None):
        s = fileString + '_' + s
    fig, axs = plt.subplots(2,4)
    fig.set_size_inches(16,7)
    k = 0
    for i in range(2):
        for j in range(4):
            k_feature_data_0 = DC0[k,:]
            k_feature_data_1 = DC1[k,:]
            axs[i][j].set_xlabel(feature_names[k], fontsize="small", fontweight="semibold")
            axs[i][j].hist(k_feature_data_0, bins=25, density=True, color='blue', alpha = 0.5,edgecolor='black')
            axs[i][j].hist(k_feature_data_1, bins=25, density=True, color='red', alpha = 0.5,edgecolor='black')
            k+=1
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
            Line2D([0], [0], color='red', lw=2)]
    fig.legend(custom_lines, [label_names[0], label_names[1]], loc='upper center')
    plt.savefig('../images/histogram_plots/' + s + ".pdf", format="pdf")
    plt.show()
       
#gaussianization
def compute_gaussianized_feature(x):
    
    r = []
    for i in range(x.size):
        
        p = numpy.array(x[i])
        indicator_sum = (p>x).sum()
        r_xi = (indicator_sum + 1) / (x.size+2)
        
        r.append(r_xi)
    r = numpy.array(r)
    y = scipy.stats.norm.ppf(r)
    return y
"""
Gaussianization on the training set
"""
def compute_gaussianized_matrix(D):
    Y = []
    for k in range(D.shape[0]):
        x = D[k,:]
        Y.append(compute_gaussianized_feature(x))
    return numpy.vstack(Y)


def zNorm(D):
    u = numpy.mean(D, axis = 1)
    std = numpy.std(D, axis = 1)
    DZ = (D-vcol(u)) / (vcol(std))
    return DZ

def generate_correlation_heatmap(D, L, fileString= None):
    C = numpy.absolute(numpy.corrcoef(D))
    C0 = numpy.absolute(numpy.corrcoef(D[:, L==0]))
    C1 = numpy.absolute(numpy.corrcoef(D[:, L==1]))
    s = 'corr_%s.pdf'
    if (fileString != None):
        s = fileString + '_' +s
    data ={
        "Whole dataset": ("Greys",C),
        "Class positive": ("Reds", C0),
        "Class negative": ("Blues", C1)
        }
    for k in data:
        plt.figure()
        plt.title(k)
        seaborn.heatmap(data[k][1], cmap=data[k][0], square=True, linewidths=0.2, annot= True)
        plt.savefig('../images/heatmap_correlations/' + s % k.replace(" ", "_"), format="pdf")

def compute_PCA(D, m):
    u = compute_empirical_mean(D)
    C = compute_empirical_covariance(D)

    # eigenvalues and vectors sorted from smallest to greatest
    s, U = numpy.linalg.eigh(C)
    # get the largest m eigenvectors
    P = U[:, ::-1][:, 0:m]
    return numpy.dot(P.T, D)
"""
FOR THE TEST SET
"""
def compute_PCA_given_C(D, m, C):
    s, U = numpy.linalg.eigh(C)
    # get the largest m eigenvectors
    P = U[:, ::-1][:, 0:m]
    return numpy.dot(P.T, D)

def compute_zNorm_given_mean(D, u, std):
    DZ = (D-vcol(u)) / (vcol(std))
    return DZ
"""
Gaussianized features for the test set
"""
def compute_gaussianized_matrix_for_test(D, DE):
    Y = []
    for k in range(D.shape[0]):
        x = D[k,:]
        r = []
        for i in range(DE.shape[1]):
            
            p = numpy.array(DE[k][i])
            indicator_sum = (p>x).sum()

            r_xi = (indicator_sum + 1) / (x.size+2)
            
            r.append(r_xi)
        r = numpy.array(r)
        y = scipy.stats.norm.ppf(r)
        Y.append(y)
    return numpy.vstack(Y)

if __name__ == "__main__":
    D, L = load("../Pulsar_Detection/Train.txt")
    plot_hist_for_feature(D,L,'raw')
    DZ= zNorm(D)
    Y = compute_gaussianized_matrix(DZ)
    generate_correlation_heatmap(D, L, 'raw')
    generate_correlation_heatmap(Y, L, 'gaussianized')
    plot_hist_for_feature(Y,L, 'gaussianized')
    generate_correlation_heatmap(Y, L, "gauss_")
    plot_hist_for_feature(DZ, L, 'znorm')
    D_PCA_5 = compute_PCA(DZ, 5)
    D_PCA_6 = compute_PCA(DZ, 6)
    D_PCA_7 = compute_PCA(DZ, 7)
    Y_PCA_5 = compute_PCA(Y, 5)
    Y_PCA_6 = compute_PCA(Y, 6)
    Y_PCA_7 = compute_PCA(Y, 7)
    numpy.save("../npy_data/data/raw_data", D)
    numpy.save("../npy_data/data/znorm_data", DZ)
    numpy.save("../npy_data/data/gaussianized_data", Y)
    numpy.save("../npy_data/data/labels", L)
    numpy.save("../npy_data/data/raw_data_PCA_5", D_PCA_5)
    numpy.save("../npy_data/data/raw_data_PCA_6", D_PCA_6)
    numpy.save("../npy_data/data/raw_data_PCA_7", D_PCA_7)
    numpy.save("../npy_data/data/gaussianized_data_PCA_5", Y_PCA_5)
    numpy.save("../npy_data/data/gaussianized_data_PCA_6", Y_PCA_6)
    numpy.save("../npy_data/data/gaussianized_data_PCA_7", Y_PCA_7)