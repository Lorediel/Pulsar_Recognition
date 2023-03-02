from utils import shuffle_dataset
import numpy 
import utils
import scipy.optimize
import minDCF as dcf
import matplotlib
import matplotlib.pyplot as plt

"""
kfold for the linear logistic regression
"""
def kfold(D, L, k, classifier, l, prior):
    idx = range(D.shape[1])
    splits = numpy.array_split(idx, k)
    total_scores = []
    total_samples_tested = D.shape[1]
    total_correct_predictions = 0
    for iteration in range(k):
        trainIdxList = splits[:iteration] + splits[iteration+1:]
        trainIdx = numpy.concatenate(trainIdxList, axis=0)
        DTest = D[:, splits[iteration]]
        DTrain = D[:, trainIdx]
        LTest = L[splits[iteration]]
        LTrain = L[trainIdx]
        scores, pred = classifier(DTrain, LTrain, DTest, LTest, prior, l)
        total_scores += list(scores)
        total_correct_predictions += (pred == LTest).sum()
        accuracy = total_correct_predictions / total_samples_tested
        err_rate = 1 - accuracy
    return numpy.array(total_scores)

def logreg_obj_wrap(DTR, LTR, prior, l):
    Z = LTR * 2.0 - 1.0
    M = DTR.shape[0]
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    Z0 = Z[LTR == 0]
    Z1 = Z[LTR == 1]
    def logreg_obj(v): 
        w = utils.vcol(v[0:-1])
        b = v[-1]
        S0 = numpy.dot(w.T, DTR0) + b
        S1 = numpy.dot(w.T, DTR1) + b
        cxe0 = (1-prior) * (numpy.logaddexp(0, -S0*Z0)).mean()
        cxe1 = prior * (numpy.logaddexp(0, -S1*Z1)).mean()
        cxe = cxe0 + cxe1

        return cxe + 0.5 * l * numpy.linalg.norm(w)**2
    return logreg_obj

def binary_logreg(DTR, LTR, DTE, LTE, prior, l):
    logreg_obj = logreg_obj_wrap(DTR, LTR, prior, l)
    v, J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
    w = utils.vcol(v[0:-1])
    b = v[-1]
    S = (numpy.dot(w.T, DTE) + b).ravel() - numpy.log(prior/(1-prior))
    LP = numpy.zeros(S.size)
    for i in range(S.size):
        LP[i] = 1 if S[i] > 0 else 0
    # S are the scores
    return S, LP

"""
to get calibrated scores
"""
def calibration_logreg(DTR, LTR, DTE, prior, l):
    logreg_obj = logreg_obj_wrap(DTR, LTR, prior, l)
    v, J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
    w = utils.vcol(v[0:-1])
    b = v[-1]
    S = (numpy.dot(w.T, DTE) + b).ravel() - numpy.log(prior/(1-prior))
    
    return S

"""
Function to tune lambda
"""
def tune_lambda(D, L, lambdas, filename):
    priors = [0.5, 0.9, 0.1]
    for prior in priors:
        i = 0
        dcfs = []
        for l in lambdas:
            print("prior:", prior, "iteration:",i )
            #pi_t = 0.5
            scores_lr = kfold(D, L, 5, binary_logreg, l, 0.5)
            minDCF = dcf.compute_min_DCF(scores_lr, L, prior, 1, 1)
            dcfs.append(minDCF)
            i+=1
        numpy.save(filename + str(prior), dcfs)

def plot_lambda(filename, lambdas):
    f = "../npy_data/lambda_tuning/" + filename
    priors = {
        "red": 0.5,
        "blue": 0.9,
        "green": 0.1
    }
    plt.figure()
    for color in priors:
        prior = priors[color]
        plt.xscale("log")
        plt.xlabel(r"$\lambda$")
        plt.ylabel("minDCF")
        data = numpy.load(f + str(prior) + ".npy")
        plt.plot(lambdas, data, color=color, label = r"$\widetilde{\pi}=$" + str(prior))
    #plt.legend()
    plt.savefig("../images/linear_LR/" + filename.replace("_", "") + ".pdf", format="pdf")
    plt.show()

"""
minDCF in tabular form for the report
"""
def tabular_minDCF(D, L):
    chosen_l = 1e-5
    pi_ts = [0.5, 0.1, 0.9]
    effective_pi = [0.5, 0.1, 0.9]
    D, L = shuffle_dataset(D, L)
    for pi_t in pi_ts:
        scores_lr = kfold(D, L, 5, binary_logreg, chosen_l, pi_t)
        for ef_pi in effective_pi:
            minDCF = dcf.compute_min_DCF(scores_lr, L, ef_pi, 1, 1)
            print(round(minDCF,3), end=" ")
        print()


"""
MODEL BUILDING
"""

"""
train with the whole dataset, returns the model parameters (v)
"""
def binary_logreg_train(D, L, prior, l):
    logreg_obj = logreg_obj_wrap(D, L, prior, l)
    v, J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(D.shape[0]+1), approx_grad=True)
    return v

"""
Given the model parameter, obtain the scores.
"""
def compute_scores(v, DTE, prior):
    w = utils.vcol(v[0:-1])
    b = v[-1]
    S = (numpy.dot(w.T, DTE) + b).ravel() - numpy.log(prior/(1-prior))
    LP = numpy.zeros(S.size)
    for i in range(S.size):
        LP[i] = 1 if S[i] > 0 else 0
    # S are the scores
    return S, LP
"""
Obtain and save v, the model parameter
"""
def train(D, L, prior, filename=None):
    v = binary_logreg_train(D, L, prior, 1e-5)
    numpy.save(filename + str(prior), v)
"""
Train the model for the different datasets
"""
def train_models():
    L = numpy.load("../npy_data/data/labels.npy")
    D = numpy.load("../npy_data/data/znorm_data.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    D_PCA_7 = numpy.load("../npy_data/data/raw_data_PCA_7.npy")
    Y_PCA_7 = numpy.load("../npy_data/data/gaussianized_data_PCA_7.npy")
    priors = [0.5,0.1,0.9]
    for prior in priors:
        train(D, L, prior, "../model_parameters/llr/v_raw_")
        train(Y, L, prior, "../model_parameters/llr/v_gauss_")
        train(D_PCA_7, L, prior, "../model_parameters/llr/v_raw_pca7_")
        train(Y_PCA_7, L, prior, "../model_parameters/llr/v_gauss_pca7_")
"""
Compute the final DCFs
"""
def minDCF_final(D, L, dataset_type):
    print(dataset_type)
    classifiers= ["0.5", "0.1", "0.9"]
    applications = [0.5, 0.1, 0.9]
    for classifier in classifiers:
        f = "../model_parameters/llr/v_" + dataset_type + "_" +classifier + ".npy"
        v = numpy.load(f, allow_pickle=True)
        dcfs = []
        for prior in applications:
            scores, lPred = compute_scores(v, D, prior)
            minDCF = dcf.compute_min_DCF(scores, L, prior, 1, 1)
            dcfs.append(minDCF)
        for d in dcfs:
            print(round(d,3), end = " ")
        print()



if __name__ == "__main__":
    L = numpy.load("../npy_data/data/labels.npy")
    D = numpy.load("../npy_data/data/znorm_data.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    D_PCA_7 = numpy.load("../npy_data/data/raw_data_PCA_7.npy")
    Y_PCA_7 = numpy.load("../npy_data/data/gaussianized_data_PCA_7.npy")
    lambdas = numpy.logspace(-5, 5, num=50)
    D, LD = shuffle_dataset(D,L)
    Y, LY = shuffle_dataset(Y,L)
    Y_PCA_7, LPCAY = shuffle_dataset(Y_PCA_7, L)
    D_PCA_7, LPCAD = shuffle_dataset(Y_PCA_7, L)
    # TUNING
    tune_lambda(D, LD, lambdas, "../npy_data/lambda_tuning/lrD_")
    tune_lambda(Y, LY, lambdas, "../npy_data/lambda_tuning/lrY_")
    tune_lambda(Y_PCA_7, LPCAY, lambdas, "../npy_data/lambda_tuning/lrY_PCA_7")
    tune_lambda(D_PCA_7, LPCAD, lambdas, "../npy_data/lambda_tuning/lrD_PCA_7")
    # PLOTTING
    plot_lambda("lrD_", lambdas)
    plot_lambda("lrY_", lambdas)
    plot_lambda("lrD_PCA_7", lambdas)
    plot_lambda("lrY_PCA_7", lambdas)
    # min DCF over training/validation set
    L = numpy.load("../npy_data/data/labels.npy")
    D = numpy.load("../npy_data/data/znorm_data.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    D_PCA_7 = numpy.load("../npy_data/data/raw_data_PCA_7.npy")
    Y_PCA_7 = numpy.load("../npy_data/data/gaussianized_data_PCA_7.npy")
    print("raw")
    tabular_minDCF(D, L)
    print("gauss")
    tabular_minDCF(Y, L)
    print("raw PCA 7")
    tabular_minDCF(D_PCA_7, L)
    print("gauss PCA 7")
    tabular_minDCF(Y_PCA_7, L)
    """
    TRAIN
    """
    train_models()
    