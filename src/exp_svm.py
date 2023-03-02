import utils
import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import minDCF as dcf

"""
Function to calculate the bounds which are the constraints to be satisfied
"""
def calculate_bound(DTR, LTR, pi_t, C):
    n = DTR.shape[1]
    nt = DTR[:, LTR==1].shape[1]
    nf = DTR[:, LTR==0].shape[1]
    emp_pi_t = nt/n
    emp_pi_f = nf/n
    pi_f = 1-pi_t
    Ct = C * pi_t / emp_pi_t
    Cf = C * pi_f / emp_pi_f
    bounds = []
    for i in range(DTR.shape[1]):
        if LTR[i] == 1:
            bounds.append((0,Ct))
        else:
            bounds.append((0,Cf))
    return bounds


#what changes is the scores formula, we do need to apply k(xi,xt)
def train_SVM_exp_kernel(DTR, LTR, C, pi_t, gamma, balanced):
    
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    Dist = utils.vcol((DTR**2).sum(0)) + utils.vrow((DTR**2).sum(0)) - 2 * numpy.dot(DTR.T, DTR)
    
    Dist = gamma * Dist
    H = numpy.exp(-Dist) + 1
    H = utils.vcol(Z) * utils.vrow(Z) * H
    b = [(0,C)] * DTR.shape[1]
    if (balanced):
        b = calculate_bound(DTR, LTR, pi_t, C)
    
    def JDual(alpha):
        Ha = numpy.dot(H, utils.vcol(alpha))
        aHa = numpy.dot(utils.vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)
    
    #since L-BFGS-B computes the minimum but we need the maximum we return
    # the loss and grad but with inverse sign
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    
    #alphaStar is the position of the minimum
    # _x is the value of the minimum
    # _y given information on the computation
    alphaStar,_x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds = b,
        factr=1.0,
        maxiter=100000,
        maxfun = 100000
        )
    
    wStar = (utils.vcol(alphaStar) * utils.vcol(Z)).ravel()
    #Dual loss
    JD = JDual(alphaStar)[0]
    #print(JD)
    
    
    return wStar, alphaStar, Z

#Perform classification
def perform_classification_SVM_exp_kernel(DTR, DTE, LTE, alphaStar, Z, gamma):
    Dist = utils.vcol((DTE**2).sum(0)) + utils.vrow((DTR**2).sum(0)) - 2 * numpy.dot(DTE.T, DTR)
    Dist = Dist * gamma
    k_func = numpy.exp(-Dist) + 1
    S = numpy.dot(k_func, utils.vcol(alphaStar)*utils.vcol(Z)).ravel()
    pred = numpy.int32(S>=0)
    accuracy = (LTE == pred).sum() / LTE.size
    err_rate = 1 - accuracy
    return S, pred
    


def kfold_exp_svm(D, L, k, C, pi_t, gamma, balanced):
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
        w, a, Z = train_SVM_exp_kernel(DTrain, LTrain, C, pi_t, gamma, balanced)
        scores, pred = perform_classification_SVM_exp_kernel(DTrain, DTest , LTest, a, Z, gamma)
        total_scores += list(scores)
        total_correct_predictions += (pred == LTest).sum()
        accuracy = total_correct_predictions / total_samples_tested
        err_rate = 1 - accuracy
    return numpy.array(total_scores)

def C_tuning(D, L, balanced, filename):
    Cs = numpy.logspace(-2, 1, num=10)
    D, L = utils.shuffle_dataset(D, L)
    gammas = [1e-1, 1e-2, 1e-3]
    pi_t = 0.5
    for g in gammas:
        dcfs= []
        i=0
        for C in Cs:
            print("iteration:", i, "C:", C)
            scores = kfold_exp_svm(D, L, 5, C, pi_t, g, balanced)
            minDCF = dcf.compute_min_DCF(scores, L, 0.5, 1, 1)
            print(minDCF)
            dcfs.append(minDCF)
            i+=1
        numpy.save(filename + str(g), dcfs)

def plot_exp_svm():
    Cs = numpy.logspace(-2, 1, num=10)
    f_balanced_raw = "../npy_data/exp_svm/exp_svm_raw_balanced_gamma_"
    f_unbalanced_raw = "../npy_data/exp_svm/exp_svm_raw_unbalanced_gamma_"
    f_balanced_gauss = "../npy_data/exp_svm/exp_svm_gauss_balanced_gamma_"
    f_unbalanced_gauss = "../npy_data/exp_svm/exp_svm_gauss_unbalanced_gamma_"
    files = [f_balanced_raw, f_unbalanced_raw, f_balanced_gauss, f_unbalanced_gauss]
    
    titles = ["raw balanced", "raw unbalanced", "gaussianized balanced", "gaussianized unbalanced"]
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(12,7)
    k=0
    for i in range(2):
        for j in range(2):
            axs[i][j].set_xscale("log")
            axs[i][j].set_title(titles[k])
            line1 = axs[i][j].plot(Cs, numpy.load(files[k] + "0.1.npy"), color="red" , label = r"$\gamma=$" + "0.1")
            line2 = axs[i][j].plot(Cs, numpy.load(files[k] + "0.01.npy"), color="orange", label = r"$\gamma=$" + "0.01")
            line3 = axs[i][j].plot(Cs, numpy.load(files[k] + "0.001.npy"), color="yellow", label = r"$\gamma=$" + "0.001")
            k+=1
    axs[0][0].legend()
    fig.tight_layout()
    plt.savefig("../images/exp_svm/C_tuning.pdf", format="pdf")
    plt.show()


def final_min_DCF_balanced(D, L, gamma):
    D, L = utils.shuffle_dataset(D, L)
    C = 10
    eff_priors = [0.5, 0.1, 0.9]
    pi_ts = [0.5, 0.1, 0.9]
    for pi_t in pi_ts:
        dcfs = []
        for p in eff_priors:
            scores = kfold_exp_svm(D, L, 5, C, pi_t, gamma, True)
            minDCF = dcf.compute_min_DCF(scores, L, p, 1, 1)
            
            dcfs.append(minDCF)
        print(dcfs)
    
def final_min_DCF_unbalanced(D, L, gamma):
    D, L = utils.shuffle_dataset(D, L)
    C = 10
    eff_priors = [0.5, 0.1, 0.9]
    dcfs = []
    for p in eff_priors:
        scores = kfold_exp_svm(D, L, 5, C, 0.5, gamma, False)
        minDCF = dcf.compute_min_DCF(scores, L, p, 1, 1)
        dcfs.append(minDCF)
    print(dcfs)

def tabular_minDCF():
    L = numpy.load("../npy_data/data/labels.npy")
    D = numpy.load("../npy_data/data/znorm_data.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    #Balanced:
    print("Balanced raw")
    final_min_DCF_balanced(D, L, 0.01)
    print("Balanced gaussian")
    final_min_DCF_balanced(Y, L, 0.1)
    #unbalanced
    print("Unbalanced raw")
    final_min_DCF_unbalanced(D, L, 0.01)
    print("Unbalanced gaussian")
    final_min_DCF_unbalanced(Y, L, 0.1)

"""
MODEL BUILDING
"""

def train(DTR, LTR, C, pi_t, balanced, filename, gamma):
    w = train_SVM_exp_kernel(DTR, LTR, C, pi_t, gamma, balanced)
    numpy.save(filename, w)

def train_models():
    L = numpy.load("../npy_data/data/labels.npy")
    D = numpy.load("../npy_data/data/znorm_data.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    C = 10
    gammaD = 0.01
    gammaY = 0.1
    priors = [0.5,0.1,0.9]
    print("raw unbalanced")
    train(D, L, C, 0.5, False, "../model_parameters/exp_svm/w_raw_u", gammaD)
    print("gauss unbalanced")
    train(Y, L, C, 0.5, False, "../model_parameters/exp_svm/w_gauss_u", gammaY)
    for prior in priors:
        print(prior)
    
        print("raw balanced")
        train(D, L, C, prior, True, "../model_parameters/exp_svm/w_raw_b_" + str(prior), gammaD)
    


def minDCF_final(D, L, dataset_type, balanced):
    if (dataset_type == "raw"):
        gamma = 0.01
        DTR = numpy.load("../npy_data/data/znorm_data.npy")
    elif (dataset_type == "gauss"):
        gamma = 0.1
        DTR = numpy.load("../npy_data/data/gaussianized_data.npy")
    print(dataset_type, balanced)
    classifiers= ["0.5", "0.1", "0.9"]
    applications = [0.5, 0.1, 0.9]
    if balanced == "b":
        for classifier in classifiers:
            f = "../model_parameters/exp_svm/w_" + dataset_type + "_" + balanced + "_" +classifier + ".npy"
            params = numpy.load(f, allow_pickle=True)
            w, a, Z = params[0], params[1], params[2]
            dcfs = []
            for prior in applications:
                scores, lPred = perform_classification_SVM_exp_kernel(DTR, D, L, a, Z, gamma)
                minDCF = dcf.compute_min_DCF(scores, L, prior, 1, 1)
                dcfs.append(minDCF)
            for d in dcfs:
                print(round(d,3), end = " ")
            print()
    else:
        f = "../model_parameters/exp_svm/w_" + dataset_type + "_" + balanced + ".npy"
        params = numpy.load(f, allow_pickle=True)
        w, a, Z = params[0], params[1], params[2]
        dcfs = []
        for prior in applications:
            scores, lPred = perform_classification_SVM_exp_kernel(DTR, D, L, a, Z, gamma)
            minDCF = dcf.compute_min_DCF(scores, L, prior, 1, 1)
            dcfs.append(minDCF)
        for d in dcfs:
            print(round(d,3), end = " ")
        print()


if __name__ == "__main__":
    L = numpy.load("../npy_data/data/labels.npy")
    D = numpy.load("../npy_data/data/znorm_data.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    #TUNING
    C_tuning(D, L, True, "../npy_data/exp_svm/exp_svm_raw_balanced_gamma_")
    C_tuning(D, L, False, "../npy_data/exp_svm/exp_svm_raw_unbalanced_gamma_")
    C_tuning(Y, L, True, "../npy_data/exp_svm/exp_svm_gauss_balanced_gamma_")
    C_tuning(Y, L, False, "../npy_data/exp_svm/exp_svm_gauss_unbalanced_gamma_")
    #PLOTTING
    plot_exp_svm()
    #minDCF over training/validation set
    tabular_minDCF()
    #TRAIN over all training data
    train_models()