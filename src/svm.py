import utils
import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import minDCF as dcf


"""
KFOLD for linear SVM
"""
def kfold_linear_SVM(D, L, k, C, pi_t, balanced):
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
        w = train_SVM_linear(DTrain, LTrain, C, pi_t, balanced)
        scores, pred = perform_classification_SVM(DTest,LTest,w, K=1)
        total_scores += list(scores)
        total_correct_predictions += (pred == LTest).sum()
        accuracy = total_correct_predictions / total_samples_tested
        err_rate = 1 - accuracy
    return numpy.array(total_scores)

"""
Train the model and obtain w (model parameters)
"""
def train_SVM_linear(DTR, LTR, C, pi_t, balanced, K=1):
    
    DTREXT = numpy.vstack([DTR, numpy.ones((1, DTR.shape[1]))*K])
    
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    # H(i,j) = Z[i] * Z[j] * X[i].T * X[J]
    H = numpy.dot(DTREXT.T, DTREXT)
    H = utils.vcol(Z) * utils.vrow(Z) * H
    
    #Returns the modified primal and its gradient
    def JDual(alpha):
        Ha = numpy.dot(H, utils.vcol(alpha))
        aHa = numpy.dot(utils.vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)
    
    bounds = [(0,C)] * DTR.shape[1]
    #Calculate Ct and Cf
    if balanced:
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
    #since L-BFGS-B computes the minimum but we need the maximum we return
    # the loss and grad but with inverse sign
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    
    #To see if the computation has gone well
    def JPrimal(w):
        S = numpy.dot(utils.vrow(w), DTREXT)
        loss = numpy.maximum(numpy.zeros(S.shape), 1-Z*S).sum()
        return 0.5 * numpy.linalg.norm(w)**2 + C * loss
    
    #alphaStar is the position of the minimum
    # _x is the value of the minimum
    # _y given information on the computation
    alphaStar,_x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds = bounds,
        factr=1.0,
        maxiter=100000,
        maxfun = 100000
        )
    wStar = numpy.dot(DTREXT, utils.vcol(alphaStar) * utils.vcol(Z))
    #Primal loss
    JP = JPrimal(wStar)
    #Dual loss
    JD = JDual(alphaStar)[0]
    #slightly different results
    Duality_gap = JP - JD
    print(Duality_gap)
    return wStar

"""
Use w to perform classification given a dataset and labels 
"""
def perform_classification_SVM(DTE,LTE,w, K=1):
    DTEEXT = numpy.vstack([DTE, numpy.ones((1, DTE.shape[1]))*K])

    S = numpy.dot(w.T,DTEEXT).ravel()
    
    # class 1 if score >= 0, else 0
    pred = numpy.int32(S>=0)
    accuracy = (LTE == pred).sum() / LTE.size
    err_rate = 1 - accuracy
    return S, pred

"""
Tune C using kfold
"""
def C_tuning(D, L, balanced, filename, datatype):
    Cs = numpy.logspace(-3, 0, num=10)
    i = 0
    dcfs = []
    D, L = utils.shuffle_dataset(D, L)
    eff_priors = [0.5, 0.1, 0.9]
    if balanced:
        f = filename + "/linear_svm_"+ datatype + "_balanced_"
    else:
        f = filename + "/linear_svm_"+ datatype + "_unbalanced_"
    for ef_pi in eff_priors:
        i = 0
        dcfs = []
        for C in Cs:
            print("iteration:",i, "C:",C)
            scores_lr = kfold_linear_SVM(D, L, 5, C, 0.5, balanced)
            minDCF = dcf.compute_min_DCF(scores_lr, L, ef_pi, 1, 1)
            dcfs.append(minDCF)
            i+=1
            print(minDCF)
        numpy.save(f + str(ef_pi), dcfs)


def plot_linear_svm():
    Cs = numpy.logspace(-3, 0, num=10)
    f_balanced_raw = "../npy_data/linear_svm/linear_svm_raw_balanced_"
    f_unbalanced_raw = "../npy_data/linear_svm/linear_svm_raw_unbalanced_"
    f_balanced_gauss = "../npy_data/linear_svm/linear_svm_gauss_balanced_"
    f_unbalanced_gauss = "../npy_data/linear_svm/linear_svm_gauss_unbalanced_"
    files = [f_balanced_raw, f_unbalanced_raw, f_balanced_gauss, f_unbalanced_gauss]
    
    titles = ["raw balanced", "raw unbalanced", "gaussianized balanced", "gaussianized unbalanced"]
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(12,7)
    k=0
    for i in range(2):
        for j in range(2):
            axs[i][j].set_xscale("log")
            axs[i][j].set_title(titles[k])
            axs[i][j].set_xlabel("C")
            axs[i][j].set_ylabel("minDCF")
            line1 = axs[i][j].plot(Cs, numpy.load(files[k] + "0.5.npy"), color="red" , label = r"$\widetilde{\pi}=$" + "0.5")
            line2 = axs[i][j].plot(Cs, numpy.load(files[k] + "0.1.npy"), color="blue", label = r"$\widetilde{\pi}=$" + "0.1")
            line3 = axs[i][j].plot(Cs, numpy.load(files[k] + "0.9.npy"), color="green", label = r"$\widetilde{\pi}=$" + "0.9")
            k+=1
    axs[0][0].legend()
    fig.tight_layout()
    plt.savefig("../images/svm_linear/C_tuning", format="pdf")
    plt.show()
"""
Compute minDCF for the validation set
"""
def compute_final_min_DCF(D, L, pi_t, balanced):
    D, L = utils.shuffle_dataset(D, L)
    eff_priors = [0.5, 0.1, 0.9]
    C = 0.1
    minDCFs = []

    for ef_pi in eff_priors:
        
        scores_lr = kfold_linear_SVM(D, L, 5, C, pi_t, balanced)
        minDCF = dcf.compute_min_DCF(scores_lr, L, ef_pi, 1, 1)
        
        minDCFs.append(minDCF)
    print(minDCFs)

def minDCF_tabular():
    L = numpy.load("../npy_data/data/labels.npy")
    D = numpy.load("../npy_data/data/znorm_data.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    pits = [0.5,0.1,0.9]
    #Balanced
    for pi_t in pits:
        print("raw Balanced")
        compute_final_min_DCF(D, L, pi_t, True)
        print("gauss Balanced")
        compute_final_min_DCF(Y, L, pi_t, True)
    #Unbalanced => 0.5 is useless
    print("raw unbalanced")
    compute_final_min_DCF(D, L, 0.5, False)
    print("gauss unbalanced")
    compute_final_min_DCF(Y, L, 0.5, False)
"""
MODEL BUILDING
"""
"""
Obtain w using all the data and save it
"""
def train(DTR, LTR, C, pi_t, balanced, filename):
    w = train_SVM_linear(DTR, LTR, C, pi_t, balanced)
    numpy.save(filename, w)

"""
Train all the models
"""
def train_models():
    L = numpy.load("../npy_data/data/labels.npy")
    D = numpy.load("../npy_data/data/znorm_data.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    C = 0.1
    priors = [0.5,0.1,0.9]
    print("raw unbalanced")
    train(D, L, C, 0.5, False, "../model_parameters/linear_svm/w_raw_u")
    print("gauss unbalanced")
    train(Y, L, C, 0.5, False, "../model_parameters/linear_svm/w_gauss_u")
    for prior in priors:
        print(prior)
    
        print("raw balanced")
        train(D, L, C, prior, True, "../model_parameters/linear_svm/w_raw_b" + str(prior))
        
        print("gauss balanced")
        train(Y, L, C, prior, True, "../model_parameters/linear_svm/w_gauss_b" + str(prior))

"""
Use the saved model parameters to compute the min DCF
"""
def minDCF_final(D, L, dataset_type, balanced):
    print(dataset_type, balanced)
    classifiers= ["0.5", "0.1", "0.9"]
    applications = [0.5, 0.1, 0.9]
    if balanced == "b":
        for classifier in classifiers:
            f = "../model_parameters/linear_svm/w_" + dataset_type + "_" + balanced +classifier + ".npy"
            w = numpy.load(f, allow_pickle=True)
            dcfs = []
            for prior in applications:
                scores, lPred = perform_classification_SVM(D, L, w)
                minDCF = dcf.compute_min_DCF(scores, L, prior, 1, 1)
                dcfs.append(minDCF)
            for d in dcfs:
                print(round(d,3), end = " ")
            print()
    else:
        f = "../model_parameters/linear_svm/w_" + dataset_type + "_" + balanced + ".npy"
        w = numpy.load(f, allow_pickle=True)
        dcfs = []
        for prior in applications:
            scores, lPred = perform_classification_SVM(D, L, w)
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
    C_tuning(D, L, True, "../npy_data/linear_svm", "raw")
    C_tuning(D, L, False, "../npy_data/linear_svm", "raw")
    C_tuning(Y, L, True, "../npy_data/linear_svm", "gauss")
    C_tuning(Y, L, False, "../npy_data/linear_svm", "gauss")
    #PLOTTING
    plot_linear_svm()
    #minDCF over training/validation
    minDCF_tabular()
    #Train models and save model parameters
    train_models()

