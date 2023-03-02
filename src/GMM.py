import utils
import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.special
import minDCF as dcf

def compute_empirical_mean(D):
    return utils.vcol(D.mean(1))


def compute_empirical_covariance(D):
    # mean by columns
    mu = compute_empirical_mean(D)
    # center the data exploiting broadcasting
    DC = D - mu
    # compute C
    C = numpy.dot(DC, DC.T)
    C = C / D.shape[1]
    return C

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

#Computes the likelihood for each sample, so it returns an array of log likelihoods
# that are the log likelihoods for each of the samples contained in X
def logpdf_GMM(X, gmm):
    G = len(gmm)
    N = X.shape[1]
    S = numpy.zeros((G,N))
    # For each component we calculate the likelihood for each sample. We add the
    # prior of the component and so S is the matrix of joint log likelihoods for sample and component
    for g in range(G):
        S[g,:] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
    return scipy.special.logsumexp(S, axis=0)

def GMM_EM(X, gmm, sigma_type="full", psi= 0.01): 
    #old likelihood and new, so we know how much it increases
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    #E STEP
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G,N))
        #we calculate the matrix of joint densities like we did in the prev function
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis = 0)
        #log likelihood for the whole dataset, we sum cause the samples are indipendent
        # then we divide by N cause we want the average log likelihood
        llNew = SM.sum()/N
        #Posterior, we take the joint and remove the marginal (cause log domain)
        # exactly how we computed class posterior even though here is component posterior
        # but we're basically doing the same thing
        P = numpy.exp(SJ-SM)
        gmmNew = []
        # We compute the updated parameters of our gmm (statistics)
        #M STEP
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (utils.vrow(gamma)*X).sum(1)
            S = numpy.dot(X, (utils.vrow(gamma)*X).T)
            w = Z/N
            mu = utils.vcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            if sigma_type == "diagonal":
                Sigma = Sigma * numpy.eye(Sigma.shape[0])
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s<psi] = psi
            Sigma = numpy.dot(U, utils.vcol(s)*U.T)
            gmmNew.append((w, mu, Sigma))
        
        gmm = gmmNew
        """if (llOld != None and llNew != None):
            print(llNew-llOld)"""
    return gmm

# G is the number of clusters
# alpha 0.1
# X is the dataset
def LBG(X, alpha, G, sigma_type="full", psi=0.01):
    mu_g = compute_empirical_mean(X)
    Sigma_g = compute_empirical_covariance(X)
    if sigma_type == "diagonal":
        Sigma_g = Sigma_g * numpy.eye(Sigma_g.shape[0])
    U, s, _ = numpy.linalg.svd(Sigma_g)
    s[s<psi] = psi
    Sigma_g = numpy.dot(U, utils.vcol(s)*U.T)
    #wg, mug, Cg
    gmm =[(1.0, mu_g, Sigma_g)]
    g = 1

    while g < G:
        newGmm = []
        if sigma_type == "diagonal":
            gmm = GMM_EM(X, gmm, "diagonal", psi)
        elif sigma_type == "full":
            gmm = GMM_EM(X, gmm, "full", psi)
        for g_params in gmm:
            U, s, Vh = numpy.linalg.svd(g_params[2])
            d = U[:, 0:1] * s[0]**0.5 * alpha
            wg = g_params[0]/2
            mu_g_1 = g_params[1] - d
            mu_g_2 = g_params[1] + d
            Sigma_g = g_params[2]
            newGmm.append((wg, mu_g_1, g_params[2]))
            newGmm.append((wg, mu_g_2, g_params[2]))
            gmm = newGmm
        g*=2
    if sigma_type == "diagonal":
            gmm = GMM_EM(X, gmm, "diagonal", psi)
    elif sigma_type == "full":
        gmm = GMM_EM(X, gmm, "full", psi)
    
    return gmm

def GMM_EM_tied(X, gmm, psi= 0.01, sigma_type="tied"): 
    #old likelihood and new, so we know how much it increases
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    #E STEP
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G,N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis = 0)
        llNew = SM.sum()/N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        #M STEP
        tempSum = 0
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (utils.vrow(gamma)*X).sum(1)
            S = numpy.dot(X, (utils.vrow(gamma)*X).T)
            w = Z/N
            mu = utils.vcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)  
            tempSum += Z * Sigma
            gmmNew.append((w, mu, Sigma))
        tied_sigma = tempSum / N
        if (sigma_type == "diagonal"):
                tied_sigma = tied_sigma * numpy.eye(tied_sigma.shape[0])
        U, s, _ = numpy.linalg.svd(tied_sigma)
        s[s<psi] = psi
        tied_sigma = numpy.dot(U, utils.vcol(s)*U.T)
        gmm = []
        for gmmTuple in gmmNew:
            gmm.append((gmmTuple[0], gmmTuple[1], tied_sigma))
        """if (llOld != None and llNew != None):
            print(llNew-llOld)"""
    return gmm

def LBG_tied(X, alpha, G, psi=0.01, sigma_type="tied"):
    mu_g = compute_empirical_mean(X)

    Sigma_g = compute_empirical_covariance(X)
    gmm = [(1.0, mu_g, Sigma_g)]
    N = X.shape[1]
    SJ = numpy.zeros((1,N))
    SJ[0, :] = logpdf_GAU_ND(X, gmm[0][1], gmm[0][2]) + numpy.log(gmm[0][0])
    SM = scipy.special.logsumexp(SJ, axis = 0)
    P = numpy.exp(SJ-SM)
    gamma = P[0, :]
    Z = gamma.sum()
    Sigma_g = (Sigma_g * Z)/N
    if (sigma_type == "diagonal"):
         Sigma_g = Sigma_g * numpy.eye(Sigma_g.shape[0])
    U, s, _ = numpy.linalg.svd(Sigma_g)
    s[s<psi] = psi
    Sigma_g = numpy.dot(U, utils.vcol(s)*U.T)


    #wg, mug, Cg
    gmm =[(1.0, mu_g, Sigma_g)]
    g = 1

    while g < G:
        newGmm = []
        gmm = GMM_EM_tied(X, gmm, psi, sigma_type)
        for g_params in gmm:
            U, s, Vh = numpy.linalg.svd(g_params[2])
            d = U[:, 0:1] * s[0]**0.5 * alpha
            wg = g_params[0]/2
            mu_g_1 = g_params[1] - d
            mu_g_2 = g_params[1] + d
            Sigma_g = g_params[2]
            newGmm.append((wg, mu_g_1, g_params[2]))
            newGmm.append((wg, mu_g_2, g_params[2]))
            gmm = newGmm
        g*=2
    gmm = GMM_EM_tied(X,gmm, psi, sigma_type)
    return gmm

"""
Compute classification for kfold
"""
def GMM_classification(DTR, LTR, DTE, LTE, G, sigma_type = "full"):
    logSJoint = numpy.zeros((2, DTE.shape[1]))
    if sigma_type == "full":
        for label in [0,1]:
            gmm = LBG(DTR[:,LTR==label], 0.1, G)
            logSJoint[label, :] = logpdf_GMM(DTE, gmm)
    elif sigma_type == "diagonal":
        for label in [0,1]:
            gmm = LBG(DTR[:,LTR==label], 0.1, G, sigma_type="diagonal")
            logSJoint[label, :] = logpdf_GMM(DTE, gmm)
    elif sigma_type == "tied":
        for label in [0,1]:
            gmm = LBG_tied(DTR[:,LTR==label], 0.1, G)
            logSJoint[label, :] = logpdf_GMM(DTE, gmm)
    elif sigma_type == "tied_diagonal":
        for label in [0,1]:
            gmm = LBG_tied(DTR[:,LTR==label], 0.1, G, 0.01, "diagonal")
            logSJoint[label, :] = logpdf_GMM(DTE, gmm)
    logSMarginal = utils.vrow(scipy.special.logsumexp(logSJoint, axis = 0))
    logPost = logSJoint -logSMarginal
    post = numpy.exp(logPost)
    llrs = logPost[1,:] - logPost[0,:]
    lPred = post.argmax(0)
    accuracy = (LTE == lPred).sum() / LTE.size
    error_rate = 1-accuracy
    #print(round(error_rate,2)*100)
    return llrs, lPred

def kfold_GMM(D, L, k, G, sigma_type):
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
        scores, pred = GMM_classification(DTrain, LTrain, DTest, LTest, G, sigma_type)
        total_scores += list(scores)
        total_correct_predictions += (pred == LTest).sum()
        accuracy = total_correct_predictions / total_samples_tested
        err_rate = 1 - accuracy
    return numpy.array(total_scores)

def g_tuning(D, L, sigma_type, filename):
    eff_prior = 0.5
    dcfs = []
    D, L = utils.shuffle_dataset(D, L)
    for g in [1,2,4,8,16,32,64]:
        print("current g:",g)
        scores = kfold_GMM(D, L, 5, g, sigma_type)
        minDCF = dcf.compute_min_DCF(scores, L, eff_prior, 1,1)
        print(minDCF)
        dcfs.append(minDCF)
    numpy.save(filename, dcfs)

def g_tuning_all_models():
    D = numpy.load("../npy_data/data/znorm_data.npy")
    L = numpy.load("../npy_data/data/labels.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    L1 = numpy.load("../npy_data/data/labels.npy")
    D, L = utils.shuffle_dataset(D, L)
    Y, L1 = utils.shuffle_dataset(Y, L1)
    g_tuning(D, L, "full", "../npy_data/gmm/gmm_raw_full_cov")
    g_tuning(Y, L1, "full", "../npy_data/gmm/gmm_gauss_full_cov")
    g_tuning(D, L, "diagonal", "../npy_data/gmm/gmm_raw_diagonal_cov")
    g_tuning(Y, L1, "diagonal", "../npy_data/gmm/gmm_gauss_diagonal_cov")
    g_tuning(D, L, "tied", "../npy_data/gmm/gmm_raw_tied_cov")
    g_tuning(Y, L1, "tied", "../npy_data/gmm/gmm_gauss_tied_cov")
    g_tuning(D, L, "tied_diagonal", "../npy_data/gmm/gmm_raw_tied_diagonal_cov")
    g_tuning(Y, L1, "tied_diagonal", "../npy_data/gmm/gmm_gauss_tied_diagonal_cov")

def plot_g_tuning():
    vals = ['1','2','4','8','16','32','64']
    fullCovRaw = numpy.load("../npy_data/gmm/gmm_raw_full_cov.npy")
    fullCovGauss = numpy.load("../npy_data/gmm/gmm_gauss_full_cov.npy")
    diagonalRaw = numpy.load("../npy_data/gmm/gmm_raw_diagonal_cov.npy")
    diagonalGauss = numpy.load("../npy_data/gmm/gmm_gauss_diagonal_cov.npy")
    tiedRaw = numpy.load("../npy_data/gmm/gmm_raw_tied_cov.npy")
    tiedGauss = numpy.load("../npy_data/gmm/gmm_gauss_tied_cov.npy")
    tiedDiagonalRaw = numpy.load("../npy_data/gmm/gmm_raw_tied_diagonal_cov.npy")
    tiedDiagonalGauss = numpy.load("../npy_data/gmm/gmm_gauss_tied_diagonal_cov.npy")
    data = [(fullCovRaw, fullCovGauss, "Full Covariance"), (diagonalRaw, diagonalGauss, "Diagonal"), (tiedRaw, tiedGauss, "Tied"), (tiedDiagonalRaw, tiedDiagonalGauss, "Tied Diagonal")]
    X_axis = numpy.arange(len(vals))
    fig, axs = plt.subplots(2,2, sharey=True)
    fig.set_size_inches(9,7)
    k = 0
    for i in range(2):
        for j in range(2):
            line1 = axs[i][j].bar(X_axis-0.2, data[k][0], 0.4, label = "raw")
            line2 = axs[i][j].bar(X_axis+0.2, data[k][1], 0.4, label = "gaussianized")
            axs[i][j].set_title(data[k][2])
            axs[i][j].set_xticks(X_axis, vals)
            k+=1
    fig.legend([line1, line2], ["raw", "gaussianized"], loc='upper center')
    fig.tight_layout()
    plt.savefig("../images/gmm/g_tuning.pdf", format = "pdf")
    plt.show()

def compute_last_min_dcf(D, L, G, s_type):
    priors = [0.5, 0.1, 0.9]
    D, L = utils.shuffle_dataset(D, L)
    dcfs = []
    for p in priors:
        scores = kfold_GMM(D, L, 5, G, s_type)
        minDCF = dcf.compute_min_DCF(scores, L, p, 1,1)
        dcfs.append(minDCF)
    print(dcfs)

"""
minDCF tabular form training/validation set
"""
def minDCF_tabular():
    D = numpy.load("../npy_data/data/znorm_data.npy")
    L = numpy.load("../npy_data/data/labels.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    print("raw 16 full cov")
    compute_last_min_dcf(D, L, 16, "full")
    print("raw 32 diag cov")
    compute_last_min_dcf(D, L, 32, "diagonal")
    print("guass 4 full")
    compute_last_min_dcf(Y, L, 4, "full")

"""
MODEL BUILDING
"""

def compute_scores(DTE, LTE, gmm):
    logSJoint = numpy.zeros((2, DTE.shape[1]))
    for label in [0,1]:
        logSJoint[label, :] = logpdf_GMM(DTE, gmm[label])
    logSMarginal = utils.vrow(scipy.special.logsumexp(logSJoint, axis = 0))
    logPost = logSJoint -logSMarginal
    post = numpy.exp(logPost)
    llrs = logPost[1,:] - logPost[0,:]
    lPred = post.argmax(0)
    accuracy = (LTE == lPred).sum() / LTE.size
    error_rate = 1-accuracy
    #print(round(error_rate,2)*100)
    return llrs, lPred


def train(DTR, LTR, G, sigma_type, dataset):
    def compute_gmms(DTR, LTR, G):
        gmm_0 = LBG(DTR[:,LTR==0], 0.1, G)
        gmm_1 = LBG(DTR[:,LTR==1], 0.1, G)
        gmm_dict = {
            0: gmm_0,
            1: gmm_1
        }
        return gmm_dict
    
    if sigma_type == "full":
        gmm = compute_gmms(DTR, LTR, G)
    elif sigma_type == "diagonal":
        gmm = compute_gmms(DTR, LTR, G)
    elif sigma_type == "tied":
        gmm = compute_gmms(DTR, LTR, G)
    elif sigma_type == "tied_diagonal":
        gmm = compute_gmms(DTR, LTR, G)
    numpy.save("../model_parameters/gmm/" + dataset + "_" + sigma_type + "_g_" + str(G), gmm)
    
def train_models():
    L = numpy.load("../npy_data/data/labels.npy")
    D = numpy.load("../npy_data/data/znorm_data.npy")
    Y = numpy.load("../npy_data/data/gaussianized_data.npy")
    print("raw full g = 16")
    train(D, L, 16, "full", "raw")
    print("raw diagonal g = 32")
    train(D, L, 32, "diagonal", "raw")
    print("gauss full g = 4")
    train(Y, L, 4, "full", "gauss")

def minDCF_final(D, L, dataset_type, sigma_type, G):
    gmm = numpy.load("../model_parameters/gmm/" + dataset_type + "_" + sigma_type + "_g_" + str(G) + ".npy", allow_pickle=True)
    gmm = gmm.item()
    applications = [0.5, 0.1, 0.9]
    dcfs = []
    for prior in applications:
        scores, pred = compute_scores(D, L, gmm)
        minDCF = dcf.compute_min_DCF(scores, L, prior, 1, 1)
        dcfs.append(minDCF)
    for d in dcfs:
        print(round(d,3), end = " ")
    print()

if __name__ == "__main__":
    g_tuning_all_models()
    plot_g_tuning()
    minDCF_tabular()
    train_models()



