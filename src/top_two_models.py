import pylab
import numpy
import minDCF as dcf
import utils
import scipy.optimize
import exp_svm as rbf
import matplotlib
import matplotlib.pyplot as plt
import quadratic_LR as qLR


def roc_plot(llrs, L,color, label):
    thresholds = numpy.array(llrs)
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    FPR = numpy.zeros(thresholds.size)
    TPR = numpy.zeros(thresholds.size)
    #t è la threshold
    for idx, t in enumerate(thresholds):  
        Pred = numpy.int32(llrs>t)
        Conf = dcf.compute_confusion_matrix_binary(Pred, L)
        TPR[idx] = Conf[1,1] / (Conf[1,1] + Conf[0,1])
        FPR[idx] = Conf[1,0] / (Conf[1,0] + Conf[0,0])

    pylab.plot(FPR, TPR, color= color, label = label)

#Computing the scores
def compute_scores_qLR(D, L):
    D, L = utils.shuffle_dataset(D, L)
    DE = qLR.compute_expanded_features(D)
    scoresqLR = qLR.kfold(DE, L, 5, qLR.binary_logreg, 1e-5, 0.1)
    numpy.save("../npy_data/scores_top_two/scoresqLR", scoresqLR)

def compute_scores_rbf(D, L):
    D, L = utils.shuffle_dataset(D, L)
    scoresrbf = rbf.kfold_exp_svm(D, L, 5, 10, 0.5, 0.01, True)
    numpy.save("../npy_data/data/shuffled_labels", L)
    numpy.save("../npy_data/scores_top_two/scoresRBF", scoresrbf)

def plot_bayes_errors(D, L, sRBF, slQR):
    D, L = utils.shuffle_dataset(D,L)
    p = numpy.linspace(-3, 3, 21)
    matplotlib.pyplot.ylim(0,1.1)
    plt.xlabel(r'$\log{\frac{\widetilde{\pi}}{1-\widetilde{\pi}}}$')
    plt.ylabel("DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, sRBF, L, minCost=False), color='r', label="SVM - act DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, sRBF, L, minCost=True),'--', color='r',label="SVM - min DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, slQR, L, minCost=False), color='b', label="Quad log reg - act DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, slQR, L, minCost=True),'--', color='b', label="Quad log reg - min DCF")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../images/top_two/bayes.pdf", format="pdf" )
    plt.show()


def roc_plots(D, L, sRBF, slQR):
    D, L = utils.shuffle_dataset(D,L)
    roc_plot(sRBF, L, "red", label="SVM")
    roc_plot(slQR, L, "blue", label = "Quad log reg")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("../images/top_two/roc.pdf", format="pdf" )

def compute_act_dcf(D, L, scores):
    D, L = utils.shuffle_dataset(D, L)
    priors = [0.5, 0.1, 0.9]
    act_dcfs = []
    for p in priors:
        act_DCF = dcf.compute_act_DCF(scores, L, p, 1, 1)
        act_dcfs.append(act_DCF)
    for a in act_dcfs:
        print(round(a,3), end=" ")
    print()
    return act_dcfs

if __name__ == "__main__":
    D = numpy.load("../npy_data/data/znorm_data.npy")
    L = numpy.load("../npy_data/data/labels.npy")
    """
    
    """
    compute_scores_rbf(D, L)
    compute_scores_qLR(D, L)
    """
    ACTUAL DCFs
    """
    sRBF = numpy.load("../npy_data/scores_top_two/scoresRBF.npy")
    slQR = numpy.load("../npy_data/scores_top_two/scoresqLR.npy")
    print("act dcf rbf")
    compute_act_dcf(D, L, sRBF)
    print("act dcf rbf")
    compute_act_dcf(D, L, slQR)
    plot_bayes_errors(D, L, sRBF, slQR)


