from xmlrpc.client import MAXINT
import numpy
import utils
import logistic_regression as LR
import minDCF as dcf
import matplotlib.pyplot as plt
import pylab
import top_two_models as ttm

"""
SCORE CALIBRATION
"""
def shuffle_scores(scores, L, seed=0):
    numpy.random.seed(seed)
    idx = numpy.random.permutation(len(scores))
    shuffled_scores = scores[idx]
    shuffled_L = L[idx]
    return shuffled_scores, shuffled_L

def split_scores(scores, L):
    half = int(len(scores)/2)
    return (scores[:half], L[:half]), (scores[half:], L[half:])

def shuffle_and_split(scores, L):
    s, L = shuffle_scores(scores, L)
    return split_scores(s,L)


def calibrate_scores(scores, L, filename):
    (est_scores, est_labels), (eval_scores, eval_labels) = shuffle_and_split(scores,L)
    priors = [0.5, 0.1, 0.9]
    pit = [0.5, 0.1, 0.9]
    #minDCFs
    print("minDCF uncalibrated")
    for p in priors:
        min_dcf = dcf.compute_min_DCF(eval_scores, eval_labels, p, 1,1)
        print(round(min_dcf,3), end=" ")
    print()
    print("Actual DCF uncalibrated")
    #uncalibrated
    for p in priors:
        uncalibrated_min_dcf = dcf.compute_act_DCF(eval_scores, eval_labels, p, 1,1 )
        print(round(uncalibrated_min_dcf,3), end=" ")
    print()
    est_scores = utils.vrow(est_scores)
    eval_scores = utils.vrow(eval_scores)
    l = 1e-5
    #calibrated
    print("Actual DCF calibrated")
    for pi_t in pit:
        calibrated_scores = LR.calibration_logreg(est_scores, est_labels, eval_scores, pi_t, l)      
        numpy.save(filename + "_" + str(pi_t), calibrated_scores)      
        for p in priors:
            act_dcf = dcf.compute_act_DCF(calibrated_scores, eval_labels, p ,1,1)
            
            print(round(act_dcf,3), end=" ")
        print()
    calibrated_scores = LR.calibration_logreg(est_scores, est_labels, eval_scores, 0.5, l)
    return calibrated_scores, eval_labels

def plot_bayes_errors(L, sRBF, slQR, filename):
    p = numpy.linspace(-3, 3, 21)
    plt.ylim(0,1.1)
    plt.xlabel(r'$\log{\frac{\widetilde{\pi}}{1-\widetilde{\pi}}}$')
    plt.ylabel("DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, sRBF, L, minCost=False), color='r', label="SVM - act DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, sRBF, L, minCost=True),'--', color='r',label="SVM - min DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, slQR, L, minCost=False), color='b', label="Quad log reg - act DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, slQR, L, minCost=True),'--', color='b', label="Quad log reg - min DCF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format="pdf" )
    plt.show()

#Computes calibrated scores and plots them on a bayes error plot
def plot_bayeses():
    L = numpy.load("../npy_data/data/shuffled_labels.npy")
    scoresRBF = numpy.load("../npy_data/scores_top_two/scoresRBF.npy")
    scoresqLR = numpy.load("../npy_data/scores_top_two/scoresqLR.npy")
    calRBF, labRBF = calibrate_scores(scoresRBF, L, "../npy_data/calibration/RBFcal")
    calqLR, labqLR = calibrate_scores(scoresqLR, L, "../npy_data/calibration/lQRcal")
    plot_bayes_errors(labRBF, calRBF, calqLR, "../images/calibration/bayes_2.pdf")

def plot_bayeses_uncalibrated():
    srbf = numpy.load("../npy_data/fusion/srbf.npy")
    sqlr = numpy.load("../npy_data/fusion/sqlr.npy")
    l = numpy.load("../npy_data/fusion/lte_labels.npy")
    plot_bayes_errors(l, srbf, sqlr, "../images/calibration/bayes_uncalibrated_2.pdf")

"""
FUSION MODEL
"""

def fusion():
    L = numpy.load("../npy_data/data/shuffled_labels.npy")
    scoresRBF = numpy.load("../npy_data/scores_top_two/scoresRBF.npy")
    scoresqLR = numpy.load("../npy_data/scores_top_two/scoresqLR.npy")
    (DTR1, LTR1), (DTE1, LTE1) = shuffle_and_split(scoresRBF, L)
    (DTR2, LTR2), (DTE2, LTE2) = shuffle_and_split(scoresqLR, L)
    DTR = numpy.vstack([DTR1, DTR2])
    DTE = numpy.vstack([DTE1,DTE2])
    
    fused_scores = LR.calibration_logreg(DTR, LTR1, DTE, 0.5, 1e-5)
    numpy.save("../npy_data/fusion/srbf", DTE1)
    numpy.save("../npy_data/fusion/sqlr", DTE2)
    numpy.save("../npy_data/fusion/sf", fused_scores)
    numpy.save("../npy_data/fusion/lte_labels", LTE1)
    #fused_scores, sRBF, sqLR, LTE = fusion(scoresqLR, scoresqLR, L)
    #roc_plots(fused_scores, DTE1, DTE2, LTE1)
    return fused_scores, LTE1

def compute_dcf_for_fusion(fused_scores, L):
    #minDCFs
    priors = [0.5,0.1,0.9]
    for p in priors:
        min_dcf = dcf.compute_min_DCF(fused_scores, L, p, 1,1)
        print("minDCF:", round(min_dcf,3))
    for p in priors:
        min_dcf = dcf.compute_act_DCF(fused_scores, L, p, 1,1)
        print("actDCF:", round(min_dcf,3))


    

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

def roc_plots(sFusion, sRBF, slQR, L):
    roc_plot(sFusion, L, "green", label="Fusion")
    roc_plot(sRBF, L, "red", label="SVM")
    roc_plot(slQR, L, "blue", label = "Quad log reg")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../images/fusion/roc.pdf", format="pdf")
    plt.show()



def plot_bayeses_fusion(srbf, sqlr, fused_scores, labels, filename):
    p = numpy.linspace(-3, 3, 21)
    plt.ylim(0,1.1)
    plt.xlabel(r'$\log{\frac{\widetilde{\pi}}{1-\widetilde{\pi}}}$')
    plt.ylabel("DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, srbf, labels, minCost=False), color='r', label="SVM (cal) - act DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, sqlr, labels, minCost=False), color='b', label="Quad log reg (cal) - act DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, fused_scores, labels, minCost=False), color='g',label="Fusion - act DCF")
    pylab.plot(p, dcf.bayes_error_plot(p, fused_scores, labels, minCost=True),'--', color='g', label="Fusion - min DCF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format="pdf" )
    plt.show()

"""srbf = numpy.load("../npy_data/calibration/lQRcal_0.5.npy")
sqlr = numpy.load("../npy_data/calibration/RBFcal_0.5.npy")
fused_scores = numpy.load("../npy_data/fusion/sf.npy")
labels = numpy.load("../npy_data/fusion/lte_labels.npy")
f = "../images/fusion/bayes.pdf"
roc_plots(fused_scores, srbf, sqlr, labels)"""
#plot_bayeses_fusion(srbf, sqlr, fused_scores, labels)
"""
OPTIMAL THRESHOLD, 1st METHOD
"""
def estimate_optimal_threshold(scores, labels, pi, Cfn, Cfp):
    t = numpy.array(scores)
    t.sort()
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    min_DCF = MAXINT
    opt_t = None
    for _th in t:
        d = dcf.compute_act_DCF(scores, labels, pi, Cfn, Cfp, th = _th)
        if d < min_DCF:
            min_DCF = d
            opt_t = _th
    return opt_t, min_DCF


def estimate_threshold():
    L = numpy.load("../npy_data/data/shuffled_labels.npy")
    scoresRBF = numpy.load("../npy_data/scores_top_two/scoresRBF.npy")
    scoresqLR = numpy.load("../npy_data/scores_top_two/scoresqLR.npy")
    print(scoresRBF.size)
    (DTR1, LTR1), (DTE1, LTE1) = shuffle_and_split(scoresRBF, L)
    (DTR2, LTR2), (DTE2, LTE2) = shuffle_and_split(scoresqLR, L)
    priors = [0.5, 0.1,0.9]
    print("min DCF RBF")
    for p in priors:
        min_DCF = dcf.compute_min_DCF(DTE1, LTE1, p, 1,1)
        print(round(min_DCF,3))
    print("min DCF qLR")
    for p in priors:
        min_DCF = dcf.compute_min_DCF(DTE2, LTE2, p, 1,1)
        print(round(min_DCF,3))
    print("act DCF RBF")
    for p in priors:
        act_dcf = dcf.compute_act_DCF(DTE1, LTE1, p, 1,1)
        print(round(act_dcf,3))
    print("min DCF qLR")
    for p in priors:
        act_dcf = dcf.compute_min_DCF(DTE2, LTE2, p, 1,1)
        print(round(act_dcf,3))
    print("act DCF using optimal threshold RBF")
    for p in priors:
        opt_t, min_DCF = estimate_optimal_threshold(DTR1, LTR1, p, 1,1)
        act_DCF = dcf.compute_act_DCF(DTE1, LTE1, p, 1,1,opt_t)
        print(round(act_DCF,3))
    print("act DCF using optimal threshold qLR")
    for p in priors:
        opt_t, min_DCF = estimate_optimal_threshold(DTR2, LTR2, p, 1,1)
        act_DCF = dcf.compute_act_DCF(DTE2, LTE2, p, 1,1,opt_t)
        print(round(act_DCF,3))
    
    

"""
MODEL BUILDING
"""

def model_calibration():
    L = numpy.load("../npy_data/data/shuffled_labels.npy")
    scoresRBF = numpy.load("../npy_data/scores_top_two/scoresRBF.npy")
    scoresqLR = numpy.load("../npy_data/scores_top_two/scoresqLR.npy")
    vRBF = LR.binary_logreg_train(utils.vrow(scoresRBF), L, 0.5, 1e-5 )
    vQLR = LR.binary_logreg_train(utils.vrow(scoresqLR), L, 0.5, 1e-5 )
    numpy.save("../model_parameters/calibration/vRBF", vRBF)
    numpy.save("../model_parameters/calibration/vQLR", vQLR)
    
def fusion_model():
    L = numpy.load("../npy_data/data/shuffled_labels.npy")
    scoresRBF = numpy.load("../npy_data/scores_top_two/scoresRBF.npy")
    scoresqLR = numpy.load("../npy_data/scores_top_two/scoresqLR.npy")
    DTR = numpy.vstack([scoresRBF,scoresqLR])
    v = LR.binary_logreg_train(DTR, L, 0.5, 1e-5)
    numpy.save("../model_parameters/calibration/fusion_v", v)

def estimate_opt_threshold():
    L = numpy.load("../npy_data/data/shuffled_labels.npy")
    scoresRBF = numpy.load("../npy_data/scores_top_two/scoresRBF.npy")
    scoresqLR = numpy.load("../npy_data/scores_top_two/scoresqLR.npy")
    priors = [0.5, 0.1, 0.9]
    opt_thresholds = []
    for p in priors:
        opt_rbf, _ = estimate_optimal_threshold(scoresRBF, L, p, 1, 1)
        opt_qlr, _ = estimate_optimal_threshold(scoresqLR, L, p, 1,1)
        opt_thresholds.append(opt_rbf)
        opt_thresholds.append(opt_qlr)
    numpy.save("../npy_data/calibration/opt_thresholds", opt_thresholds)

if __name__ == "__main__":
    #1st method
    estimate_threshold()
    """estimate_opt_threshold()
    #FUSION
    fusion()
    fused_scores = numpy.load("../npy_data/fusion/sf.npy")
    labels = numpy.load("../npy_data/fusion/lte_labels.npy")
    #calibrated scores
    plot_bayeses_uncalibrated()
    plot_bayeses()

    compute_dcf_for_fusion(fused_scores, labels)
    srbf = numpy.load("../npy_data/calibration/lQRcal_0.5.npy")
    sqlr = numpy.load("../npy_data/calibration/RBFcal_0.5.npy")
    roc_plots(fused_scores, srbf, sqlr, labels)
    plot_bayeses_fusion(srbf, sqlr, fused_scores, labels, "../images/fusion/bayes.pdf")"""