import feature_pre_processing as fpp
import gaussian as gss
import logistic_regression as llr
from minDCF import compute_act_DCF
import numpy
import matplotlib.pyplot as plt
import quadratic_LR as qlr
import svm as lsvm
import quadratic_svm as qsvm
import exp_svm as rbf
import GMM as gmm
import utils
import top_two_models as ttm
import score_calibration as cal
import minDCF as dcf


"""
DATA
"""

def load_and_save_test_data():
    D, L = fpp.load("../Pulsar_Detection/Train.txt")
    DZ = fpp.zNorm(D)
    u = fpp.compute_empirical_mean(D)
    C = fpp.compute_empirical_covariance(DZ)
    std = numpy.std(D, axis = 1)

    DE, LE = fpp.load("../Pulsar_Detection/Test.txt")

    DZE = fpp.compute_zNorm_given_mean(DE, u, std)
    D_PCA_5 = fpp.compute_PCA_given_C(DZE, 5, C)
    D_PCA_6 = fpp.compute_PCA_given_C(DZE, 6, C)
    D_PCA_7 = fpp.compute_PCA_given_C(DZE, 7, C)
    print(DZE.shape)
    Y = fpp.compute_gaussianized_matrix(DZ)
    YE = fpp.compute_gaussianized_matrix_for_test(DZ, DZE)
    print(YE.shape)
    CY = fpp.compute_empirical_covariance(Y)
    Y_PCA_5 = fpp.compute_PCA_given_C(YE, 5, CY)
    Y_PCA_6 = fpp.compute_PCA_given_C(YE, 6, CY)
    Y_PCA_7 = fpp.compute_PCA_given_C(YE, 7, CY)
    numpy.save("../npy_data/test/data/raw_data", DE)
    numpy.save("../npy_data/test/data/labels",LE)
    numpy.save("../npy_data/test/data/gaussianized_data", YE)
    numpy.save("../npy_data/test/data/znorm_data", DZE)
    numpy.save("../npy_data/test/data/raw_data_PCA_5", D_PCA_5)
    numpy.save("../npy_data/test/data/raw_data_PCA_6", D_PCA_6)
    numpy.save("../npy_data/test/data/raw_data_PCA_7", D_PCA_7)
    numpy.save("../npy_data/test/data/gaussianized_data_PCA_5", Y_PCA_5)
    numpy.save("../npy_data/test/data/gaussianized_data_PCA_6", Y_PCA_6)
    numpy.save("../npy_data/test/data/gaussianized_data_PCA_7", Y_PCA_7)

"""
MVG
"""
def minDCF_MVG():
    gss.minDCF_final(D, L, "raw_data")
    gss.minDCF_final(Y, L, "gaussianized_data")
    gss.minDCF_final(D_PCA_5, L, "raw_PCA_5")
    gss.minDCF_final(D_PCA_6, L, "raw_PCA_6")
    gss.minDCF_final(D_PCA_7, L, "raw_PCA_7")
    gss.minDCF_final(Y_PCA_5, L, "gaussianized_PCA_5")
    gss.minDCF_final(Y_PCA_6, L, "gaussianized_PCA_6")
    gss.minDCF_final(Y_PCA_7, L, "gaussianized_PCA_7")

"""
LINEAR LOGISTIC REGRESSION
"""
def plot_tune_lambda_llr():
    lambdas = numpy.logspace(-5, 5, num=50)
    plot_lambda_linear("lrD_", lambdas, True)
    plot_lambda_linear("lrY_", lambdas, False)
    plot_lambda_linear("lrD_PCA_7", lambdas,False)
    plot_lambda_linear("lrY_PCA_7", lambdas, False)

def tune_lambda_llr():
    lambdas = numpy.logspace(-5, 5, num=50)
    llr.tune_lambda(D, L, lambdas, "../npy_data/test/llr/lrD_")
    llr.tune_lambda(Y, L, lambdas, "../npy_data/test/llr/lrY_")
    llr.tune_lambda(D_PCA_7, L, lambdas, "../npy_data/test/llr/lrD_PCA_7")
    llr.tune_lambda(D_PCA_7, L, lambdas, "../npy_data/test/llr/lrY_PCA_7")


def plot_lambda_linear(filename, lambdas, legend):
    file_training = "../npy_data/lambda_tuning/" + filename
    file_test = "../npy_data/test/llr/" + filename
    print(file_test)
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
        data_train = numpy.load(file_training + str(prior) + ".npy")
        data_test = numpy.load(file_test + str(prior)+ ".npy")
        plt.plot(lambdas, data_train,'--', color=color, label = "minDCF" + r"($\widetilde{\pi}=$" + str(prior) + "[Val]")
        plt.plot(lambdas, data_test, color=color, label = "minDCF" + r"($\widetilde{\pi}=$" + str(prior) + "[Eval]")
    if legend:
        plt.legend()
    plt.savefig("../images/test/llr/llr" + filename.replace("_", "") + ".pdf", format="pdf")
    plt.show()

def minDCF_llr():
    llr.minDCF_final(D, L, "raw")
    llr.minDCF_final(Y, L, "gauss")
    llr.minDCF_final(D_PCA_7,L, "raw_pca7")
    llr.minDCF_final(Y_PCA_7, L, "gauss_pca7")

"""
QUADRATIC LOGISTIC REGRESSION
"""
def plot_tune_lambda_qlr():
    plot_lambda_quadratic("quadLR_raw_", True)
    plot_lambda_quadratic("quadLR_gauss_", False)


def tune_lambda_qlr():
    lambdas = numpy.logspace(-5, 3, num=30)
    qlr.tune_lambda(D, L, lambdas, "../npy_data/test/qlr/quadLR_raw_")
    qlr.tune_lambda(Y, L, lambdas, "../npy_data/test/qlr/quadLR_gauss_")

def plot_lambda_quadratic(filename, legend):
    lambdas = numpy.logspace(-5, 3, num=30)
    file_training = "../npy_data/quad_LR/" + filename
    file_test = "../npy_data/test/qlr/" + filename
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
        data_train = numpy.load(file_training + str(prior) + ".npy")
        data_test = numpy.load(file_test + str(prior)+ ".npy")
        plt.plot(lambdas, data_train,'--', color=color, label = "minDCF" + r"($\widetilde{\pi}=$" + str(prior) + "[Val]")
        plt.plot(lambdas, data_test, color=color, label = "minDCF" + r"($\widetilde{\pi}=$" + str(prior) + "[Eval]")
    if legend:
        plt.legend()
    plt.savefig("../images/test/qlr/" + filename + ".pdf", format="pdf")
    plt.show()

def minDCF_qlr():
    qlr.minDCF_final(D, L, "raw")
    qlr.minDCF_final(Y, L, "gauss")
    qlr.minDCF_final(D_PCA_7,L, "raw_pca7")
    qlr.minDCF_final(Y_PCA_7, L, "gauss_pca7")

"""
LINEAR SVM
"""
def plot_C(files, save_file, Cs):
    titles = ["raw unbalanced", "gaussianized unbalanced"]
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(12,7)
    k=0
    for i in range(2):
        axs[i].set_xscale("log")
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("C")
        axs[i].set_ylabel("minDCF")
        axs[i].plot(Cs, numpy.load(files[k] + "0.5.npy"), '--', color="red" , label = "minDCF" + r"($\widetilde{\pi}=$" + str(0.5) + ")[Val]")
        axs[i].plot(Cs, numpy.load(files[k] + "0.1.npy"),'--', color="blue", label  = "minDCF" + r"($\widetilde{\pi}=$" + str(0.1) + ")[Val]")
        axs[i].plot(Cs, numpy.load(files[k] + "0.9.npy"),'--', color="green", label = "minDCF" + r"($\widetilde{\pi}=$" + str(0.9) + ")[Val]")
        axs[i].plot(Cs, numpy.load(files[k+1] + "0.5.npy"), color="red" , label = "minDCF" + r"($\widetilde{\pi}=$" + str(0.5) + ")[EVal]")
        axs[i].plot(Cs, numpy.load(files[k+1] + "0.1.npy"), color="blue", label  = "minDCF" + r"($\widetilde{\pi}=$" + str(0.1) + ")[EVal]")
        axs[i].plot(Cs, numpy.load(files[k+1] + "0.9.npy"), color="green", label  = "minDCF" + r"($\widetilde{\pi}=$" + str(0.9) + ")[EVal]")
        k+=2
    axs[0].legend()
    fig.tight_layout()
    plt.savefig(save_file, format="pdf")
    plt.show()


def C_tuning_svm_linear():
    #unbalanced
    lsvm.C_tuning(D,L,False,"../npy_data/test/linear_svm","raw")
    lsvm.C_tuning(Y,L,False,"../npy_data/test/linear_svm","gauss")

def plot_C_linear():
    Cs = numpy.logspace(-3, 0, num=10)
    f_unbalanced_raw = "../npy_data/linear_svm/linear_svm_raw_unbalanced_"
    f_unbalanced_gauss = "../npy_data/linear_svm/linear_svm_gauss_unbalanced_"
    f_unbalanced_raw_test = "../npy_data/test/linear_svm/linear_svm_raw_unbalanced_"
    f_unbalanced_gauss_test = "../npy_data/test/linear_svm/linear_svm_gauss_unbalanced_"
    files = [f_unbalanced_raw, f_unbalanced_raw_test, f_unbalanced_gauss, f_unbalanced_gauss_test]
    savefile = "../images/test/svm_linear/C_tuning.pdf"
    plot_C(files, savefile, Cs)

def minDCF_lsvm():
    lsvm.minDCF_final(D, L, "raw", "u")
    lsvm.minDCF_final(D, L, "raw", "b")
    lsvm.minDCF_final(Y, L, "gauss", "u")


"""
QUADRATIC SVM
"""

def C_tuning_quadratic():
    qsvm.C_tuning(D, L, False, "../npy_data/test/quadratic_svm", "raw")
    qsvm.C_tuning(Y, L, False, "../npy_data/test/quadratic_svm", "gauss")

def plot_C_quadatric():
    Cs = numpy.logspace(-3, 0, num=10)
    f_unbalanced_raw = "../npy_data/quadratic_svm/quadratic_svm_raw_unbalanced_"
    f_unbalanced_gauss = "../npy_data/quadratic_svm/quadratic_svm_gauss_unbalanced_"
    f_unbalanced_raw_test = "../npy_data/test/quadratic_svm/quadratic_svm_raw_unbalanced_"
    f_unbalanced_gauss_test = "../npy_data/test/quadratic_svm/quadratic_svm_gauss_unbalanced_"
    files = [f_unbalanced_raw, f_unbalanced_raw_test, f_unbalanced_gauss, f_unbalanced_gauss_test]
    savefile = "../images/test/quadratic_svm/C_tuning.pdf"
    plot_C(files, savefile, Cs)

def minDCF_qsvm():
    qsvm.minDCF_final(D, L, "raw", "u")
    qsvm.minDCF_final(D, L, "raw", "b")
    qsvm.minDCF_final(Y, L, "gauss", "u")


"""
RBF
"""

def C_tuning_exp():
    rbf.C_tuning(D, L, False, "../npy_data/test/exp_svm/exp_svm_raw_unbalanced_gamma_")
    rbf.C_tuning(Y,L,False, "../npy_data/test/exp_svm/exp_svm_gauss_unbalanced_gamma_")


def plot_C_rbf():
    Cs = numpy.logspace(-2, 1, num=10)
    f_unbalanced_raw = "../npy_data/exp_svm/exp_svm_raw_unbalanced_gamma_"
    f_unbalanced_gauss = "../npy_data/exp_svm/exp_svm_gauss_unbalanced_gamma_"
    f_unbalanced_raw_test = "../npy_data/test/exp_svm/exp_svm_raw_unbalanced_gamma_"
    f_unbalanced_gauss_test = "../npy_data/test/exp_svm/exp_svm_gauss_unbalanced_gamma_"
    files = [f_unbalanced_raw, f_unbalanced_raw_test, f_unbalanced_gauss, f_unbalanced_gauss_test]
    save_file = "../images/test/exp_svm/C_tuning.pdf"
    titles = ["raw unbalanced", "gaussianized unbalanced"]
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(12,7)
    k=0
    for i in range(2):
        axs[i].set_xscale("log")
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("C")
        axs[i].set_ylabel("minDCF")
        axs[i].plot(Cs, numpy.load(files[k] + "0.1.npy"), '--', color="red" , label = "minDCF" + r"($\gamma=$" + str(0.1) + ")[Val]")
        axs[i].plot(Cs, numpy.load(files[k] + "0.01.npy"),'--', color="orange", label  = "minDCF" + r"($\gamma=$" + str(0.01) + ")[Val]")
        axs[i].plot(Cs, numpy.load(files[k] + "0.001.npy"),'--', color="yellow", label = "minDCF" + r"($\gamma=$" + str(0.001) + ")[Val]")
        axs[i].plot(Cs, numpy.load(files[k+1] + "0.1.npy"), color="red" , label = "minDCF" + r"($\gamma=$" + str(0.1) + ")[EVal]")
        axs[i].plot(Cs, numpy.load(files[k+1] + "0.01.npy"), color="orange", label  = "minDCF" + r"($\gamma=$" + str(0.01) + ")[EVal]")
        axs[i].plot(Cs, numpy.load(files[k+1] + "0.001.npy"), color="yellow", label  = "minDCF" + r"($\gamma=$" + str(0.001) + ")[EVal]")
        k+=2
    axs[0].legend()
    fig.tight_layout()
    plt.savefig(save_file, format="pdf")
    plt.show()

def minDCF_rbf():
    rbf.minDCF_final(D, L, "raw", "u")
    rbf.minDCF_final(D, L, "raw", "b")
    rbf.minDCF_final(Y, L, "gauss", "u")


"""
GMM 
"""

def gmm_tuning():
    gmm.g_tuning(D, L, "full", "../npy_data/test/gmm/gmm_raw_full_cov")
    gmm.g_tuning(D, L, "diagonal", "../npy_data/test/gmm/gmm_raw_diagonal_cov")
    gmm.g_tuning(D, L, "tied", "../npy_data/test/gmm/gmm_raw_tied_cov")
    gmm.g_tuning(D, L, "tied_diagonal", "../npy_data/test/gmm/gmm_raw_tied_diagonal_cov")
    gmm.g_tuning(Y, L, "full", "../npy_data/test/gmm/gmm_gauss_full_cov")
    gmm.g_tuning(Y, L, "diagonal", "../npy_data/test/gmm/gmm_gauss_diagonal_cov")
    gmm.g_tuning(Y, L, "tied", "../npy_data/test/gmm/gmm_gauss_tied_cov")
    gmm.g_tuning(Y, L, "tied_diagonal", "../npy_data/test/gmm/gmm_gauss_tied_diagonal_cov")


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

    test_fullCovRaw = numpy.load("../npy_data/test/gmm/gmm_raw_full_cov.npy")
    test_fullCovGauss = numpy.load("../npy_data/test/gmm/gmm_gauss_full_cov.npy")
    test_diagonalRaw = numpy.load("../npy_data/test/gmm/gmm_raw_diagonal_cov.npy")
    test_diagonalGauss = numpy.load("../npy_data/test/gmm/gmm_gauss_diagonal_cov.npy")
    test_tiedRaw = numpy.load("../npy_data/test/gmm/gmm_raw_tied_cov.npy")
    test_tiedGauss = numpy.load("../npy_data/test/gmm/gmm_gauss_tied_cov.npy")
    test_tiedDiagonalRaw = numpy.load("../npy_data/test/gmm/gmm_raw_tied_diagonal_cov.npy")
    test_tiedDiagonalGauss = numpy.load("../npy_data/test/gmm/gmm_gauss_tied_diagonal_cov.npy")

    data = [(fullCovRaw, fullCovGauss,test_fullCovRaw, test_fullCovGauss, "Full Covariance"),
     (diagonalRaw, diagonalGauss,test_diagonalRaw, test_diagonalGauss, "Diagonal"),
      (tiedRaw, tiedGauss,test_tiedRaw, test_tiedGauss, "Tied"),
       (tiedDiagonalRaw, tiedDiagonalGauss,test_tiedDiagonalRaw, test_tiedDiagonalGauss, "Tied Diagonal")]
    X_axis = numpy.arange(len(vals))
    fig, axs = plt.subplots(2,2, sharey=True)
    fig.set_size_inches(9,7)
    k = 0
    for i in range(2):
        for j in range(2):
            line1 = axs[i][j].bar(X_axis-0.3, data[k][0], 0.2, label = "Raw [Val]")
            line2 = axs[i][j].bar(X_axis-0.1, data[k][1], 0.2, label = "Gaussianized [Val]")
            line3 = axs[i][j].bar(X_axis+0.1, data[k][2], 0.2, label = "Raw [Eval]")
            line4 = axs[i][j].bar(X_axis+0.3, data[k][3], 0.2, label = "Gaussianized [Eval]")
            axs[i][j].set_title(data[k][4])
            axs[i][j].set_xticks(X_axis, vals)
            k+=1
    fig.legend([line1, line2, line3, line4], ["Raw [Val]", "Gaussianized [Val]", "Raw [Eval]", "Gaussianized [Eval]"], loc='upper center')
    fig.tight_layout()
    plt.savefig("../images/test/gmm/g_tuning", format = "pdf")
    plt.show()

def minDCF_gmm():
    gmm.minDCF_final(D, L, "raw", "full", 16)
    gmm.minDCF_final(D, L, "raw", "diagonal", 32)
    gmm.minDCF_final(Y, L, "gauss", "full", 4)




"""
CALIBRATION 1st METHOD
"""
def get_scores():
    fqlr = "../model_parameters/qlr/v_raw_0.1.npy"
    v = numpy.load(fqlr, allow_pickle=True)
    scores_qlr, pred = qlr.compute_scores(v, D, 0.1)
    numpy.save("../npy_data/test/calibration/scoresqLR", scores_qlr)

    frbf = "../model_parameters/exp_svm/w_raw_b_0.5.npy"
    w, a, Z =  numpy.load(frbf, allow_pickle=True)
    DTR = numpy.load("../npy_data/data/znorm_data.npy")
    scores_rbf, pred = rbf.perform_classification_SVM_exp_kernel(DTR, D, L, a, Z, 0.01)
    numpy.save("../npy_data/test/calibration/scoresRBF", scores_rbf)

def act_dcf_opt_threshold():
    slQR = numpy.load("../npy_data/test/calibration/scoresqLR.npy")
    sRBF = numpy.load("../npy_data/test/calibration/scoresRBF.npy")
    opt_th = numpy.load("../npy_data/calibration/opt_thresholds.npy")
    priors = [0.5, 0.1, 0.9]
    k = 0
    for p in priors:
        act_dcf_rbf = dcf.compute_act_DCF(sRBF, L, p, 1,1,opt_th[k])
        print(round(act_dcf_rbf,3))
        act_dcf_qlr = dcf.compute_act_DCF(slQR, L, p, 1,1,opt_th[k+1])
        print(round(act_dcf_qlr,3))
        k+=2
"""
ACTUAL DCF AND CALIBRATION
"""
def calibrate():
    slQR = numpy.load("../npy_data/test/calibration/scoresqLR.npy")
    sRBF = numpy.load("../npy_data/test/calibration/scoresRBF.npy")
    vLQR = numpy.load("../model_parameters/calibration/vQLR.npy")
    vRBF = numpy.load("../model_parameters/calibration/vRBF.npy")
    calScoreslQR, pred = llr.compute_scores(vLQR, utils.vrow(slQR), 0.5)
    calScoresRBF, pred = llr.compute_scores(vRBF, utils.vrow(sRBF), 0.5)
    numpy.save("../npy_data/test/calibration/calibrated_lqr", calScoreslQR)
    numpy.save("../npy_data/test/calibration/calibrated_rbf", calScoresRBF)

def compute_dcfs(u_scores, c_scores):
    priors = [0.5,0.1,0.9]
    print("minDCF uncalibrated")
    for p in priors:
        min_dcf = dcf.compute_min_DCF(u_scores, L, p, 1,1)
        print(round(min_dcf,3), end=" ")
    print()
    print("Actual DCF uncalibrated")
    #uncalibrated
    for p in priors:
        uncalibrated_min_dcf = dcf.compute_act_DCF(u_scores, L, p, 1,1 )
        print(round(uncalibrated_min_dcf,3), end=" ")
    print()
    print("min dcf calibrated")
    for p in priors:
        calibrated_min_dcf = dcf.compute_min_DCF(c_scores, L, p, 1,1 )
        print(round(calibrated_min_dcf,3), end=" ")
    print()
    print("Actual DCF calibrated")
    #calibrated
    for p in priors:
        calibrated_act_dcf = dcf.compute_act_DCF(c_scores, L, p, 1,1 )
        print(round(calibrated_act_dcf,3), end=" ")
    print()

def min_and_act_dcfs_cal():
    sqlr_u = numpy.load("../npy_data/test/calibration/scoresqLR.npy")
    srbf_u = numpy.load("../npy_data/test/calibration/scoresRBF.npy")
    srbf_c = numpy.load("../npy_data/test/calibration/calibrated_rbf.npy")
    sqlr_c = numpy.load("../npy_data/test/calibration/calibrated_lqr.npy")
    compute_dcfs(srbf_u, srbf_c)
    compute_dcfs(sqlr_u, sqlr_c)

"""
FUSION
"""


def fusion_scores():
    slQR = numpy.load("../npy_data/test/calibration/calibrated_lqr.npy")
    sRBF = numpy.load("../npy_data/test/calibration/calibrated_rbf.npy")
    v = numpy.load("../model_parameters/calibration/fusion_v.npy")
    DTE = numpy.vstack([sRBF, slQR])
    fused_scores, _ = llr.compute_scores(v, DTE, 0.5)
    numpy.save("../npy_data/test/fusion/fs", fused_scores)

def minDCF_fusion():
    fs = numpy.load("../npy_data/test/fusion/fs.npy")
    cal.compute_dcf_for_fusion(fs, L)



def roc_plots(L, sRBF, slQR, sf):
    cal.roc_plot(sRBF, L, "red", label="SVM")
    cal.roc_plot(slQR, L, "blue", label = "Quad log reg")
    cal.roc_plot(sf, L, "green", label = "Fusion")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../images/test/top_two_and_fusion/roc.pdf", format="pdf" )
    plt.show()


def roc():
    slQR = numpy.load("../npy_data/test/calibration/scoresqLR.npy")
    sRBF = numpy.load("../npy_data/test/calibration/scoresRBF.npy")
    sf = numpy.load("../npy_data/test/fusion/fs.npy")
    roc_plots(L, sRBF, slQR, sf)

"""
BAYES ERROR PLOTS
"""
def bayes_uncalibrated():
    
    slQR = numpy.load("../npy_data/test/calibration/scoresqLR.npy")
    sRBF = numpy.load("../npy_data/test/calibration/scoresRBF.npy")
    cal.plot_bayes_errors(L, sRBF, slQR, "../images/test/calibration/bayes_top_two_unc.pdf")

def bayes_calibrated():

    srbf = numpy.load("../npy_data/test/calibration/calibrated_rbf.npy")
    sqlr = numpy.load("../npy_data/test/calibration/calibrated_lqr.npy")
    
    cal.plot_bayes_errors(L, srbf, sqlr, "../images/test/calibration/bayes_top_two_cal.pdf")

def bayes_fusion():
    srbf = numpy.load("../npy_data/test/calibration/calibrated_rbf.npy")
    sqlr = numpy.load("../npy_data/test/calibration/calibrated_lqr.npy")
    fs = numpy.load("../npy_data/test/fusion/fs.npy")
    cal.plot_bayeses_fusion(srbf, sqlr, fs, L, "../images/test/calibration/fusion_bayes.pdf")

if __name__ == "__main__":
    #DATA
    load_and_save_test_data()
    L = numpy.load("../npy_data/test/data/labels.npy")
    D = numpy.load("../npy_data/test/data/znorm_data.npy")
    Y = numpy.load("../npy_data/test/data/gaussianized_data.npy")
    D_PCA_5 = numpy.load("../npy_data/test/data/raw_data_PCA_5.npy")
    D_PCA_6 =numpy.load("../npy_data/test/data/raw_data_PCA_6.npy")
    D_PCA_7 =numpy.load("../npy_data/test/data/raw_data_PCA_7.npy")
    Y_PCA_5 =numpy.load("../npy_data/test/data/gaussianized_data_PCA_5.npy")
    Y_PCA_6 = numpy.load("../npy_data/test/data/gaussianized_data_PCA_6.npy")
    Y_PCA_7 = numpy.load("../npy_data/test/data/gaussianized_data_PCA_7.npy")
    #MVG
    minDCF_MVG()
    #LINEAR LOGISTIC REGRESSION
    tune_lambda_llr()
    plot_tune_lambda_llr()
    minDCF_llr()
    #QUADRATIC LOGISTIC REGRESSION
    tune_lambda_qlr()
    plot_tune_lambda_qlr()
    minDCF_qlr()
    #LINEAR SVM
    C_tuning_svm_linear()
    plot_C_linear()
    minDCF_lsvm()
    #QUADRATIC SVM
    C_tuning_quadratic()
    plot_C_quadatric()
    minDCF_qsvm()
    #RBF SVM
    C_tuning_exp()
    plot_C_rbf()
    minDCF_rbf()
    #GMM
    gmm_tuning()
    plot_g_tuning()
    minDCF_gmm()
    #calibration 1st method (opt threshold for application)
    get_scores()
    act_dcf_opt_threshold()
    #calibration 2nd method (llr over scores)
    calibrate()
    min_and_act_dcfs_cal()
    #FUSION
    fusion_scores()
    minDCF_fusion()
    roc()
    #BAYES ERROR PLOTS
    bayes_uncalibrated()
    bayes_calibrated()
    bayes_fusion()
    