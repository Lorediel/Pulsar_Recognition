import numpy

def compute_confusion_matrix_binary(pred, labels):
    C = numpy.zeros((2,2))
    C[0,0] = ((pred == 0) * (labels == 0)).sum()
    C[0,1] = ((pred == 0) * (labels == 1)).sum()
    C[1,0] = ((pred == 1) * (labels == 0)).sum()
    C[1,1] = ((pred == 1) * (labels == 1)).sum()
    return C

def compute_optimal_bayes_decisions_binary(scores, labels, pi1, Cfn, Cfp, th=None):
    #This is the theoritical threshold that gives us optimal bayes decisions
    # works only if scores are llrs.
    if th is None:
        th = -numpy.log(pi1 * Cfn) + numpy.log((1-pi1) * Cfp)
    
    pred = scores > th
    return numpy.int32(pred)

def compute_fnr_fpr(CM):
    fnr = CM[0,1] / (CM[0,1] + CM[1,1])
    fpr = CM[1,0] / (CM[0,0] + CM[1,0])
    return fnr, fpr


def compute_emp_Bayes_Binary(CM, pi, Cfn, Cfp):
    fnr, fpr = compute_fnr_fpr(CM)
    return pi * Cfn * fnr + (1-pi) * Cfp * fpr
    

def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp):
    empBayes = compute_emp_Bayes_Binary(CM, pi, Cfn, Cfp)
    return empBayes / min(pi*Cfn, (1-pi)*Cfp)



# assume scores are llrs, cause otherwise we have issues.
def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = compute_optimal_bayes_decisions_binary(scores, labels, pi, Cfn, Cfp, th=th)
    CM = compute_confusion_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)


def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    t = numpy.array(scores)
    t.sort()
    #I consider all threshold such that FNR and FPR change. Just need to consider those.
    #same as the ROC
    #I put one score too much depending if I use s>t or s>=t. I put both +inf and -inf to be sure I don't miss one
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    #bayes cost list
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th = _th))
    return numpy.array(dcfList).min()


# pArray is an array of priors of the target class
def bayes_error_plot(pArray, scores, labels, minCost = False):
    y = []
    #for every application:
    for p in pArray:
        pi = 1.0 / (1.0 + numpy.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1))
    return numpy.array(y)



