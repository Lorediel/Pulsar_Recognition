import numpy

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

def shuffle_dataset(D, L, seed = 0):
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    shuffled_D = D[:, idx]
    shuffled_L = L[idx]
    return shuffled_D, shuffled_L

    
