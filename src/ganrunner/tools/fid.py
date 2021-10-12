import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(act1, act2):
    """
    Calculates the Frechet inception distance:
    d^2 = ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2))
    """
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means,
    ssdiff = np.sum((mu1 - mu2) ** 2.0)  # ||mu_1 – mu_2||^2
    # calculate sqrt of product between cov
    covmean = 2.0 * sqrtm(sigma1.dot(sigma2))  # 2*sqrt(C_1*C_2)
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - covmean)
   # print("FID:", fid)
    return fid
