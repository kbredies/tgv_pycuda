from tikhonov_pycuda import tv_tikhonov, tgv_tikhonov
from linop_pycuda import ConvolutionOperator


def tv_deblur(f, mask, alpha=0.1, maxiter=500, vis=-1):
    """Perform deblurring of f convolved with mask
    with total variation regularization with parameter
    alpha and maxiter iterations."""

    K = ConvolutionOperator(mask)
    return tv_tikhonov(K, f, alpha, maxiter, vis)


def tgv_deblur(f, mask, alpha=0.1, fac=2.0, maxiter=500, vis=-1):
    """Perform deblurring of f convolved with mask
    with second-order total variation regularization with parameters
    (fac*alpha, alpha) and maxiter iterations."""

    K = ConvolutionOperator(mask)
    return tgv_tikhonov(K, f, alpha, fac, maxiter, vis)
