from common_pycuda import *
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
from numpy import *
from matplotlib.pyplot import *

gpuarray_update_r2 = ElementwiseKernel("float *dest, float alpha, float *x, float tau, float *y",
                                       "dest[i] -= alpha*(x[i] + tau*y[i])",
                                       "update_r2")

gpuarray_update_x = ElementwiseKernel("float *x, float alpha, float *p",
                                       "x[i] += alpha*p[i]",
                                       "update_x")

gpuarray_update_p = ElementwiseKernel("float *p, float beta, float *r",
                                       "p[i] = r[i] + beta*p[i]",
                                       "update_p")

gpuarray_normsqr = ReductionKernel('float32', neutral="0",
                                   reduce_expr="a+b", map_expr="x[i]*x[i]",
                                   arguments="float *x")

def cg_update_r1(K, tau, p, temp_dest):
    # compute Kp
    K.apply(p, temp_dest)

    # compute <p, (id + tau K*K) p>
    scp = float32(gpuarray_normsqr(p).get()) + tau*float32(gpuarray_normsqr(temp_dest).get())
    return scp

def cg_update_r2(K, tau, r, alpha, p, temp_src, temp_dest):
    # compute K*Kp
    K.adjoint(temp_dest, temp_src)

    # update r <- r - alpha*(p + tau K*Kp)
    gpuarray_update_r2(r, alpha, p, tau, temp_src)
    
def solve_id_tauKtK(K, tau, y, x, maxiter=100, temp_src=None, temp_dest=None):
    # initialize temporary variables, if necessary
    if (temp_src == None):
        temp_src = gpuarray.zeros(x.shape, 'float32', order='F')
    if (temp_dest == None):
        temp_dest = gpuarray.zeros(K.get_dest_shape(x.shape), 
                                   'float32', order='F')

    # compute p = r = y - (I + tau K*K)x
    r = gpuarray_copy(y)
    ptAp = cg_update_r1(K, tau, x, temp_dest)
    cg_update_r2(K, tau, r, 1.0, x, temp_src, temp_dest)
    p = gpuarray_copy(r)
    
    # initialize <r, r>
    rtr = float32(gpuarray_normsqr(r).get())

    # perform CG iteration
    for i in range(maxiter):
        # compute update for r
        ptAp = cg_update_r1(K, tau, p, temp_dest)
        alpha = rtr/ptAp
        cg_update_r2(K, tau, r, alpha, p, temp_src, temp_dest)

        # compute update for x
        gpuarray_update_x(x, alpha, p)
        
        # compute update for p
        rtr_old = rtr
        rtr = float32(gpuarray_normsqr(r).get())
        beta = rtr/rtr_old
        print("beta="+repr(beta)+" alpha="+repr(alpha)+" rtr="+repr(rtr)+" ptAp="+repr(ptAp))
        
        gpuarray_update_p(p, beta, r)

    
