# -*- coding: utf-8 -*-
#
#   PyCUDA primal-dual algorithms for TV/TGV-constrained imaging problems
#
#   Copyright (C) 2013, 2024 Kristian Bredies (kristian.bredies@uni-graz.at)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   If you consider this code useful, please cite:
#
#   Kristian Bredies. Recovering piecewise smooth multichannel images by
#   minimization of convex functionals with total generalized variation
#   penalty. Lecture Notes in Computer Science, 8293:44-77, 2014.

import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
from numpy import float32
from common_pycuda import gpuarray_copy


gpuarray_update_r2 = ElementwiseKernel(
    "float *dest, float alpha, float *x, float tau, float *y",
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
    scp = float32(gpuarray_normsqr(p).get()) + tau * \
        float32(gpuarray_normsqr(temp_dest).get())
    return scp


def cg_update_r2(K, tau, r, alpha, p, temp_src, temp_dest):
    # compute K*Kp
    K.adjoint(temp_dest, temp_src)

    # update r <- r - alpha*(p + tau K*Kp)
    gpuarray_update_r2(r, alpha, p, tau, temp_src)


def solve_id_tauKtK(K, tau, y, x, maxiter=100, temp_src=None, temp_dest=None):
    # initialize temporary variables, if necessary
    if temp_src is None:
        temp_src = gpuarray.zeros(x.shape, 'float32', order='F')
    if temp_dest is None:
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
        alpha = rtr / ptAp
        cg_update_r2(K, tau, r, alpha, p, temp_src, temp_dest)

        # compute update for x
        gpuarray_update_x(x, alpha, p)

        # compute update for p
        rtr_old = rtr
        rtr = float32(gpuarray_normsqr(r).get())
        beta = rtr / rtr_old
        print(f"Iteration {i}: beta={beta}, alpha={alpha}, rtr={rtr}, ptAp={ptAp}")

        gpuarray_update_p(p, beta, r)
