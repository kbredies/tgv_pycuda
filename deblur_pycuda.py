# -*- coding: utf-8 -*-
#
#   PyCUDA primal-dual algorithms for TV/TGV-constrained imaging problems
#
#   Copyright (C) 2013, 2025 Kristian Bredies (kristian.bredies@uni-graz.at)
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
