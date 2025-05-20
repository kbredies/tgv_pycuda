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

import scipy.io
from pycuda import gpuarray
from numpy import maximum
from matplotlib.pyplot import imread, imsave, cm
from tikhonov_pycuda import tv_tikhonov, tgv_tikhonov
from linop_pycuda import AccumulationOperator


def do_reconstruction(K, y, samples, alpha, maxiter, name):
    print(f"\nTV reconstruction ({samples} samples)\n-----------------")
    v = tv_tikhonov(K, y[0:samples], alpha, maxiter=maxiter, vis=100)
    v = maximum(0.0, v[::-1, :])
    imsave("results/" + name + "_" + repr(samples) + "_samples_tv.png",
           v, cmap=cm.gray)

    print(f"\nTGV reconstruction ({samples} samples)\n------------------")
    v = tgv_tikhonov(K, y[0:samples], alpha, 2.0, maxiter=maxiter, vis=100)
    v = maximum(0.0, v[::-1, :])
    imsave("results/" + name + "_" + repr(samples) + "_samples_tgv.png",
           v, cmap=cm.gray)


print("Compressed sensing test (mug, 64 x 64)\n-----------------------")

# read data
Phi = scipy.io.loadmat("compressed_sensing/Phi_64.mat").get("Phi")
y = scipy.io.loadmat("compressed_sensing/Mug_64.mat").get("y")

# construct operator
K = AccumulationOperator(Phi, (64, 64))

# do reconstruction
do_reconstruction(K, y, 768, -0.001, 15000, "mug_64")
do_reconstruction(K, y, 384, -0.001, 20000, "mug_64")
do_reconstruction(K, y, 256, -0.001, 30000, "mug_64")
do_reconstruction(K, y, 192, -0.001, 30000, "mug_64")


print("\nCompressed sensing test (violin, 128 x 128)\n-----------------------")

# read data
Phi = scipy.io.loadmat("compressed_sensing/Phi_128.mat").get("Phi")

# construct operator
K = AccumulationOperator(Phi, (128, 128))

# generate data
x = imread('test_data/violin_gray128.png')
x = x[::-1, :]
x_gpu = gpuarray.to_gpu(x.astype('float32').copy(order='F'))
y = gpuarray.zeros(K.get_dest_shape(x.shape), 'float32', order='F')
K.apply(x_gpu, y)
y = y.get()

# do reconstruction
do_reconstruction(K, y, 3072, -10, 40000, "violin_cs_128")
do_reconstruction(K, y, 1536, -10, 40000, "violin_cs_128")
do_reconstruction(K, y, 1024, -10, 40000, "violin_cs_128")
do_reconstruction(K, y, 768, -2, 40000, "violin_cs_128")
