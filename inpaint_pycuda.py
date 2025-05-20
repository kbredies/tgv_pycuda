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

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import compiler
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from numpy import float32, int32, zeros, sqrt
from common_pycuda import block_size_x, block_size_y, block_, \
    get_grid, gpuarray_copy, prepare_data, display, visualize


# kernel code
kernels = """
    __global__ void tv_update_p(float *u, float *p,
                                float tau_d, int nx, int ny, int chans) {
    __shared__ float l_px[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];
    __shared__ float l_py[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      float pabs;
      pabs = 0.0f;

      for (int i=0; i<chans; i++) {
        float u00 =              u[(i*ny+y)  *nx+x];
        float u10 = (x < nx-1) ? u[(i*ny+y)  *nx+x+1] : u00;
        float u01 = (y < ny-1) ? u[(i*ny+y+1)*nx+x] : u00;

        float dxp = u10-u00;
        float dyp = u01-u00;

        float px = p[((2*i)  *ny+y)*nx+x] + tau_d*dxp;
        float py = p[((2*i+1)*ny+y)*nx+x] + tau_d*dyp;

        l_px[threadIdx.x][threadIdx.y][i] = px;
        l_py[threadIdx.x][threadIdx.y][i] = py;

        pabs += px*px + py*py;
      }

      pabs = sqrtf(pabs);
      pabs = (pabs > 1.0f) ? 1.0f/pabs : 1.0f;

      for (int i=0; i<chans; i++) {
        p[((2*i)  *ny+y)*nx+x] = l_px[threadIdx.x][threadIdx.y][i]*pabs;
        p[((2*i+1)*ny+y)*nx+x] = l_py[threadIdx.x][threadIdx.y][i]*pabs;
      }
    }
    }

    __global__ void tv_update_u(float *u, float *p, int *mask,
                                float tau_p, int nx, int ny, int chans) {

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      if (mask[y*nx+x] == 0) {
        for (int i=0; i<chans; i++) {
          float div = 0.0f;
          if (x < nx-1) div += p[((2*i  )*ny+y  )*nx+x];
          if (x > 0)    div -= p[((2*i  )*ny+y  )*nx+x-1];
          if (y < ny-1) div += p[((2*i+1)*ny+y  )*nx+x];
          if (y > 0)    div -= p[((2*i+1)*ny+y-1)*nx+x];

          u[(i*ny+y)*nx+x] += tau_p*div;
        }
      }
    }
    }

    __global__ void tgv_update_p(float *u, float *w, float *p,
                                 float tau_d,
                                 int nx, int ny, int chans) {

    __shared__ float l_px[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];
    __shared__ float l_py[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      float pabs;
      pabs = 0.0f;

      for (int i=0; i<chans; i++) {
        float u00 =              u[(i*ny+y)  *nx+x];
        float u10 = (x < nx-1) ? u[(i*ny+y)  *nx+x+1] : u00;
        float u01 = (y < ny-1) ? u[(i*ny+y+1)*nx+x]   : u00;

        float dxp = u10-u00;
        float dyp = u01-u00;

        float px = p[((2*i)  *ny+y)*nx+x]
                   + tau_d*(dxp - w[((2*i)  *ny+y)*nx+x]);
        float py = p[((2*i+1)*ny+y)*nx+x]
                   + tau_d*(dyp - w[((2*i+1)*ny+y)*nx+x]);

        l_px[threadIdx.x][threadIdx.y][i] = px;
        l_py[threadIdx.x][threadIdx.y][i] = py;

        pabs += px*px + py*py;
      }

      pabs = sqrtf(pabs);
      pabs = (pabs > 1.0f) ? 1.0f/pabs : 1.0f;

      for (int i=0; i<chans; i++) {
        p[((2*i)  *ny+y)*nx+x] = l_px[threadIdx.x][threadIdx.y][i]*pabs;
        p[((2*i+1)*ny+y)*nx+x] = l_py[threadIdx.x][threadIdx.y][i]*pabs;
      }
    }
    }

    __global__ void tgv_update_q(float *w, float *q, float alpha_inv,
                                 float tau_d, int nx, int ny, int chans) {

    __shared__ float l_qxx[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];
    __shared__ float l_qyy[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];
    __shared__ float l_qxy[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      float qabs;
      qabs = 0.0f;

      for (int i=0; i<chans; i++) {
        float wx00 =              w[((2*i)*ny+y  )*nx+x];
        float wx10 = (x < nx-1) ? w[((2*i)*ny+y  )*nx+x+1] : wx00;
        float wx01 = (y < ny-1) ? w[((2*i)*ny+y+1)*nx+x]   : wx00;

        float wy00 =              w[((2*i+1)*ny+y  )*nx+x];
        float wy10 = (x < nx-1) ? w[((2*i+1)*ny+y  )*nx+x+1] : wy00;
        float wy01 = (y < ny-1) ? w[((2*i+1)*ny+y+1)*nx+x]   : wy00;

        float wxx = wx10-wx00;
        float wyy = wy01-wy00;
        float wxy = 0.5f*(wx01-wx00+wy10-wy00);

        float qxx = q[((3*i  )*ny+y)*nx+x] + tau_d*wxx;
        float qyy = q[((3*i+1)*ny+y)*nx+x] + tau_d*wyy;
        float qxy = q[((3*i+2)*ny+y)*nx+x] + tau_d*wxy;

        l_qxx[threadIdx.x][threadIdx.y][i] = qxx;
        l_qyy[threadIdx.x][threadIdx.y][i] = qyy;
        l_qxy[threadIdx.x][threadIdx.y][i] = qxy;

        qabs += qxx*qxx + qyy*qyy + 2.0f*qxy*qxy;
      }

      qabs = sqrtf(qabs)*alpha_inv;
      qabs = (qabs > 1.0f) ? 1.0f/qabs : 1.0f;

      for (int i=0; i<chans; i++) {
        q[((3*i)  *ny+y)*nx+x] = l_qxx[threadIdx.x][threadIdx.y][i]*qabs;
        q[((3*i+1)*ny+y)*nx+x] = l_qyy[threadIdx.x][threadIdx.y][i]*qabs;
        q[((3*i+2)*ny+y)*nx+x] = l_qxy[threadIdx.x][threadIdx.y][i]*qabs;
      }
    }
    }

    __global__ void tgv_update_w(float *w, float *p, float *q,
                                 float tau_p, int nx, int ny, int chans) {

    float div1, div2;

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      for (int i=0; i<chans; i++) {
        div1 = p[((2*i)*ny+y)*nx+x];
        if (x < nx-1) div1 += q[((3*i  )*ny+y  )*nx+x];
        if (x > 0)    div1 -= q[((3*i  )*ny+y  )*nx+x-1];
        if (y < ny-1) div1 += q[((3*i+2)*ny+y  )*nx+x];
        if (y > 0)    div1 -= q[((3*i+2)*ny+y-1)*nx+x];

        div2 = p[((2*i+1)*ny+y)*nx+x];
        if (x < nx-1) div2 += q[((3*i+2)*ny+y  )*nx+x];
        if (x > 0)    div2 -= q[((3*i+2)*ny+y  )*nx+x-1];
        if (y < ny-1) div2 += q[((3*i+1)*ny+y  )*nx+x];
        if (y > 0)    div2 -= q[((3*i+1)*ny+y-1)*nx+x];

        w[((2*i  )*ny+y)*nx+x] += tau_p*div1;
        w[((2*i+1)*ny+y)*nx+x] += tau_p*div2;
      }
    }
    }
    """
kernels = kernels % {
    'BLOCK_SIZE_X': block_size_x,
    'BLOCK_SIZE_Y': block_size_y,
}

# compile kernels
module = compiler.SourceModule(kernels)
tv_update_p_func = module.get_function("tv_update_p")
tv_update_u_func = module.get_function("tv_update_u")
tv_update_u_ = ElementwiseKernel("float *u_, float *u",
                                 "u_[i] = 2.0f*u[i] - u_[i]",
                                 "extragradient_update")
tgv_update_p_func = module.get_function("tgv_update_p")
tgv_update_q_func = module.get_function("tgv_update_q")
tgv_update_u_func = module.get_function("tv_update_u")
tgv_update_w_func = module.get_function("tgv_update_w")
tgv_update_u_ = tv_update_u_
tgv_update_w_ = tgv_update_u_


def tv_update_p(u, p, tau_d):
    tv_update_p_func(u, p, float32(tau_d),
                     int32(u.shape[0]), int32(u.shape[1]), int32(u.shape[2]),
                     block=block_, grid=get_grid(u))


def tv_update_u(u, p, mask, tau_p):
    tv_update_u_func(u, p, mask, float32(tau_p),
                     int32(u.shape[0]), int32(u.shape[1]), int32(u.shape[2]),
                     block=block_, grid=get_grid(u))


def tv_inpaint(f, mask, maxiter=1000, vis=False):
    """Perform total variation inpainting of f of zero regions of mask
    with maxiter iterations."""

    f = prepare_data(f)
    f_gpu = gpuarray.to_gpu(f)

    mask = mask.astype('int32').copy(order='F')
    mask_gpu = gpuarray.to_gpu(mask)

    u = gpuarray_copy(f_gpu)
    u_ = gpuarray_copy(u)
    p = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')

    L2 = 8.0
    k = 0

    if vis:
        display("Inpainting image")
        u_vis = zeros(f.shape, 'float32', order='F')
    else:
        u_vis = None

    while k < maxiter:
        tau_p = sqrt(L2)  # /(k+50.0)
        tau_d = 1.0 / tau_p / L2

        #############
        # dual update
        tv_update_p(u_, p, tau_d)

        ###############
        # primal update
        cuda.memcpy_dtod(u_.gpudata, u.gpudata, u.nbytes)
        tv_update_u(u, p, mask_gpu, tau_p)

        ######################
        # extragradient update
        tv_update_u_(u_, u)

        if vis and (k % 50 == 0):
            display(".")
            cuda.memcpy_dtoh(u_vis.data, u.gpudata)
            visualize(u_vis)

        k += 1

    if vis:
        display("\n")

    return u.get().squeeze()


def tgv_update_p(u, w, p, tau_d):
    tgv_update_p_func(u, w, p, float32(tau_d),
                      int32(u.shape[0]), int32(u.shape[1]), int32(u.shape[2]),
                      block=block_, grid=get_grid(u))


def tgv_update_q(w, q, tau_d, alpha):
    tgv_update_q_func(w, q, float32(1.0 / alpha), float32(tau_d),
                      int32(w.shape[0]), int32(w.shape[1]),
                      int32(w.shape[2] / 2),
                      block=block_, grid=get_grid(w))


def tgv_update_u(u, p, mask, tau_p):
    tgv_update_u_func(u, p, mask, float32(tau_p),
                      int32(u.shape[0]), int32(u.shape[1]), int32(u.shape[2]),
                      block=block_, grid=get_grid(u))


def tgv_update_w(w, p, q, tau_p):
    tgv_update_w_func(w, p, q, float32(tau_p),
                      int32(w.shape[0]), int32(w.shape[1]),
                      int32(w.shape[2] / 2),
                      block=block_, grid=get_grid(w))


def tgv_inpaint(f, mask, fac=2.0, maxiter=1000, vis=False):
    """Perform second-order total generalized variation inpainting of f
    of zero regions of mask parameters (alpha0,alpha1) = (fac,1)
    and maxiter iterations."""

    # primal variables
    f = prepare_data(f)
    f_gpu = gpuarray.to_gpu(f)

    mask = mask.astype('int32').copy(order='F')
    mask_gpu = gpuarray.to_gpu(mask)

    u = gpuarray_copy(f_gpu)
    u_ = gpuarray_copy(u)
    w = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')
    w_ = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                        'float32', order='F')

    # dual variables
    p = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')
    q = gpuarray.zeros([u.shape[0], u.shape[1], 3 * u.shape[2]],
                       'float32', order='F')

    alpha0 = fac
    L2 = 12.0

    k = 0

    if vis:
        display("Inpainting image")
        u_vis = zeros(f.shape, 'float32', order='F')
    else:
        u_vis = None

    while k < maxiter:
        tau_p = sqrt(L2) / (2000.0)
        tau_d = 1.0 / tau_p / L2

        #############
        # dual update

        tgv_update_p(u_, w_, p, tau_d)
        tgv_update_q(w_, q, tau_d, alpha0)

        ###############
        # primal update

        cuda.memcpy_dtod(u_.gpudata, u.gpudata, u.nbytes)
        cuda.memcpy_dtod(w_.gpudata, w.gpudata, w.nbytes)

        tgv_update_u(u, p, mask_gpu, tau_p)
        tgv_update_w(w, p, q, tau_p)

        ######################
        # extragradient update
        tgv_update_u_(u_, u)
        tgv_update_w_(w_, w)

        if vis and (k % 500 == 0):
            display(".")
            cuda.memcpy_dtoh(u_vis.data, u.gpudata)
            visualize(u_vis)

        k += 1

    if vis:
        display("\n")
    return u.get().squeeze()
