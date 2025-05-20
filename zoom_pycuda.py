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
from pycuda import gpuarray, compiler
from pycuda.elementwise import ElementwiseKernel
from numpy import float32, int32, zeros, sqrt, array
from common_pycuda import block_size_x, block_size_y, block_, get_grid, \
    prepare_data, enlarge_next_power_of_2, display, visualize, gpuarray_copy
import pyfft.cuda as pyfft


# kernel code
kernels = """
    __global__ void tv_update_p(float *u, float *p, float alpha_inv,
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

      pabs = sqrtf(pabs)*alpha_inv;
      pabs = (pabs > 1.0f) ? 1.0f/pabs : 1.0f;

      for (int i=0; i<chans; i++) {
        p[((2*i)  *ny+y)*nx+x] = l_px[threadIdx.x][threadIdx.y][i]*pabs;
        p[((2*i+1)*ny+y)*nx+x] = l_py[threadIdx.x][threadIdx.y][i]*pabs;
      }
    }
    }

    __global__ void tv_update_u_dct(float *u, float *p,
                                    float tau_p, int nx, int ny,
                                    int chans) {

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
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

    __global__ void tv_update_u_avg(float *u, float *p, float *f, int power,
                                    float tau_p, int nx, int ny,
                                    int chans) {

    __shared__ float l_u[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];
    __shared__ float l_avg[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      for (int i=0; i<chans; i++) {
        float div = 0.0f;
        if (x < nx-1) div += p[((2*i  )*ny+y  )*nx+x];
        if (x > 0)    div -= p[((2*i  )*ny+y  )*nx+x-1];
        if (y < ny-1) div += p[((2*i+1)*ny+y  )*nx+x];
        if (y > 0)    div -= p[((2*i+1)*ny+y-1)*nx+x];

        l_u[threadIdx.x][threadIdx.y][i] = u[(i*ny+y)*nx+x] + tau_p*div;
        l_avg[threadIdx.x][threadIdx.y][i] = l_u[threadIdx.x][threadIdx.y][i];
      }
    }

    // compute averages
    int step = 1;
    int mask = 0;
    for (int j=0; j<power; j++) {
      mask = (mask << 1) | 1;

      for (int i=0; i<chans; i++) {
        // reduce in x direction
        __syncthreads();
        if ((threadIdx.x & mask) == 0)
          l_avg[threadIdx.x][threadIdx.y][i] +=
                   l_avg[threadIdx.x+step][threadIdx.y][i];

        // reduce in y direction
        __syncthreads();
        if ((threadIdx.y & mask) == 0)
          l_avg[threadIdx.x][threadIdx.y][i] +=
                   l_avg[threadIdx.x][threadIdx.y+step][i];
      }

      step *= 2;
    }

    // write projection
    float reduction = 1.0f/((float)step*step);
    __syncthreads();
    if ((x < nx) && (y < ny)) {
      for (int i=0; i<chans; i++) {
        u[(i*ny+y)*nx+x] = l_u[threadIdx.x][threadIdx.y][i]
            - l_avg[threadIdx.x & ~mask][threadIdx.y & ~mask][i]*reduction
            + f[(i*(ny >> power) + (y >> power))*(nx >> power) + (x >> power)];
      }
    }
    }

    __global__ void tgv_update_p(float *u, float *w, float *p,
                                 float alpha_inv, float tau_d,
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

      pabs = sqrtf(pabs)*alpha_inv;
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

    __global__ void set_dct_coeff(float *dest, float *src, int power,
                                  int nx, int ny, int chans) {

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      int x_src = x - nx/2; int x_dest = x_src;
      if (x_src < 0) {
        x_src += nx;
        x_dest += (nx << power);
      }

      int y_src = y - ny/2; int y_dest = y_src;
      if (y_src < 0) {
        y_src += ny;
        y_dest += (ny << power);
      }

      for (int i=0; i<chans; i++) {
        dest[(i*(ny << power) + y_dest)*(nx << power) + x_dest]
          = src[(i*ny + y_src)*nx + x_src];
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
complex_assign = ElementwiseKernel("pycuda::complex<float> *v, float* u",
                                   "v[i] = u[i]",
                                   "complex_assign")
real_assign = ElementwiseKernel("float *u, pycuda::complex<float> *v",
                                "u[i] = v[i].real()",
                                "real_assign")
tv_update_p_func = module.get_function("tv_update_p")
tv_update_u_avg_func = module.get_function("tv_update_u_avg")
tv_update_u_dct_func = module.get_function("tv_update_u_dct")
tv_update_u_ = ElementwiseKernel("float *u_, float *u",
                                 "u_[i] = 2.0f*u[i] - u_[i]",
                                 "extragradient_update")
tgv_update_p_func = module.get_function("tgv_update_p")
tgv_update_q_func = module.get_function("tgv_update_q")
tgv_update_u_avg_func = module.get_function("tv_update_u_avg")
tgv_update_u_dct_func = module.get_function("tv_update_u_dct")
tgv_update_w_func = module.get_function("tgv_update_w")
tgv_update_u_ = tv_update_u_
tgv_update_w_ = tgv_update_u_
set_dct_coeff_func = module.get_function("set_dct_coeff")


def tv_update_p(u, p, alpha, tau_d):
    tv_update_p_func(u, p, float32(1.0 / alpha), float32(tau_d),
                     int32(u.shape[0]), int32(u.shape[1]), int32(u.shape[2]),
                     block=block_, grid=get_grid(u))


def tv_update_u_avg(u, p, f, power, tau_p):
    tv_update_u_avg_func(u, p, f, int32(power), float32(tau_p),
                         int32(u.shape[0]), int32(u.shape[1]),
                         int32(u.shape[2]),
                         block=block_, grid=get_grid(u))


def tv_update_u_dct(u_real, u_imag, p, fhat_real, fhat_imag,
                    power, tau_p, plan):
    tv_update_u_dct_func(u_real, p, float32(tau_p),
                         int32(u_real.shape[0]), int32(u_real.shape[1]),
                         int32(u_real.shape[2]),
                         block=block_, grid=get_grid(u_real))

    # forward transform + setting coefficients + inverse transform
    plan.execute(u_real, u_imag, batch=u_real.shape[2])
    set_dct_coeff_func(u_real, fhat_real, int32(power),
                       int32(fhat_real.shape[0]), int32(fhat_real.shape[1]),
                       int32(fhat_real.shape[2]),
                       block=block_, grid=get_grid(fhat_real))
    set_dct_coeff_func(u_imag, fhat_imag, int32(power),
                       int32(fhat_imag.shape[0]), int32(fhat_imag.shape[1]),
                       int32(fhat_imag.shape[2]),
                       block=block_, grid=get_grid(fhat_imag))
    plan.execute(u_real, u_imag, inverse=True, batch=u_real.shape[2])


def tv_zoom(f, power, alpha=0.1, maxiter=500, vis=-1):
    """Perform zooming of f of the factor 2^power with minimal
    total variation weighted with the parameter alpha and maxiter
    iterations."""

    # copy data f on the gpu (fortran order)
    f = prepare_data(f)
    f_gpu = gpuarray.to_gpu(f)

    # get shape of solution
    src_shape = array(f.shape)
    src_shape[0:2] *= (1 << power)
    src_shape = [int(a) for a in src_shape]

    # set up variables
    u = gpuarray.zeros(src_shape, 'float32', order='F')
    u_ = gpuarray_copy(u)
    p = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')

    L2 = 8.0
    k = 0

    if vis > 0:
        display("Zooming image")
        u_vis = zeros(u.shape, 'float32', order='F')
    else:
        u_vis = None

    while k < maxiter:
        tau_p = 1.0 / sqrt(L2)
        tau_d = 1.0 / tau_p / L2

        #############
        # dual update
        tv_update_p(u_, p, alpha, tau_d)

        ###############
        # primal update
        cuda.memcpy_dtod(u_.gpudata, u.gpudata, u.nbytes)
        tv_update_u_avg(u, p, f_gpu, power, tau_p)

        ######################
        # extragradient update
        tv_update_u_(u_, u)

        if (vis > 0) and (k % vis == 0):
            display(".")
            cuda.memcpy_dtoh(u_vis.data, u.gpudata)
            visualize(u_vis)

        k += 1

    if vis > 0:
        display("\n")

    return u.get().squeeze()


def tv_zoom_dct(f, power, alpha=0.1, maxiter=500, vis=-1):
    """Perform DCT-based zooming of f of the factor 2^power with minimal
    total variation weighted with the parameter alpha and maxiter
    iterations."""

    # copy data f on the gpu (fortran order)
    f = prepare_data(f)
    f = enlarge_next_power_of_2(f)
    f_gpu_real = gpuarray.to_gpu(f)
    f_gpu_imag = gpuarray.zeros(f_gpu_real.shape, 'float32', order='F')

    # make plan for data
    plan_f = pyfft.Plan((f.shape[1], f.shape[0]), float32,
                        scale=1.0 / (f.shape[0] * f.shape[1]))
    plan_f.execute(f_gpu_real, f_gpu_imag, batch=f.shape[2])

    # get shape of solution
    src_shape = array(f.shape)
    src_shape[0:2] *= (1 << power)
    src_shape = [int(a) for a in src_shape]

    # set up variables
    u = gpuarray.zeros(src_shape, 'float32', order='F')
    u_imag = gpuarray.zeros(src_shape, 'float32', order='F')
    u_ = gpuarray_copy(u)
    p = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')

    # make plan for solution
    plan_u = pyfft.Plan((u.shape[1], u.shape[0]), float32,
                        scale=1.0 / (u.shape[0] * u.shape[1]))

    L2 = 8.0
    k = 0

    if vis > 0:
        display("Zooming image")
        u_vis = zeros(u.shape, 'complex64', order='F')
    else:
        u_vis = None

    while k < maxiter:
        tau_p = 1.0 / sqrt(L2)
        tau_d = 1.0 / tau_p / L2

        #############
        # dual update
        tv_update_p(u_, p, alpha, tau_d)

        ###############
        # primal update
        cuda.memcpy_dtod(u_.gpudata, u.gpudata, u.nbytes)
        tv_update_u_dct(u, u_imag, p, f_gpu_real, f_gpu_imag,
                        power, tau_p, plan_u)

        ######################
        # extragradient update
        tv_update_u_(u_, u)

        if (vis > 0) and (k % vis == 0):
            display(".")
            cuda.memcpy_dtoh(u_vis.data, u.gpudata)
            visualize(u_vis.real())

        k += 1

    if vis > 0:
        display("\n")

    return u.get().squeeze()


def tgv_update_p(u, w, p, tau_d, alpha):
    tgv_update_p_func(u, w, p, float32(1.0 / alpha), float32(tau_d),
                      int32(u.shape[0]), int32(u.shape[1]), int32(u.shape[2]),
                      block=block_, grid=get_grid(u))


def tgv_update_q(w, q, tau_d, alpha):
    tgv_update_q_func(w, q, float32(1.0 / alpha), float32(tau_d),
                      int32(w.shape[0]), int32(w.shape[1]),
                      int32(w.shape[2] / 2),
                      block=block_, grid=get_grid(w))


def tgv_update_w(w, p, q, tau_p):
    tgv_update_w_func(w, p, q, float32(tau_p),
                      int32(w.shape[0]), int32(w.shape[1]),
                      int32(w.shape[2] / 2),
                      block=block_, grid=get_grid(w))


def tgv_zoom(f, power, alpha1=0.1, fac=2.0, maxiter=500, vis=-1):
    """Perform zooming of f of the factor 2^power with minimal
    total general variation weighted with the parameter
    (fac*alpha1, alpha1) and maxiter iterations."""

    # copy data f on the gpu (fortran order)
    f = prepare_data(f)
    f_gpu = gpuarray.to_gpu(f)

    # get shape of solution
    src_shape = array(f.shape)
    src_shape[0:2] *= (1 << power)
    src_shape = [int(a) for a in src_shape]

    # set up primal variables
    u = gpuarray.zeros(src_shape, 'float32', order='F')
    u_ = gpuarray_copy(u)
    w = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')
    w_ = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                        'float32', order='F')

    # set up dual variables
    p = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')
    q = gpuarray.zeros([u.shape[0], u.shape[1], 3 * u.shape[2]],
                       'float32', order='F')

    L2 = 12.0
    alpha0 = fac * alpha1

    k = 0

    if vis > 0:
        display("Zooming image")
        u_vis = zeros(u.shape, 'float32', order='F')
    else:
        u_vis = None

    while k < maxiter:
        tau_p = 1.0 / sqrt(L2)
        tau_d = 1.0 / tau_p / L2

        #############
        # dual update

        tgv_update_p(u_, w_, p, tau_d, alpha1)
        tgv_update_q(w_, q, tau_d, alpha0)

        ###############
        # primal update

        cuda.memcpy_dtod(u_.gpudata, u.gpudata, u.nbytes)
        cuda.memcpy_dtod(w_.gpudata, w.gpudata, w.nbytes)

        tv_update_u_avg(u, p, f_gpu, power, tau_p)
        tgv_update_w(w, p, q, tau_p)

        ######################
        # extragradient update
        tgv_update_u_(u_, u)
        tgv_update_w_(w_, w)

        if (vis > 0) and (k % vis == 0):
            display(".")
            cuda.memcpy_dtoh(u_vis.data, u.gpudata)
            visualize(u_vis)

        k += 1

    if vis > 0:
        display("\n")
    return u.get().squeeze()


def tgv_zoom_dct(f, power, alpha1=0.1, fac=2.0, maxiter=500, vis=-1):
    """Perform DCT-based zooming of f of the factor 2^power with minimal
    total general variation weighted with the parameter
    (fac*alpha1, alpha1) and maxiter iterations."""

    # copy data f on the gpu (fortran order)
    f = prepare_data(f)
    f = enlarge_next_power_of_2(f)
    f_gpu_real = gpuarray.to_gpu(f)
    f_gpu_imag = gpuarray.zeros(f_gpu_real.shape, 'float32', order='F')

    # make plan for data
    plan_f = pyfft.Plan((f.shape[1], f.shape[0]), float32,
                        scale=1.0 / (f.shape[0] * f.shape[1]))
    plan_f.execute(f_gpu_real, f_gpu_imag, batch=f.shape[2])

    # get shape of solution
    src_shape = array(f.shape)
    src_shape[0:2] *= (1 << power)
    src_shape = [int(a) for a in src_shape]

    # set up primal variables
    u = gpuarray.zeros(src_shape, 'float32', order='F')
    u_imag = gpuarray.zeros(src_shape, 'float32', order='F')
    u_ = gpuarray_copy(u)
    w = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')
    w_ = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                        'float32', order='F')

    # set up dual variables
    p = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')
    q = gpuarray.zeros([u.shape[0], u.shape[1], 3 * u.shape[2]],
                       'float32', order='F')

    # make plan for solution
    plan_u = pyfft.Plan((u.shape[1], u.shape[0]), float32,
                        scale=1.0 / (u.shape[0] * u.shape[1]))

    L2 = 12.0
    alpha0 = fac * alpha1

    k = 0

    if vis > 0:
        display("Zooming image")
        u_vis = zeros(u.shape, 'float32', order='F')
    else:
        u_vis = None

    while k < maxiter:
        tau_p = 1.0 / sqrt(L2)
        tau_d = 1.0 / tau_p / L2

        #############
        # dual update

        tgv_update_p(u_, w_, p, tau_d, alpha1)
        tgv_update_q(w_, q, tau_d, alpha0)

        ###############
        # primal update

        cuda.memcpy_dtod(u_.gpudata, u.gpudata, u.nbytes)
        cuda.memcpy_dtod(w_.gpudata, w.gpudata, w.nbytes)

        tv_update_u_dct(u, u_imag, p, f_gpu_real, f_gpu_imag,
                        power, tau_p, plan_u)
        tgv_update_w(w, p, q, tau_p)

        ######################
        # extragradient update
        tgv_update_u_(u_, u)
        tgv_update_w_(w_, w)

        if (vis > 0) and (k % vis == 0):
            display(".")
            cuda.memcpy_dtoh(u_vis.data, u.gpudata)
            visualize(u_vis)

        k += 1

    if vis > 0:
        display("\n")
    return u.get().squeeze()
