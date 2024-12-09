import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from numpy import *
from matplotlib.pyplot import *
import sys

block_size_x = 16
block_size_y = 16
block_ = (block_size_x, block_size_y, 1)

def get_grid(u, offset=0):
    grid = (((u.shape[0+offset] + block_[0] - 1)//block_[0]),
            ((u.shape[1+offset] + block_[1] - 1)//block_[1]))
    return(grid)

def gpuarray_copy(u):
    v = gpuarray.zeros_like(u)
    v.strides = u.strides
    cuda.memcpy_dtod(v.gpudata, u.gpudata, u.nbytes)
    return(v)

def prepare_data(f):
    if (len(f.shape) < 3):
        f = f.reshape(f.shape[0], f.shape[1], 1)
    f = f.astype('float32').copy(order='F')
    return(f)

def enlarge_next_power_of_2(f):
    # determine larger shape
    new_shape = array(f.shape)
    new_shape[0:2] = pow(2.0, ceil(log2(new_shape[0:2]))).astype('int32')
    if (new_shape == f.shape).all():
        return(f)

    # zero pad
    offset = (new_shape - array(f.shape))/2
    g = zeros(list(new_shape), 'float32', order='F')
    g[offset[0]:offset[0]+f.shape[0],offset[1]:offset[1]+f.shape[1]] = f
    return(g)

def display(str):
    sys.stdout.write(str)
    sys.stdout.flush()

def visualize(u):
    ion()
    clf()
    if (u.shape[2] > 1):
        imshow(clip(u,0,1))
    else:
        imshow(u.squeeze(), cmap=cm.gray)
    draw()
    ioff()
