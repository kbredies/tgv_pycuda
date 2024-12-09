from common_pycuda import *
from tikhonov_pycuda import *
from linop_pycuda import *
import scipy.io

def do_reconstruction(K, y, samples, alpha, maxiter, name):
    v = tgv_tikhonov(K, y[0:samples], alpha, 2.0, maxiter=maxiter, vis=100)
    v = maximum(0.0, v[::-1,:])
    imsave("results/"+name+"_"+repr(samples)+"_samples_tgv.png",
           v,cmap=cm.gray)

    v = tv_tikhonov(K, y[0:samples], alpha, maxiter=maxiter, vis=100)
    v = maximum(0.0, v[::-1,:])
    imsave("results/"+name+"_"+repr(samples)+"_samples_tv.png",
           v,cmap=cm.gray)

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



# read data
Phi = scipy.io.loadmat("compressed_sensing/Phi_128.mat").get("Phi")
#y = scipy.io.loadmat("compressed_sensing/R_128.mat").get("y")

# construct operator
K = AccumulationOperator(Phi, (128, 128))

# generate data
x = imread('test_data/violin_gray128.png')
x = x[::-1,:]
x_gpu = gpuarray.to_gpu(x.astype('float32').copy(order='F'))
y = gpuarray.zeros(K.get_dest_shape(x.shape),'float32',order='F')
K.apply(x_gpu, y)
y = y.get()


# do reconstruction
do_reconstruction(K, y, 3072, -10, 40000, "violin_cs_128")
do_reconstruction(K, y, 1536, -10, 40000, "violin_cs_128")
do_reconstruction(K, y, 1024, -10, 40000, "violin_cs_128")
do_reconstruction(K, y, 768, -2, 40000, "violin_cs_128")


#do_reconstruction(K, y, 1536, -0.05, 40000, "r_128")
#do_reconstruction(K, y, 768, -0.05, 40000, "r_128")
#do_reconstruction(K, y, 512, -0.05, 40000, "r_128")
#do_reconstruction(K, y, 384, -0.05, 40000, "r_128")
#do_reconstruction(K, y, 192, -0.05, 40000, "r_128")


#do_reconstruction(K, y, 1536, -0.005, 20000, "r_128")
#do_reconstruction(K, y, 768, 0.005, 25000, "r_128")
#do_reconstruction(K, y, 384, -0.005, 30000, "r_128")
#do_reconstruction(K, y, 192, -0.005, 35000, "r_128")

