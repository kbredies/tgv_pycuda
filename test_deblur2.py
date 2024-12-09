from deblur_pycuda import *
from linop_pycuda import *
from numpy import *
import time
import Image

def imwrite(fname, data):
    data = clip(255*data,0,255).astype('uint8')
    im = Image.fromarray(data)
    im.save(fname)

def MSE(f,g):
    diff = f.ravel()-g.ravel()
    numel = len(diff)
    return(sum(diff*diff)/numel)

def PSNR(f,g):
    return(10*log10(1.0/MSE(f,g)))
    
def test_range(K, f, f_orig, alphas, tgv=False):
    maxiter = 1000
    vis = -1

    best_psnr = 0
    for alpha in alphas:
        print(("Trying alpha="+repr(alpha)+":"))
        if (tgv):
            tic = time.time()
            u = tgv_tikhonov(K, f, alpha, 4.0, maxiter, vis)
            toc = time.time() - tic
        else:
            tic = time.time()
            u = tv_tikhonov(K, f, alpha, maxiter, vis)
            toc = time.time() - tic

        cur_psnr = PSNR(f_orig,u)
        print(("PSNR: "+repr(cur_psnr)))
        ion(); clf(); imshow(clip(u,0,1)); ioff(); draw()

        if (cur_psnr > best_psnr):
            u_best = u
            best_alpha = alpha
            best_psnr = cur_psnr
            best_time = toc
            
    return((u_best, best_alpha, best_psnr, best_time))

base="peppers"

# create linear operator
x = list(range(-7,8))
[X,Y] = meshgrid(x,x)
mask = (X*X + Y*Y <= 7*7).astype('float32')
mask = mask/sum(mask[:])
K = ConvolutionOperator(mask)

# blur image and add noise
f = imread("test_data/"+base+".png")
f_gpu = gpuarray.to_gpu(f.astype('float32').copy(order='F'))
Kf = gpuarray.zeros(K.get_dest_shape(f.shape),'float32',order='F')
K.apply(f_gpu,Kf)
Kf = Kf.get()
random.seed(14031621)
noise = random.randn(*Kf.shape)*0.05
g = Kf + noise
imwrite("results/"+base+"_deblur_noisy.png", g)

alphas = linspace(0.006,0.007,11)
(u_tgv, alpha_tgv, psnr_tgv, time_tgv) = test_range(K, g, f, alphas, tgv=True)
print(("TGV Best parameter: alpha="+repr(alpha_tgv)+", PSNR="+repr(psnr_tgv)
      +", time="+repr(time_tgv)))
imwrite("results/"+base+"_deblurred_tgv.png", u_tgv)

alphas = linspace(0.0075,0.0085,11)
(u_tv, alpha_tv, psnr_tv, time_tv) = test_range(K, g, f, alphas)
print(("TV Best parameter: alpha="+repr(alpha_tv)+", PSNR="+repr(psnr_tv)
      +", time="+repr(time_tv)))
imwrite("results/"+base+"_deblurred_tv.png", u_tv)
