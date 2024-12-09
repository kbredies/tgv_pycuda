from dequantize_pycuda import *
from numpy import *
import time
from PIL import Image

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
    
def test_range(lower, upper, f_orig, alphas, tgv=False):
    maxiter = 2000
    vis = -1

    best_psnr = 0
    for alpha in alphas:
        print(("Trying alpha="+repr(alpha)+":"))
        if (tgv):
            tic = time.time()
            u = tgv_dequantize(lower, upper, alpha, 2.0, maxiter, vis)
            toc = time.time() - tic
        else:
            tic = time.time()
            u = tv_dequantize(lower, upper, alpha, maxiter, vis)
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

base="oak_leaf"
bins = 6.0

f = imread("test_data/"+base+".png")
lower = floor(f*bins)/bins
upper = ceil(f*bins)/bins
imwrite("results/"+base+"_quantized.png", 0.5*(lower+upper))

alphas = linspace(0.4,0.5,11)
(u_tgv, alpha_tgv, psnr_tgv, time_tgv) = test_range(lower, upper, 
                                                    f, alphas, tgv=True)
print(("TGV Best parameter: alpha="+repr(alpha_tgv)+", PSNR="+repr(psnr_tgv)
      +", time="+repr(time_tgv)))
imwrite("results/"+base+"_dequantized_tgv.png", u_tgv)

alphas = linspace(0.4,0.5,11)
(u_tv, alpha_tv, psnr_tv, time_tv) = test_range(lower, upper, f, alphas)
print(("TV Best parameter: alpha="+repr(alpha_tv)+", PSNR="+repr(psnr_tv)
      +", time="+repr(time_tv)))
imwrite("results/"+base+"_dequantized_tv.png", u_tv)
