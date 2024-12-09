import time
from PIL import Image
from numpy import clip, linspace, log10, sum
from matplotlib.pyplot import imread, ion, ioff, clf, draw, imshow
import pycuda.gpuarray as gpuarray
from zoom_pycuda import tv_zoom, tgv_zoom
from linop_pycuda import ZoomingOperator


def imwrite(fname, data):
    data = clip(255 * data, 0, 255).astype('uint8')
    im = Image.fromarray(data)
    im.save(fname)


def MSE(f, g):
    diff = f.ravel() - g.ravel()
    numel = len(diff)
    return sum(diff * diff) / numel


def PSNR(f, g):
    return 10 * log10(1.0 / MSE(f, g))


def test_range(power, f, f_orig, alphas, tgv=False):
    maxiter = 4000
    vis = -1

    best_psnr = 0
    for alpha in alphas:
        print(f"Trying alpha={alpha}:")
        if tgv:
            tic = time.time()
            u = tgv_zoom(f, power, alpha, 3.0, maxiter, vis)
            toc = time.time() - tic
        else:
            tic = time.time()
            u = tv_zoom(f, power, alpha, maxiter, vis)
            toc = time.time() - tic

        cur_psnr = PSNR(f_orig, u)
        print(f"PSNR: {cur_psnr}")
        ion()
        clf()
        imshow(clip(u, 0, 1))
        ioff()
        draw()

        if cur_psnr > best_psnr:
            u_best = u
            best_alpha = alpha
            best_psnr = cur_psnr
            best_time = toc

    return u_best, best_alpha, best_psnr, best_time


base = "violin"

# read image
f = imread("test_data/" + base + ".png")
# f = enlarge_next_power_of_2(f)

# set up operator
power = 3
# K = DCTZoomingOperator(power, (f.shape[0] >> power, f.shape[1] >> power))
K = ZoomingOperator(power)

# shrink image
f_gpu = gpuarray.to_gpu(f.astype('float32').copy(order='F'))
u = gpuarray.zeros(K.get_dest_shape(f.shape), 'float32', order='F')
K.apply(f_gpu, u)
g = u.get()  # /float(1 << power)

imwrite("results/" + base + "_small.png", g)  # , vmin=0.0, vmax=1.0)

alphas = linspace(0.1, 0.1, 1)
u_tv, alpha_tv, psnr_tv, time_tv = test_range(power, g, f, alphas)
print(f"TV best parameter: alpha={alpha_tv}, PSNR={psnr_tv}, time={time_tv}")
imwrite("results/" + base + "_zoomed_tv.png", u_tv)

alphas = linspace(0.1, 0.1, 1)
u_tgv, alpha_tgv, psnr_tgv, time_tgv = test_range(
    power, g, f, alphas, tgv=True)
print(
    f"TGV best parameter: alpha={alpha_tgv}, PSNR={psnr_tgv}, time={time_tgv}")
imwrite("results/" + base + "_zoomed_tgv.png", u_tgv)
