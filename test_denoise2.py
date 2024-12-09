import time
from PIL import Image
from numpy import clip, linspace, log10, random, sum
from matplotlib.pyplot import imread, ion, ioff, clf, imshow, draw
from denoise_pycuda import tv_denoise, tgv_denoise


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


def test_range(f, f_orig, alphas, tgv=False):
    maxiter = 500
    vis = -1

    best_psnr = 0
    for alpha in alphas:
        print(f"Trying alpha={alpha}:")
        if tgv:
            tic = time.time()
            u = tgv_denoise(f, alpha, 2.0, 2, maxiter, vis)
            toc = time.time() - tic
        else:
            tic = time.time()
            u = tv_denoise(f, alpha, 2, maxiter, vis)
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


base = "wet_leaf"

f = imread("test_data/" + base + ".png")
random.seed(14031621)
noise = random.randn(*f.shape) * 0.2
g = f + noise
imwrite("results/" + base + "_noisy.png", g)  # , vmin=0.0, vmax=1.0)

alphas = linspace(0.305, 0.315, 11)
(u_tv, alpha_tv, psnr_tv, time_tv) = test_range(g, f, alphas)
print(f"TV best parameter: alpha={alpha_tv}, PSNR={psnr_tv}, time={time_tv}")
imwrite("results/" + base + "_denoised_tv.png", u_tv)

alphas = linspace(0.285, 0.295, 11)
(u_tgv, alpha_tgv, psnr_tgv, time_tgv) = test_range(g, f, alphas, tgv=True)
print(
    f"TGV best parameter: alpha={alpha_tgv}, PSNR={psnr_tgv}, time={time_tgv}")
imwrite("results/" + base + "_denoised_tgv.png", u_tgv)
