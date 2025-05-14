# PyCUDA primal-dual algorithms for TV/TGV-constrained imaging problems

Algorithms, examples and tests for denoising, deblurring, zooming, dequantization and compressed sensing with total variation and second-order total generalized variation regularization. Python implementation with GPU acceleration using PyCUDA.

The code reproduces, in particular, the numerical experiments in the associated publication:

> Kristian Bredies. Recovering piecewise smooth multichannel images by minimization of convex functionals with total generalized variation penalty. *Lecture Notes in Computer Science*, 8293:44-77, 2014. doi:[10.1007/978-3-642-54774-4_3](https://doi.org/10.1007/978-3-642-54774-4_3)
 
## Getting started

One easy way of getting started is to create a Python virtual environment, install the dependencies and to call a test script. For instance, run in the project folder:

```bash
  python -m venv venv
  source ./venv/bin/activate
  pip install -r requirements.txt
  python test_denoise.py
```
Please note that a working CUDA installation is required, in particular, a CUDA-enabled GPU.
