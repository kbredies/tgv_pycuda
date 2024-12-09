from common_pycuda import *
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
import pycuda.curandom as curandom
from pycuda.elementwise import ElementwiseKernel
import reikna.cluda as cluda
import reikna.fft as pyfft
from numpy import *
from matplotlib.pyplot import *
import sys


# kernel code
kernels = """
    __global__ void inpaint_project(float *src, float *dest, int *mask,
                                    int nx, int ny, int chans) {

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny))
        for (int i=0; i<chans; i++)
          dest[(i*ny+y)*nx+x] = mask[y*nx+x] ? 0.0f : src[(i*ny+y)*nx+x]; 
    }

    __global__ void zoom_forward(float *src, float *dest, int power,
                                 int nx, int ny, int chans) {

    __shared__ float l_src[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    // read data in shared memory
    if ((x < nx) && (y < ny))
      for (int i=0; i<chans; i++)
        l_src[threadIdx.x][threadIdx.y][i] = src[(i*ny+y)*nx+x];
    else 
      for (int i=0; i<chans; i++)
        l_src[threadIdx.x][threadIdx.y][i] = 0.0f;

    // reduce data
    int step = 1;
    int mask = 0;
    for (int j=0; j<power; j++) {
      mask = (mask << 1) | 1;

      for (int i=0; i<chans; i++) {
        // reduce in x direction
        __syncthreads();
        if ((threadIdx.x & mask) == 0)
          l_src[threadIdx.x][threadIdx.y][i] += 
                   l_src[threadIdx.x+step][threadIdx.y][i];

        // reduce in y direction
        __syncthreads();
        if ((threadIdx.y & mask) == 0)
          l_src[threadIdx.x][threadIdx.y][i] += 
                   l_src[threadIdx.x][threadIdx.y+step][i];
      }

      step *= 2;
    }
      
    // write result  
    __syncthreads();

    float reduction = 1.0f/((float)step*step);
    x = x >> power; y = y >> power;
    nx = (nx + mask) >> power; ny = (ny + mask) >> power;

    if (((threadIdx.x & mask) == 0) && ((threadIdx.y & mask) == 0))
      for (int i=0; i < chans; i++)
        dest[(i*ny+y)*nx+x] = l_src[threadIdx.x][threadIdx.y][i]*reduction;
    }

    __global__ void zoom_adjoint(float *src, float *dest, int power,
                                 int nx, int ny, int chans) {
    
    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      int step = (1 << power);
      float reduction = 1.0f/((float)step*step);

      int s_nx = (nx + step-1) >> power;
      int s_ny = (ny + step-1) >> power;

      int s_x = x >> power;
      int s_y = y >> power;
      for (int i=0; i < chans; i++) 
        dest[(i*ny+y)*nx+x] = src[(i*s_ny+s_y)*s_nx+s_x]*reduction;
    }
    }

    __device__ float read_array(float *ary, int x, int y, int nx, int ny)
    {
      if ((x >= 0) && (y >= 0) && (x < nx) && (y < ny))
        return(ary[y*nx+x]);
      return(0);
    }
  
    __device__ void write_array(float *ary, float val, int x, int y, int nx, int ny)
    {
      if ((x >= 0) && (y >= 0) && (x < nx) && (y < ny))
        ary[y*nx+x] = val;
    }

    __global__ void convolve_valid(float *src, float *mask, float *dest,
                                   int nx, int ny, int chans, 
                                   int m_nx, int m_ny) {

    __shared__ float l_src[2*%(BLOCK_SIZE_X)s][2*%(BLOCK_SIZE_Y)s];
    __shared__ float l_mask[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s];
    __shared__ float l_dest[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;    

    if ((threadIdx.x <= 2*m_nx) && (threadIdx.y <= 2*m_ny))
      l_mask[threadIdx.x][threadIdx.y] 
                         = mask[threadIdx.y*(2*m_nx+1)+threadIdx.x];

    for (int i=0; i < chans; i++) {

      // read source into shared memory
      __syncthreads();
      l_src[threadIdx.x][threadIdx.y] 
             = read_array(src, x, y, nx, ny);
      l_src[threadIdx.x+%(BLOCK_SIZE_X)s][threadIdx.y] 
             = read_array(src, x+%(BLOCK_SIZE_X)s, y, nx, ny);
      l_src[threadIdx.x][threadIdx.y+%(BLOCK_SIZE_Y)s] 
             = read_array(src, x, y+%(BLOCK_SIZE_Y)s, nx, ny);
      l_src[threadIdx.x+%(BLOCK_SIZE_X)s][threadIdx.y+%(BLOCK_SIZE_Y)s] 
             = read_array(src, x+%(BLOCK_SIZE_X)s, y+%(BLOCK_SIZE_Y)s, nx, ny);

      // do convolution
      __syncthreads();
      l_dest[threadIdx.x][threadIdx.y] = 0;
      for(int yy=-m_ny; yy <= m_ny; yy++)
        for(int xx=-m_nx; xx <= m_nx; xx++)
          l_dest[threadIdx.x][threadIdx.y] += l_mask[xx+m_nx][yy+m_ny]
                        *l_src[m_nx+threadIdx.x-xx][m_ny+threadIdx.y-yy];

      // write results
      write_array(dest, l_dest[threadIdx.x][threadIdx.y], x, y,
                  nx-2*m_nx, ny-2*m_ny);

      // update array pointer
      src += nx*ny;
      dest += (nx-2*m_nx)*(ny-2*m_ny);
    }
    }

    __global__ void convolve_full(float *src, float *mask, float *dest,
                                  int nx, int ny, int chans, 
                                  int m_nx, int m_ny) {

    __shared__ float l_src[2*%(BLOCK_SIZE_X)s][2*%(BLOCK_SIZE_Y)s];
    __shared__ float l_mask[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s];
    __shared__ float l_dest[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;    

    if ((threadIdx.x <= 2*m_nx) && (threadIdx.y <= 2*m_ny))
      l_mask[threadIdx.x][threadIdx.y] 
                         = mask[threadIdx.y*(2*m_nx+1)+threadIdx.x];

    for (int i=0; i < chans; i++) {
      // read source into shared memory
      __syncthreads();
      l_src[threadIdx.x][threadIdx.y] 
            = read_array(src, x-%(BLOCK_SIZE_X)s, y-%(BLOCK_SIZE_Y)s, 
                         nx-2*m_nx, ny-2*m_ny);
      l_src[threadIdx.x+%(BLOCK_SIZE_X)s][threadIdx.y] 
            = read_array(src, x, y-%(BLOCK_SIZE_Y)s, 
                         nx-2*m_nx, ny-2*m_ny);
      l_src[threadIdx.x][threadIdx.y+%(BLOCK_SIZE_Y)s] 
            = read_array(src, x-%(BLOCK_SIZE_X)s, y, 
                         nx-2*m_nx, ny-2*m_ny);
      l_src[threadIdx.x+%(BLOCK_SIZE_X)s][threadIdx.y+%(BLOCK_SIZE_Y)s] 
            = read_array(src, x, y, 
                          nx-2*m_nx, ny-2*m_ny);

      // do convolution
      __syncthreads();
      l_dest[threadIdx.x][threadIdx.y] = 0;
      for(int yy=0; yy <= 2*m_ny; yy++)
        for(int xx=0; xx <= 2*m_nx; xx++)
          l_dest[threadIdx.x][threadIdx.y] += l_mask[xx][yy]
                        *l_src[threadIdx.x-xx+%(BLOCK_SIZE_X)s]
                              [threadIdx.y-yy+%(BLOCK_SIZE_Y)s];

      // write results
      write_array(dest, l_dest[threadIdx.x][threadIdx.y], x, y, nx, ny);

      // update array pointer
      src += (nx-2*m_nx)*(ny-2*m_ny);
      dest += nx*ny;
    }
    }

    __global__ void copy_low_coeff(float *src0, float *dest0,
                                   float *src1, float *dest1,
                                   int power, int nx, int ny) {

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;
   
    if ((x < nx) && (y < ny)) {
      int x_src = x - (nx >> 1); int x_dest = x_src;
      if (x_src < 0) {
        x_src += (nx << power);
        x_dest += nx;
      }

      int y_src = y - (ny >> 1); int y_dest = y_src;
      if (y_src < 0) {
        y_src += (ny << power);
        y_dest += ny;
      }

      dest0[y_dest*nx + x_dest] = src0[y_src*(nx << power) + x_src];
      dest1[y_dest*nx + x_dest] = src1[y_src*(nx << power) + x_src];
      }
    }

    __global__ void copy_low_coeff_ad(float *src0, float *dest0,  
                                      float *src1, float *dest1,
                                      int power, int nx, int ny) {

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;
   
    if ((x < nx) && (y < ny)) {
      int x_src = x - (nx >> 1); int x_dest = x_src;
      if (x_src < 0) {
        x_src += nx;
        x_dest += (nx << power);
      }

      int y_src = y - (ny >> 1); int y_dest = y_src;
      if (y_src < 0) {
        y_src += ny;
        y_dest += (ny << power);
      }

      dest0[y_dest*(nx << power) + x_dest] = src0[y_src*nx + x_src];
      dest1[y_dest*(nx << power) + x_dest] = src1[y_src*nx + x_src];
      }
    }

    __global__ void accumulate_pixels(char *ary, float *src, float *dest,
                                      int nx, int ny, int chans, int stride) {
    int idx = (blockIdx.x * %(BLOCK_SIZE_Y)s + threadIdx.y) 
              * %(BLOCK_SIZE_X)s + threadIdx.x;

    float acc[4];

    if (idx < ny) {
      ary += idx;

      // reset
      for (int i=0; i<chans; i++) 
        acc[i] = 0.0f;

      // accumulate
      for (int j=0; j<nx; j++) {
        if (ary[0]) 
          for (int i=0; i<chans; i++)
             acc[i] += src[i*nx+j];
        ary += stride;
      }
  
      // write
      for (int i=0; i<chans; i++) 
        dest[i*ny+idx] = acc[i];
    }
    }
    """
kernels = kernels % {
    'BLOCK_SIZE_X': block_size_x,
    'BLOCK_SIZE_Y': block_size_y, 
    }

# compile kernels
module = compiler.SourceModule(kernels)
inpaint_project_func = module.get_function("inpaint_project")
zoom_forward_func = module.get_function("zoom_forward")
zoom_adjoint_func = module.get_function("zoom_adjoint")
convolve_forward_func = module.get_function("convolve_valid")
convolve_adjoint_func = module.get_function("convolve_full")
copy_low_coeff_func = module.get_function("copy_low_coeff")
copy_low_coeff_ad_func = module.get_function("copy_low_coeff_ad")
accumulate_pixels_func = module.get_function("accumulate_pixels")

# initialize cluda thread
api = cluda.cuda_api()
thrd = api.Thread(pycuda.autoinit.context)

def get_channels(u):
    return 1 if len(u.shape) <= 2 else u.shape[2]

def create_slice_view(u, num):
    v = u.ravel()
    v.shape = (u.shape[0], u.shape[1])
    v.strides = (u.strides[0], u.strides[1])
    v.gpudata = int(u.gpudata) + u.strides[2]*num
    v.nbytes = u.strides[2]
    v.mem_size = u.shape[0]*u.shape[1]
    return(v)

# prototype for a linear operator on the GPU
class LinearOperator:
    """Prototype for a linear operator."""

    def get_dest_shape(self, shape):
        """Returns the shape of the image of <self> applied to data with
        shape <shape>."""
        return(shape)
    
    def get_src_shape(self, shape):
        """Returns the shape of the image of the adjoint <self> 
        applied to data with shape <shape>."""
        return(shape)

    def check_shapes(self, src, dest):
        """Checks whether shape of <src> and <dest> match."""
        return ((self.get_dest_shape(src.shape) == tuple(dest.shape)) and
                (self.get_src_shape(dest.shape) == tuple(src.shape)))

    def norm_est(self):
        """Return a norm estimate for <self>."""
        return(0)

    def apply(self, src, dest):
        """Applies <self> to <src> and stores the result in <dest>."""
        dest = 0

    def adjoint(self, src, dest):
        """Applies the adjoint of <self> to <src> and stores the result 
        in <dest>."""
        dest = 0

# the identity operator on the GPU
class IdentityOperator(LinearOperator):
    """The identity operator."""

    def get_dest_shape(self, shape):
        """Returns the shape of the image of <self> applied to data with
        shape <shape>."""
        return(shape)
    
    def get_src_shape(self, shape):
        """Returns the shape of the image of the adjoint <self> 
        applied to data with shape <shape>."""
        return(shape)

    def norm_est(self):
        """Return a norm estimate for <self>."""
        return(1.0)

    def apply(self, src, dest):
        """Applies <self> to <src> and stores the result in <dest>."""
        if (not self.check_shapes(src,dest)):
            raise TypeError("Size mismatch")
        cuda.memcpy_dtod(dest.gpudata, src.gpudata, src.nbytes)

    def adjoint(self, src, dest):
        """Applies the adjoint of <self> to <src> and stores the result 
        in <dest>."""
        self.apply(src, dest)

# inpainting operator
class InpaintingOperator(LinearOperator):
    """Inpainting operator which projects on the space spanned by
    the elements given by the non-zero elements of an inpainting mask."""

    def __init__(self, mask):
        """Initializes operator with given <mask>."""
        mask = mask.astype('int32').copy(order='F')
        self.mask = gpuarray.to_gpu(mask)

    def get_dest_shape(self, shape):
        """Returns the shape of the image of <self> applied to data with
        shape <shape>."""
        dshape = list(shape)
        dshape[0:2] = self.mask.shape[0:2]
        return(tuple(dshape))

    def get_src_shape(self, shape):
        """Returns the shape of the image of the adjoint <self> 
        applied to data with shape <shape>."""
        dshape = list(shape)
        dshape[0:2] = self.mask.shape[0:2]
        return(tuple(dshape))

    def norm_est(self):
        """Return a norm estimate for <self>."""
        return(1.0)

    def apply(self, src, dest):
        """Applies the projection operator to <src> storing the result
        in <dest>."""
        if (not self.check_shapes(src,dest)):
            raise TypeError("Size mismatch")
        inpaint_project_func(src, dest, self.mask, 
                             int32(src.shape[0]), int32(src.shape[1]), 
                             int32(get_channels(src)),
                             block=block_, grid=get_grid(src))

    def adjoint(self, src, dest):
        """Applies the projection operator to <src> storing the result
        in <dest>."""
        if (not self.check_shapes(src,dest)):
            raise TypeError("Size mismatch")
        inpaint_project_func(src, dest, self.mask, 
                             int32(src.shape[0]), int32(src.shape[1]), 
                             int32(get_channels(src)),
                             block=block_, grid=get_grid(src))

# zooming operator
class ZoomingOperator(LinearOperator):
    """Zooming operator which averages over 2^p x 2^p pixel blocks
    where p is in [0,1,2,3,4]."""

    def __init__(self, power):
        """Initializes operator with factor 2^<power>."""
        allowed_powers = list(range(5))
        if (not power in allowed_powers):
            raise TypeError("Power not in allowed range.")
        self.power = power
        self.factor = pow(2, power)

    def get_dest_shape(self, shape):
        """Returns the shape of the image of <self> applied to data with
        shape <shape>."""
        dshape = list(shape)
        for i in range(2):
            dshape[i] = (dshape[i] + self.factor-1)//self.factor
        return(tuple(dshape))

    def get_src_shape(self, shape):
        """Returns the shape of the image of the adjoint <self> 
        applied to data with shape <shape>."""
        sshape = list(shape)
        for i in range(2):
            sshape[i] *= self.factor
        return(tuple(sshape))

    def norm_est(self):
        """Returns a norm estimate for <self>."""
        return(1.0)

    def apply(self, src, dest):
        """Applies the zooming operator to <src> and stores the result 
        in <dest>."""
        if (not self.get_dest_shape(src.shape) == dest.shape):
            raise TypeError("Size mismatch")
        zoom_forward_func(src, dest, int32(self.power), 
                          int32(src.shape[0]), int32(src.shape[1]), 
                          int32(get_channels(src)),
                          block=block_, grid=get_grid(src))

    def adjoint(self, src, dest):
        """Applies the adjoint zooming operator to <src> and stores 
        the result in <dest>."""
        if (not self.get_dest_shape(dest.shape) == src.shape):
            raise TypeError("Size mismatch")
        zoom_adjoint_func(src, dest, int32(self.power), 
                          int32(dest.shape[0]), int32(dest.shape[1]), 
                          int32(get_channels(dest)),
                          block=block_, grid=get_grid(dest))

# convolution operator
class ConvolutionOperator(LinearOperator):
    """Convolution operator performs a convolution with respect to
       a mask of size l x m with l,m in [1,3,5,7,9,11,13,15]."""

    def __init__(self, mask):
        """Initializes operator with <mask>."""
        # enlarge if necessary
        if (mask.shape[0] % 2 == 0):
            mask = r_[mask, zeros([1,mask.shape[1]])]
        if (mask.shape[1] % 2 == 0):
            mask = c_[mask, zeros([mask.shape[0],1])]

        # range check
        allowed_range = [1,3,5,7,9,11,13,15]
        if ((not mask.shape[0] in allowed_range)
            or (not mask.shape[1] in allowed_range)):
            raise TypeError("Mask shape not in allowed range.")

        # copy mask to gpu
        mask = mask.astype('float32').copy(order='F')
        mask_ad = mask[::-1,::-1].copy(order='F')
        self.mask = gpuarray.to_gpu(mask)
        self.mask_ad = gpuarray.to_gpu(mask_ad)
        self.normest = sum(abs(mask))

    def get_dest_shape(self, shape):
        """Returns the shape of the image of <self> applied to data with
        shape <shape>."""
        dshape = list(shape)
        for i in range(2):
            dshape[i] = max(0, dshape[i]-self.mask.shape[i]+1)
        return(tuple(dshape))

    def get_src_shape(self, shape):
        """Returns the shape of the image of the adjoint <self> 
        applied to data with shape <shape>."""
        sshape = list(shape)
        for i in range(2):
            sshape[i] += self.mask.shape[i]-1
        return(tuple(sshape))

    def norm_est(self):
        """Returns a norm estimate for <self>."""
        return(self.normest)

    def apply(self, src, dest):
        """Applies the convolution/restriction operator to <src> and 
        stores the result in <dest>."""
        if (not self.get_dest_shape(src.shape) == dest.shape):
            raise TypeError("Size mismatch")
        convolve_forward_func(src, self.mask, dest, 
                              int32(src.shape[0]), int32(src.shape[1]), 
                              int32(get_channels(src)), 
                              int32(self.mask.shape[0]/2), 
                              int32(self.mask.shape[1]/2), 
                              block=block_, grid=get_grid(dest))
        
    def adjoint(self, src, dest):
        """Applies the extension/adjoint convolution operator to <src> 
        and stores the result in <dest>."""
        if (not self.get_dest_shape(dest.shape) == src.shape):
            raise TypeError("Size mismatch")
        convolve_adjoint_func(src, self.mask_ad, dest,  
                              int32(dest.shape[0]), int32(dest.shape[1]), 
                              int32(get_channels(dest)),
                              int32(self.mask_ad.shape[0]/2), 
                              int32(self.mask_ad.shape[1]/2), 
                              block=block_, grid=get_grid(dest))

# DCT zooming operator
class DCTZoomingOperator(LinearOperator):
    """Zooming operator which performs DCT-lowpass filter
    over 2^p x 2^p pixel blocks."""

    def __init__(self, power, shape):
        """Initializes operator with destination shape <shape> 
        and factor 2^<power>."""

        # get shapes
        dest_shape = (shape[0], shape[1])
        src_shape = (shape[0] << power, shape[1] << power)

        # create plans and additional data
        self.dest_hat = gpuarray.zeros(dest_shape, 'complex64', order='F')
        dest_plan = pyfft.FFT(self.dest_hat, axes=(0, 1))
        self.dest_fft = dest_plan.compile(thrd)

        
        
        self.dest_plan = pyfft.Plan((dest_shape[1], dest_shape[0]), float32,
                                    scale=1.0/(dest_shape[0]*dest_shape[1]
                                               *(1 << power)))
        self.dest_imag0 = gpuarray.zeros(dest_shape, 'float32', order='F')
        self.dest_real1 = gpuarray.zeros(dest_shape, 'float32', order='F')
        self.dest_imag1 = gpuarray.zeros(dest_shape, 'float32', order='F')
        self.src_plan = pyfft.Plan((src_shape[1], src_shape[0]), float32,
                                   scale=1.0/(src_shape[0]*src_shape[1]))
        self.src_imag0 = gpuarray.zeros(src_shape, 'float32', order='F')
        self.src_real1 = gpuarray.zeros(src_shape, 'float32', order='F')
        self.src_imag1 = gpuarray.zeros(src_shape, 'float32', order='F')

        # store variables
        self.dest_shape = dest_shape
        self.src_shape = src_shape
        self.power = power
        self.factor = (1 << power)

    def get_dest_shape(self, shape):
        """Returns the shape of the image of <self> applied to data with
        shape <shape>."""
        dshape = list(shape)
        dshape[0:2] = self.dest_shape
        return(tuple(dshape))

    def get_src_shape(self, shape):
        """Returns the shape of the image of the adjoint <self> 
        applied to data with shape <shape>."""
        sshape = list(shape)
        sshape[0:2] = self.src_shape
        return(tuple(sshape))

    def norm_est(self):
        """Returns a norm estimate for <self>."""
        return(1.0)

    def apply(self, src, dest):
        """Applies the zooming operator to <src> and stores the result 
        in <dest>."""
        if (not self.check_shapes(src,dest)):
            raise TypeError("Size mismatch")

        channels = get_channels(src)
    
        # compute for each channel
        for i in range(channels):
            src_ = create_slice_view(src, i)
            dest_ = create_slice_view(dest, i)

            # transform
            self.src_plan.execute(src_, self.src_imag0,
                                  self.src_real1, self.src_imag1)
            
            # copy into self.dest_*1
            copy_low_coeff_func(self.src_real1, self.dest_real1,
                                self.src_imag1, self.dest_imag1,
                                int32(self.power),
                                int32(self.dest_real1.shape[0]), 
                                int32(self.dest_real1.shape[1]),
                                block=block_, grid=get_grid(self.dest_real1))

            # transform back
            self.dest_plan.execute(self.dest_real1, self.dest_imag1,
                                   dest_, self.dest_imag0, inverse=True)
            
    def adjoint(self, src, dest):
        """Applies the adjoint zooming operator to <src> and stores 
        the result in <dest>."""
        if (not self.check_shapes(dest, src)):
            raise TypeError("Size mismatch")

        # change source and dest shape
        channels = get_channels(src)
    
        # compute for each channel
        for i in range(channels):
            src_ = create_slice_view(src, i)
            dest_ = create_slice_view(dest, i)

            # transform
            self.dest_plan.execute(src_, self.dest_imag0,
                                   self.dest_real1, self.dest_imag1)
            
            # copy into self.src_*1
            self.src_real1.fill(0)
            self.src_imag1.fill(0)
            copy_low_coeff_ad_func(self.dest_real1, self.src_real1,
                                   self.dest_imag1, self.src_imag1,
                                   int32(self.power),
                                   int32(self.dest_real1.shape[0]), 
                                   int32(self.dest_real1.shape[1]),
                                   block=block_, grid=get_grid(self.dest_real1))

            # transform back
            self.src_plan.execute(self.src_real1, self.src_imag1,
                                  dest_, self.src_imag0, inverse=True)

# Pixel accumulation operator
class AccumulationOperator(LinearOperator):
    """Operator which sums up pixels according to a specified pattern."""

    def __init__(self, ary, shape=None):
        self.src_len = ary.shape[1]
        self.src_shape = shape if shape != None else (ary.shape[1], 1) 
        self.dest_len = ary.shape[0]

        ary = (ary != 0).astype('uint8').copy(order='F')
        ary_ad = ary.transpose().copy(order='F')
        self.ary = gpuarray.to_gpu(ary)
        self.ary_ad = gpuarray.to_gpu(ary_ad)
        
        self.norm_estimate = sqrt(sum(ary))

    def get_dest_shape(self, shape):
        """Returns the shape of the image of <self> applied to data with
        shape <shape>."""
        dshape = list(shape)
        dshape[0] = self.dest_len
        dshape[1] = 1
        return(tuple(dshape))

    def get_src_shape(self, shape):
        """Returns the shape of the image of the adjoint <self> 
        applied to data with shape <shape>."""
        sshape = list(shape)
        sshape[0:2] = self.src_shape
        return(tuple(sshape))

    def norm_est(self):
        return(self.norm_estimate)

    def check_shapes(self, src, dest):
        return((src.shape[0]*src.shape[1] == self.src_len) and
               (dest.shape[0]*dest.shape[1] <= self.dest_len) and
               (get_channels(src) == get_channels(dest)))

    def apply(self, src, dest):
        if (not self.check_shapes(src, dest)):
            raise TypeError("Size mismatch")

        chans = get_channels(src)
        dest_len = dest.shape[0]*dest.shape[1]
        acc_block = (block_size_x*block_size_y, 1, 1)
        acc_grid = ((dest_len + acc_block[0] - 1)/acc_block[0], 1)
        
        accumulate_pixels_func(self.ary, src, dest, 
                               int32(self.src_len), int32(dest_len), 
                               int32(chans), int32(self.dest_len), 
                               block=acc_block, grid=acc_grid)

    def adjoint(self, src, dest):
        if (not self.check_shapes(dest, src)):
            raise TypeError("Size mismatch")

        chans = get_channels(src)
        src_len = src.shape[0]*src.shape[1]
        acc_block = (block_size_x*block_size_y, 1, 1)
        acc_grid = ((self.src_len + acc_block[0] - 1)/acc_block[0], 1)
        
        accumulate_pixels_func(self.ary_ad, src, dest, 
                               int32(src_len), int32(self.src_len), 
                               int32(chans), int32(self.src_len),
                               block=acc_block, grid=acc_grid)

# test operator K for adjointness
def test_adjoint(K, shape, iter=10):
    """Test the linear operator <K> for adjointness using source
    data with <shape>."""
    u = gpuarray.zeros(shape,'float32',order='F')
    Ku = gpuarray.zeros(K.get_dest_shape(u.shape),'float32',order='F')
    v = gpuarray.zeros(K.get_dest_shape(u.shape),'float32',order='F')
    Kadv = gpuarray.zeros(shape,'float32',order='F')

    generator = curandom.XORWOWRandomNumberGenerator()

    for i in range(iter):
        # fill with random numbers
        generator.fill_uniform(u)
        generator.fill_uniform(v)

        # apply operators
        K.apply(u,Ku)
        K.adjoint(v,Kadv)

        # compute scalar products
        scp1 = float(gpuarray.dot(Ku,v).get())
        scp2 = float(gpuarray.dot(u,Kadv).get())

        # print result
        print("Test "+repr(i)+": <Ku,v>="+repr(scp1)+", <u,Kadv>="+repr(scp2)+". Error="+repr(abs(scp1-scp2)))

