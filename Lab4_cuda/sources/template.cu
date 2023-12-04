#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))


//@@ INSERT CODE HERE
// __constant__ float* __restrict__ deviceMaskData;

__global__ void convolution(float* I, const float* __restrict__ M, float* P, int channels, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int ti = threadIdx.y * TILE_WIDTH + threadIdx.x;

  __shared__ float Ns[w][w][3];

  while (ti < w * w) {
    int Si = ti / w, Sj = ti % w;
    int Ii = blockIdx.y * blockDim.y + Si - Mask_radius;
    int Ij = blockIdx.x * blockDim.x + Sj - Mask_radius;
    if (Ii >= 0 && Ii < height && Ij >= 0 && Ij < width) {
      for (int m = 0; m < channels; m++) {
        Ns[Si][Sj][m] = I[(Ii * width + Ij) * channels + m];
      }
    } else {
      for (int m = 0; m < channels; m++) {
        Ns[Si][Sj][m] = 0.0f;
      }
    }
    ti += TILE_WIDTH * TILE_WIDTH;
  }
  __syncthreads();

  float out[3] = { 0.0f, 0.0f, 0.0f };

  if (row < height && col < width) {
    for (int i = 0; i < Mask_width; i++) {
      for (int j = 0; j < Mask_width; j++) {
        for (int m = 0; m < channels; m++) {
          out[m] += Ns[threadIdx.y + i][threadIdx.x + j][m] * M[i * Mask_width + j]; 
        }
      }
    }
  }

  if (row < height && col < width) {
    for (int m = 0; m < channels; m++) {
      P[(row * width + col) * channels + m] = out[m];
    }
  }
}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = gpuTKArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = gpuTKArg_getInputFile(arg, 0);
  inputMaskFile  = gpuTKArg_getInputFile(arg, 1);

  inputImage   = gpuTKImport(inputImageFile);
  hostMaskData = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = gpuTKImage_getWidth(inputImage);
  imageHeight   = gpuTKImage_getHeight(inputImage);
  imageChannels = gpuTKImage_getChannels(inputImage);

  outputImage = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  size_t imageSize = sizeof(float) * imageWidth * imageHeight * imageChannels;
  cudaMalloc((void**)&deviceInputImageData, imageSize);
  cudaMalloc((void**)&deviceOutputImageData, imageSize);
  cudaMalloc((void**)&deviceMaskData, Mask_width * Mask_width * sizeof(float));

  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, Mask_width * Mask_width * sizeof(float), cudaMemcpyHostToDevice);
  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE

  dim3 dimGrid((imageWidth + TILE_WIDTH - 1) / TILE_WIDTH, (imageHeight  + TILE_WIDTH - 1) / TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

  convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
                                     deviceOutputImageData, imageChannels,
                                     imageWidth, imageHeight);
  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);
  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(arg, outputImage);

  //@@ Insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}
