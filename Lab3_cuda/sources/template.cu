#include <gputk.h>


__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < len) {
    out[index] = in1[index] + in2[index];
  }
}

#ifndef STREAM
#define stream 4
#endif

int main(int argc, char **argv) {
  gpuTKArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  unsigned int numStreams;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);

  gpuTKTime_start(GPU, "Allocating Pinned memory.");

  //@@ Allocate GPU memory here using pinned memory here
  cudaMallocHost((void **)&deviceInput1, inputLength * sizeof(float));
  cudaMallocHost((void **)&deviceInput2, inputLength * sizeof(float));
  cudaMallocHost((void **)&deviceOutput, inputLength * sizeof(float));

  //@@ Create and setup streams
  numStreams = STREAM;
  cudaStream_t streams[numStreams];
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }
  
  //@@ Calculate data segment size of input data processed by each stream 
  int streamSizes[numStreams];
  int offsets[numStreams];
  int streamSizeBase = inputLength / numStreams;
  for (int i = 0; i < numStreams; i++) {
    if (i < numStreams - 1)
      streamSizes[i] = streamSizeBase;
    else
      streamSizes[i] = inputLength - streamSizeBase * (numStreams - 1);
    offsets[i] = i * streamSizeBase;
  }

  int blockSize = 256;
  int numBlocks = (inputLength + blockSize - 1) / blockSize;

 
  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Perform parallel vector addition with different streams. 
  for (unsigned int s = 0; s<numStreams; s++){
    //@@ Asynchronous copy data to the device memory in segments 
    //@@ Calculate starting and ending indices for per-stream data
    int offset = offsets[s];
    int streamSize = streamSizes[s];
    cudaStream_t stream = streams[s];

    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], 
      streamSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset],
      streamSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    //@@ Invoke CUDA Kernel
    //@@ Determine grid and thread block sizes (consider ococupancy)
    vecAdd<<<numBlocks, blockSize, 0, stream>>>
      (&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], streamSize);   

    //@@ Asynchronous copy data from the device memory in segments
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset],
      streamSize * sizeof(float), cudaMemcpyDeviceToHost, stream);

  }

  //@@ Synchronize
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  gpuTKTime_stop(Compute, "Performing CUDA computation");


  gpuTKTime_start(GPU, "Freeing Pinned Memory");
  //@@ Destory cudaStream
  for (int i = 0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }

  //@@ Free the GPU memory here
  cudaFreeHost(deviceInput1);
  cudaFreeHost(deviceInput2);
  cudaFreeHost(deviceOutput);

  gpuTKTime_stop(GPU, "Freeing Pinned Memory");

  gpuTKSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
