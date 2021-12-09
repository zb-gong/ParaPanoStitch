#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda.h>

using namespace std;
using namespace cv;

__global__ void DogKernel(float *gaussian, float *dogs, int height, int width, int num_levels) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int level = threadIdx.z; 

  float *curr_dog = dogs + level * height * width;
  float *g_0 = gaussian + level * height * width;
  float *g_1 = gaussian + (level+1) *height * width;

  curr_dog[index] = g_1[index] - g_0[index];
}


float **DodogCUDA(float **gaussian, int height, int width, int num_levels) {
  float *dogs_cuda;
  cudaMalloc(&dogs_cuda, height * width * (num_levels-1) * sizeof(float));

  float *gaussian_1D = new float[height * width * num_levels];
  for (int i=0; i<num_levels; i++) {
    for (int j=0; j<height; j++) {
      for (int k=0; k<width; k++) {
        gaussian_1D[i*height*width + j*width + k] = gaussian[i][j*width + k];
      }
    }
  }
  float *gaussian_cuda;
  cudaMalloc(&gaussian_cuda, height * width * num_levels * sizeof(float));
  cudaMemcpy(gaussian_cuda, gaussian_1D, height * width * num_levels * sizeof(float), cudaMemcpyHostToDevice);
  delete [] gaussian_1D;
  
  const int threadsPerBlock = 128;
  dim3 gridDim((height*width + threadsPerBlock - 1)/ threadsPerBlock);
  dim3 blockDim(threadsPerBlock, num_levels-1);
  DogKernel<<<gridDim, blockDim>>>(gaussian_cuda, dogs_cuda, height, width, num_levels);
  
  float **dogs = new float *[num_levels-1];
  for (int i=0; i<num_levels-1; i++) {
    dogs[i] = new float[height *width];
    cudaMemcpy(dogs[i], dogs_cuda+i*height*width, height*width*sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaFree(dogs_cuda);
  cudaFree(gaussian_cuda);
  return dogs;
}