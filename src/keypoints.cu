#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda.h>

using namespace std;
using namespace cv;
                        
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

// on device; convolves image with gaussian filter
__global__ void ConvolveKernel(float *image, int height, int width, float *cudaFilter, int filter_size, float *cudaGauss) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  float sum = 0.0;
  int h_mid = filter_size / 2;
	int w_mid = filter_size / 2;

  if (row < height && col < width) {
    for (int fi = -h_mid; fi < filter_size - h_mid; fi++) {
      for (int fj = -w_mid; fj < filter_size - w_mid; fj++) {
        int r = row + fi;
        int c = col + fj;
        float val = 0.0;
        if (r >= 0 && r < height && c >= 0 && c < width)
        {
            val = image[c + r * width];
        }
        sum += val * cudaFilter[(fi + h_mid) * filter_size + (fj + w_mid)];
      }
    }
    cudaGauss[col + row * width] = sum;
  }
}

// // Creates num_levels gaussian pyramids
// float **createGaussianPyramidCUDA(float *image, int height, int width, float k, float sigma_0, int num_levels, vector<int> levels) {
//   float **gaussian_pyramid = new float *[num_levels];
//   /*for (int i = 0; i < num_levels; i++) {
//         gaussian_pyramid[i] = new float[height * width];
//   }*/

//   float *cudaGauss;
//   cudaMalloc(&cudaGauss, sizeof(float) * height * width);

//   float *cudaImage; // device
//   cudaMalloc(&cudaImage, sizeof(float) * height * width);
//   cudaMemcpy(cudaImage, image, sizeof(float) * height * width, cudaMemcpyHostToDevice);
  
//   for (int i = 0; i < num_levels; i++) {
//     float sigma = sigma_0 * pow(k, levels[i]);
//     int filter_size = ceil(6 * sigma);
//     float* filter = createFilter(filter_size, sigma);

//     // device filter
//     float *cudaFilter;
//     cudaMalloc(&cudaFilter, sizeof(float) * filter_size * filter_size);
//     cudaMemcpy(cudaFilter, filter, sizeof(float) * filter_size * filter_size,
//                 cudaMemcpyHostToDevice);

//     // separate image into blocks
//     // NOTE: may need to tweak BLOCK_WIDTH & BLOCK_HEIGHT
//     dim3 gridDim(width / BLOCK_WIDTH, height / BLOCK_HEIGHT);
//     dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
//     convolveCUDA<<<gridDim, blockDim>>>(cudaImage, height, width, cudaFilter, filter_size, cudaGauss);

//     // Moves level i's gaussian res from device to host
//     cudaMemcpy(gaussian_pyramid[i], cudaGauss, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

//     delete[] filter;
//     cudaFree(cudaFilter);
//   }    
//   cudaFree(cudaGauss);
//   cudaFree(cudaImage);

//   return gaussian_pyramid;
// }

float *ConvolveCUDA(float *image, float *filter, int height, int width, int filter_size) {
  float *image_cuda;
  cudaMalloc(&image_cuda, sizeof(float) * height * width);
  cudaMemcpy(image_cuda, image, sizeof(float) * height * width, cudaMemcpyHostToDevice);
  
  float *filter_cuda;
  cudaMalloc(&filter_cuda, sizeof(float) * filter_size * filter_size);
  cudaMemcpy(filter_cuda, filter, sizeof(float) * filter_size * filter_size, cudaMemcpyHostToDevice);
  
  float *gaussian_cuda;
  cudaMalloc(&gaussian_cuda, sizeof(float) * height * width);

  dim3 gridDim(width / BLOCK_WIDTH, height / BLOCK_HEIGHT);
  dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
  ConvolveKernel<<<gridDim, blockDim>>>(image_cuda, height, width, filter_cuda, filter_size, gaussian_cuda);

  float *curr_gaussian = new float[height * width];
  cudaMemcpy(curr_gaussian, gaussian_cuda, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

  cudaFree(image_cuda);
  cudaFree(filter_cuda);
  cudaFree(gaussian_cuda);
  return curr_gaussian;
}