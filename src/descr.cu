#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda.h>
#include <math.h>

using namespace std;
using namespace cv;

__global__ void DescriptorKernel(float *dogs, float *key_x, float *key_y, int *dog_index, float *descr, 
                                 int height, int width, int num_levels, int key_size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= key_size) {return;}

  int curr_dog_index = dog_index[index];
  float *curr_dog = dogs + curr_dog_index*height*width; 

  int curr_x = key_x[index];
  int curr_y = key_y[index];

  float hist[128] = {};
  int post_index = 0, orient_index = 0, loc_x = 0, loc_y = 0;
  float dx = 0., dy = 0., magnitude = 0., orient = 0.;
  for (int c = curr_y - 8; c <= curr_y + 7; c++) {
    for (int r = curr_x - 8; r <= curr_x + 7; r++) {
      if ((r - curr_x) * (r - curr_x) + (c - curr_y) * (c - curr_y) > 64) // maybe need adjust
        continue;
      if (r - 9 < 0 || r + 8 >= width || c - 8 < 0 || c + 7 >= height)
        continue;
      loc_x = r - curr_x + 8;
      loc_y = c - curr_y + 8;
      post_index = loc_y * 16 + loc_x;
      dx = curr_dog[c*width + r+1] - curr_dog[c*width + r-1];
      dy = curr_dog[(c+1)*width + r] - curr_dog[(c-1)*width + r];
      magnitude = sqrt(dx * dx + dy * dy); // maybe sqrtf 
      orient = (atan2(dy, dx) == M_PI) ? -M_PI : atan2(dy, dx); //maybe atan2f
      orient = (orient + M_PI) * 180 / M_PI; 
      orient_index = floor(orient / 45); //maybe floorf
      orient_index %= 7;
      loc_x /= 4;
      loc_y /= 4;
      post_index = loc_y * 4 + loc_x;
      hist[post_index*8 + orient_index] += magnitude;
    }
  }

  // cudaMemcpy(descr+index*128, hist, 128*sizeof(float), cudaMemcpyDeviceToDevice);
  for (int i=0; i<128; i++)
    descr[index*128+i] = hist[i];
}


void CalcDescriptorCUDA(float **dogs, int height, int width, int num_levels, vector<Point2f> &key, 
                        vector<int> &dog_index, Mat &descriptor) {
  assert(key.size() == dog_index.size());
  int key_size = key.size();

  float *descriptor_cuda;
  cudaMalloc(&descriptor_cuda, key_size * 128 * sizeof(float));

  float *key_x = new float[key_size];
  float *key_y = new float[key_size];
  for (int i=0; i<key_size; i++) {
    key_x[i] = key[i].x;
    key_y[i] = key[i].y;
  }
  float *key_x_cuda;
  float *key_y_cuda;
  cudaMalloc(&key_x_cuda, key_size * sizeof(float));
  cudaMalloc(&key_y_cuda, key_size * sizeof(float));
  cudaMemcpy(key_x_cuda, key_x.data, key_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(key_y_cuda, key_y.data, key_size * sizeof(float), cudaMemcpyHostToDevice);
  delete[] key_x;
  delete[] key_y;

  int *dog_index_cuda;
  cudaMalloc(&dog_index_cuda, key_size * sizeof(int));
  cudaMemcpy(dog_index_cuda, dog_index.data, key_size * sizeof(int), cudaMemcpyHostToDevice);

  float *dogs_1D = new float[height * width * (num_levels-1)];
  for (int i=0; i<num_levels-1; i++) {
    for (int j=0; j<height; j++) {
      for (int k=0; k<width; k++) {
        dogs_1D[i*height*width + j*width + k] = dogs[i][j*width + k]
      }
    }
  }
  float *dogs_cuda;
  cudaMalloc(&dogs_cuda, height * width * (num_levels-1) * sizeof(float));
  cudaMemcpy(dogs_cuda, dogs_1D, height * width * (num_levels-1) * sizeof(float), cudaMemcpyHostToDevice);
  delete[] dogs_1D;

  const int threadsPerBlock = 32;
  const int blocks = (key_size + threadsPerBlock - 1) / threadsPerBlock;
  DescriptorKernel<<<blocks, threadsPerBlock>>>(dogs_cuda, key_x_cuda, key_y_cuda, dog_index_cuda, 
                                                descriptor_cuda, height, width, num_levels, key_size);

  float *tmp_descr = new float[key_size * 128];
  cudaMemcpy(tmp_descr, descriptor_cuda, key_size * 128 * sizeof(float), cudaMemcpyDeviceToHost);
	descriptor = Mat(key_size, 128, CV_32F, tmp_descr);
}
