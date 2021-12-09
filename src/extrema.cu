#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda.h>

using namespace std;
using namespace cv;
                        
#define BLOCK_WIDTH 4
#define BLOCK_HEIGHT 4

__device__ bool isLocalMinMaxKernel(float *curr_dog, float *prev_dog, float *next_dog, int i, int j, int height, int width) {
	int size = 3 / 2;
	float point = curr_dog[j + i * width];
	bool local_max = true;
	bool local_min = true;

	for (int fi = -size; fi <= size; fi++) {
		for (int fj = -size; fj <= size; fj++) {
			int r = i + fi;
			int c = j + fj;
			if (r >= 0 && r < height && c >= 0 && c < width) {
				if (curr_dog[c + r * width] > point)
					local_max = false;
				if (curr_dog[c + r * width] < point)
					local_min = false;
				if ((prev_dog != nullptr) && (prev_dog[c + r * width] > point))
					local_max = false;
				if ((prev_dog != nullptr) && (prev_dog[c + r * width] < point))
					local_min = false;
				if ((next_dog != nullptr) && (next_dog[c + r * width] > point))
					local_max = false;
				if ((next_dog != nullptr) && (next_dog[c + r * width] < point))
					local_min = false;
			}
		}
	}
	return local_max || local_min;
}

__device__ float dxKernel(float *image, int height, int width, int r, int c) {
	// dx_i = (I(i+1) - I(i-1)) / 2
	if (r >= 0 && r < height && c >= 0 && c < width) {
		float a = 0.0;
		float b = 0.0;
		if (c > 0)
			a = image[c + r * width - 1];
		if (c < width - 1)
			b = image[c + r * width + 1];
		return (b - a) / 2;
	}
	return 0.0;
}

__device__ float dyKernel(float *image, int height, int width, int r, int c) {
	if (r >= 0 && r < height && c >= 0 && c < width) {
		float a = 0.0;
		float b = 0.0;
		if (r > 0)
			a = image[c + (r - 1) * width];
		if (r < height - 1)
			b = image[c + (r + 1) * width];
		return (a - b) / 2;
	}
	return 0.0;
}

__device__ float CurvatureKernel(float *curr_dog, int height, int width, int i, int j) {
  float dx1 = dxKernel(curr_dog, height, width, i, j - 1);
	float dx2 = dxKernel(curr_dog, height, width, i, j + 1);
	float dxx = (dx2 - dx1) / 2;

	float dy1 = dyKernel(curr_dog, height, width, i - 1, j);
	float dy2 = dyKernel(curr_dog, height, width, i + 1, j);
	float dyy = (dy1 - dy2) / 2;

	float dx3 = dxKernel(curr_dog, height, width, i - 1, j);
	float dx4 = dxKernel(curr_dog, height, width, i + 1, j);
	float dxy = (dx3 - dx4) / 2;
	float dyx = dxy;

	float det = dxx * dyy - dxy * dyx;
	float trace = dxx + dyy;
	return trace * trace / det;
}

__global__ void ExtremaKernel(float *dog, int *key, int *dog_index, int height, int width, 
                              int num_levels, int curr_level, float contrast_thresh, int edge_thresh) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  float *curr_dog = dog + curr_level*height*width;
  float *prev_dog = nullptr;
  float *next_dog = nullptr;
  if (curr_level > 0)
    prev_dog = curr_dog - height*width;
  if (curr_level + 1 < num_levels - 1)
    next_dog = curr_dog + height*width;

  
  if (abs(curr_dog[col + row*width]) < contrast_thresh)
    return;
  if (isLocalMinMaxKernel(curr_dog, prev_dog, next_dog, row, col, height, width)) {
    float curvature = CurvatureKernel(curr_dog, height, width, row, col);
    if (curvature <= edge_thresh) {
      key[col + row*width] = 1;
      dog_index[col + row*width] = curr_level;
    }
  }

}

void GetLocalExtremaCUDA(float **dog, int height, int width, int num_levels, vector<Point2f> &keypoints, vector<int> &dog_index, float contrast_thresh, int edge_thresh) {
  float *dogs_1D = new float[height * width * (num_levels-1)];
  for (int i=0; i<num_levels-1; i++) {
    for (int j=0; j<height; j++) {
      for (int k=0; k<width; k++) {
        dogs_1D[i*height*width + j*width + k] = dog[i][j*width + k];
      }
    }
  }
  float *dogs_cuda;
  cudaMalloc(&dogs_cuda, height * width * (num_levels-1) * sizeof(float));
  cudaMemcpy(dogs_cuda, dogs_1D, height * width * (num_levels-1) * sizeof(float), cudaMemcpyHostToDevice);
  delete[] dogs_1D;

	int *tmp_keypoints = new int[height * width]; // bool will lead to error
  int *tmp_dog_index = new int [height * width];
  int *tmp_keypoints_cuda;
  int *tmp_dog_index_cuda;
  cudaMalloc(&tmp_keypoints_cuda, height * width * sizeof(int));
  cudaMalloc(&tmp_dog_index_cuda, height * width * sizeof(int));
  for (int k = 0; k < num_levels - 1; k++) {
    for (int i=0; i<height*width; i++) {
      tmp_keypoints[i] = -1;
      tmp_dog_index[i] = -1;
    }
    cudaMemcpy(tmp_keypoints_cuda, tmp_keypoints, height * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_dog_index_cuda, tmp_dog_index, height * width * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 gridDim((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (height + BLOCK_HEIGHT - 1)/ BLOCK_HEIGHT);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
    ExtremaKernel<<<gridDim, blockDim>>>(dogs_cuda, tmp_keypoints_cuda, tmp_dog_index_cuda, 
                                         height, width, num_levels, k, contrast_thresh, edge_thresh);

    cudaMemcpy(tmp_keypoints, tmp_keypoints_cuda, height * width * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp_dog_index, tmp_dog_index_cuda, height * width * sizeof(int), cudaMemcpyDeviceToHost);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++){
				if (tmp_keypoints[j + i * width] != -1)
					keypoints.push_back(Point2f(j, i));
				if (tmp_dog_index[j + i * width] != -1)
					dog_index.push_back(tmp_dog_index[j + i * width]);
			}
		}
	}
  delete[] tmp_keypoints;
  delete[] tmp_dog_index;
  cudaFree(tmp_keypoints_cuda);
  cudaFree(tmp_dog_index_cuda);
}