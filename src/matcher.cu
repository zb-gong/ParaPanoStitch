#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda.h>

using namespace std;
using namespace cv;

#define DESCR_LEN 128
#define FLOAT_MAX 9999.0
#define THREADS_PER_BLOCK 32

__global__ void MatcherKernel(float *d1, float *d2, int *match_index, int N1, int N2, int d1_size, int d2_size, float ratio) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= N1) {return;}

  // __shared__ float shared_d1[]; // maybe no need, only read once 
  // __shared__ float shared_d2[]; // (2500, 128)->1280KB maybe too large to fit in shared memory
  // for (int i=0; i<d1_size; i++)
  //   shared_d1[i] = d1[i];
  // for (int i=0; i<d2_size; i++)
  //   shared_d2[i] = d2[i];

  float *curr_d1 = d1 + index * DESCR_LEN;
  float *curr_d2;
  

  float curr_min_dist = FLOAT_MAX-1, curr_sec_dist = FLOAT_MAX, curr_tmp_dist = 0;
  int curr_min_index;
  for (int i = 0; i < N2; i++) {
    curr_tmp_dist = 0;
    curr_d2 = d2 + i * DESCR_LEN;
    for (int j = 0; j < DESCR_LEN; j++) {
      curr_tmp_dist += sqrt((*(curr_d1+j) - *(curr_d2+j)) * (*(curr_d1+j) - *(curr_d2+j)));
    }

    if (curr_tmp_dist < curr_min_dist) {
      curr_sec_dist = curr_min_dist;
      curr_min_index = i;
      curr_min_dist = curr_tmp_dist;
    } else if (curr_tmp_dist < curr_sec_dist) {
      curr_sec_dist = curr_tmp_dist;
    }
  }
  if (curr_min_dist < ratio * curr_sec_dist) {
    match_index[index] = curr_min_index;
  } else {
    match_index[index] = -1;
  }
}


void BruteForceMatcherCUDA(Mat &descriptor1, Mat &descriptor2, vector<pair<int, int> > &indexes, float ratio) {
  int N1 = descriptor1.rows; // image 1 number of keypoints
  int N2 = descriptor2.rows; // image 2 number of keypoints 
  int *match_index_cuda;
  cudaMalloc(&match_index_cuda, N1 * sizeof(int));

  float *descr1_pt = (float *)descriptor1.ptr<float>();
  float *descr1_cuda;
  int descr1_size = N1 * descriptor1.cols;
  cudaMalloc(&descr1_cuda, descr1_size * sizeof(float));
  cudaMemcpy(descr1_cuda, descr1_pt, descr1_size * sizeof(float), cudaMemcpyHostToDevice);
  float *descr2_pt = (float *)descriptor2.ptr<float>();
  float *descr2_cuda;
  int descr2_size = N2 * descriptor2.cols;
  cudaMalloc(&descr2_cuda, descr2_size * sizeof(float));
  cudaMemcpy(descr2_cuda, descr2_pt, descr2_size * sizeof(float), cudaMemcpyHostToDevice);
  
  const int threadsPerBlock = 32;
  const int blocks = (N1 + threadsPerBlock - 1) / threadsPerBlock;
  MatcherKernel<<<blocks, threadsPerBlock>>>(descr1_cuda, descr2_cuda, match_index_cuda, N1, N2, descr1_size, descr2_size, ratio);
  
  int *match_index = new int[N1];
  cudaMemcpy(match_index, match_index_cuda, N1 * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N1; i++) {
		if (match_index[i] != -1)
			indexes.push_back(pair<int, int>{i, match_index[i]});
	}
  delete[] match_index;
  cudaFree(match_index_cuda);
  cudaFree(descr1_cuda);
  cudaFree(descr2_cuda);
}
