#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
<<<<<<< HEAD
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <omp.h>
// #include "util.h"
=======
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
>>>>>>> fb95a16c8567d13a1a0225a4a15ae143fdf75c89

using namespace cv;
using namespace std;

#define SIFT_WIDTH 4
#define SIFT_BIN 8
#define PI 3.1415926535897932384626433832795
#define MAX_DOUBLE 99999.0

typedef chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;


float *createFilter(int filter_size, float sigma) {
    float *filter = new float[filter_size * filter_size];
    float sum = 0.0;
    int center_x = filter_size / 2;
    int center_y = filter_size / 2;
    float div = 1.0 / (2 * M_PI * sigma * sigma);

    // make Guassian filter
    for (int i = 0; i < filter_size; i++) {
        for (int j = 0; j < filter_size; j++) {
            int x = j - center_x;
            int y = i - center_y;
            filter[j + i * filter_size] = div * exp(-1.0 * (x * x + y * y) / (2 * sigma * sigma));
            sum += filter[j + i * filter_size];
        }
    }

    // normalize
    for (int i = 0; i < filter_size; i++) {
        for (int j = 0; j < filter_size; j++) {
            filter[j + i * filter_size] /= sum;
        }
    }
    return filter;
}

float *convolve(float *image, float *filter, int height, int width, int filter_size) {
    float *res = new float[height * width];
    int h_mid = filter_size / 2;
    int w_mid = filter_size / 2;

    int i, j, fi, fj;
    // #pragma omp parallel
    // #pragma omp for private(i, j, fi, fj)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            float sum = 0.0;
            for (fi = -h_mid; fi < filter_size - h_mid; fi++) {
                for (fj = -w_mid; fj < filter_size - w_mid; fj++) {
                    int r = i + fi;
                    int c = j + fj;
                    float val = 0.0;
                    if (r >= 0 && r < height && c >= 0 && c < width) {
                        val = image[c + r * width];
                    }
                    sum += val * filter[(fi + h_mid) * filter_size + (fj + w_mid)];
                }
            }
            res[j + i * width] = sum;
        }
    }
    return res;
}
// http://www.cse.psu.edu/~rtc12/CSE486/lecture10.pdf
// https://www.cs.utah.edu/~srikumar/cv_spring2017_files/Keypoints&Descriptors.pdf
// https://www.cs.toronto.edu/~mangas/teaching/320/slides/CSC320L10.pdf
// https://medium.com/analytics-vidhya/a-beginners-guide-to-computer-vision-part-4-pyramid-3640edeffb00
// https://www.cs.utexas.edu/~grauman/courses/fall2009/papers/local_features_synthesis_draft.pdf
float **createGaussianPyramid(float *image, int height, int width, float k, float sigma_0, int num_levels, vector<int> levels) {
  float **gaussian_pyramid = new float *[num_levels];

//   #pragma omp parallel for
  for (int i = 0; i < num_levels; i++) {
    float sigma = sigma_0 * pow(k, levels[i]);
    // https://stackoverflow.com/questions/3149279/optimal-sigma-for-gaussian-filtering-of-an-image
    int filter_size = ceil(6 * sigma); // maybe constant 5x5 filter
    float *filter = createFilter(filter_size, sigma);
    gaussian_pyramid[i] = convolve(image, filter, height, width, filter_size);
    delete[] filter;
  }
  return gaussian_pyramid;
}

// D(x,y,sigma) = L(x,y,k*sigma) - L(x,y,sigma)
float **doDoG(float **gaussian_pyramid, int height, int width, int num_levels) {
    float **dogs = new float *[num_levels - 1];
    // #pragma omp parallel
    {
      int i, ii, jj;
      // #pragma omp for ?
      for (i = 0; i < num_levels - 1; i++) {
        float *dog = new float[height * width];
        float *g_0 = gaussian_pyramid[i];
        float *g_1 = gaussian_pyramid[i + 1];
        
        // #pragma omp for collapse(2)
        for (ii = 0; ii < height; ii++) {
          for (jj = 0; jj < width; jj++) {
            dog[jj + ii * width] = g_1[jj + ii * width] - g_0[jj + ii * width];
          }
        }
        dogs[i] = dog;
      }
    }
    
    return dogs;
}

void denormalize_img(float *image, int height, int width) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      image[j + i * width] *= 255.0;
    }
  }
}

bool isLocalMinMax(float *curr_dog, float *prev_dog, float *next_dog, int i, int j, int height, int width) {
    int size = 3 / 2;
    float point = curr_dog[j + i * width];
    bool local_max = true;
    bool local_min = true;

    int fi, fj;
    // #pragma omp parallel for collapse(2)
    for (fi = -size; fi <= size; fi++) {
        for (fj = -size; fj <= size; fj++) {
            int r = i + fi;
            int c = j + fj;
            if (r >= 0 && r < height && c >= 0 && c < width) {
                if (curr_dog[c + r * width] > point) {
                    local_max = false;
                }
                if (curr_dog[c + r * width] < point) {
                    local_min = false;
                }
                if ((prev_dog != nullptr) && (prev_dog[c + r * width] > point)) {
                    local_max = false;
                }
                if ((prev_dog != nullptr) && (prev_dog[c + r * width] < point)) {
                    local_min = false;
                }
                if ((next_dog != nullptr) && (next_dog[c + r * width] > point)) {
                    local_max = false;
                }
                if ((next_dog != nullptr) && (next_dog[c + r * width] < point)) {
                    local_min = false;
                }
            }
        }
    }
    return local_max || local_min;
}

float dx(float *image, int height, int width, int r, int c) {
    // dx_i = (I(i+1) - I(i-1)) / 2
    if (r >= 0 && r < height && c >= 0 && c < width) {
        float a = 0.0;
        float b = 0.0;
        if (c > 0) {
            a = image[c + r * width - 1];
        }
        if (c < width - 1) {
            b = image[c + r * width + 1];
        }
        return (b - a) / 2;
    }
    return 0.0;
}

float dy(float *image, int height, int width, int r, int c) {
  if (r >= 0 && r < height && c >= 0 && c < width) {
      float a = 0.0;
      float b = 0.0;
      if (r > 0) {
        a = image[c + (r - 1) * width];
      }
      if (r < height - 1) {
        b = image[c + (r + 1) * width];
      }
      return (a - b) / 2;
  }
  return 0.0;
}

// use 2x2 Hessian matrix to compute principle curvature
float computePrincipleCurvature(float *curr_dog, int height, int width, int i, int j) {
    /*  Hessian
     *  [dxx    dxy]
     *  [dyx    dyy]
     */
    float dx1 = dx(curr_dog, height, width, i, j - 1);
    float dx2 = dx(curr_dog, height, width, i, j + 1);
    float dxx = (dx2 - dx1) / 2;

    float dy1 = dy(curr_dog, height, width, i - 1, j);
    float dy2 = dy(curr_dog, height, width, i + 1, j);
    float dyy = (dy1 - dy2) / 2;

    float dx3 = dx(curr_dog, height, width, i - 1, j);
    float dx4 = dx(curr_dog, height, width, i + 1, j);
    float dxy = (dx3 - dx4) / 2;
    float dyx = dxy;

    // Tr(H) / Det(H)
    float det = dxx * dyy - dxy * dyx;
    float trace = dxx + dyy;
    return trace * trace / det;
}

void getLocalExtrema(float **dog, int height, int width, int num_levels, vector<Point2f> &keypoints, float contrast_thresh, int edge_thresh) {
//   #pragma omp parallel
  { 
    int i, j, k;
    for (k = 0; k < num_levels - 1; k++) {
      float *curr_dog = dog[k];
      float *prev_dog = nullptr;
      float *next_dog = nullptr;
      if (k > 0) {
        prev_dog = dog[k - 1];
      }
      if (k + 1 < num_levels - 1) {
        next_dog = dog[k + 1];
      }

    //   #pragma omp for collapse(2)
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          if (abs(curr_dog[j + i * width]) < contrast_thresh) {
            continue;
          }
          if (isLocalMinMax(curr_dog, prev_dog, next_dog, i, j, height,
                            width)) {
            float curvature =
                computePrincipleCurvature(curr_dog, height, width, i, j);
            if (curvature > edge_thresh) {
              continue;
            }
            keypoints.push_back(Point2f(j, i));
          }
        }
      }
    }
  }
  
}

void detectKeyPoints(float *image, int height, int width,
                    vector<Point2f> &keypoints) {
  // create Guassian pyramid
  float k = sqrt(2);
  float sigma_0 = 1.6;
  int num_levels = 5;
  vector<int> levels = {-1, 0, 1, 2, 3};
  float **gaussian_pyramid = createGaussianPyramid(image, height, width, k,
                                                   sigma_0, num_levels, levels);

  float **dogs = doDoG(gaussian_pyramid, height, width, num_levels);

  // detect keypoints with keypoint localization
  float contrast_thresh = 0.03;
  int edge_thresh = 10;
  getLocalExtrema(dogs, height, width, num_levels, keypoints, contrast_thresh, edge_thresh);
  printf("num keypoints detected: %lu \n", keypoints.size());
}

void CalcDescriptor(Mat &image, vector<Point2f> &key, Mat &descriptor) {
    // #pragma omp parallel for
    for (int i=0; i<key.size(); i++) {
        /* init */
        double hist[16][8];
        memset(hist, 0, sizeof(hist));
        int post_index = 0, orient_index = 0, loc_x = 0, loc_y = 0;
        double dx = 0., dy = 0., magnitude = 0., orient = 0.;

        int curr_x = key[i].x;
        int curr_y = key[i].y;
        if (curr_x-8 < 0 || curr_x+7>=image.rows || curr_y-8<0 || curr_y+7>=image.cols)
            continue;
        for (int c=curr_y-8; c<=curr_y+7; c++) {
            for (int r=curr_x-8; r<=curr_x+7; r++) {
                if (abs(r-curr_x)*abs(r-curr_x) + abs(c-curr_y)*abs(c-curr_y) > 64) // maybe need adjust
                    continue;
                int loc_x = (r-curr_x+8)/4;
                int loc_y = (c-curr_y+8)/4;
                post_index = loc_y * 4 + loc_x;
                dx = (double)image.at<uchar>(curr_x+1, curr_y) - (double)image.at<uchar>(curr_x-1, curr_y);
                dy = (double)image.at<uchar>(curr_x, curr_y+1) - (double)image.at<uchar>(curr_x, curr_y-1);
                magnitude = sqrt(dx*dx + dy*dy);
                orient = (atan2(dy,dx) == PI)? -PI : atan2(dy,dx);
                orient = (orient+PI) * 180 / PI;
                orient_index = floor(orient / 45);
                orient_index %= 7;
                hist[post_index][orient_index] += magnitude;
                // cout << "dx:" << dx << "dy:" << dy << "magnitude:" << magnitude << "orient:" << orient << endl;
                // cout << "post_index:" << post_index << "orient_index:" << orient_index << "hist:" << hist[post_index][orient_index] << endl;
            }
        }

        Mat tmp(1, 128, CV_64F);
        memcpy(tmp.data, hist, sizeof(hist));
        tmp.copyTo(descriptor.row(i));
        // cout <<"descripter" << i << ":" << descriptor.row(i) << endl;
    }
}

void BruteForceMacher(Mat &descritor1, Mat &descritor2, vector<pair<int, int> > &indexes) {
    double ratio = 0.8;
    vector<int> match_index(descritor1.rows);
    #pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int i=0; i<descritor1.rows; i++) {
        Mat d2, d1 = descritor1.row(i);
        double curr_min_dist = MAX_DOUBLE-1, curr_sec_dist = MAX_DOUBLE, curr_tmp_dist = MAX_DOUBLE;
        int curr_min_index; 
        for (int j=0; j<descritor2.rows; j++) {
            d2 = descritor2.row(j);
            curr_tmp_dist = norm(d1-d2, NORM_L2);
            if (curr_tmp_dist < curr_min_dist) {
                curr_sec_dist = curr_min_dist;
                curr_min_index = j;
                curr_min_dist = curr_tmp_dist;
            } else if (curr_tmp_dist < curr_sec_dist) {
                curr_sec_dist = curr_tmp_dist;
            }
        }
        if (curr_min_dist < ratio * curr_sec_dist) {
            match_index[i] = curr_min_index;
        } else {
            match_index[i] = -1;
        }
    }

    for (int i=0; i<descritor1.rows; i++) {
        if (match_index[i] != -1)
            indexes.push_back(pair<int, int> {i, match_index[i]});
    }
}


// reference:http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
Mat CalcHomography(vector<Point2f> &key1, vector<Point2f> &key2, int iter=1000, int thr = 10) {
    assert(key1.size() == key2.size());
    
    vector<int> inliners(iter, 0);
    vector<Mat> homograpy(iter);
    int keypoints_size = key1.size();
    cout << "keypoints size" << keypoints_size << endl;

    #pragma omp parallel num_threads(2)
    {
        unsigned int seed = 25234 + 17*omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i=0; i<iter; i++) {
            Mat A_matrix(8, 9, CV_64F);
            A_matrix.setTo(Scalar(0.0));
            double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
            int index[4];
            for (int j=0; j<4; j++) {
                index[j] = rand_r(&seed) % keypoints_size;
                double *matrix_ptr1 = A_matrix.ptr<double>(j*2, 0);
                double *matrix_ptr2 = A_matrix.ptr<double>(j*2+1, 0);
                x1 = key1[index[j]].x;
                y1 = key1[index[j]].y;
                x2 = key2[index[j]].x;
                y2 = key2[index[j]].y;
                *matrix_ptr1 = x1;
                *(matrix_ptr1+1) = y1;
                *(matrix_ptr1+2) = 1;
                *(matrix_ptr1+6) = -x1 * x2;
                *(matrix_ptr1+7) = -y1 * x2;
                *(matrix_ptr1+8) = -x2;
                *(matrix_ptr2+3) = x1;
                *(matrix_ptr2+4) = y1;
                *(matrix_ptr2+5) = 1;
                *(matrix_ptr2+6) = -x1 * y2;
                *(matrix_ptr2+7) = -y1 * y2;
                *(matrix_ptr2+8) = -y2;
            }
            Mat matD, matU, matVt;
            SVD::compute(A_matrix, matD, matU, matVt, 4);
            Mat curr_homography = matVt.row(8).reshape(0,3);
            curr_homography.convertTo(curr_homography, CV_64F);
            curr_homography = (1 / curr_homography.at<double>(2,2)) * curr_homography;
            homograpy[i] = curr_homography;

            for (int j=0; j<keypoints_size; j++) {
                double p1[] = {key1[j].x, key1[j].y, 1.0};
                double p2[] = {key2[j].x, key2[j].y, 1.0};
                Mat mp1 = Mat(3, 1, CV_64F, p1);
                Mat mp2 = Mat(3, 1, CV_64F, p2);
                mp1 = curr_homography * mp1;
                if (norm(mp1 - mp2) < thr)
                    inliners[i]++;            
            }
        }
    }

    int max_index = max_element(inliners.begin(), inliners.end()) - inliners.begin();
    return homograpy[max_index];
}


int main(int argc, char *argv[]) {

    //TODO: getopt

    /* read image */
    // auto start_time = Clock::now();
    // // Mat image1 = imread("/home/zibo/Pictures/pic1.png", IMREAD_COLOR); // needs to be absolute path
    // // Mat image2 = imread("/home/zibo/Pictures/pic2.png", IMREAD_COLOR);
    // Mat image1 = imread("/home/zibo/Pictures/pic1.png", IMREAD_GRAYSCALE);
    // Mat image2 = imread("/home/zibo/Pictures/pic2.png", IMREAD_GRAYSCALE);
    // // cout << "image1:" << (double)image1.at<uchar>(0,0) << endl;
    // cout << "image1 size: " << image1.size << " image2 size: " << image2.size 
    //      << " channel:" << image2.channels() << endl;
    // imshow("p1", image1);
    // imshow("p2", image2);
    
    // auto ck0_time = Clock::now(); 
    // /* detect features */
    // Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    // // Ptr<FeatureDetector> detector = ORB::create();
    // vector<KeyPoint> keypoints1, keypoints2;
    // detector->detect(image1, keypoints1);
    // detector->detect(image2, keypoints2);
    // cout << "keypoints1 size:" << keypoints1.size();
    // cout << " keypoints1[0].x:" << keypoints1[0].pt.x << endl;
    // cout << "keypoints2 size:" << keypoints2.size() << endl;
    // Mat image_keypoints1;
    // Mat image_keypoints2;
    // drawKeypoints(image1, keypoints1, image_keypoints1);
    // drawKeypoints(image2, keypoints2, image_keypoints2);
    // imshow("p3", image_keypoints1);
    // imshow("p4", image_keypoints2);
    // auto ck1_time = Clock::now();
    // cout << "detector time spent:" << chrono::duration_cast<dsec>(ck1_time - ck0_time).count() << endl;


    Mat image1_gray = imread("/home/zibo/Pictures/pic1.png", IMREAD_GRAYSCALE);
    // imshow("image1 gray", image1_gray);
    Mat image1;
    // normalized grayscale image to image1
    image1_gray.convertTo(image1, CV_32F, 1.0 / 255, 0);
    // imshow("image1 normalized", image1);
    int height = image1.rows;
    int width = image1.cols;
    float *image = (float *)image1.ptr<float>();
    
    vector<Point2f> keypoints;
    vector<Mat> keyimages;
    int num_octaves = 1;
    // #pragma omp parallel for
    for (int octave = 0; octave < num_octaves; octave++) {
      printf("height: %d, width: %d\n", height, width);
      detectKeyPoints(image, height, width, keypoints, keyimages);

      Mat res(height, width, CV_32F);
      memcpy(res.data, image, height * width * sizeof(float));
      string label = "octave " + to_string(octave);
      imshow(label, res);

      int height_half = height / 2;
      int width_half = width / 2;
      // printf("new height: %d, new width; %d\n", height_half, width_half);
      Mat half_res(height_half, width_half, CV_32F);
      resize(res, half_res, Size(height_half, width_half), INTER_LINEAR);

      height = height_half;
      width = width_half;
      image = (float *)half_res.ptr<float>();
    }

    for (Point2f p : keypoints) {
        circle(image1_gray, p, 1, CV_RGB(255, 0, 0), 3);
    }
    imshow("keypoints", image1_gray);



    // /* extract descriptors */
    // Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
    // // Ptr<DescriptorExtractor> extractor = ORB::create();
    // Mat descriptor1, descriptor2;
    // extractor->compute(image1, keypoints1, descriptor1);
    // extractor->compute(image2, keypoints2, descriptor2);
    // vector<Point2f> k1;
    // for (auto k:keypoints1) {
    //     Point2f tmp(k.pt.x, k.pt.y);
    //     k1.push_back(tmp);
    // }
    // // CalcDescriptor(image1, k1, descriptor1);
    // // CalcDescriptor(image2, k2, descriptor2);
    // cout << "descriptor size:" << descriptor1.size() << endl;
    // auto ck2_time = Clock::now();
    // cout << "extractor time spent:" << chrono::duration_cast<dsec>(ck2_time - ck1_time).count() << endl;

    // /* Flann needs the descriptor to be of type CV_32F */
    // // descriptor1.convertTo(descriptor1, CV_32F);
    // // descriptor2.convertTo(descriptor2, CV_32F);

    // /* use lowe ratio to sift matches points */
    // vector<pair<int, int> > indexes;
    // BruteForceMacher(descriptor1, descriptor2, indexes);
    // auto ck3_time = Clock::now();
    // cout << "match time spent:" << chrono::duration_cast<dsec>(ck3_time - ck2_time).count() << endl;

    // /* use RANSAC to get homography matrix */
    // cout << "indexes size:" << indexes.size() << endl;
    // assert(indexes.size() >= 4);
    // vector<Point2f> key1;
    // vector<Point2f> key2;
    // for (int i=0; i<indexes.size(); i++) {
    //     key1.push_back(keypoints1[indexes[i].first].pt);
    //     key2.push_back(keypoints2[indexes[i].second].pt);
    // }
    // Mat self_Homo = CalcHomography(key1, key2);
    // cout << "self Homo:" << self_Homo;
    // auto ck4_time = Clock::now();
    // cout << "matcher time spent:" << chrono::duration_cast<dsec>(ck4_time - ck3_time).count() << endl;

    // /* stitch the images */
    // Mat output;
    // warpPerspective(image1, output, self_Homo, Size(image1.size[1] + image2.size[1], image1.size[0]));
    // image2.copyTo(output.rowRange(0,image2.size[0]).colRange(0, image2.size[1]));
    // imshow("p6", output);

    waitKey(0);
    // cout << "total time spent:" << chrono::duration_cast<dsec>(Clock::now() - start_time).count() << endl;
    // cout << "computation time spent:" << chrono::duration_cast<dsec>(Clock::now() - ck0_time).count() << endl;
    return 0;
}