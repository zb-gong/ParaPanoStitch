#ifndef __UTIL_H__
#define __UTIL_H__

#include <chrono>
#include <vector>

using namespace std;
using namespace cv;

#define MAX_DOUBLE 99999.0
typedef chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;
extern int number_of_threads;

void CalcDescriptor(float **dogs, int height, int width, 
                    vector<Point2f> &key, vector<int> &dog_index, Mat &descriptor);
void detectKeyPoints(float *image, int height, int width,
										 vector<Point2f> &keypoints, Mat &descriptor);
void BruteForceMatcher(Mat &descritor1, Mat &descritor2, 
                      vector<pair<int, int> > &indexes, float ratio);
Mat CalcHomography(vector<Point2f> &key1, vector<Point2f> &key2, 
                   int iter = 2000, int thr = 10);


#endif