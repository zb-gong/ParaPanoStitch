#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <unistd.h>
#include "util.h"

void BruteForceMatcherCUDA(Mat &descritor1, Mat &descritor2, vector<pair<int, int> > &indexes, float ratio);

int main(int argc, char *argv[]) {

	//TODO: getopt
	int tag;
	bool cuda_flag = false;
	while ((tag =getopt(argc, argv, "cn:")) != -1) {
		switch (tag) {
		case 'n': {
			number_of_threads = atoi(optarg);
			break;
		}
		case 'c': {
			cuda_flag = true;
			break;
		}
		default: {
			number_of_threads = 1;
			break;
		}
		}
	}
	cout << "number of threads:" << number_of_threads << endl;

	auto start_time = Clock::now();
	Mat image1_color = imread("/home/gzb/ParaPanoStitch/inputs/12.jpg", IMREAD_COLOR);
	Mat image2_color = imread("/home/gzb/ParaPanoStitch/inputs/11.jpg", IMREAD_COLOR);
	Mat image1_gray = imread("/home/gzb/ParaPanoStitch/inputs/12.jpg", IMREAD_GRAYSCALE);
	Mat image2_gray = imread("/home/gzb/ParaPanoStitch/inputs/11.jpg", IMREAD_GRAYSCALE);
	Mat image1, image2;
	image1_gray.convertTo(image1, CV_32F, 1.0 / 255, 0);
	image2_gray.convertTo(image2, CV_32F, 1.0 / 255, 0);
	int height1 = image1.rows, width1 = image1.cols;
	int height2 = image2.rows, width2 = image2.cols;
	float *image1_pt = (float *)image1.ptr<float>();
	float *image2_pt = (float *)image2.ptr<float>();

	auto ck0_time = Clock::now(); 
	vector<Point2f> keypoints1, keypoints2;
	Mat descriptor1, descriptor2;
	detectKeyPoints(image1_pt, height1, width1, keypoints1, descriptor1, cuda_flag);
	detectKeyPoints(image2_pt, height2, width2, keypoints2, descriptor2, cuda_flag);
	auto ck1_time = Clock::now();
	cout << "whole detect time spent:" << chrono::duration_cast<dsec>(ck1_time - ck0_time).count() << endl;

	// Mat image1_color_copy; 
	// Mat image2_color_copy;
	// image1_color.copyTo(image1_color_copy); 
	// image2_color.copyTo(image2_color_copy); 
	// for (Point2f p : keypoints1) {
	// 	circle(image1_color_copy, p, 1, CV_RGB(255, 0, 0), 3);
	// }
	// imshow("keypoints1", image1_color_copy);
	// for (Point2f p : keypoints2) {
	// 	circle(image2_color_copy, p, 1, CV_RGB(255, 0, 0), 3);
	// }
	// imshow("keypoints2", image2_color_copy);

	/* use lowe ratio to sift matches points */
	vector<pair<int, int> > indexes;
	if (cuda_flag)
		BruteForceMatcherCUDA(descriptor1, descriptor2, indexes, 0.8);
	else
		BruteForceMatcher(descriptor1, descriptor2, indexes, 0.8);
	auto ck2_time = Clock::now();
	cout << "match time spent:" << chrono::duration_cast<dsec>(ck2_time - ck1_time).count() << endl;

	// /* use RANSAC to get homography matrix */
	assert(indexes.size() >= 4);
	vector<Point2f> key1;
	vector<Point2f> key2;
	for (int i=0; i<indexes.size(); i++) {
	    key1.push_back(keypoints1[indexes[i].first]);
	    key2.push_back(keypoints2[indexes[i].second]);
	}
	Mat Homo;
	if (cuda_flag) {
		Homo = findHomography(key1, key2, RANSAC);
		Homo.convertTo(Homo, CV_32F);
	}
	else 
		Homo = CalcHomography(key1, key2);
	cout << "self Homo:" << Homo;
	auto ck3_time = Clock::now();
	cout << "homo time spent:" << chrono::duration_cast<dsec>(ck3_time - ck2_time).count() << endl;

	/* warp the images */
	Mat output;
	float ty = 500;
	Mat A = Mat::eye(3,3,CV_32F); A.at<float>(1,2) = ty;
	warpPerspective(image1_color, output, A*Homo, Size(image1_color.size[1] + image2_color.size[1]*2, image1_color.size[0] + image2_color.size[0]));
	image2_color.copyTo(output.rowRange(ty,image2_color.size[0]+ty).colRange(0, image2_color.size[1]));
	// imshow("p6", output);
	imwrite("../outputs/lake.jpg", output);

	// waitKey(0);
	cout << "total time spent:" << chrono::duration_cast<dsec>(Clock::now() - start_time).count() << endl;
	cout << "computation time spent:" << chrono::duration_cast<dsec>(Clock::now() - ck0_time).count() << endl;
	return 0;
}