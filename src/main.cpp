#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <ctime> 
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <omp.h>
// #include "util.h"

using namespace cv;
using namespace std;

#define SIFT_WIDTH 4
#define SIFT_BIN 8
#define PI 3.1415926535897932384626433832795
#define MAX_DOUBLE 99999.0

void CalcDescriptor(Mat &image, vector<Point2f> &key, Mat &descriptor) {
    double hist[16][8];
    int post_index, orient_index;
    int loc_x, loc_y;
    double dx, dy, magnitude, orient;
    for (int i=0; i<key.size(); i++) {
        /* init */
        memset(hist, 0, sizeof(hist));
        post_index = orient_index = loc_x = loc_y = 0;
        dx = dy = magnitude = orient = 0;

        int curr_x = key[i].x;
        int curr_y = key[i].y;
        if (curr_x-8 < 0 || curr_x+7>=image.rows || curr_y-8<0 || curr_y+7>=image.cols)
            continue;
        for (int c=curr_y-8; c<=curr_y+7; c++) {
            for (int r=curr_x-8; r<=curr_x+7; r++) {
                int loc_x = (r-curr_x+8)/4;
                int loc_y = (c-curr_y+8)/4;
                post_index = loc_y * 4 + loc_x;
                dx = image.at<double>(curr_x+1, curr_y) - image.at<double>(curr_x-1, curr_y);
                dy = image.at<double>(curr_x, curr_y+1) - image.at<double>(curr_x, curr_y-1);
                magnitude = sqrt(dx*dx + dy*dy);
                orient = (atan2(dy,dx) == PI)? -PI : atan2(dy,dx);
                orient_index = floor(orient / 45);
                orient_index %= 7;
                hist[post_index][orient_index] += magnitude; 
            }
        }
        memcpy(descriptor.row(i).data, hist, sizeof(hist));
    }
}

void BruteForceMacher(Mat &descritor1, Mat &descritor2, vector<pair<int, int> > &indexes) {
    Mat d1, d2;
    double curr_min_dist, curr_sec_dist, curr_tmp_dist;
    int curr_min_index, curr_sec_index; 
    double ratio = 0.8;
    for (int i=0; i<descritor1.rows; i++) {
        d1 = descritor1.row(i);
        curr_min_dist = MAX_DOUBLE-1, curr_sec_dist = MAX_DOUBLE, curr_tmp_dist = MAX_DOUBLE;
        for (int j=0; j<descritor2.rows; j++) {
            d2 = descritor2.row(j);
            curr_tmp_dist = norm(d1-d2, NORM_L2);
            if (curr_tmp_dist < curr_min_dist) {
                curr_sec_index = curr_min_index;
                curr_sec_dist = curr_min_dist;
                curr_min_index = j;
                curr_min_dist = curr_tmp_dist;
                } else if (curr_tmp_dist < curr_sec_dist) {
                curr_sec_index = j;
                curr_sec_dist = curr_tmp_dist;
            }
        }
        // cout << "i" << i << endl;
        // cout << "current_sec_index:" << curr_sec_index << "dist:" << curr_sec_dist << endl;
        // cout << "current_min_index:" << curr_min_index << "dist:" << curr_min_dist << endl;    
        if (curr_min_dist < ratio * curr_sec_dist) {
            // cout << "hi there" << endl;
            indexes.push_back(pair<int, int> {i, curr_min_index});
        }
    }
}


// reference:http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
Mat CalcHomography(vector<Point2f> &key1, vector<Point2f> &key2, int iter=1000, int thr = 10) {
    assert(key1.size() == key2.size());
    
    int max_inliners = 0;
    Mat max_homograpy;
    int keypoints_size = key1.size();
    cout << "keypoints size" << keypoints_size << endl;
    Mat A_matrix(8, 9, CV_64F);

    // #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<iter; i++) {
        A_matrix.setTo(Scalar(0.0));
        double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        int index[4];
        for (int j=0; j<4; j++) {
            index[j] = rand() % keypoints_size;
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
        Mat homography = matVt.row(8).reshape(0,3);
        homography.convertTo(homography, CV_64F);
        homography = (1 / homography.at<double>(2,2)) * homography;

        int curr_inliners = 0;
        for (int j=0; j<keypoints_size; j++) {
            double p1[] = {key1[j].x, key1[j].y, 1.0};
            double p2[] = {key2[j].x, key2[j].y, 1.0};
            Mat mp1 = Mat(3, 1, CV_64F, p1);
            Mat mp2 = Mat(3, 1, CV_64F, p2);
            mp1 = homography * mp1;
            if (norm(mp1 - mp2) < thr)
                curr_inliners++;            
        }
        if (curr_inliners > max_inliners) {
            max_inliners = curr_inliners;
            max_homograpy = homography;
        }
    }
    return max_homograpy;
}


int main(int argc, char *argv[]) {
    /* read image */
    clock_t start_time = clock();
    Mat image1 = imread("/home/zibo/Pictures/pic1.png", IMREAD_COLOR); // needs to be absolute path
    Mat image2 = imread("/home/zibo/Pictures/pic2.png", IMREAD_COLOR);
    // Mat image1 = imread("/home/zibo/Pictures/pic1.png", IMREAD_GRAYSCALE);
    // Mat image2 = imread("/home/zibo/Pictures/pic2.png", IMREAD_GRAYSCALE);
    cout << "image1: " << image1.size << endl;
    cout << "image1[0]: " << image1.ptr<Vec3b>(0, 0)[0] << endl;
    cout << "image2 size: " << image2.size << " channel:" << image2.channels() << endl;
    imshow("p1", image1);
    imshow("p2", image2);
    
    clock_t ck0_time = clock(); 
    /* detect features */
    Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    // Ptr<FeatureDetector> detector = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);
    cout << "keypoints1 size:" << keypoints1.size();
    cout << " keypoints1[0].x:" << keypoints1[0].pt.x << endl;
    cout << "keypoints2 size:" << keypoints2.size() << endl;
    Mat image_keypoints1;
    Mat image_keypoints2;
    drawKeypoints(image1, keypoints1, image_keypoints1);
    drawKeypoints(image2, keypoints2, image_keypoints2);
    imshow("p3", image_keypoints1);
    imshow("p4", image_keypoints2);
    clock_t ck1_time = clock();
    cout << "detector time spent:" <<float(ck1_time - ck0_time) / CLOCKS_PER_SEC << endl;

    /* extract descriptors */
    Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
    // Ptr<DescriptorExtractor> extractor = ORB::create();
    Mat descriptor1, descriptor2;
    extractor->compute(image1, keypoints1, descriptor1);
    extractor->compute(image2, keypoints2, descriptor2);
    cout << "descriptor size:" << descriptor1.size() << endl;
    clock_t ck2_time = clock();
    cout << "extractor time spent:" <<float(ck2_time - ck1_time) / CLOCKS_PER_SEC << endl;

    /* Flann needs the descriptor to be of type CV_32F */
    // descriptor1.convertTo(descriptor1, CV_32F);
    // descriptor2.convertTo(descriptor2, CV_32F);

    /* match descriptors */
    // Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    // Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, false);
    // vector<vector<DMatch> > matches;
    // matcher->knnMatch(descriptor1, descriptor2, matches, 2);
    // Mat img_matches;
    // drawMatches(image1, keypoints1, image2, keypoints2, matches, img_matches);
    // imshow("p5", img_matches);
    // cout << "mathces[0]distance:" << matches[0][0].distance  
    //      << "mathces[1]distance:" << matches[0][1].distance 
    //      << "mathces size:" << matches.size() << "mathces[0] size:" << matches[0].size() << endl;
    clock_t ck3_time = clock();
    // cout << "matcher time spent:" <<float(ck3_time - ck2_time) / CLOCKS_PER_SEC << endl;

    /* use lowe ratio to sift matches points */
    // double ratio = 0.8;
    vector<pair<int, int> > indexes;
    // for (int i=0; i<matches.size(); i++) {
    //     assert(matches[i].size() == 2);
    //     cout << "mathces[0]distance:" << matches[i][0].distance  
    //          << "mathces[1]distance:" << matches[i][1].distance << endl;
    //     if (matches[i][0].distance < ratio * matches[i][1].distance) {
    //         indexes.push_back(pair<int, int> {matches[i][0].queryIdx, matches[i][0].trainIdx});
    //     }
    // }
    BruteForceMacher(descriptor1, descriptor2, indexes);
    clock_t ck4_time = clock();
    cout << "lowe time spent:" <<float(ck4_time - ck3_time) / CLOCKS_PER_SEC << endl;

    /* use RANSAC to get homography matrix */
    cout << "indexes size:" << indexes.size() << endl;
    assert(indexes.size() >= 4);
    vector<Point2f> key1;
    vector<Point2f> key2;
    // #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<indexes.size(); i++) {
        key1.push_back(keypoints1[indexes[i].first].pt);
        key2.push_back(keypoints2[indexes[i].second].pt);
    }
    // Mat Homo = findHomography(key1, key2, RANSAC);
    // cout << "Homo:" << Homo;
    Mat self_Homo = CalcHomography(key1, key2);
    cout << "self Homo:" << self_Homo;
    clock_t ck5_time = clock();
    cout << "matcher time spent:" <<float(ck5_time - ck4_time) / CLOCKS_PER_SEC << endl;

    /* stitch the images */
    Mat output;
    warpPerspective(image1, output, self_Homo, Size(image1.size[1] + image2.size[1], image1.size[0]));
    image2.copyTo(output.rowRange(0,image2.size[0]).colRange(0, image2.size[1]));
    imshow("p6", output);

    waitKey(1);
    cout << "total time spent:" <<float(clock() - start_time) / CLOCKS_PER_SEC << endl;
    cout << "computation time spent:" <<float(clock() - ck0_time) / CLOCKS_PER_SEC << endl;
    return 0;
}