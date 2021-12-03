// #include <iostream>
// #include <vector>
// #include <cstdlib>
// #include <opencv2/opencv.hpp>
// #include <omp.h>
// #include "util.h"

// using namespace std;
// using namespace cv;

// Mat CalcHomography(vector<Point2f> &key1, vector<Point2f> &key2, int iter=5000, int thr = 500) {
//     assert(key1.size() == key2.size());
    
//     int max_inliners = 0;
//     Mat max_homograpy;
//     int keypoints_size = key1.size();
//     Mat A_matrix(Size(8, 9), CV_64F);
    
//     // #pragma omp parallel for schedule(dynamic)
//     for (int i=0; i<iter; i++) {
//         A_matrix.setTo(Scalar(0.0));
//         double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
//         vector<int> index(4);
//         for (int j=0; j<4; j++) {
//             index[j] = rand() % keypoints_size;
//             double *matrix_ptr1 = A_matrix.ptr<double>(j*2, 0);
//             double *matrix_ptr2 = A_matrix.ptr<double>(j*2+1, 0);
//             x1 = key1[index[j]].x;
//             y1 = key1[index[j]].y;
//             x2 = key2[index[j]].x;
//             y2 = key2[index[j]].y;
//             *matrix_ptr1 = x1;
//             *(matrix_ptr1+1) = y1;
//             *(matrix_ptr1+2) = 1;
//             *(matrix_ptr1+6) = -x1 * x2;
//             *(matrix_ptr1+7) = -y1 * x2;
//             *(matrix_ptr1+8) = -x2;
//             *(matrix_ptr2+3) = x1;
//             *(matrix_ptr2+4) = y1;
//             *(matrix_ptr2+5) = 1;
//             *(matrix_ptr2+6) = -x1 * y2;
//             *(matrix_ptr2+7) = -y1 * y2;
//             *(matrix_ptr2+8) = -y2;
//         }
//         SVD homosvd(A_matrix, SVD::FULL_UV);
//         cout << "homo matrix size:" << homosvd.w.size() << endl;
//         Mat homography = homosvd.w.row(8).reshape(3, 3);
//         homography = (1 / *homography.ptr<uchar>(2,2)) * homography;

//         int curr_inliners = 0;
//         for (int j=0; j<keypoints_size; j++) {
//             double p1[] = {key1[j].x, key1[j].y, 1};
//             double p2[] = {key2[j].x, key2[j].y, 2};
//             Mat mp1 = Mat(3, 1, CV_64F, p1);
//             Mat mp2 = Mat(3, 1, CV_64F, p2);
//             mp1 = homography * mp1;
//             if (norm(mp1 - mp2) < 500)
//                 curr_inliners++;            
//         }
//         if (curr_inliners > max_inliners) {
//             max_inliners = curr_inliners;
//             max_homograpy = homography;
//         }
//     }
//     return max_homograpy;
// }