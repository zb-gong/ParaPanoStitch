#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    /* read image */
    Mat image1 = imread("/home/zibo/Pictures/pic1.png", IMREAD_COLOR); // needs to be absolute path
    Mat image2 = imread("/home/zibo/Pictures/pic2.png", IMREAD_COLOR);
    // Mat image1 = imread("/home/zibo/Pictures/pic1.png", IMREAD_GRAYSCALE);
    // Mat image2 = imread("/home/zibo/Pictures/pic2.png", IMREAD_GRAYSCALE);
    cout << "image1: " << image1.size << endl;
    cout << "image1[0]: " << image1.ptr<Vec3b>(0, 0)[0] << endl;
    cout << "image2 size: " << image2.size << " channel:" << image2.channels() << endl;
    imshow("p1", image1);
    imshow("p2", image2);
    
    /* detect features */
    // Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    Ptr<FeatureDetector> detector = ORB::create();
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

    /* extract descriptors */
    // Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
    Ptr<DescriptorExtractor> extractor = ORB::create();
    Mat descriptor1, descriptor2;
    extractor->compute(image1, keypoints1, descriptor1);
    extractor->compute(image2, keypoints2, descriptor2);

    /* Flann needs the descriptor to be of type CV_32F */
    descriptor1.convertTo(descriptor1, CV_32F);
    descriptor2.convertTo(descriptor2, CV_32F);

    /* match descriptors */
    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    vector<vector<DMatch> > matches;
    matcher->knnMatch(descriptor1, descriptor2, matches, 2);
    Mat img_matches;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, img_matches);
    imshow("p5", img_matches);
    cout << "mathces[0]distance:" << matches[0][0].distance  
         << "mathces[1]distance:" << matches[0][1].distance 
         << "mathces size:" << matches.size() << "mathces[0] size:" << matches[0].size() << endl;

    /* use lowe ratio to sift matches points */
    double ratio = 0.8;
    vector<pair<int, int> > indexes;
    for (int i=0; i<matches.size(); i++) {
        assert(matches[i].size() == 2);
        if (matches[i][0].distance < ratio * matches[i][1].distance) {
            indexes.push_back(pair<int, int> {matches[i][0].queryIdx, matches[i][0].trainIdx});
        }
    }

    /* use RANSAC to get homography matrix */
    cout << "indexes size:" << indexes.size() << endl;
    assert(indexes.size() >= 4);
    vector<Point2f> key1;
    vector<Point2f> key2;
    for (int i=0; i<indexes.size(); i++) {
        key1.push_back(keypoints1[indexes[i].first].pt);
        key2.push_back(keypoints2[indexes[i].second].pt);
    }
    Mat Homo = findHomography(key1, key2, RANSAC);

    /* stitch the images */
    Mat output;
    warpPerspective(image1, output, Homo, Size(image1.size[1] + image2.size[1], image1.size[0]));
    image2.copyTo(output.rowRange(0,image2.size[0]).colRange(0, image2.size[1]));
    imshow("po", output);

    waitKey(0);
    return 0;
}