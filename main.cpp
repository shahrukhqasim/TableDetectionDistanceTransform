#include <iostream>
// Tesseract headers
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <set>
// Open CV headers
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;


void binarizeShafait(Mat &gray, Mat &binary, int w, double k) {
    Mat sum, sumsq;
    gray.copyTo(binary);
    int half_width = w >> 1;
    integral(gray, sum, sumsq, CV_64F);
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            int x_0 = (i > half_width) ? i - half_width : 0;
            int y_0 = (j > half_width) ? j - half_width : 0;
            int x_1 = (i + half_width >= gray.rows) ? gray.rows - 1 : i + half_width;
            int y_1 = (j + half_width >= gray.cols) ? gray.cols - 1 : j + half_width;
            double area = (x_1 - x_0) * (y_1 - y_0);
            double mean = (sum.at<double>(x_0, y_0) + sum.at<double>(x_1, y_1) - sum.at<double>(x_0, y_1) -
                           sum.at<double>(x_1, y_0)) / area;
            double sq_mean = (sumsq.at<double>(x_0, y_0) + sumsq.at<double>(x_1, y_1) - sumsq.at<double>(x_0, y_1) -
                              sumsq.at<double>(x_1, y_0)) / area;
            double stdev = sqrt(sq_mean - (mean * mean));
            double threshold = mean * (1 + k * ((stdev / 128) - 1));
            if (gray.at<uchar>(i, j) > threshold)
                binary.at<uchar>(i, j) = 255;
            else
                binary.at<uchar>(i, j) = 0;
        }
    }
}


int main() {

    std::vector<cv::String> filenames;
    cv::String folder = "/home/azka/Desktop/data/transformed/Images1";
    glob(folder, filenames);
    int j=0;
    for(size_t i = 0; i < filenames.size(); ++i) {
        string bname = basename(filenames[i].c_str());

        Mat src = imread(filenames[i]);
        cv::Mat bgr[3], channel_blue, channel_green, channel_red;   //destination array

        cv::split(src, bgr);//split source

        channel_blue = bgr[0];
        channel_green = bgr[1];
        channel_red = bgr[2];
        cout << channel_red.channels();

        Mat bw;
        cvtColor(src, bw, CV_BGR2GRAY);
        threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        cout << bw.channels();

        // Perform the distance transform algorithm
        Mat dist_red, dist_blue, dist_green;
        distanceTransform(channel_red, dist_blue, CV_DIST_L1, 3);
        distanceTransform(channel_blue, dist_green, CV_DIST_L1, 3);
        distanceTransform(channel_green, dist_red, CV_DIST_L1, 3);

    //    normalize(dist_blue, dist_blue, 0, 1., NORM_MINMAX);
    //    normalize(dist_blue, dist_green, 0, 1., NORM_MINMAX);
    //    normalize(dist_blue, dist_red, 0, 1., NORM_MINMAX);

        std::vector<cv::Mat> array_to_merge;

        array_to_merge.push_back(dist_blue);
        array_to_merge.push_back(dist_green);
        array_to_merge.push_back(dist_red);

        cv::Mat color;

        cv::merge(array_to_merge, color);

        imwrite(bname, color);

    }

    return 0;
}