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


int main() {

    std::vector<cv::String> filenames;
    cv::String folder = "/home/azka/Desktop/data/transformed/Images1";
    glob(folder, filenames);
    int j=0;
    double alpha = 0.33;
    double beta=1-alpha;
    for(size_t i = 0; i < filenames.size(); ++i) {

        string bname = basename(filenames[i].c_str());
        Mat src = imread(filenames[i],0);

        Mat dist_red, dist_blue, dist_green,dist_new;
        distanceTransform(src, dist_blue, CV_DIST_L1, 3);
        distanceTransform(src, dist_green, CV_DIST_L2, 3);
        distanceTransform(src, dist_new, CV_DIST_C, 3);

        src.convertTo(dist_red,dist_green.type());

        Mat avg_blue,avg_green,avg_red;
        addWeighted( dist_blue, alpha, dist_red, beta, 0.0, avg_blue);
        addWeighted( dist_green, alpha, dist_red, beta, 0.0, avg_green);
        addWeighted( dist_new, alpha, dist_red, beta, 0.0, avg_red);

        cout<<"RED CHANNELS "<<avg_red.channels();
        std::vector<cv::Mat> array_to_merge;
        array_to_merge.push_back(avg_red);
        array_to_merge.push_back(avg_green);
        array_to_merge.push_back(avg_blue);

        cv::Mat colored;
        cv::merge(array_to_merge, colored);
        cout<<"HEELOOO"<<colored.depth();
        imwrite(bname,colored);

    }

    return 0;
}