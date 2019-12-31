#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/stitching/stitcher.hpp"

using namespace cv;
using namespace std;

extern bool StitcherPrepared;//拼接准备标志位

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

class MatStitcher
{
public:
	virtual Stitcher::Status stitch(InputArray images, OutputArray pano)=0;
	virtual ~MatStitcher(){};
};

class TestStitcher: public MatStitcher
{
public:
	Stitcher::Status stitch(InputArray images, OutputArray pano);
};

four_corners_t CalcCorners(const Mat& H, const Mat& src);
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst, four_corners_t corners);