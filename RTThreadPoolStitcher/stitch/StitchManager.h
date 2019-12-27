#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include "MatStitcher.h"
#include "RS_define.h"
using namespace cv;
using namespace std;

/*
�����Ĺ�����
���������ƴ�ӵĶ��󣬶����ṩ��̬�ķ��ʷ���;
*/

class StitchManager
{
	/*��������*/
	SINGLETON_DECLARE(StitchManager)
public:
	bool initStitchObject(InputArray images);
	// ִ�кϲ�
	Stitcher::Status stitch(InputArray images, OutputArray pano);

private:
	StitchManager();
	~StitchManager();
	//����ƴ�ӵĶ���
	MatStitcher * pMS;

};

