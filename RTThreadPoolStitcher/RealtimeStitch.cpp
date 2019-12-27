// RealtimeStitch.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"

#pragma comment(lib, "libvlc.lib")
#pragma comment(lib, "libvlccore.lib")


#ifdef _DEBUG  
#pragma comment(lib, "opencv_core2413d.lib")   
#pragma comment(lib, "opencv_imgproc2413d.lib")   //MAT processing  
#pragma comment(lib, "opencv_highgui2413d.lib")  
//need by cv::Stitcher
#pragma comment(lib, "opencv_stitching2413d.lib")
//need by surf stitcher
#pragma comment(lib, "opencv_flann2413d.lib")    
#pragma comment(lib, "opencv_features2d2413d.lib")  
#pragma comment(lib, "opencv_nonfree2413d.lib")
#pragma comment(lib, "opencv_legacy2413d.lib")
#pragma comment(lib, "opencv_calib3d2413d.lib")

#else
#pragma comment(lib, "opencv_core2413.lib")  
#pragma comment(lib, "opencv_imgproc2413.lib")  
#pragma comment(lib, "opencv_highgui2413.lib")  
//need by cv::Stitcher
#pragma comment(lib, "opencv_stitching2413.lib")
//need by surf stitcher
#pragma comment(lib, "opencv_flann2413.lib")  
#pragma comment(lib, "opencv_features2d2413.lib")  
#pragma comment(lib, "opencv_nonfree2413.lib")  
#pragma comment(lib, "opencv_legacy2413.lib")
#pragma comment(lib, "opencv_calib3d2413.lib")
#endif 

#include "frame\CaptureManager.h"
#include "stitch\StitchManager.h"
#include "control\ComposeManager.h"


#include "control\ComposeManager.h"

int main(int argc, char* argv[])
{
	std::cout << "Starting...\n";
	bool r = CaptureManager::getInstance()->initCaptures();
	
	vector<cv::Mat> * pV = CaptureManager::getInstance()->getFrames();
	
	StitchManager * p = StitchManager::getInstance();
	bool b = p->initStitchObject(*pV);
	if (!b)
	{
		delete pV;
		std::cout << "StitchManager initStitchObject ERROR\n";
		cv::waitKey(0);
		return 1;
	}
	else
	{
		delete pV;
	}
	std::cout << "Start Compose\n";
	ComposeManager comMng(5);

	cv::Mat pano, image;

	while (1)
	{
		comMng.getNextCaptureFrame(pano);
		CV_Assert(pano.rows > 0 && pano.cols > 0);
		if (!(pano.rows > 0 && pano.cols > 0))
		{
			continue;
		}
		//resize(pano, image, Size(pano.cols*0.5, pano.rows*0.5));
		cv::imshow("pano", pano);
		cv::waitKey(10);
	}
	std::cout << "...End\n";
	
	cv::waitKey(0);
	return 0;
}