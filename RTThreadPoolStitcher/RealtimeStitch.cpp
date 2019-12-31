// RealtimeStitch.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include <Windows.h>
#include <iostream>

#pragma comment(lib, "libvlc.lib")
#pragma comment(lib, "libvlccore.lib")

#include "frame\CaptureManager.h"
#include "stitch\StitchManager.h"
#include "control\ComposeManager.h"

int main(int argc, char* argv[])
{
	printf("Starting...\n");

	bool r = CaptureManager::getInstance()->initCaptures();
	vector<cv::Mat> * pV = CaptureManager::getInstance()->getFrames();
	StitchManager * p = StitchManager::getInstance();
	bool b = p->initStitchObject(*pV);
	if (!b)
	{
		delete pV;
		printf("StitchManager initStitchObject ERROR\n");
		cv::waitKey(0);
		return 1;
	}
	else
	{
		delete pV;
	}
	while (!StitcherPrepared) {
		Sleep(10);
		printf(".");
	}
	printf("\nAlready prepared, Start Compose\n");
	ComposeManager comMng(2);//初始化线程数

	cv::Mat pano, image;

	bool is_preview_ = true;
	string window_name = "pano";
	if (is_preview_) {
		cv::namedWindow(window_name, CV_WINDOW_NORMAL);//可缩放预览
		resizeWindow(window_name, 1920, 1080);//设置默认窗口尺寸
	}		
	double show_scale = 1.0, scale_interval = 0.03;
	while (1)
	{
		comMng.getNextCaptureFrame(pano);
		CV_Assert(pano.rows > 0 && pano.cols > 0);
		if (!(pano.rows > 0 && pano.cols > 0))
		{
			continue;
		}
		//resize(pano, image, Size(pano.cols*0.5, pano.rows*0.5));
		cv::imshow(window_name, pano);
		cv::waitKey(5);
	}
	std::cout << "...End\n";
	
	cv::waitKey(0);
	return 0;
}