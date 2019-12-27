#include "stdafx.h"
#include "CaptureManager.h"
#include "RTSPCapture.h"
#include "VLCRTSPCapture.h"
#include "opencv2/opencv.hpp"

SINGLETON_IMPLEMENT(CaptureManager)

CaptureManager::CaptureManager()
{
}
CaptureManager::~CaptureManager()
{
}

bool CaptureManager::initCaptures()
{
	FrameCapture *p1 = new
		//VLCRTSPCapture("rtsp://admin:1234@192.168.1.202:554/live1.sdp", cv::Size(1920, 1080));
		//RTSPCapture("rtsp://admin:1234@192.168.1.201:554/live1.sdp");
		RTSPCapture("C:\\Users\\Administrator\\Desktop\\Master's_thesis\\ʵ������\\Vedio\\video_left.mp4");
		//RTSPCapture("D:\\�ҹ���\\ѧϰ\\videostitch\\RealtimeStitch\\x64\\files\\bigbuckbunny_480x272.h264");

	m_frameCptVec.push_back(p1);


	FrameCapture *p2 = new 
		//VLCRTSPCapture("rtsp://admin:1234@192.168.1.201:554/live1.sdp", cv::Size(1920, 1080));
		//RTSPCapture("rtsp://admin:1234@192.168.1.202:554/live1.sdp");
		RTSPCapture("C:\\Users\\Administrator\\Desktop\\Master's_thesis\\ʵ������\\Vedio\\video_right.mp4");
		//RTSPCapture("D:\\�ҹ���\\ѧϰ\\videostitch\\RealtimeStitch\\x64\\files\\src_60.h264");	
	m_frameCptVec.push_back(p2);
	return true;
}

std::vector<cv::Mat>* CaptureManager::getFrames()
{
	vector<cv::Mat>* pVec = new vector < cv::Mat >;

	//*
	//��ȡ��Ƶ֡
	size_t i, num;
	cv::Mat frame;
	num = m_frameCptVec.size();
	for (i = 0; i < num; i++)
	{
		m_frameCptVec[i]->next(frame);
		pVec->push_back(frame);
	}
	/*/
	cv::Mat frame1,frame2;
	//frame1 = imread("E:\\visual studio 2013\\Projects\\Still-Image-Stitching-Test-Using-OpenCV-master\\Stitching_CV\\stitching_img\\m11.jpg", 1);    //��ͼ
	//frame2 = imread("E:\\visual studio 2013\\Projects\\Still-Image-Stitching-Test-Using-OpenCV-master\\Stitching_CV\\stitching_img\\m10.jpg", 1);    //��ͼ
	frame1 = cv::imread("D:\\�ҹ���\\ѧϰ\\videostitch\\RealtimeStitch\\x64\\files\\2right.jpg", 1);    //��ͼ
	frame2 = cv::imread("D:\\�ҹ���\\ѧϰ\\videostitch\\RealtimeStitch\\x64\\files\\2left.jpg", 1);    //��ͼ
	pVec->push_back(frame1);
	pVec->push_back(frame2);
	//*/

	return pVec;
}