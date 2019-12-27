#pragma once
#include "opencv2/stitching/stitcher.hpp"
#include <vector>
#include <Windows.h>

#include "threadpool/ThreadPoolTask.h"

typedef cv::Stitcher::Status(*ComposeFunc)(cv::InputArray images, cv::OutputArray pano);



class ComposeTask :
	public ThreadPoolTask
{
public:
	ComposeTask(HANDLE* event);
	~ComposeTask();

	bool execute();

	int status; //0��δ��ɵģ�1���ϲ��ɹ���ɵģ�-1: �ϲ�����
	std::vector<cv::Mat>* imgs;
	cv::Mat pano;//�ϲ����
	HANDLE* finishEvent;
};

