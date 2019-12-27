#pragma once
#include <queue>
#include<iostream>
#include "opencv2/opencv.hpp"
#include <mutex>
#include <Windows.h>

#define FrameQueueMaxSize  20

using namespace std;

class FrameCapture
{
public:
	FrameCapture();
	virtual~FrameCapture();
	//��ȡ��Ƶ֡
	//�ɷ��������ø�Ϊ��������
	bool next(cv::Mat & frame);
protected:
	bool pushFrame(cv::Mat & frame);

private:
	std::queue<cv::Mat> m_matQue;
	//std::mutex mutexQue;
	CRITICAL_SECTION m_queCS;//��mutex����Ч
	HANDLE m_hEvent;//��ȡ��֡�¼�
	//��Ƶ֡�������
	
};

