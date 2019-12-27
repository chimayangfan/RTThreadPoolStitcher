#pragma once
#include <vector>
#include "FrameCapture.h"
#include "RS_define.h"
/*
����ץȡ���������
*/
class CaptureManager
{
	SINGLETON_DECLARE(CaptureManager)
public:
	bool initCaptures();

	/*��ȡ ��Ƶץȡ������б�*/
	const std::vector<FrameCapture*>* getFrameCaptureVecter(){ return &m_frameCptVec; };

	/*��ȡ ץȡ������Ƶ֡��ÿ�η��ص������У������Ŵ�m_frameCptVec�е�ÿ�������ȡһ֡
		�������ã���ȡ����ʱ����һֱ�ȴ�;*/
	std::vector<cv::Mat>* getFrames();
	

protected:
	CaptureManager();
	~CaptureManager();

private:

	std::vector<FrameCapture*>  m_frameCptVec;

};

