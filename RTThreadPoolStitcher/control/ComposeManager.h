#pragma once
#include "../frame/FrameCapture.h"
#include "opencv2/stitching/stitcher.hpp"

#include <queue>
#include <vector>
#include <Windows.h>

#include "threadpool/ThreadPool.h"
#include "ComposeTask.h"

/*
����֡�ϲ��߳�
	1.��FrameCapture�б��л�ȡ��Ч֡�б�
	2.ѡȡ�����߳�ִ�кϲ����������ƺϲ���֡�Ĵ��λ��
*/

class ComposeManager
{
public:
	ComposeManager(int threadNum);
	~ComposeManager();

	//�������ã����غϲ����֡
	bool getNextCaptureFrame(cv::Mat & frame);

	//�������ã���ȡ��Ƶ֡����,�����̳߳�ִ��
	bool executeCompose();
private:

	std::queue<ComposeTask*> m_ComposeQue;//�ϲ���������
	int m_queMaxCount;
	HANDLE m_hEvent;//�ϲ�����¼�

	ThreadPool* m_pThreadPool;
	HANDLE m_hdlThread;
};


