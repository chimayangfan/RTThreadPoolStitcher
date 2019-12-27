#pragma once
/*
�̳߳�
1��������������У�������ʱ���ҿ����̴߳���û�п����߳̾͵ȴ�

*/
#include <vector>
#include "ExecuteThread.h"
#include "ThreadPoolTask.h"

class ThreadPool
{
public:
	ThreadPool(int threadNum);
	~ThreadPool();
	bool initThreadPool();
	//�������ã�ֱ���ȵ������߳�ȥִ������
	bool addJob(ThreadPoolTask * pTask);
protected:
	//��������
	ExecuteThread * getIdleThread();
private:
	int m_threadNum; //�������߳���
	std::vector<ExecuteThread *> m_vecThread;

	HANDLE m_hEvent;//�߳̿����¼��������߳�ִ��������������ź�
	int m_curIndex; //��һ�β��ҿ����̴߳Ӹ�����λ�ÿ�ʼ
};

