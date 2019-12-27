#include "stdafx.h"
#include "ThreadPool.h"


ThreadPool::ThreadPool(int threadNum = 0)
{
	m_threadNum = threadNum;
	m_hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);//�߳̿����¼����ֶ���λ
}


ThreadPool::~ThreadPool()
{
	int i;
	for (i = 0; i < m_threadNum; i++)
	{
		delete m_vecThread[i];
	}
	m_threadNum = 0;
}

bool ThreadPool::initThreadPool()
{
	if (m_threadNum == 0)
	{
		//Error
		return false;
	}
	int i;
	for (i = 0; i < m_threadNum; i++)
	{
		ExecuteThread * pET = new ExecuteThread(&m_hEvent);
		m_vecThread.push_back(pET);
	}
	m_curIndex = 0;
	return true;
}

bool ThreadPool::addJob(ThreadPoolTask * pTask)
{
	//ѡ������߳�
	ExecuteThread * pET = NULL;
	while (1)
	{
		int i, num;
		for (i = m_curIndex, num = 0; num < m_threadNum; num++)
		{
			if (m_vecThread[i]->getStatus() == ExecuteThread::IDLE)
			{
				pET = m_vecThread[i];
				m_curIndex = (i + 1) % m_threadNum;
				break;
			}
			else
			{
				i = (i + 1) % m_threadNum;
			}
		}
		if (num < m_threadNum)
		{
			break;
		}
		else
		{
			ResetEvent(m_hEvent);
			//�ȴ������ź�
			WaitForSingleObject(m_hEvent, INFINITE);//�ȴ�
		}
	}
	//ִ��
	pET->setParam(pTask);
	return true;
}
