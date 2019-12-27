#include "stdafx.h"
#include "ExecuteThread.h"
#include <iostream>

static DWORD WINAPI threadFun(LPVOID lpParameter);

int ExecuteThread::m_sCount = 0;

ExecuteThread::ExecuteThread(HANDLE * phIdleEvent)
{
	m_pTask = NULL;
	m_phIdleEvent = phIdleEvent;
	
	//���������̵߳��¼�����ʼ״̬���ź�,��WaitForSingleObject��,ϵͳ�Զ�����¼��ź�
	m_hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	
	//�����߳�
	m_hdlThread = CreateThread(NULL, 0, threadFun, this, 0, NULL);
	m_sCount++;
	m_index = m_sCount;
	//����Ϊ����״̬
	m_status = IDLE;
}


ExecuteThread::~ExecuteThread()
{
	//�ر��߳̾��
	CloseHandle(m_hdlThread);

	CloseHandle(m_hEvent);
}

bool ExecuteThread::setParam(ThreadPoolTask * pTask)
{
	//����Ϊ�ǿ���״̬
	m_status = BUSY;
	m_pTask = pTask;
	//����ִ��
	SetEvent(m_hEvent);
	return true;
}

void ExecuteThread::run()
{
	while (1)
	{
		WaitForSingleObject(m_hEvent, INFINITE);//�ȴ�
		//���úϲ�
		std::cout << "ExecuteThread " << m_index << " runing"<<std::endl;
		m_pTask->execute();
		m_pTask = NULL;

		//����Ϊ����״̬
		m_status = IDLE;
		//���Ϳ����¼�������ThreadPool���������̺߳���ɵ�֡
		SetEvent(*m_phIdleEvent);
	}
}

ExecuteThread::Status ExecuteThread::getStatus()
{
	return m_status;
}

static DWORD WINAPI threadFun(LPVOID lpParameter)
{
	ExecuteThread* pEctThd = (ExecuteThread *)lpParameter;
	pEctThd->run();
	return 1;
}

