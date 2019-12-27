#pragma once

/*
//������̣�
1�������߳�
2���ȴ�
3�����ò����������߳�ִ��
4��ִ����ɣ�������ThreadPool
5������2��
*/

#include <Windows.h>
#include "ThreadPoolTask.h"

class ExecuteThread
{
public:
	enum Status {
		BUSY,
		IDLE
	};
	static int m_sCount;//ͳ����

	/*phIdleEvent ����ʱ���͵��¼��ź�*/
	ExecuteThread(HANDLE * phIdleEvent);
	~ExecuteThread();
	//���ò�����ʼִ��
	bool setParam(ThreadPoolTask * pTask);
	Status getStatus();
	void run();
private:
	HANDLE m_hdlThread;
	ThreadPoolTask *m_pTask;

	HANDLE m_hEvent;//ִ���¼�
	HANDLE * m_phIdleEvent;//�߳̿����¼�
	Status m_status; //�߳�״̬�����У�æµ
	int m_index;//�̺߳ţ������������
	
};

