#include "stdafx.h"
#include "ComposeManager.h"
#include "../frame/CaptureManager.h"
static DWORD WINAPI threadFun(LPVOID lpParameter);

ComposeManager::ComposeManager(int threadNum)
{
	m_queMaxCount = 100;// m_queMaxCount = 3 * threadNum;

	m_hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);//�ֶ���λ

	//�����̳߳�
	m_pThreadPool = new ThreadPool(threadNum);
	if (!m_pThreadPool->initThreadPool())
	{
		//error
		cout << "EEROR:ComposeManager::ComposeManager  initThreadPool error\n";
	}

	//������ȡ��Ƶ֡������������߳�
	m_hdlThread = CreateThread(NULL, 0, threadFun, this, 0, NULL);
	if (m_hdlThread == NULL)
	{
		cout << "ERROR:CreateThread fail in ComposeManager" << endl;
	}
}

ComposeManager::~ComposeManager()
{
	CloseHandle(m_hdlThread);
	CloseHandle(m_hEvent);
	delete m_pThreadPool;
}

//�������ã����غϲ����֡
bool ComposeManager::getNextCaptureFrame(cv::Mat & frame)
{
	ComposeTask* pTask = NULL;

	do
	{
		if (!m_ComposeQue.empty())
		{
			pTask = m_ComposeQue.front();
			if (pTask->status == 0) //�������
			{
				pTask = NULL; //����ѭ���ȴ���һ������ź�
			}
			else if(pTask->status == 1)
			{
				CV_Assert(pTask->pano.rows > 0 && pTask->pano.cols > 0);
				frame = pTask->pano;
				m_ComposeQue.pop();
				delete pTask;
				break;
			}
			else
			{
				//compose error
				m_ComposeQue.pop();
				delete pTask;
				pTask = NULL; //����ѭ���ȴ���һ������ź�
			}
		}
		ResetEvent(m_hEvent);
		//�ȴ�ֱ���ϲ�ִ�����
		WaitForSingleObject(m_hEvent, INFINITE);//�ȴ��ϲ�ִ������¼�������
	}while (pTask == NULL);

	cout << "ComposeManager::getNextCaptureFrame end\n";
	return true;
}

/*
��m_composeFunc ��ȡ��֡�������̳߳ؽ��кϲ�
*/
bool ComposeManager::executeCompose()
{
	ComposeTask* pTask = new ComposeTask(&m_hEvent);

	//��ȡ��Ƶ֡
	pTask->imgs = CaptureManager::getInstance()->getFrames();
	/*
	֡��ȡ�ٶ� > �ϲ��ٶ�ʱ��m_ComposeQue���ѹ��Խ��Խ��
	���⣬��ʾ�ٶ� < �ϲ��ٶ�ʱ��m_ComposeQue���ѹ��Խ��Խ�󣬵���ȡʵʱ��Ƶ֡ʱ�����ᣬ
	*/
	if (m_ComposeQue.size() > m_queMaxCount)//���ƺϲ����еĴ�С
	{
		delete pTask;//����������֡������ʣ��������֡����֤ʵʱ��
		return false;
	}//*/
	//����ϲ�����
	m_ComposeQue.push(pTask);
	//std::cout << "add a ComposeItem,que size "<< m_ComposeQue.size() << std::endl;
	//�̳߳��첽ִ������
	return m_pThreadPool->addJob(pTask);
}

static DWORD WINAPI threadFun(LPVOID lpParameter)
{
	ComposeManager* pEctThd = (ComposeManager *)lpParameter;
	while (1)
	{
		pEctThd->executeCompose();
	}
	return 1;
}