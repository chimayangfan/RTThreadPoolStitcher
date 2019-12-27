#pragma once
//#include "stdafx.h"

/*�������ٵĵ�����*/

/*����
m_garbo ����һ���ڲ���ľ�̬����         
���ö�������ʱ��˳�����ͷ�myInstanceָ��Ķ�����Դ*/   


#define SINGLETON_DECLARE(className)      \
public:                                   \
	static className * getInstance();     \
private:                                  \
	static className * myInstance;        \
	/* ����һ���ڲ���*/                   \
	class CGarbo{                         \
	public:                               \
		CGarbo(){};                       \
		~CGarbo()                         \
		{                                 \
			if (nullptr != myInstance)    \
			{                             \
				delete myInstance;        \
				myInstance = nullptr;     \
			}                             \
		}                                 \
	};                                    \
	static CGarbo m_garbo;                


/*ʵ��*/
#define  SINGLETON_IMPLEMENT(className)  \
className* className::myInstance = nullptr; \
className* className::getInstance()      \
{                                        \
	if (myInstance == nullptr)          \
	{                                    \
		myInstance = new className();    \
	}                                    \
	return myInstance;                   \
}                                        
