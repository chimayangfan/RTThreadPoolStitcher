#pragma once

/*�̳߳���Ҫִ������Ļ���
	
*/
class ThreadPoolTask
{
public:
	virtual bool execute() = 0;
	virtual ~ThreadPoolTask(){};
};

