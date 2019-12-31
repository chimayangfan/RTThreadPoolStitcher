#include "stdafx.h"
#include "StitchManager.h"
#include "OpencvStitcher.h"
#include "SurfStitcher.h"
#include "CosiftStitcher.h"


//����ʵ��
SINGLETON_IMPLEMENT(StitchManager)

StitchManager::StitchManager()
{
	pMS = nullptr;
}


StitchManager::~StitchManager()
{
	if (pMS != nullptr)
	{
		delete pMS;
		pMS = nullptr;
	}
}
bool StitchManager::initStitchObject(InputArray images)
{
	cout << "StitchManager::initStitchObject" << endl;
	//*
	//pMS = new  TestStitcher(); //TestStitcher();// SurfStitcher(); //���ڲ���
	//pMS = new  SurfStitcher();//Surfƴ��
	//pMS = new OpencvStitcher();//ORBƴ��
	pMS = new CosiftStitcher(images);//Cosiftƴ��
	/*/
	OpencvStitcher *p = new OpencvStitcher();
	if (p->init(images))
	{
		pMS = p;
	}
	else
	{
		//error
		return false;
	}
	//*/
	return true;
}
// ִ�кϲ�
Stitcher::Status StitchManager::stitch(InputArray images, OutputArray pano)
{
	if (pMS != nullptr)
	{
		return pMS->stitch(images, pano);
	}
	return Stitcher::Status::ERR_NEED_MORE_IMGS;
}
