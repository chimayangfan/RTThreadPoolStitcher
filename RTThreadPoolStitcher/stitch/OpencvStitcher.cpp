#include "stdafx.h"
#include "OpencvStitcher.h"

static Stitcher stitcher = Stitcher::createDefault(false);

OpencvStitcher::OpencvStitcher()
{
	is_mapping = false;
	StitcherPrepared = true;//׼�����
}

OpencvStitcher::~OpencvStitcher()
{
}

bool OpencvStitcher::init(InputArray images)
{
	std::cout << "OpencvStitcher::init\n";
	Stitcher::Status rst = stitcher.estimateTransform(images);
	if (Stitcher::OK == rst)
	{
		is_mapping = true;
		return true;
	}
	else
	{
		return false;
	}
}

Stitcher::Status OpencvStitcher::stitch(InputArray images, OutputArray pano)
{
	std::cout << "OpencvStitcher::stitch\n";

	Mat image01, image02;
	vector<Mat> mVec;
	images.getMatVector(mVec);
	image01 = mVec[0];
	image02 = mVec[1];
	if (!is_mapping)
	{
		if (Stitcher::OK == mapping(images, projection_matrix))
		{
			is_mapping = true;
		}
		else
			return Stitcher::ERR_NEED_MORE_IMGS;
	}

	//ͼ����׼  
	Mat imageTransform1, imageTransform2;
	warpPerspective(image01, imageTransform1, projection_matrix, Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
	//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
	//imshow("ֱ�Ӿ���͸�Ӿ���任", imageTransform1);
	//imwrite("trans1.jpg", imageTransform1);

	//����ƴ�Ӻ��ͼ,����ǰ����ͼ�Ĵ�С
	int dst_width = imageTransform1.cols;  //ȡ���ҵ�ĳ���Ϊƴ��ͼ�ĳ���
	int dst_height = image02.rows;
	//cout << "dst_height: " << dst_height << "   dst_width: " << dst_width << endl;
	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	image02.copyTo(dst(Rect(0, 0, image02.cols, image02.rows)));

	//imshow("b_dst", dst);
	OptimizeSeam(image02, imageTransform1, dst, this->corners);

	Mat &pano_ = pano.getMatRef();
	dst.convertTo(pano_, CV_8U);
	return Stitcher::OK;



	//if (!status) {
	//	is_estimate = true;
	//}
	//else {
	//	is_estimate = false;
	//}
	if (!is_mapping)
	{
		std::cout << "OpencvStitcher::stitch fail\n";
		return Stitcher::ERR_NEED_MORE_IMGS;
	}
	else
	{
		std::cout << "OpencvStitcher::stitch succeed\n";
		return stitcher.composePanorama(images,pano);
	}
}

Stitcher::Status OpencvStitcher::mapping(InputArray images, OutputArray pano)
{
	Mat image01, image02;
	vector<Mat> mVec;
	images.getMatVector(mVec);
	image01 = mVec[0];
	image02 = mVec[1];


	//�Ҷ�ͼת��  
	Mat image1, image2;
	cvtColor(image01, image1, CV_RGB2GRAY);
	cvtColor(image02, image2, CV_RGB2GRAY);

	//��ȡ������    
	OrbFeatureDetector Detector(3000); // ����������ֵ��������������ȣ�ֵԽ���Խ�٣�Խ��׼
	vector<KeyPoint> keyPoint1, keyPoint2;
	Detector.detect(image1, keyPoint1);
	Detector.detect(image2, keyPoint2);

	//������������Ϊ�±ߵ�������ƥ����׼��    
	OrbDescriptorExtractor Descriptor;
	Mat imageDesc1, imageDesc2;
	Descriptor.compute(image1, keyPoint1, imageDesc1);
	Descriptor.compute(image2, keyPoint2, imageDesc2);

	flann::Index flannIndex(imageDesc1, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

	vector<DMatch> GoodMatchePoints;

	Mat macthIndex(imageDesc2.rows, 2, CV_32SC1), matchDistance(imageDesc2.rows, 2, CV_32FC1);
	flannIndex.knnSearch(imageDesc2, macthIndex, matchDistance, 2, flann::SearchParams());

#if 0
	// Lowe's algorithm,��ȡ����ƥ���
	for (int i = 0; i < matchePoints.size(); i++)
	{
		if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
		{
			GoodMatchePoints.push_back(matchePoints[i][0]);
		}
	}
#else
	//MatchesSet matches;
	set<pair<int, int>> matches;

	// Lowe's algorithm,��ȡ����ƥ���
	for (int i = 0; i < matchDistance.rows; i++)
	{
		if (matchDistance.at<float>(i, 0) < 0.4 * matchDistance.at<float>(i, 1))
		{
			DMatch dmatches(i, macthIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
			GoodMatchePoints.push_back(dmatches);
		}
	}
	cout << "\n1->2 matches: " << GoodMatchePoints.size() << endl;
#endif 

	/*Mat first_match;
	drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
	imshow("first_match ", first_match);*/

	vector<Point2f> imagePoints1, imagePoints2;

	for (int i = 0; i<GoodMatchePoints.size(); i++)
	{
		imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
		imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
	}
	if (imagePoints1.size() <= 10 || imagePoints2.size() <= 10)
	{
		return Stitcher::ERR_NEED_MORE_IMGS;
	}
	cout << "imagePoints1 count: " << imagePoints1.size() << "  imagePoints2 count: " << imagePoints2.size() << endl;
	//��ȡͼ��1��ͼ��2��ͶӰӳ����� �ߴ�Ϊ3*3
	Mat &homo = pano.getMatRef();
	homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	//Ҳ����ʹ��getPerspectiveTransform�������͸�ӱ任���󣬲���Ҫ��ֻ����4���㣬Ч���Բ�  
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	//cout << "�任����Ϊ��\n" << homo << endl << endl; //���ӳ�����      

	//������׼ͼ���ĸ���������
	this->corners = CalcCorners(homo, image01);
	//cout << "left_top:" << corners.left_top << endl;
	//cout << "left_bottom:" << corners.left_bottom << endl;
	//cout << "right_top:" << corners.right_top << endl;
	//cout << "right_bottom:" << corners.right_bottom << endl;
	return Stitcher::OK;
}
