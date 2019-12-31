#include "stdafx.h"
#include "CosiftStitcher.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"  

#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <Windows.h>

cudaError_t gCudaStatus;

#define CUDA_CHECK_CALL(fun, err_msg, return_code)					\
	gCudaStatus = fun;												\
	if(gCudaStatus != cudaSuccess){									\
		fprintf(stderr, "error_code%d: %s", gCudaStatus, err_msg);	\
		return return_code;											\
	}

ConstDataGPU *const_data;
__constant__ ConstDataGPU dev_const_data[100];
ImageSize pano_size_;
ImageXYMap *dev_maps_;
ImageWeight *dev_weights_;
GPUImageData *dev_imgs_;
static int image_num_;
unsigned char *dev_pano_;

#define USE_STREAM 1
#define DST_IMAGE_CHANNEL 3

static const float WEIGHT_EPS = 1e-10f;

bool StitcherPrepared = false;//ƴ��׼����־λ��ʼ��

//Function of Class MyExposureCompensator
void MyExposureCompensator::createWeightMaps(const vector<cv::Point> &corners, const vector<Mat> &images,
	const vector<Mat> &masks, vector<Mat_<float>> &ec_maps)
{
	vector<pair<Mat, uchar> > level_masks;
	for (size_t i = 0; i < masks.size(); ++i)
		level_masks.push_back(make_pair(masks[i], 255));
	createWeightMaps(corners, images, level_masks, ec_maps);
}

void MyExposureCompensator::createWeightMaps(const vector<cv::Point> &corners, const vector<Mat> &images,
	const vector<pair<Mat, uchar>> &masks, vector<Mat_<float>> &ec_maps)
{
	CV_Assert(corners.size() == images.size() && images.size() == masks.size());

	const int num_images = static_cast<int>(images.size());

	vector<Size> bl_per_imgs(num_images);
	vector<cv::Point> block_corners;
	vector<Mat> block_images;
	vector<pair<Mat, uchar> > block_masks;

	// Construct blocks for gain compensator
	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		Size bl_per_img((images[img_idx].cols + bl_width_ - 1) / bl_width_,
			(images[img_idx].rows + bl_height_ - 1) / bl_height_);
		int bl_width = (images[img_idx].cols + bl_per_img.width - 1) / bl_per_img.width;
		int bl_height = (images[img_idx].rows + bl_per_img.height - 1) / bl_per_img.height;
		bl_per_imgs[img_idx] = bl_per_img;
		for (int by = 0; by < bl_per_img.height; ++by)
		{
			for (int bx = 0; bx < bl_per_img.width; ++bx)
			{
				cv::Point bl_tl(bx * bl_width, by * bl_height);
				cv::Point bl_br(min(bl_tl.x + bl_width, images[img_idx].cols),
					min(bl_tl.y + bl_height, images[img_idx].rows));

				block_corners.push_back(corners[img_idx] + bl_tl);
				block_images.push_back(images[img_idx](Rect(bl_tl, bl_br)));
				block_masks.push_back(make_pair(masks[img_idx].first(Rect(bl_tl, bl_br)), masks[img_idx].second));
			}
		}
	}
	//ʵ����GainCompensator��ÿ���鶼Ӧ�����油������
	GainCompensator compensator;
	compensator.feed(block_corners, block_images, block_masks);//�õ��鲹��ϵ��
	vector<double> gains = compensator.gains();//�õ�����ϵ��
	ec_maps.resize(num_images);//ȫ�ֱ���ec_maps��ʾ���п������

	Mat_<float> ker(1, 3);
	ker(0, 0) = 0.25; ker(0, 1) = 0.5; ker(0, 2) = 0.25;

	int bl_idx = 0;
	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		Size bl_per_img = bl_per_imgs[img_idx];
		ec_maps[img_idx].create(bl_per_img);

		for (int by = 0; by < bl_per_img.height; ++by)
			for (int bx = 0; bx < bl_per_img.width; ++bx, ++bl_idx)
				ec_maps[img_idx](by, bx) = static_cast<float>(gains[bl_idx]);

		sepFilter2D(ec_maps[img_idx], ec_maps[img_idx], CV_32F, ker, ker);
		sepFilter2D(ec_maps[img_idx], ec_maps[img_idx], CV_32F, ker, ker);
	}

	double max_ec = 1.0f;
	double max_ec_i, min_ec_i;
	for (int i = 0; i < num_images; i++)
	{
		cv::minMaxIdx(ec_maps[i], &min_ec_i, &max_ec_i);
		max_ec = std::max(max_ec, max_ec_i);
	}
	for (int i = 0; i < num_images; i++)
		ec_maps[i] = ec_maps[i] / ((float)(max_ec));
	ec_maps_ = ec_maps;
}

void MyExposureCompensator::feed(const vector<cv::Point> &corners, const vector<Mat> &images, vector<Mat> &masks)
{
	vector<pair<Mat, uchar> > level_masks;
	for (size_t i = 0; i < masks.size(); ++i)
		level_masks.push_back(make_pair(masks[i], 255));
	createWeightMaps(corners, images, level_masks, ec_maps_);
}

void MyExposureCompensator::gainMapResize(vector<Size> sizes_, vector<Mat_<float>> &ec_maps)
{
	int n = sizes_.size();
	for (int i = 0; i < n; i++)
	{
		Mat_<float> gain_map;
		resize(ec_maps[i], gain_map, sizes_[i], 0, 0, INTER_LINEAR);
		ec_maps[i] = gain_map.clone();
	}
}

void MyExposureCompensator::apply(int index, Mat &image)
{
	CV_Assert(image.type() == CV_8UC3);

	Mat_<float> gain_map;
	if (ec_maps_[index].size() == image.size())
		gain_map = ec_maps_[index];
	else
		resize(ec_maps_[index], gain_map, image.size(), 0, 0, INTER_LINEAR);

	for (int y = 0; y < image.rows; ++y)
	{
		const float* gain_row = gain_map.ptr<float>(y);
		cv::Point3_<uchar>* row = image.ptr<cv::Point3_<uchar> >(y);
		for (int x = 0; x < image.cols; ++x)
		{
			row[x].x = saturate_cast<uchar>(row[x].x * gain_row[x]);
			row[x].y = saturate_cast<uchar>(row[x].y * gain_row[x]);
			row[x].z = saturate_cast<uchar>(row[x].z * gain_row[x]);
		}
	}
}


//MyFeatherBlender of Class MyExposureCompensator
void MyFeatherBlender::createWeightMaps(Rect dst_roi, vector<cv::Point> corners, vector<Mat> &masks, vector<Mat> &weight_maps)
{
	dst_weight_map_.create(dst_roi.size(), CV_32F);
	dst_weight_map_.setTo(0);

	// Ϊÿһ��ͼƬ����weight map
	int image_num = masks.size();
	weight_maps.resize(image_num);
	for (int i = 0; i < image_num; i++)
	{
		createWeightMap(masks[i], m_sharpness_, weight_maps[i]);
		//cout << weight_maps[i].size() << endl;
		int dx = corners[i].x - dst_roi.x;
		int dy = corners[i].y - dst_roi.y;
		for (int y = 0; y < weight_maps[i].rows; ++y)
		{
			float* weight_row = weight_maps[i].ptr<float>(y);
			float* dst_weight_row = dst_weight_map_.ptr<float>(dy + y);
			for (int x = 0; x < weight_maps[i].cols; ++x)
			{
				//weight_row[x] = pow(weight_row[x], 0.1f);
				dst_weight_row[dx + x] += weight_row[x];
			}
		}
	}
	for (int i = 0; i < image_num; i++)
	{
		int dx = corners[i].x - dst_roi.x;
		int dy = corners[i].y - dst_roi.y;
		for (int y = 0; y < weight_maps[i].rows; ++y)
		{
			float* weight_row = weight_maps[i].ptr<float>(y);
			float* dst_weight_row = dst_weight_map_.ptr<float>(dy + y);
			for (int x = 0; x < weight_maps[i].cols; ++x)
				weight_row[x] = weight_row[x] / (dst_weight_row[dx + x] + WEIGHT_EPS);
		}
	}
}

void MyFeatherBlender::prepare(Rect dst_roi, vector<cv::Point> corners, vector<Mat> &masks)
{
	dst_.create(dst_roi.size(), CV_16SC3);
	dst_.setTo(Scalar::all(0));
	dst_mask_.create(dst_roi.size(), CV_8U);
	dst_mask_.setTo(Scalar::all(0));
	dst_roi_ = dst_roi;

	this->createWeightMaps(dst_roi, corners, masks, weight_maps_);
}

void MyFeatherBlender::feed(const Mat &img, const Mat &mask, cv::Point tl, int img_idx)
{
	CV_Assert(img.type() == CV_16SC3);
	CV_Assert(mask.type() == CV_8U);

	int dx = tl.x - dst_roi_.x;
	int dy = tl.y - dst_roi_.y;

	for (int y = 0; y < img.rows; ++y)
	{
		const cv::Point3_<short>* src_row = img.ptr<cv::Point3_<short> >(y);
		cv::Point3_<short>* dst_row = dst_.ptr<cv::Point3_<short> >(dy + y);
		const float* weight_row = weight_maps_[img_idx].ptr<float>(y);

		for (int x = 0; x < img.cols; ++x)
		{
			dst_row[dx + x].x += static_cast<short>(src_row[x].x * weight_row[x]);
			dst_row[dx + x].y += static_cast<short>(src_row[x].y * weight_row[x]);
			dst_row[dx + x].z += static_cast<short>(src_row[x].z * weight_row[x]);
		}
	}
}

void MyFeatherBlender::blend(Mat &dst, Mat &dst_mask)
{
	dst_mask_ = dst_weight_map_ > WEIGHT_EPS;
	dst = dst_;
	dst_mask = dst_mask_;
}


//Function of Class CosiftStitcher

#define STREAM_NUM 2

/*====================================================================
������      :ucharToMat
����        :��uchar���͵�����ת��ΪMat����
�������˵��:
����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
ע�ͣ������ڵ��Թ۲�
******************************************************************************/
void ucharToMat(uchar *p2, Mat& src)
{
	int nr = src.rows;
	int nc = src.cols * src.channels();//ÿһ�е�Ԫ�ظ���
	for (int j = 0; j < nr; j++)
	{
		uchar* data = src.ptr<uchar>(j);
		for (int i = 0; i < nc; i++)
		{
			*data++ = *p2++;
		}
	}
}

//Kernel of Stitch
__global__ void Stitch_kernel(unsigned char *image, ImageXYMap xymap, ImageWeight weight, unsigned char *dst, int img_idx, ImageSize pano_size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i < dev_const_data[img_idx].warped_width) && (j < dev_const_data[img_idx].warped_height))
	{
		int data_idx = j * dev_const_data[img_idx].warped_width + i;
		float map_x = xymap.xmap[data_idx];
		int map_x1 = (int)map_x;
		if (map_x1 >= 0)
		{
			float map_y = xymap.ymap[data_idx];
			int map_y1 = (int)map_y;
			int map_x2 = map_x1 + 1;
			int map_y2 = map_y1 + 1;

			int dst_data_idx = ((j + dev_const_data[img_idx].corner_y) * pano_size.width + i + dev_const_data[img_idx].corner_x) * DST_IMAGE_CHANNEL;

			float dx1 = map_x - map_x1;
			float dy1 = map_y - map_y1;
			float dx2 = map_x2 - map_x;
			float dy2 = map_y2 - map_y;
			int img_data_idx11 = (map_y1 * dev_const_data[img_idx].width + map_x1) * 3;
			int img_data_idx12 = (map_y2 * dev_const_data[img_idx].width + map_x1) * 3;
			int img_data_idx21 = (map_y1 * dev_const_data[img_idx].width + map_x2) * 3;
			int img_data_idx22 = (map_y2 * dev_const_data[img_idx].width + map_x2) * 3;
			float total_weight = weight.total_weight[data_idx];

			for (int channel = 0; channel < 3; channel++)
			{
				dst[dst_data_idx + channel] += (unsigned char)((
					image[img_data_idx11 + channel] * dx2 * dy2 +
					image[img_data_idx12 + channel] * dx2 * dy1 +
					image[img_data_idx21 + channel] * dx1 * dy2 +
					image[img_data_idx22 + channel] * dx1 * dy1
					) * total_weight);
			}
		}
	}
}

CosiftStitcher::CosiftStitcher(InputArray images)
{
	is_mapping = false;

	is_preview_ = true;
	is_save_video_ = true;//Save Output Stitched Vedio
	start_frame_index_ = 0;
	end_frame_index_ = -1;
	is_try_gpu_ = true;
	is_debug_ = false;
	trim_type_ = CosiftStitcher::TRIM_NO;

	work_megapix_ = 1.0;//-1;//
	seam_megapix_ = 0.2;//-1;//
	is_prepared_ = false;
	conf_thresh_ = 1.f;
	features_type_ = "orb";//"surf";//
	ba_cost_func_ = "ray";
	ba_refine_mask_ = "xxxxx";
	is_do_wave_correct_ = true;
	wave_correct_ = detail::WAVE_CORRECT_HORIZ;
	is_save_graph_ = false;
	warp_type_ = "cylindrical";//"plane";//"apap";//"paniniA2B1";//"transverseMercator";//"spherical";//
	expos_comp_type_ = ExposureCompensator::GAIN_BLOCKS;//ExposureCompensator::GAIN;//
	match_conf_ = 0.3f;
	seam_find_type_ = "gc_color";//"voronoi";//
	blend_type_ = Blender::FEATHER;//Blender::MULTI_BAND;//Blender::NO;//
	blend_strength_ = 5;

	//	��ȡ��ǰϵͳ�ĺ���
	SYSTEM_INFO sys_info;
	GetSystemInfo(&sys_info);
	parallel_num_ = sys_info.dwNumberOfProcessors;

	// ����ʱ���õ�һ֡�궨ƴ�Ӳ���

	vector<Mat> srcVec;
	images.getMatVector(srcVec);
	printf("Stitcher is preparing ");
	Prepare(srcVec);//Ĭ��ʹ��Classical�㷨
	if (is_prepared_) {
		StitcherPrepared = true;//׼�����
	}
	else {
		perror("\nError in preparation!\n");
	}
}

CosiftStitcher::~CosiftStitcher()
{
	DevFree();
}

cudaError_t CosiftStitcher::DevMalloc(int num_images)
{
	cudaError_t cudaStatus = cudaSetDevice(0);//Ĭ��0��GPU
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaError_t::cudaErrorLaunchFailure;
	}
	image_num_ = num_images;
	const_data = (ConstDataGPU *)(malloc(num_images * sizeof(ConstDataGPU)));
	dev_maps_ = (ImageXYMap *)(malloc(num_images * sizeof(ImageXYMap)));
	dev_weights_ = (ImageWeight *)(malloc(num_images * sizeof(ImageWeight)));
	dev_imgs_ = (GPUImageData *)(malloc(num_images * sizeof(GPUImageData)));//dev_imgs_[0].data = 0;
	return cudaError_t::cudaSuccess;
}

int CosiftStitcher::DevDataUpload(C2GInitData *c2g_data, int pano_height, int pano_width)
{
	for (int i = 0; i < image_num_; i++)
	{
		const_data[i].warped_height = c2g_data[i].warped_height;
		const_data[i].warped_width = c2g_data[i].warped_width;
		const_data[i].height = c2g_data[i].height;
		const_data[i].width = c2g_data[i].width;
		const_data[i].corner_x = c2g_data[i].corner_x;
		const_data[i].corner_y = c2g_data[i].corner_y;

		int xy_map_size = c2g_data[i].warped_height * c2g_data[i].warped_width * sizeof(float);
		int img_size = c2g_data[i].height * c2g_data[i].width * 3 * sizeof(unsigned char);

		//	��xmap��ymap���Դ��Ϸ���ռ�
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_maps_[i].xmap), xy_map_size), "cudaMalloc failed!\n", -2);
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_maps_[i].ymap), xy_map_size), "cudaMalloc failed!\n", -2);

		//	��Ȩ�ؾ������Դ��Ϸ���ռ�
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_weights_[i].ec_weight), xy_map_size), "cudaMalloc failed!\n", -2);
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_weights_[i].blend_weight), xy_map_size), "cudaMalloc failed!\n", -2);
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_weights_[i].total_weight), xy_map_size), "cudaMalloc failed!\n", -2);

		//	��ÿһ֡ͼ������Դ�
		CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_imgs_[i].data), img_size), "cudaMalloc failed!\n", -2);

		//	��������
		CUDA_CHECK_CALL(cudaMemcpy(dev_maps_[i].xmap, c2g_data[i].xmap, xy_map_size, cudaMemcpyHostToDevice),
			"cudaMemcpy xmap failed!\n", -2);
		CUDA_CHECK_CALL(cudaMemcpy(dev_maps_[i].ymap, c2g_data[i].ymap, xy_map_size, cudaMemcpyHostToDevice),
			"cudaMemcpy ymap failed!\n", -2);
		CUDA_CHECK_CALL(cudaMemcpy(dev_weights_[i].ec_weight, c2g_data[i].ec_weight, xy_map_size, cudaMemcpyHostToDevice),
			"cudaMemcpy ec_weight failed!\n", -2);
		CUDA_CHECK_CALL(cudaMemcpy(dev_weights_[i].blend_weight, c2g_data[i].blend_weight, xy_map_size, cudaMemcpyHostToDevice),
			"cudaMemcpy blend_weight failed!\n", -2);
		CUDA_CHECK_CALL(cudaMemcpy(dev_weights_[i].total_weight, c2g_data[i].total_weight, xy_map_size, cudaMemcpyHostToDevice),
			"cudaMemcpy blend_weight failed!\n", -2);
	}
	//	�����洢��
	CUDA_CHECK_CALL(cudaMemcpyToSymbol(dev_const_data, const_data, image_num_ * sizeof(ConstDataGPU)),
		"cudaMemcpyToSymbol failed\n", -2);

	pano_size_.height = pano_height;
	pano_size_.width = pano_width;
	int pano_malloc_size = pano_height * pano_width * DST_IMAGE_CHANNEL * sizeof(unsigned char);
	//	��ȫ��ͼ������Դ��Ϸ���ռ�
	CUDA_CHECK_CALL(cudaMalloc((void**)&(dev_pano_), pano_malloc_size), "cudaMalloc failed!\n", -2);
	return 0;
}

int CosiftStitcher::Cuda_Stitch(GPUImageData *images, unsigned char *dst)
{
	int pano_malloc_size = pano_size_.height * pano_size_.width * DST_IMAGE_CHANNEL * sizeof(unsigned char);
	CUDA_CHECK_CALL(cudaMemset(dev_pano_, 0, pano_malloc_size), "cudaMemset failed!\n", -2);

	for (int i = 0; i < image_num_; i++)
	{
		int img_size = const_data[i].height * const_data[i].width * 3 * sizeof(unsigned char);
		CUDA_CHECK_CALL(cudaMemcpy(dev_imgs_[i].data, images[i].data, img_size, cudaMemcpyHostToDevice),
			"cudaMemcpy images failed\n", -2);		//	2ms/f
		dim3 dimBlock(32, 16);
		dim3 dimGrid((const_data[i].warped_width + dimBlock.x - 1) / dimBlock.x,
			(const_data[i].warped_height + dimBlock.y - 1) / dimBlock.y);
		Stitch_kernel << <dimGrid, dimBlock >> >(dev_imgs_[i].data, dev_maps_[i], dev_weights_[i], dev_pano_, i, pano_size_);		//	4.1ms/f
	}

	CUDA_CHECK_CALL(cudaThreadSynchronize(), "cudaThreadSynchronize failed!\n", -2);
	CUDA_CHECK_CALL(cudaMemcpy(dst, dev_pano_, pano_malloc_size, cudaMemcpyDeviceToHost),
		"cudaMemcpy to dst failed\n", -2);			//	1.4ms/f
	return 0;
}

Stitcher::Status CosiftStitcher::stitch(InputArray images, OutputArray pano)
{
	vector<Mat> srcVec;
	images.getMatVector(srcVec);
	Mat &dst = pano.getMatRef();
	if (dst.empty())
		dst.create(dst_roi_.size(), CV_8UC3);

	int image_num = srcVec.size();
	GPUImageData *srcimages = new GPUImageData[image_num];
	for (int i = 0; i < image_num; i++)
		srcimages[i].data = srcVec[i].ptr<uchar>(0);

	//������
	Mat t(srcVec[0].rows, srcVec[0].cols, CV_8UC3, Scalar(0, 0, 0));
	ucharToMat(srcVec[0].data, t);
	Mat t1(srcVec[0].rows, srcVec[0].cols, CV_8UC3, Scalar(0, 0, 0));
	ucharToMat(srcVec[1].data, t1);

	//int flag = Cuda_Stitch(srcimages, dst.ptr<uchar>(0));
	StitchFrame(srcVec, dst);

	free(srcimages);
	return Stitcher::OK;
}

int CosiftStitcher::DevFree()
{
	for (int i = 0; i < image_num_; i++)
	{
		cudaFree(dev_maps_[i].xmap);
		cudaFree(dev_maps_[i].ymap);
		cudaFree(dev_weights_[i].ec_weight);
		cudaFree(dev_weights_[i].blend_weight);
		cudaFree(dev_weights_[i].total_weight);
	}
	free(const_data);
	free(dev_imgs_);
	free(dev_maps_);
	free(dev_weights_);
	return 0;
}

Stitcher::Status CosiftStitcher::mapping(InputArray images, OutputArray pano)
{

	return Stitcher::OK;
}

//int CosiftStitcher::stitch(vector<VideoCapture> &captures, string &writer_file_name)
//{
//	int video_num = captures.size();
//	vector<Mat> src(video_num);
//	Mat frame, dst, show_dst;
//
//	//	Debug����Ϣ
//	bool is_save_input_frames = false;
//	bool is_save_output_frames = true;
//
//	double fps = captures[0].get(CV_CAP_PROP_FPS);
//
//	// skip some frames
//	for (int j = 0; j < video_num; j++)
//		for (int i = 0; i < start_frame_index_; i++)
//			captures[j].read(frame);
//
//	// ��һ֡����һЩ��ʼ��������ȷ�������Ƶ�ķֱ���
//	for (int j = 0; j < video_num; j++)
//	{
//		if (!captures[j].read(frame))
//			return -1;
//		frame.copyTo(src[j]);
//		if (is_debug_)
//		{
//			char img_save_name[100];
//			sprintf(img_save_name, "/%d.jpg", j + 1);
//			imwrite(debug_dir_path_ + img_save_name, src[j]);
//		}
//	}
//
//	long prepare_start_clock = clock();
//	int prepare_status = Prepare(src);
//	//	����ORB�������ԣ�����Ļ���ʹ��SURF����Ȼ�����򱨴�������Ƶ����������
//	if (prepare_status == STITCH_CONFIG_ERROR)
//	{
//		cout << "video stitch config error!" << endl;
//		return -1;
//	}
//	if (prepare_status != STITCH_SUCCESS)
//	{
//		features_type_ = "surf";
//		cout << "video stitch first try failed, second try ... " << endl;
//		if (Prepare(src) != STITCH_SUCCESS)
//		{
//			cout << "videos input are invalid. Initialization failed." << endl;
//			return -1;
//		}
//	}
//	long prepare_end_clock = clock();
//	cout << "prepare time: " << prepare_end_clock - prepare_start_clock << "ms" << endl;
//	long first_frame_stitching_start = clock();
//	StitchFrame(src, dst);
//	long first_frame_stitching_end = clock();
//	cout << "first_frame time: " << first_frame_stitching_end - first_frame_stitching_start << "ms" << endl;
//	if (is_debug_)	//�����һ֡ƴ�ӽ����mask
//	{
//		imwrite(debug_dir_path_ + "/res.jpg", dst);
//		vector<Mat> img_masks(video_num);
//		for (int i = 0; i < video_num; i++)
//		{
//			img_masks[i].create(src[i].rows, src[i].cols, CV_8UC3);
//			img_masks[i].setTo(Scalar::all(255));
//		}
//		Mat dst_mask;
//		StitchFrame(img_masks, dst_mask);
//		imwrite(debug_dir_path_ + "/mask.jpg", dst_mask);
//	}
//
//	// ���������Ƶ
//	VideoWriter writer;
//	if (is_save_video_)
//	{
//		writer.open(writer_file_name, CV_FOURCC('D', 'I', 'V', '3'), 20, Size(dst.cols, dst.rows));
//		writer.write(dst);
//	}
//
//
//	// ��ʼƴ��
//	double stitch_time = 0;
//
//	FrameInfo frame_info;
//	frame_info.src.resize(video_num);
//
//	int frameidx = 1;
//
//	cout << "Stitching..." << endl;
//
//	string window_name = "��Ƶƴ��";
//	if (is_preview_)
//		namedWindow(window_name, CV_WINDOW_NORMAL);//������Ԥ��
//	double show_scale = 1.0, scale_interval = 0.03;
//	int frame_show_interval = cvFloor(1000 / fps);
//
//	int failed_frame_count = 0;
//
//	char log_string[1000];
//	char log_file_name[200];
//	SYSTEMTIME sys_time = { 0 };
//	GetLocalTime(&sys_time);
//	sprintf(log_file_name, "%d%02d%02d-%02d%02d%02d.log",
//		sys_time.wYear, sys_time.wMonth, sys_time.wDay, sys_time.wHour, sys_time.wMinute, sys_time.wSecond);
//	ofstream log_file;
//	if (is_debug_)
//		log_file.open(debug_dir_path_ + log_file_name);
//	long long startTime = clock();
//	while (true)
//	{
//		long frame_time = 0;
//		//	�ɼ�
//		long cap_start_clock = clock();
//		int j;
//		for (j = 0; j < video_num; j++)
//		{
//			if (!captures[j].read(frame))
//				break;
//			frame.copyTo(frame_info.src[j]);
//		}
//		frame_info.frame_idx = frameidx;
//		frameidx++;
//		if (j != video_num || (end_frame_index_ >= 0 && frameidx >= end_frame_index_))	//��һ����ƵԴ��������ֹͣƴ��
//			break;
//
//		//	ƴ��
//		long stitch_start_clock = clock();
//		frame_info.stitch_status = StitchFrame(frame_info.src, frame_info.dst);
//		long stitch_clock = clock();
//		sprintf(log_string, "\tframe %d: stitch(%dms), capture(%dms)",
//			frame_info.frame_idx, stitch_clock - stitch_start_clock, stitch_start_clock - cap_start_clock);
//		printf("%s", log_string);
//		if (is_debug_)
//			log_file << log_string << endl;
//		stitch_time += stitch_clock - stitch_start_clock;
//		frame_time += stitch_clock - cap_start_clock;
//
//		//	ƴ��ʧ��
//		if (frame_info.stitch_status != 0)
//		{
//			cout << "failed\n";
//			if (is_debug_)
//				log_file << "failed" << endl;
//			failed_frame_count++;
//			break;
//		}
//
//		//	������Ƶ
//		if (is_save_video_)
//		{
//			cout << ", write(";
//			if (is_save_output_frames)
//			{
//				char img_save_name[100];
//				sprintf(img_save_name, "/images/%d.jpg", frame_info.frame_idx);
//				imwrite(debug_dir_path_ + img_save_name, frame_info.dst);
//			}
//			long write_start_clock = clock();
//			writer.write(frame_info.dst);
//			long write_clock = clock();
//			cout << write_clock - write_start_clock << "ms)";
//			frame_time += write_clock - write_start_clock;
//		}
//		cout << endl;
//
//		//	��ʾ---
//		if (is_preview_)
//		{
//			int key = waitKey(std::max(1, (int)(frame_show_interval - frame_time)));
//			if (key == 27)	//	ESC(ASCII = 27)
//				break;
//			else if (key == 61 || key == 43)	//	+
//				show_scale += scale_interval;
//			else if (key == 45)				//	-
//				if (show_scale >= scale_interval)
//					show_scale -= scale_interval;
//			resize(frame_info.dst, show_dst, Size(show_scale * dst.cols, show_scale * dst.rows));
//			imshow(window_name, show_dst);
//		}
//	}
//	long long endTime = clock();
//	cout << "test " << endTime - startTime << endl;
//	cout << "\nStitch over" << endl;
//	cout << failed_frame_count << " frames failed." << endl;
//	cout << "\tfull view angle is " << cvRound(view_angle_) << "��" << endl;
//	if (is_debug_)
//		log_file << "\tfull view angle is " << cvRound(view_angle_) << "��" << endl;
//	writer.release();
//
//	cout << "\ton average: stitch time = " << stitch_time / (frameidx - 1) << "ms" << endl;
//	cout << "\tcenter: (" << -dst_roi_.x << ", " << -dst_roi_.y << ")" << endl;
//	if (is_debug_)
//	{
//		log_file << "\ton average: stitch time = " << stitch_time / (frameidx - 1) << "ms" << endl;
//		log_file << "\tcenter: (" << -dst_roi_.x << ", " << -dst_roi_.y << ")" << endl;
//		log_file.close();
//	}
//
//	return 0;
//}

void CosiftStitcher::InitMembers(int num_images)
{
}

/*
*	��ʼͼ����ֱܷ��ʺܸߣ�����һ�����������������ʱ��Ч��
*/
void CosiftStitcher::SetScales(vector<Mat> &src)
{
	if (work_megapix_ < 0)
		work_scale_ = 1.0;
	else
		work_scale_ = min(1.0, sqrt(work_megapix_ * 1e6 / src[0].size().area()));

	if (seam_megapix_ < 0)
		seam_scale_ = 1.0;
	else
		seam_scale_ = min(1.0, sqrt(seam_megapix_ * 1e6 / src[0].size().area()));
}

/*
*	������ȡ��֧��SURF��ORB
*/
int CosiftStitcher::FindFeatures(vector<Mat> &src, vector<ImageFeatures> &features)
{
	Ptr<FeaturesFinder> finder;
	if (features_type_ == "surf")
	{
#ifdef HAVE_OPENCV_GPU
		if (is_try_gpu_ && gpu::getCudaEnabledDeviceCount() > 0)
			finder = new SurfFeaturesFinderGpu();
		else
#endif
			finder = new SurfFeaturesFinder();
	}
	else if (features_type_ == "orb")
	{
		finder = new OrbFeaturesFinder();//Size(3,1), 1500, 1.3, 5);
	}
	else
	{
		cout << "Unknown 2D features type: '" << features_type_ << "'.\n";
		return Stitcher::ERR_NEED_MORE_IMGS;
	}

	int num_images = static_cast<int>(src.size());
	Mat full_img, img;

	for (int i = 0; i < num_images; ++i)
	{
		full_img = src[i].clone();//

		if (work_megapix_ < 0)
			img = full_img;
		else
			resize(full_img, img, Size(), work_scale_, work_scale_);

		(*finder)(img, features[i]);
		//LOGLN("Features in image #" << i+1 << "("<<img.size()<< "): " << features[i].keycv::Points.size());
		features[i].img_idx = i;
	}

	finder->collectGarbage();
	full_img.release();
	img.release();

	return Stitcher::OK;
}

/*
* ����ƥ�䣬Ȼ��ȥ������ͼƬ��������ʵ��ʱ��һ����������ͼƬ������ֹ�㷨
* ����ֵ��
*		0	����	����
*		-2	����	��������ͼƬ
*/
int CosiftStitcher::MatchImages(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches)
{
	int total_num_images = static_cast<int>(features.size());

	BestOf2NearestMatcher matcher(is_try_gpu_, match_conf_);
	matcher(features, pairwise_matches);
	matcher.collectGarbage();

	// ȥ������ͼ��
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh_);

	// һ����������ͼƬ������ֹ�㷨
	int num_images = static_cast<int>(indices.size());
	if (num_images != total_num_images)
	{
		LOGLN(total_num_images - num_images << " videos are invaild");
		return Stitcher::ERR_NEED_MORE_IMGS;
	}

	return Stitcher::OK;
}

/*
* ������궨
*/
int CosiftStitcher::CalibrateCameras(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras)
{
	HomographyBasedEstimator estimator;
	Ptr<detail::BundleAdjusterBase> adjuster;
	Mat_<uchar> refine_mask;
	vector<double> focals;

	estimator(features, pairwise_matches, cameras);

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		LOGLN("Initial intrinsics #" << i << ":\n" << cameras[i].K());
	}

	if (ba_cost_func_ == "reproj") adjuster = new detail::BundleAdjusterReproj();
	else if (ba_cost_func_ == "ray") adjuster = new detail::BundleAdjusterRay();
	else
	{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func_ << "'.\n";
		return Stitcher::ERR_NEED_MORE_IMGS;
	}
	adjuster->setConfThresh(conf_thresh_);
	refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask_[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask_[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask_[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask_[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask_[4] == 'x') refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);
	(*adjuster)(features, pairwise_matches, cameras);

	// Find median focal length
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		focals.push_back(cameras[i].focal);
		LOGLN("Camera #" << i + 1 << ":\n" << cameras[i].t << cameras[i].R);
	}

	sort(focals.begin(), focals.end());
	if (focals.size() % 2 == 1)
		median_focal_len_ = static_cast<float>(focals[focals.size() / 2]);
	else
		median_focal_len_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (is_do_wave_correct_)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R);
		waveCorrect(rmats, wave_correct_);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}

	if (is_debug_)
		this->saveCameraParam(debug_dir_path_ + "/camera_param.dat");

	return Stitcher::OK;
}

/*
*	����ˮƽ�ӽǣ������ж��Ƿ�������ƽ��ͶӰ
*/
double CosiftStitcher::GetViewAngle(vector<Mat> &src, vector<CameraParams> &cameras)
{
	Ptr<WarperCreator> warper_creator = new cv::CylindricalWarper();
	Ptr<RotationWarper> warper = warper_creator->create(median_focal_len_);

	int num_images = static_cast<int>(src.size());
	vector<cv::Point> corners;
	vector<Size> sizes;
	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		Rect roi = warper->warpRoi(Size(src[i].cols * work_scale_, src[i].rows * work_scale_), K, cameras[i].R);
		corners.push_back(roi.tl());
		sizes.push_back(roi.size());
	}
	Rect result_roi = resultRoi(corners, sizes);
	double view_angle = result_roi.width * 180.0 / (median_focal_len_  * CV_PI);
	return view_angle;
}

/*
*	����ӷ�֮ǰ����Ҫ�Ȱ�ԭʼͼ���mask�����������ͶӰ
*/
int CosiftStitcher::WarpForSeam(vector<Mat> &src, vector<CameraParams> &cameras, vector<Mat> &masks_warped, vector<Mat> &images_warped)
{
	// Warp images and their masks
#ifdef HAVE_OPENCV_GPU
	if (is_try_gpu_ && gpu::getCudaEnabledDeviceCount() > 0)
	{
		if (warp_type_ == "plane") warper_creator_ = new cv::PlaneWarperGpu();
		else if (warp_type_ == "cylindrical") warper_creator_ = new cv::CylindricalWarperGpu();
		else if (warp_type_ == "spherical") warper_creator_ = new cv::SphericalWarperGpu();
	}
	else
#endif
	{
		if (warp_type_ == "plane") warper_creator_ = new cv::PlaneWarper();
		else if (warp_type_ == "cylindrical") warper_creator_ = new cv::CylindricalWarper();
		else if (warp_type_ == "spherical") warper_creator_ = new cv::SphericalWarper();
		else if (warp_type_ == "fisheye") warper_creator_ = new cv::FisheyeWarper();
		else if (warp_type_ == "stereographic") warper_creator_ = new cv::StereographicWarper();
		else if (warp_type_ == "compressedPlaneA2B1") warper_creator_ = new cv::CompressedRectilinearWarper(2, 1);
		else if (warp_type_ == "compressedPlaneA1.5B1") warper_creator_ = new cv::CompressedRectilinearWarper(1.5, 1);
		else if (warp_type_ == "compressedPlanePortraitA2B1") warper_creator_ = new cv::CompressedRectilinearPortraitWarper(2, 1);
		else if (warp_type_ == "compressedPlanePortraitA1.5B1") warper_creator_ = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
		else if (warp_type_ == "paniniA2B1") warper_creator_ = new cv::PaniniWarper(2, 1);
		else if (warp_type_ == "paniniA1.5B1") warper_creator_ = new cv::PaniniWarper(1.5, 1);
		else if (warp_type_ == "paniniPortraitA2B1") warper_creator_ = new cv::PaniniPortraitWarper(2, 1);
		else if (warp_type_ == "paniniPortraitA1.5B1") warper_creator_ = new cv::PaniniPortraitWarper(1.5, 1);
		else if (warp_type_ == "mercator") warper_creator_ = new cv::MercatorWarper();
		else if (warp_type_ == "transverseMercator") warper_creator_ = new cv::TransverseMercatorWarper();
	}

	if (warper_creator_.empty())
	{
		cout << "Can't create the following warper '" << warp_type_ << "'\n";
		return Stitcher::ERR_NEED_MORE_IMGS;
	}

	float warp_scale = static_cast<float>(median_focal_len_ * seam_scale_ / work_scale_);
	Ptr<RotationWarper> warper = warper_creator_->create(warp_scale);
	int full_pano_width = cvFloor(warp_scale * 2 * CV_PI);

	int num_images = static_cast<int>(src.size());
	Mat img, mask;
	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_scale_ / work_scale_;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		if (seam_megapix_ < 0)
			img = src[i].clone();
		else
			resize(src[i], img, Size(), seam_scale_, seam_scale_);

		mask.create(img.size(), CV_8U);
		mask.setTo(Scalar::all(255));
		Mat tmp_mask_warped, tmp_img_warped;
		cv::Point tmp_corner;
		Size tmp_size;
		warper->warp(mask, K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, tmp_mask_warped);

		//	����360��ƴ�ӵ��������
		tmp_corner = warper->warp(img, K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, tmp_img_warped);
		//cout << "warped width = " << tmp_mask_warped.cols << ", pano width = " << full_pano_width << endl;
		if (abs(tmp_mask_warped.cols - full_pano_width) <= 10)
		{
			int x1, x2;
			FindWidestInpaintRange(tmp_mask_warped, x1, x2);
			Mat mask1, mask2, img1, img2;
			Rect rect1(0, 0, x1, tmp_mask_warped.rows), rect2(x2 + 1, 0, tmp_mask_warped.cols - 1 - x2, tmp_mask_warped.rows);
			tmp_mask_warped(rect1).copyTo(mask1);
			tmp_mask_warped(rect2).copyTo(mask2);
			masks_warped.push_back(mask1);
			masks_warped.push_back(mask2);

			tmp_img_warped(rect1).copyTo(img1);
			tmp_img_warped(rect2).copyTo(img2);
			images_warped.push_back(img1);
			images_warped.push_back(img2);

			corners_.push_back(tmp_corner);
			corners_.push_back(tmp_corner + rect2.tl());

			sizes_.push_back(rect1.size());
			sizes_.push_back(rect2.size());
		}
		else
		{
			masks_warped.push_back(tmp_mask_warped);
			corners_.push_back(tmp_corner);
			images_warped.push_back(tmp_img_warped);
			sizes_.push_back(tmp_img_warped.size());
		}

	}
	return Stitcher::OK;
}

/*
*	���360��ƴ�����⡣���ں��360��ӷ��ͼƬ���ҵ�����inpaint����[x1, x2]
*/
int CosiftStitcher::FindWidestInpaintRange(Mat mask, int &x1, int &x2)
{
	vector<int> sum_row(mask.cols);
	uchar *mask_ptr = mask.ptr<uchar>(0);
	for (int x = 0; x < mask.cols; x++)
		sum_row[x] = 0;
	for (int x = 0; x < mask.cols; x++)
		for (int y = 0; y < mask.rows; y++)
			if (mask_ptr[y * mask.cols + x] != 0)
				sum_row[x] = 1;

	int cur_x1, cur_x2, max_range = 0;
	for (int x = 1; x < mask.cols; x++)	//	����߿϶���1
	{
		if (sum_row[x - 1] == 1 && sum_row[x] == 0)
			cur_x1 = x;
		else if (sum_row[x - 1] == 0 && sum_row[x] == 1)
		{
			cur_x2 = x - 1;
			if (cur_x2 - cur_x1 > max_range)
			{
				x1 = cur_x1;
				x2 = cur_x2;
			}
		}
	}
	return 0;
}

/*
*	����ӷ�
*/
int CosiftStitcher::FindSeam(vector<Mat> &images_warped, vector<Mat> &masks_warped)
{
	int num_images = static_cast<int>(images_warped.size());
	vector<Mat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	Ptr<SeamFinder> seam_finder;

	if (seam_find_type_ == "no")
		seam_finder = new detail::NoSeamFinder();
	else if (seam_find_type_ == "voronoi")
		seam_finder = new detail::VoronoiSeamFinder();
	else if (seam_find_type_ == "gc_color")
	{
#ifdef HAVE_OPENCV_GPU
		if (is_try_gpu_ && gpu::getCudaEnabledDeviceCount() > 0)
			seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
		else
#endif
			seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	}
	else if (seam_find_type_ == "gc_colorgrad")
	{
#ifdef HAVE_OPENCV_GPU
		if (is_try_gpu_ && gpu::getCudaEnabledDeviceCount() > 0)
			seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		else
#endif
			seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	}
	else if (seam_find_type_ == "dp_color")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
	else if (seam_find_type_ == "dp_colorgrad")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
	if (seam_finder.empty())
	{
		cout << "Can't create the following seam finder '" << seam_find_type_ << "'\n";
		return Stitcher::ERR_NEED_MORE_IMGS;
	}
	seam_finder->find(images_warped_f, corners_, masks_warped);

	images_warped_f.clear();
	return Stitcher::OK;
}

/*
*	�ָ�ԭʼͼ���С
*/
int CosiftStitcher::Rescale(vector<Mat> &src, vector<CameraParams> &cameras, vector<Mat> &seam_masks)
{
	median_focal_len_ = median_focal_len_ / work_scale_;
	Ptr<RotationWarper> warper = warper_creator_->create(median_focal_len_);
	int full_pano_width = cvFloor(median_focal_len_ * 2 * CV_PI);

	//cout << "median focal length: " << median_focal_len_ << endl;

	// Update corners and sizes
	int num_images = static_cast<int>(src.size());
	Mat tmp_mask, tmp_dilated_mask, tmp_seam_mask;
	corners_.clear();
	sizes_.clear();
	for (int src_idx = 0, seam_idx = 0; src_idx < num_images; ++src_idx)
	{
		// Update intrinsics
		cameras[src_idx].focal /= work_scale_;
		cameras[src_idx].ppx /= work_scale_;
		cameras[src_idx].ppy /= work_scale_;

		Mat K;
		cameras[src_idx].K().convertTo(K, CV_32F);

		// ��������image warp������ӳ�����
		Mat tmp_xmap, tmp_ymap;
		warper->buildMaps(src[src_idx].size(), K, cameras[src_idx].R, tmp_xmap, tmp_ymap);

		// Warp the current image mask
		Mat tmp_mask_warped, tmp_final_blend_mask;
		tmp_mask.create(src[src_idx].size(), CV_8U);
		tmp_mask.setTo(Scalar::all(255));
		cv::Point tmp_corner = warper->warp(tmp_mask, K, cameras[src_idx].R, INTER_NEAREST, BORDER_CONSTANT, tmp_mask_warped);

		//	����360��ƴ�ӵ��������
		if (abs(tmp_mask_warped.cols - full_pano_width) <= 10)
		{
			int x1, x2;
			FindWidestInpaintRange(tmp_mask_warped, x1, x2);
			Mat warped_mask[2], blend_mask[2], xmap[2], ymap[2];
			Rect rect[2];
			rect[0] = Rect(0, 0, x1, tmp_mask_warped.rows);
			rect[1] = Rect(x2 + 1, 0, tmp_mask_warped.cols - 1 - x2, tmp_mask_warped.rows);
			for (int j = 0; j < 2; j++)
			{
				tmp_mask_warped(rect[j]).copyTo(warped_mask[j]);
				final_warped_masks_.push_back(warped_mask[j]);

				tmp_xmap(rect[j]).copyTo(xmap[j]);
				xmaps_.push_back(xmap[j]);

				tmp_ymap(rect[j]).copyTo(ymap[j]);
				ymaps_.push_back(ymap[j]);

				// �����ܵ�mask = warp_mask & seam_mask
				dilate(seam_masks[seam_idx], tmp_dilated_mask, Mat());	//����
				resize(tmp_dilated_mask, tmp_seam_mask, rect[j].size());
				final_blend_masks_.push_back(warped_mask[j] & tmp_seam_mask);

				corners_.push_back(tmp_corner + rect[j].tl());
				sizes_.push_back(rect[j].size());

				src_indices_.push_back(src_idx);

				seam_idx++;
			}
		}
		else
		{
			xmaps_.push_back(tmp_xmap);
			ymaps_.push_back(tmp_ymap);
			final_warped_masks_.push_back(tmp_mask_warped);
			corners_.push_back(tmp_corner);

			Size sz = tmp_mask_warped.size();
			sizes_.push_back(sz);

			//	�����ܵ�mask = warp_mask & seam_mask
			dilate(seam_masks[seam_idx], tmp_dilated_mask, Mat());	//����
			resize(tmp_dilated_mask, tmp_seam_mask, sz);
			final_blend_masks_.push_back(tmp_mask_warped & tmp_seam_mask);

			src_indices_.push_back(src_idx);

			seam_idx++;
		}
	}

	dst_roi_ = resultRoi(corners_, sizes_);
	int parts_num = sizes_.size();
	final_warped_images_.resize(parts_num);
	for (int j = 0; j < parts_num; j++)
		final_warped_images_[j].create(sizes_[j], src[src_indices_[j]].type());

	tmp_dilated_mask.release();
	tmp_seam_mask.release();
	tmp_mask.release();

	return Stitcher::OK;
}

/*
*	ƴ�ӽ�������ǲ�������״���ü��ɷ���
*/
int CosiftStitcher::TrimRect(Rect rect)
{
	// ����ÿ��ͼ���rect�����޸�xmap��ymap
	int top = rect.y;
	int left = rect.x;
	int bottom = rect.y + rect.height - 1;
	int right = rect.x + rect.width - 1;
	int num_images = xmaps_.size();
	for (int i = 0; i < num_images; i++)
	{
		int top_i, bottom_i, left_i, right_i;
		top_i = max(dst_roi_.y + top, corners_[i].y);
		left_i = max(dst_roi_.x + left, corners_[i].x);
		bottom_i = min(corners_[i].y + sizes_[i].height - 1, dst_roi_.y + bottom);
		right_i = min(corners_[i].x + sizes_[i].width - 1, dst_roi_.x + right);

		sizes_[i].height = bottom_i - top_i + 1;
		sizes_[i].width = right_i - left_i + 1;

		Rect map_rect(left_i - corners_[i].x, top_i - corners_[i].y,
			sizes_[i].width, sizes_[i].height);

		Mat tmp_map = xmaps_[i].clone();
		tmp_map(map_rect).copyTo(xmaps_[i]);
		tmp_map = ymaps_[i].clone();
		tmp_map(map_rect).copyTo(ymaps_[i]);

		Mat tmp_img = final_blend_masks_[i].clone();
		tmp_img(map_rect).copyTo(final_blend_masks_[i]);

		corners_[i].x = left_i;
		corners_[i].y = top_i;
	}

	dst_roi_.x += left;
	dst_roi_.y += top;
	dst_roi_.width = right - left + 1;
	dst_roi_.height = bottom - top + 1;
	return Stitcher::OK;
}

/*
*	�����ƽ��ͶӰ�Ļ��������Զ�ȥ��δ�������
*/
int CosiftStitcher::TrimInpaint(vector<Mat> &src)
{
	int num_images = static_cast<int>(src.size());

	// �ȼ�������ͼ���mask
	dst_roi_ = resultRoi(corners_, sizes_);
	Mat dst = Mat::zeros(dst_roi_.height, dst_roi_.width, CV_8UC1);
	for (int i = 0; i < num_images; i++)
	{
		int dx = corners_[i].x - dst_roi_.x;
		int dy = corners_[i].y - dst_roi_.y;
		int img_rows = sizes_[i].height;
		int img_cols = sizes_[i].width;
		for (int y = 0; y < img_rows; y++)
		{
			uchar *mask_row_ptr = final_warped_masks_[i].ptr<uchar>(y);
			uchar *dst_row_ptr = dst.ptr<uchar>(dy + y);
			for (int x = 0; x < img_cols; x++)
				dst_row_ptr[dx + x] += mask_row_ptr[x];
		}
	}

	int x, y;
	// top
	for (y = 0; y < dst_roi_.height; y++)
	{
		uchar *dst_row_ptr = dst.ptr<uchar>(y);
		if (!(this->IsRowCrossInpaint(dst_row_ptr, dst_roi_.width)))
			break;
	}
	int top = y;

	// bottom
	for (y = dst_roi_.height - 1; y >= 0; y--)
	{
		uchar *dst_row_ptr = dst.ptr<uchar>(y);
		if (!(this->IsRowCrossInpaint(dst_row_ptr, dst_roi_.width)))
			break;
	}
	int bottom = y;

	// left
	uchar *dst_ptr_00 = dst.ptr<uchar>(0);
	for (x = 0; x < dst_roi_.width; x++)
	{
		for (y = top; y < bottom; y++)
			if (dst_ptr_00[y * (dst_roi_.width) + x] == 0)
				break;
		if (y == bottom)
			break;
	}
	int left = x;

	// right
	for (x = dst_roi_.width - 1; x >= 0; x--)
	{
		for (y = top; y < bottom; y++)
			if (dst_ptr_00[y * (dst_roi_.width) + x] == 0)
				break;
		if (y == bottom)
			break;
	}
	int right = x;

	// ����ÿ��ͼ���rect�����޸�xmap��ymap
	for (int i = 0; i < num_images; i++)
	{
		int top_i, bottom_i, left_i, right_i;
		top_i = max(dst_roi_.y + top, corners_[i].y);
		left_i = max(dst_roi_.x + left, corners_[i].x);
		bottom_i = min(corners_[i].y + sizes_[i].height - 1, dst_roi_.y + bottom);
		right_i = min(corners_[i].x + sizes_[i].width - 1, dst_roi_.x + right);

		sizes_[i].height = bottom_i - top_i + 1;
		sizes_[i].width = right_i - left_i + 1;

		Rect rect(left_i - corners_[i].x, top_i - corners_[i].y,
			sizes_[i].width, sizes_[i].height);

		Mat tmp_map = xmaps_[i].clone();
		tmp_map(rect).copyTo(xmaps_[i]);
		tmp_map = ymaps_[i].clone();
		tmp_map(rect).copyTo(ymaps_[i]);

		Mat tmp_img = final_blend_masks_[i].clone();
		tmp_img(rect).copyTo(final_blend_masks_[i]);

		corners_[i].x = left_i;
		corners_[i].y = top_i;
	}

	dst_roi_.x += left;
	dst_roi_.y += top;
	dst_roi_.width = right - left + 1;
	dst_roi_.height = bottom - top + 1;

	return 0;
}

/*
*	�ж�һ�����Ƿ���δ�������
*/
bool CosiftStitcher::IsRowCrossInpaint(uchar *row, int width)
{
	bool is_have_entered_inpaint = false;
	int count0 = 0;
	for (int x = 1; x < width; x++)
	{
		if (row[x] == 0)
			count0++;
		if (row[x - 1] != 0 && row[x] == 0)
			is_have_entered_inpaint = true;
		if ((row[x - 1] == 0 && row[x] != 0) && is_have_entered_inpaint)
			return true;
	}
	if (count0 >= (width / 2))
		return true;
	return false;
}

int CosiftStitcher::Prepare(vector<Mat> &src, const char* warp_type_)
{
	cv::setBreakOnError(true);
	int num_images = static_cast<int>(src.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}

	//cudaDeviceProp deviceProp;
	//int deviceCount;
	//cudaError_t cudaError;
	//cudaError = cudaGetDeviceCount(&deviceCount);
	//for (int i = 0; i < deviceCount; i++)
	//{
	//	cudaError = cudaGetDeviceProperties(&deviceProp, i);

	//	cout << "�豸 " << i + 1 << " ����Ҫ���ԣ� " << endl;
	//	cout << "�豸�Կ��ͺţ� " << deviceProp.name << endl;
	//	cout << "�豸ȫ���ڴ���������MBΪ��λ���� " << deviceProp.totalGlobalMem / 1024 / 1024 << endl;
	//	cout << "�豸��һ���߳̿飨Block���п��õ�������ڴ棨��KBΪ��λ���� " << deviceProp.sharedMemPerBlock / 1024 << endl;
	//	cout << "�豸��һ���߳̿飨Block���ֿ��õ�32λ�Ĵ��������� " << deviceProp.regsPerBlock << endl;
	//	cout << "�豸��һ���߳̿飨Block���ɰ���������߳������� " << deviceProp.maxThreadsPerBlock << endl;
	//	cout << "�豸�ļ��㹦�ܼ���Compute Capability���İ汾�ţ� " << deviceProp.major << "." << deviceProp.minor << endl;
	//	cout << "�豸�϶ദ������������ " << deviceProp.multiProcessorCount << endl;
	//}

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != 0)
	{
		LOGLN("GPU acceleration failed! Error code: " << cudaStatus << " Please ensure that you have a CUDA-capable GPU installed!");
		LOGLN("Stitching with CPU next ...");
		return -1;
	}

	cudaError_t flag;
	if (warp_type_ == "apap")
		flag = PrepareAPAP(src);
	else
		flag = PrepareClassical(src);

		if (flag == cudaSuccess)
		{
			flag = DevMalloc(num_images);
			if (flag != cudaSuccess)
				return flag;
			C2GInitData *c2g_data = new C2GInitData[num_images];
			for (int i = 0; i < num_images; i++)
			{
				c2g_data[i].xmap = xmaps_[i].ptr<float>(0);
				c2g_data[i].ymap = ymaps_[i].ptr<float>(0);
				c2g_data[i].ec_weight = ec_weight_maps_[i].ptr<float>(0);
				c2g_data[i].blend_weight = blend_weight_maps_[i].ptr<float>(0);
				c2g_data[i].total_weight = total_weight_maps_[i].ptr<float>(0);
				c2g_data[i].height = src[i].rows;
				c2g_data[i].width = src[i].cols;
				c2g_data[i].warped_height = xmaps_[i].rows;
				c2g_data[i].warped_width = xmaps_[i].cols;
				c2g_data[i].corner_x = corners_[i].x - dst_roi_.x;
				c2g_data[i].corner_y = corners_[i].y - dst_roi_.y;
			}
			DevDataUpload(c2g_data, dst_roi_.height, dst_roi_.width);
		}

	if (flag == cudaSuccess)
	{
		LOGLN("\t~Prepare complete");
		is_prepared_ = true;
	}

	return flag;
}

/*
*	APAP�㷨�ĳ�ʼ��
*/
cudaError_t CosiftStitcher::PrepareAPAP(vector<Mat> &src)
{
	int num_images = static_cast<int>(src.size());

	this->InitMembers(num_images);

	// ����һЩ�����ĳ߶ȣ����������ͼ���ӷ��ʱ��Ϊ����߳���Ч�ʣ����Զ�Դͼ�����һЩ����
	work_megapix_ = -1;	//	�Ȳ����Ƿ���
	seam_megapix_ = -1;	//	�Ȳ����Ƿ���
	this->SetScales(src);

	// �������
	vector<ImageFeatures> features(num_images);
	this->FindFeatures(src, features);

	// ����ƥ�䣬��ȥ������ͼƬ
	vector<MatchesInfo> pairwise_matches;
	this->MatchImages(features, pairwise_matches);

	// APAP�㷨
	//APAPWarper apap_warper;
	//apap_warper.buildMaps(src, features, pairwise_matches, xmaps_, ymaps_, corners_);
	for (int i = 0; i < num_images; i++)
		sizes_[i] = xmaps_[i].size();
	dst_roi_ = resultRoi(corners_, sizes_);

	// ����ӷ�
	vector<Mat> seamed_masks(num_images);
	vector<Mat> images_warped(num_images);
	vector<Mat> init_masks(num_images);
	for (int i = 0; i < num_images; i++)
	{
		init_masks[i].create(src[i].size(), CV_8U);
		init_masks[i].setTo(Scalar::all(255));
		remap(src[i], images_warped[i], xmaps_[i], ymaps_[i], INTER_LINEAR);
		remap(init_masks[i], final_warped_masks_[i], xmaps_[i], ymaps_[i], INTER_NEAREST, BORDER_CONSTANT);
		seamed_masks[i] = final_warped_masks_[i].clone();
	}
	this->FindSeam(images_warped, seamed_masks);
	LOGLN("find seam");

	// �عⲹ��
	compensator_.createWeightMaps(corners_, images_warped, final_warped_masks_, ec_weight_maps_);
	// �عⲹ��ʱ��������ȨֵҲҪresizeһ��
	compensator_.gainMapResize(sizes_, ec_weight_maps_);
	LOGLN("compensate");

	images_warped.clear();

	// �����ں�ʱ�������ص�Ȩֵ
	Size dst_sz = dst_roi_.size();
	//cout << "dst size: " << dst_sz << endl;
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength_ / 100.f;
	blender_.setSharpness(1.f / blend_width);
	for (int i = 0; i < num_images; i++)
		final_blend_masks_[i] = final_warped_masks_[i] & seamed_masks[i];
	blender_.createWeightMaps(dst_roi_, corners_, seamed_masks, blend_weight_maps_);

	return cudaError_t::cudaSuccess;
}

/*
*	Classical�㷨�ĳ�ʼ��
*/
cudaError_t CosiftStitcher::PrepareClassical(vector<Mat> &src)
{
	int num_images = static_cast<int>(src.size());
	LOGLN("Preparing...");

	this->InitMembers(num_images);

	// ����һЩ�����ĳ߶ȣ����������ͼ���ӷ��ʱ��Ϊ����߳���Ч�ʣ����Զ�Դͼ�����һЩ����
	this->SetScales(src);

	if ((cameras_.size() == 0) || (cameras_.size() != num_images))
	{
		if ((cameras_.size() != 0) && (cameras_.size() != num_images))
		{
			cameras_.clear();
			LOGLN("\t~load camera parameters error! Trying to calculate again ...");
		}

		// �������
		LOGLN("\t~finding features...");
		vector<ImageFeatures> features(num_images);
		this->FindFeatures(src, features);

		// ����ƥ�䣬��ȥ������ͼƬ
		LOGLN("\t~matching images...");
		vector<MatchesInfo> pairwise_matches;
		int retrun_flag = this->MatchImages(features, pairwise_matches);
		if (retrun_flag != 0)
			return cudaError_t::cudaErrorNotReady;

		// ������궨
		LOGLN("\t~calibrating cameras...");
		cameras_.resize(num_images);
		this->CalibrateCameras(features, pairwise_matches, cameras_);
	}


	//	����ˮƽ�ӽǣ��ж�ƽ��ͶӰ�ĺϷ���
	LOGLN("\t~calculating view angle...");
	view_angle_ = this->GetViewAngle(src, cameras_);
	if (view_angle_ > 140 && warp_type_ == "plane")
		warp_type_ = "cylindrical";

	// Ϊ�ӷ�ļ�����Warp
	LOGLN("\t~warping for seaming...");
	vector<Mat> masks_warped;
	vector<Mat> images_warped;
	this->WarpForSeam(src, cameras_, masks_warped, images_warped);

	// �عⲹ��
	LOGLN("\t~compensating...");
	compensator_.createWeightMaps(corners_, images_warped, masks_warped, ec_weight_maps_);

	// ����ӷ�
	LOGLN("\t~finding seam...");
	this->FindSeam(images_warped, masks_warped);
	images_warped.clear();

	// �������������masks��ԭ��������С
	LOGLN("\t~rescaling...");
	this->Rescale(src, cameras_, masks_warped);

	// �ü���inpaint����
	if (trim_type_ == CosiftStitcher::TRIM_AUTO)
		if (warp_type_ == "plane")
			this->TrimInpaint(src);
	if (trim_type_ == CosiftStitcher::TRIM_RECTANGLE)
		this->TrimRect(trim_rect_);

	// ƴ������
	//this->RegistEvaluation(features, pairwise_matches, cameras);

	// �عⲹ��ʱ��������ȨֵҲҪresizeһ��
	LOGLN("\t~resizing compensators' weight map...");
	compensator_.gainMapResize(sizes_, ec_weight_maps_);

	// �����ں�ʱ�������ص�Ȩֵ
	LOGLN("\t~blending...");
	Size dst_sz = dst_roi_.size();
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength_ / 100.f;
	blender_.setSharpness(1.f / blend_width);
	blender_.createWeightMaps(dst_roi_, corners_, final_blend_masks_, blend_weight_maps_);

	//	������Ȩ��
	num_images = sizes_.size();
	total_weight_maps_.resize(num_images);
	for (int i = 0; i < num_images; i++)
	{
		int n_pixel = sizes_[i].height * sizes_[i].width;
		float *blend_weight_ptr = blend_weight_maps_[i].ptr<float>(0);
		float *ec_weight_ptr = ec_weight_maps_[i].ptr<float>(0);
		total_weight_maps_[i].create(sizes_[i]);
		float *total_weight_ptr = total_weight_maps_[i].ptr<float>(0);
		for (int j = 0; j < n_pixel; j++)
			total_weight_ptr[j] = blend_weight_ptr[j] * ec_weight_ptr[j];
	}
	//	����xmap��ymap������GPU�˺���ʹ��
	for (int i = 0; i < num_images; i++)
	{
		float *xmap = xmaps_[i].ptr<float>(0);
		float *ymap = ymaps_[i].ptr<float>(0);
		int n_pixel = sizes_[i].height * sizes_[i].width;
		int src_height = src[src_indices_[i]].rows;
		int src_width = src[src_indices_[i]].cols;
		for (int j = 0; j < n_pixel; j++)
		{
			float map_x = xmap[j];
			float map_y = ymap[j];
			int map_x1 = cvFloor(map_x);
			int map_y1 = cvFloor(map_y);
			int map_x2 = map_x1 + 1;
			int map_y2 = map_y1 + 1;
			if ((map_x1 < 0) || (map_y1 < 0) || (map_x2 >= src_width) || (map_y2 >= src_height))
				xmap[j] = ymap[j] = -1;
		}
	}

	//is_prepared_ = true;
	return cudaError_t::cudaSuccess;
}

int CosiftStitcher::StitchFrame(vector<Mat> &src, Mat &dst)
{
	if (!is_prepared_)
	{
		int flag = Prepare(src);
		if (flag != 0)
			return flag;
	}

	if (is_try_gpu_)
		return StitchFrameGPU(src, dst);
	else
		return StitchFrameCPU(src, dst);
}

int CosiftStitcher::StitchFrameGPU(vector<Mat> &src, Mat &dst)
{
	if (dst.empty())
		dst.create(dst_roi_.size(), CV_8UC3);

	int image_num = src.size();
	GPUImageData *images = new GPUImageData[image_num];
	for (int i = 0; i < image_num; i++)
		images[i].data = src[i].ptr<uchar>(0);
	int flag = Cuda_Stitch(images, dst.ptr<uchar>(0));
	free(images);
	return flag;
}

int CosiftStitcher::StitchFrameCPU(vector<Mat> &src, Mat &dst)
{
	bool time_debug = false;//true;//
	long start_clock, end_clock;

	if (time_debug)
		start_clock = clock();

	int64 t;
	int num_images = src_indices_.size();

	int dst_width = dst_roi_.width;
	int dst_height = dst_roi_.height;
	if (dst.empty())
		dst.create(dst_roi_.size(), CV_8UC3);
	uchar *dst_ptr_00 = dst.ptr<uchar>(0);
	memset(dst_ptr_00, 0, dst_width * dst_height * 3);

	double warp_time[100], feed_time[100];

	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		if (time_debug)
			t = getTickCount();

		// Warp the current image
		remap(src[src_indices_[img_idx]], final_warped_images_[img_idx], xmaps_[img_idx], ymaps_[img_idx],
			INTER_LINEAR);//, BORDER_REFLECT);
		if (time_debug)
			warp_time[img_idx] = 1000 * (getTickCount() - t) / getTickFrequency();

		if (time_debug)
			t = getTickCount();
		int dx = corners_[img_idx].x - dst_roi_.x;
		int dy = corners_[img_idx].y - dst_roi_.y;
		int img_rows = sizes_[img_idx].height;
		int img_cols = sizes_[img_idx].width;
		int src_rows = src[img_idx].rows;
		int src_cols = src[img_idx].cols;

		int rows_per_parallel = img_rows / parallel_num_;
#pragma omp parallel for
		for (int parallel_idx = 0; parallel_idx < parallel_num_; parallel_idx++)
		{
			int row_start = parallel_idx * rows_per_parallel;
			int row_end = row_start + rows_per_parallel;
			if (parallel_idx == parallel_num_ - 1)
				row_end = img_rows;

			uchar *dst_ptr;
			uchar *warped_img_ptr = final_warped_images_[img_idx].ptr<uchar>(row_start);
			float *total_weight_ptr = total_weight_maps_[img_idx].ptr<float>(row_start);
			for (int y = row_start; y < row_end; y++)
			{
				dst_ptr = dst_ptr_00 + ((dy + y) * dst_width + dx) * 3;
				for (int x = 0; x < img_cols; x++)
				{
					/* �عⲹ�����ںϼ�Ȩƽ�� */
					(*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
					warped_img_ptr++;
					dst_ptr++;

					(*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
					warped_img_ptr++;
					dst_ptr++;

					(*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
					warped_img_ptr++;
					dst_ptr++;

					total_weight_ptr++;
				}
			}
		}


		if (time_debug)
			feed_time[img_idx] = 1000 * (getTickCount() - t) / getTickFrequency();
	}

	if (time_debug)
		for (int i = 0; i < num_images; i++)
			cout << "\twarp " << warp_time[i] << "ms, feed " << feed_time[i] << "ms" << endl;

	if (time_debug)
		cout << "(=" << clock() - start_clock << "ms)" << endl;

	return 0;
}

void CosiftStitcher::setDebugDirPath(string dir_path)
{
	is_debug_ = true;
	debug_dir_path_ = dir_path;
}

int CosiftStitcher::RegistEvaluation(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras)
{
	int num_images = features.size();
	Ptr<RotationWarper> warper = warper_creator_->create(median_focal_len_);

	MatchesInfo matches_info;
	vector<vector<cv::Point2f>> warped_fpts;
	warped_fpts.resize(num_images);
	for (int i = 0; i < num_images; i++)
	{
		int fpts_num = features[i].keypoints.size();
		warped_fpts[i].resize(fpts_num);
		Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		for (int j = 0; j < fpts_num; j++)
			warped_fpts[i][j] = warper->warpPoint(features[i].keypoints[j].pt, K, cameras[i].R);
	}

	double final_total_error, final_total_inliners;
	final_total_inliners = final_total_error = 0;

	for (int i = 0; i < num_images; i++)
	{
		for (int j = i + 1; j < num_images; j++)
		{
			// �������
			int idx = i * num_images + j;
			matches_info = pairwise_matches[idx];

			int inliner_nums = matches_info.num_inliers;
			if (inliner_nums < 50)// || j != i+1)
				continue;

			int matches_size = matches_info.matches.size();
			double total_error = 0;
			for (int k = 0; k < matches_size; k++)
			{
				if (matches_info.inliers_mask[k])
				{
					const DMatch& m = matches_info.matches[k];
					cv::Point2f p1 = warped_fpts[i][m.queryIdx];
					cv::Point2f p2 = warped_fpts[j][m.trainIdx];
					total_error += ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
				}
			}
			final_total_error += total_error;
			final_total_inliners += inliner_nums;
			LOGLN("\t\t~Image" << i << "-" << j << ": total error(" << total_error <<
				"), total inliners(" << inliner_nums << "), average error(" <<
				sqrt(total_error / inliner_nums) << ")");
		}
	}
	LOGLN("\t\t~all pairs' total error(" << final_total_error <<
		"), total inliners(" << final_total_inliners << "), average error(" <<
		sqrt(final_total_error / final_total_inliners) << ")");

	return 0;
}

int CosiftStitcher::stitchImage(vector<Mat> &src, Mat &pano)
{
	Prepare(src);
	if (false)
	{
		char img_name[100];
		int img_num = corners_.size();
		cout << dst_roi_ << endl;
		for (int i = 0; i < img_num; i++)
		{
			cout << src_indices_[i] << ", " << corners_[i] << ", " << sizes_[i] << endl;
			sprintf(img_name, "/masks/%d.jpg", i);
			imwrite(debug_dir_path_ + img_name, this->final_blend_masks_[i]);

			sprintf(img_name, "/weight/%d.jpg", i);
			Mat weight_img_float = total_weight_maps_[i] * 255;
			Mat weight_img;
			weight_img_float.convertTo(weight_img, CV_8U);
			imwrite(debug_dir_path_ + img_name, weight_img);
		}
	}
	StitchFrame(src, pano);
	return 0;
}

//	����������������ļ���ʽ���£�
//	��һ�����м佹��median_focal_len_
//	֮��ÿһ����һ�����--
//		����������focal��aspect��ppx��ppy��R��t
void CosiftStitcher::saveCameraParam(string filename)
{
	ofstream cp_file(filename.c_str());
	cp_file << median_focal_len_ << endl;
	for (int i = 0; i < cameras_.size(); i++)
	{
		CameraParams cp = cameras_[i];
		cp_file << cp.focal << " " << cp.aspect << " " << cp.ppx << " " << cp.ppy;
		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				cp_file << " " << cp.R.at<float>(r, c);
		for (int r = 0; r < 3; r++)
			cp_file << " " << cp.t.at<double>(r, 0);
		cp_file << endl;
	}
	cp_file.close();
}

int CosiftStitcher::loadCameraParam(string filename)
{
	ifstream cp_file(filename.c_str());
	string line;

	//	median_focal_len_
	if (!getline(cp_file, line))
		return -1;
	stringstream mfl_string_stream;
	mfl_string_stream << line;
	mfl_string_stream >> median_focal_len_;

	//	ÿ��һ�������
	cameras_.clear();
	while (getline(cp_file, line))
	{
		stringstream cp_string_stream;
		cp_string_stream << line;
		CameraParams cp;
		cp.R.create(3, 3, CV_32F);
		cp.t.create(3, 1, CV_64F);
		cp_string_stream >> cp.focal >> cp.aspect >> cp.ppx >> cp.ppy;
		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				cp_string_stream >> cp.R.at<float>(r, c);
		for (int r = 0; r < 3; r++)
			cp_string_stream >> cp.t.at<double>(r, 0);
		cameras_.push_back(cp);
	}
	return 0;
}