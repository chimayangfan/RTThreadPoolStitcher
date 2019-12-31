#pragma once
#include "MatStitcher.h"

//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//OPENCV
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include <stdio.h>
#include <malloc.h>

using namespace std;
using namespace cv;
using namespace cv::detail;

typedef struct _cpu2gpu_init_data
{
	int height;
	int width;

	int warped_height;
	int warped_width;

	int corner_x;
	int corner_y;

	float *xmap;
	float *ymap;

	float *blend_weight;
	float *ec_weight;
	float *total_weight;
}
C2GInitData;

typedef struct _gpu_image_data
{
	unsigned char *data;
}
GPUImageData;

typedef struct Point
{
	int x;
	int y;
}
Point;

typedef struct _image_size
{
	int height;
	int width;
}
ImageSize;

typedef struct _image_xy_map
{
	float *xmap;
	float *ymap;
}
ImageXYMap;

typedef struct _image_weight
{
	float *blend_weight;
	float *ec_weight;
	float *total_weight;
}
ImageWeight;

typedef struct _const_data
{
	int height;
	int width;

	int warped_height;
	int warped_width;

	int corner_x;
	int corner_y;
}
ConstDataGPU;

typedef float* float_ptr;

class XYMap
{
public:
	vector<Mat> xmaps;
	vector<Mat> ymaps;
	int index;
};

/*
* 使用的是BlocksGainCompensator
*/
//definition of Class MyExposureCompensator
class MyExposureCompensator
{
public:
	MyExposureCompensator(int bl_width = 32, int bl_height = 32)
		: bl_width_(bl_width), bl_height_(bl_height) {}

	void createWeightMaps(const vector<cv::Point> &corners, const vector<Mat> &images,
		const vector<Mat> &masks, vector<Mat_<float>> &ec_maps);

	void createWeightMaps(const vector<cv::Point> &corners, const vector<Mat> &images,
		const vector<pair<Mat, uchar>> &masks, vector<Mat_<float>> &ec_maps);

	void feed(const vector<cv::Point> &corners, const vector<Mat> &images, vector<Mat> &masks);

	void gainMapResize(vector<Size> sizes_, vector<Mat_<float>> &ec_maps);

	void apply(int index, Mat &image);

private:
	int bl_width_, bl_height_;
	vector<Mat_<float> > ec_maps_;
};


//definition of Class MyFeatherBlender
class MyFeatherBlender
{
public:
	void setImageNum(int image_num)
	{
		weight_maps_.resize(image_num);
	}

	float sharpness() const { return m_sharpness_; }

	void setSharpness(float val) { m_sharpness_ = val; }

	void createWeightMaps(Rect dst_roi, vector<cv::Point> corners, vector<Mat> &masks, vector<Mat> &weight_maps);

	void prepare(Rect dst_roi, vector<cv::Point> corners, vector<Mat> &masks);

	void clear()
	{
		dst_.setTo(Scalar::all(0));
	}

	void feed(const Mat &img, const Mat &mask, cv::Point tl, int img_idx);

	void blend(Mat &dst, Mat &dst_mask);

private:
	int m_image_num;
	float m_sharpness_;
	vector<Mat> weight_maps_;
	Mat dst_weight_map_;

protected:
	Mat dst_, dst_mask_;
	Rect dst_roi_;
};

class CosiftStitcher :
	public MatStitcher
{
public:
	//CosiftStitcher(); 
	CosiftStitcher(InputArray images);//有参构造
	~CosiftStitcher();
	cudaError_t DevMalloc(int num_images);
	int DevDataUpload(C2GInitData *c2g_data, int pano_height, int pano_width);
	int Cuda_Stitch(GPUImageData *images, unsigned char *dst);
	Stitcher::Status stitch(InputArray images, OutputArray pano);
	int DevFree();

	void setPreview(bool is_preview) { is_preview_ = is_preview; };
	void setSave(bool is_save) { is_save_video_ = is_save; };
	void setRange(int start, int end = -1) { start_frame_index_ = std::max(1, start) - 1; end_frame_index_ = end; };
	void setTryGPU(bool try_gpu) { is_try_gpu_ = try_gpu; };
	void setTrim(bool is_trim) {
		if (is_trim)
			trim_type_ = CosiftStitcher::TRIM_AUTO;
		else
			trim_type_ = CosiftStitcher::TRIM_NO;
	};
	void setTrim(Rect trim_rect) { trim_rect_ = trim_rect; trim_type_ = CosiftStitcher::TRIM_RECTANGLE; };
	void setWarpType(string warp_type) { warp_type_ = warp_type; };

	//int stitch(vector<VideoCapture> &captures, string &writer_file_name);
	int stitchImage(vector<Mat> &src, Mat &pano);

	void setDebugDirPath(string dir_path);

	void saveCameraParam(string filename);
	int loadCameraParam(string filename);
private:
	Stitcher::Status mapping(InputArray images, OutputArray pano);

	bool is_mapping;
	four_corners_t corners;
	Mat projection_matrix;
	
	int Prepare(vector<Mat> &src, const char* warp_type_ = NULL);
	cudaError_t PrepareAPAP(vector<Mat> &src);
	cudaError_t PrepareClassical(vector<Mat> &src);
	int StitchFrame(vector<Mat> &src, Mat &dst);
	int StitchFrameCPU(vector<Mat> &src, Mat &dst);
	int StitchFrameGPU(vector<Mat> &src, Mat &dst);

	void InitMembers(int num_images);

	/*
	* 计算一些放缩的尺度，在特征检测和计算接缝的时候，为了提高程序效率，可以对源图像进行一些放缩
	*/
	void SetScales(vector<Mat> &src);

	int FindFeatures(vector<Mat> &src, vector<ImageFeatures> &features);

	/*
	* 特征匹配，然后去除噪声图片。本代码实现时，一旦出现噪声图片，就终止算法
	* 返回值：
	*		0	——	正常
	*		-2	——	存在噪声图片
	*/
	int MatchImages(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches);

	/*
	* 摄像机标定
	*/
	int CalibrateCameras(vector<ImageFeatures> &features,
		vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras);

	/*
	*	计算水平视角
	*/
	double GetViewAngle(vector<Mat> &src, vector<CameraParams> &cameras);


	/*
	* 为接缝的计算做Warp
	*/
	int WarpForSeam(vector<Mat> &src, vector<CameraParams> &cameras,
		vector<Mat> &masks_warped, vector<Mat> &images_warped);

	/*
	* 计算接缝
	*/
	int FindSeam(vector<Mat> &images_warped, vector<Mat> &masks_warped);

	/*
	*	把摄像机参数和masks还原到正常大小
	*/
	int Rescale(vector<Mat> &src, vector<CameraParams> &cameras, vector<Mat> &seam_masks);

	int RegistEvaluation(vector<ImageFeatures> &features,
		vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras);

	/*
	*	解决360°拼接问题。对于横跨360°接缝的图片，找到最宽的inpaint区域[x1, x2]
	*/
	int FindWidestInpaintRange(Mat mask, int &x1, int &x2);

	/*
	* 裁剪掉inpaint区域
	*/
	int TrimRect(Rect rect);
	int TrimInpaint(vector<Mat> &src);
	bool IsRowCrossInpaint(uchar *row, int width);

	/* 裁剪类型 */
	enum { TRIM_NO, TRIM_AUTO, TRIM_RECTANGLE };

	/* 参数 */
	bool is_preview_;
	bool is_save_video_;
	int start_frame_index_, end_frame_index_;
	bool is_try_gpu_;
	bool is_debug_;
	int trim_type_;
	Rect trim_rect_;

	double work_megapix_;
	double seam_megapix_;
	float conf_thresh_;
	string features_type_;
	string ba_cost_func_;
	string ba_refine_mask_;
	bool is_do_wave_correct_;
	WaveCorrectKind wave_correct_;
	bool is_save_graph_;
	string save_graph_to_;
	string warp_type_;
	int expos_comp_type_;
	float match_conf_;
	string seam_find_type_;
	int blend_type_;
	float blend_strength_;

	Ptr<WarperCreator> warper_creator_;
	double work_scale_, seam_scale_;
	double median_focal_len_;

	/* 第一帧计算出的参数，不用重复计算 */
	vector<CameraParams> cameras_;
	vector<int> src_indices_;
	vector<cv::Point> corners_;
	vector<Size> sizes_;
	Rect dst_roi_;
	vector<Mat> final_warped_masks_;	//warp的mask
	vector<Mat> xmaps_;
	vector<Mat> ymaps_;
	vector<Mat_<float>> ec_weight_maps_;		//曝光补偿
	vector<Mat> blend_weight_maps_;
	vector<Mat_<float>> total_weight_maps_;
	vector<Mat> final_blend_masks_;	//blend_mask = seam_mask & warp_mask
	double view_angle_;

	MyExposureCompensator compensator_;
	MyFeatherBlender blender_;

	/* 缓存 */
	vector<Mat> final_warped_images_;

	int cur_frame_idx_;
	int parallel_num_;
	bool is_prepared_;

	/* Debug */
	string debug_dir_path_;

};

typedef struct frameInfo_
{
	vector<Mat> src;
	Mat dst;
	int frame_idx;
	int stitch_status;
}FrameInfo;