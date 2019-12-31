#pragma once
#include "MatStitcher.h"

class OpencvStitcher :
	public MatStitcher
{
public:
	OpencvStitcher();
	~OpencvStitcher();

	bool init(InputArray images);

	Stitcher::Status stitch(InputArray images, OutputArray pano);

private:
	Stitcher::Status mapping(InputArray images, OutputArray pano);

	bool is_mapping;
	four_corners_t corners;
	Mat projection_matrix;
};

