#pragma once
#include "MatStitcher.h"

class SurfStitcher :
	public MatStitcher
{
public:
	SurfStitcher();
	~SurfStitcher();
	Stitcher::Status stitch(InputArray images, OutputArray pano);
private:
	Stitcher::Status mapping(InputArray images, OutputArray pano);

	bool is_mapping;
	four_corners_t corners;
	Mat projection_matrix;
};


