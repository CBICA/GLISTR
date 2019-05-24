///////////////////////////////////////////////////////////////////////////////////////
// EvaluateQ.cpp for BrainTumorSegmentation
// Developed by Dongjin Kwon
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014 University of Pennsylvania. All rights reserved.
// See http://www.cbica.upenn.edu/sbia/software/license.html or COYPING file.
//
// Contact: SBIA Group <sbia-software at uphs.upenn.edu>
///////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
//
#undef TRACE
#define TRACE printf
//
#include "MyUtils.h"
#include "Volume.h"
//
#include "itkImageFunction.h"
#include "vnl/algo/vnl_qr.h"
#include "vnl/vnl_math.h"


#if (!defined(WIN32) && !defined(WIN64)) || !defined(_DEBUG)
char SIMULATOR_PATH[1024];
char RESAMPLE_IMAGE_PATH[1024];
char RESAMPLE_DEFORMATION_FIELD_PATH[1024];
char REVERSE_DEFORMATION_FIELD_PATH[1024];
#endif


#ifdef USE_4_IMAGES
#define USE_FAST_LIKELIHOOD
#endif

#define PI						3.1415926535897932384626433832795
#define PIM2					6.283185307179586476925286766559
#define PI2						9.8696044010893586188344909998762
#define PI2M4					39.478417604357434475337963999505
#define eps						1e-8
#define epss					1e-32
#define MaxNumberOFTumorSeeds	10


typedef vnl_matrix_fixed<double, NumberOfImageChannels, NumberOfImageChannels> VarianceType;
typedef vnl_vector_fixed<double, NumberOfImageChannels> MeanType;
typedef std::vector<VarianceType> VarianceVectorType;
typedef std::vector<MeanType> MeanVectorType;


BOOL MakeProbForTumorAndEdema(
	FVolume& priors_TU, FVolume& priors_ED, FVolume& priors_CSF, 
#if defined(USE_9_PRIORS) || defined(USE_10_PRIORS) || defined(USE_11_PRIORS)
	FVolume& priors_VT, 
#endif
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	FVolume& priors_CB, 
#endif
	FVolume& priors_GM, FVolume& priors_WM, 
#ifdef USE_WARPED_BG
	FVolume& priors_BG, 
#endif
	FVolume& tumor_density, FVolume& tumor_disp
#ifdef USE_ED_NON_WM_PROB
	, double ed_non_wm_prob
#endif
#if defined(USE_ED_TU_DISP) || defined(USE_ED_TU_DENS)
	, double ged
#endif
#if defined(USE_ED_TU_DENS_NONE_MIX)
	, BOOL sol
#endif
#if defined(USE_ED_TU_DW)
	, double gdiffwm
#endif
	);

BOOL ComputeQ(FVolume* scan_atlas_reg_images, FVolume* scan_atlas_reg_warp_prior, 
#ifdef USE_PROB_PRIOR
	FVolume* scan_atlas_reg_probs, double prob_weight, 
#endif
	const char* scan_means_file, const char* scan_variances_file, double* pScore, double* pScoreEx, int tag, char* tag_F);

BOOL LoadMeansAndVariances(const char* means_file, const char* variances_file, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector);

static
inline void ComputeLikelihood4(double m[16], double ym[4], double* like) {
	double inv[16], det, det_1, ySIy;
	
	inv[ 0] =  m[ 5] * m[10] * m[15] - m[ 5] * m[11] * m[14] - m[ 9] * m[ 6] * m[15] + m[ 9] * m[ 7] * m[14] + m[13] * m[ 6] * m[11] - m[13] * m[ 7] * m[10];
	inv[ 4] = -m[ 4] * m[10] * m[15] + m[ 4] * m[11] * m[14] + m[ 8] * m[ 6] * m[15] - m[ 8] * m[ 7] * m[14] - m[12] * m[ 6] * m[11] + m[12] * m[ 7] * m[10];
	inv[ 8] =  m[ 4] * m[ 9] * m[15] - m[ 4] * m[11] * m[13] - m[ 8] * m[ 5] * m[15] + m[ 8] * m[ 7] * m[13] + m[12] * m[ 5] * m[11] - m[12] * m[ 7] * m[ 9];
	inv[12] = -m[ 4] * m[ 9] * m[14] + m[ 4] * m[10] * m[13] + m[ 8] * m[ 5] * m[14] - m[ 8] * m[ 6] * m[13] - m[12] * m[ 5] * m[10] + m[12] * m[ 6] * m[ 9];
	inv[ 1] = -m[ 1] * m[10] * m[15] + m[ 1] * m[11] * m[14] + m[ 9] * m[ 2] * m[15] - m[ 9] * m[ 3] * m[14] - m[13] * m[ 2] * m[11] + m[13] * m[ 3] * m[10];
	inv[ 5] =  m[ 0] * m[10] * m[15] - m[ 0] * m[11] * m[14] - m[ 8] * m[ 2] * m[15] + m[ 8] * m[ 3] * m[14] + m[12] * m[ 2] * m[11] - m[12] * m[ 3] * m[10];
	inv[ 9] = -m[ 0] * m[ 9] * m[15] + m[ 0] * m[11] * m[13] + m[ 8] * m[ 1] * m[15] - m[ 8] * m[ 3] * m[13] - m[12] * m[ 1] * m[11] + m[12] * m[ 3] * m[ 9];
	inv[13] =  m[ 0] * m[ 9] * m[14] - m[ 0] * m[10] * m[13] - m[ 8] * m[ 1] * m[14] + m[ 8] * m[ 2] * m[13] + m[12] * m[ 1] * m[10] - m[12] * m[ 2] * m[ 9];
	inv[ 2] =  m[ 1] * m[ 6] * m[15] - m[ 1] * m[ 7] * m[14] - m[ 5] * m[ 2] * m[15] + m[ 5] * m[ 3] * m[14] + m[13] * m[ 2] * m[ 7] - m[13] * m[ 3] * m[ 6];   
	inv[ 6] = -m[ 0] * m[ 6] * m[15] + m[ 0] * m[ 7] * m[14] + m[ 4] * m[ 2] * m[15] - m[ 4] * m[ 3] * m[14] - m[12] * m[ 2] * m[ 7] + m[12] * m[ 3] * m[ 6];  
	inv[10] =  m[ 0] * m[ 5] * m[15] - m[ 0] * m[ 7] * m[13] - m[ 4] * m[ 1] * m[15] + m[ 4] * m[ 3] * m[13] + m[12] * m[ 1] * m[ 7] - m[12] * m[ 3] * m[ 5];   
	inv[14] = -m[ 0] * m[ 5] * m[14] + m[ 0] * m[ 6] * m[13] + m[ 4] * m[ 1] * m[14] - m[ 4] * m[ 2] * m[13] - m[12] * m[ 1] * m[ 6] + m[12] * m[ 2] * m[ 5];   
	inv[ 3] = -m[ 1] * m[ 6] * m[11] + m[ 1] * m[ 7] * m[10] + m[ 5] * m[ 2] * m[11] - m[ 5] * m[ 3] * m[10] - m[ 9] * m[ 2] * m[ 7] + m[ 9] * m[ 3] * m[ 6];  
	inv[ 7] =  m[ 0] * m[ 6] * m[11] - m[ 0] * m[ 7] * m[10] - m[ 4] * m[ 2] * m[11] + m[ 4] * m[ 3] * m[10] + m[ 8] * m[ 2] * m[ 7] - m[ 8] * m[ 3] * m[ 6];  
	inv[11] = -m[ 0] * m[ 5] * m[11] + m[ 0] * m[ 7] * m[ 9] + m[ 4] * m[ 1] * m[11] - m[ 4] * m[ 3] * m[ 9] - m[ 8] * m[ 1] * m[ 7] + m[ 8] * m[ 3] * m[ 5];   
	inv[15] =  m[ 0] * m[ 5] * m[10] - m[ 0] * m[ 6] * m[ 9] - m[ 4] * m[ 1] * m[10] + m[ 4] * m[ 2] * m[ 9] + m[ 8] * m[ 1] * m[ 6] - m[ 8] * m[ 2] * m[ 5];

#ifndef USE_SUM_EPSS
	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];  
	if (det < eps) {
		*like = 0;
		return;
	} else {
		det_1 = 1.0 / det;
	}
#else
	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12] + epss;  
	det_1 = 1.0 / det;
#endif

	ySIy = ym[0] * (inv[ 0] * ym[0] + inv[ 1] * ym[1] + inv[ 2] * ym[2] + inv[ 3] * ym[3]) +
	       ym[1] * (inv[ 4] * ym[0] + inv[ 5] * ym[1] + inv[ 6] * ym[2] + inv[ 7] * ym[3]) +
	       ym[2] * (inv[ 8] * ym[0] + inv[ 9] * ym[1] + inv[10] * ym[2] + inv[11] * ym[3]) +
	       ym[3] * (inv[12] * ym[0] + inv[13] * ym[1] + inv[14] * ym[2] + inv[15] * ym[3]);
	ySIy *= det_1;

	*like = 1.0 / (PI2M4 * vcl_sqrt(det)) * exp (-0.5 * ySIy);
}
static
inline void GetInv4(double m[16], double inv[16], double* det) {
	inv[ 0] =  m[ 5] * m[10] * m[15] - m[ 5] * m[11] * m[14] - m[ 9] * m[ 6] * m[15] + m[ 9] * m[ 7] * m[14] + m[13] * m[ 6] * m[11] - m[13] * m[ 7] * m[10];
	inv[ 4] = -m[ 4] * m[10] * m[15] + m[ 4] * m[11] * m[14] + m[ 8] * m[ 6] * m[15] - m[ 8] * m[ 7] * m[14] - m[12] * m[ 6] * m[11] + m[12] * m[ 7] * m[10];
	inv[ 8] =  m[ 4] * m[ 9] * m[15] - m[ 4] * m[11] * m[13] - m[ 8] * m[ 5] * m[15] + m[ 8] * m[ 7] * m[13] + m[12] * m[ 5] * m[11] - m[12] * m[ 7] * m[ 9];
	inv[12] = -m[ 4] * m[ 9] * m[14] + m[ 4] * m[10] * m[13] + m[ 8] * m[ 5] * m[14] - m[ 8] * m[ 6] * m[13] - m[12] * m[ 5] * m[10] + m[12] * m[ 6] * m[ 9];
	inv[ 1] = -m[ 1] * m[10] * m[15] + m[ 1] * m[11] * m[14] + m[ 9] * m[ 2] * m[15] - m[ 9] * m[ 3] * m[14] - m[13] * m[ 2] * m[11] + m[13] * m[ 3] * m[10];
	inv[ 5] =  m[ 0] * m[10] * m[15] - m[ 0] * m[11] * m[14] - m[ 8] * m[ 2] * m[15] + m[ 8] * m[ 3] * m[14] + m[12] * m[ 2] * m[11] - m[12] * m[ 3] * m[10];
	inv[ 9] = -m[ 0] * m[ 9] * m[15] + m[ 0] * m[11] * m[13] + m[ 8] * m[ 1] * m[15] - m[ 8] * m[ 3] * m[13] - m[12] * m[ 1] * m[11] + m[12] * m[ 3] * m[ 9];
	inv[13] =  m[ 0] * m[ 9] * m[14] - m[ 0] * m[10] * m[13] - m[ 8] * m[ 1] * m[14] + m[ 8] * m[ 2] * m[13] + m[12] * m[ 1] * m[10] - m[12] * m[ 2] * m[ 9];
	inv[ 2] =  m[ 1] * m[ 6] * m[15] - m[ 1] * m[ 7] * m[14] - m[ 5] * m[ 2] * m[15] + m[ 5] * m[ 3] * m[14] + m[13] * m[ 2] * m[ 7] - m[13] * m[ 3] * m[ 6];   
	inv[ 6] = -m[ 0] * m[ 6] * m[15] + m[ 0] * m[ 7] * m[14] + m[ 4] * m[ 2] * m[15] - m[ 4] * m[ 3] * m[14] - m[12] * m[ 2] * m[ 7] + m[12] * m[ 3] * m[ 6];  
	inv[10] =  m[ 0] * m[ 5] * m[15] - m[ 0] * m[ 7] * m[13] - m[ 4] * m[ 1] * m[15] + m[ 4] * m[ 3] * m[13] + m[12] * m[ 1] * m[ 7] - m[12] * m[ 3] * m[ 5];   
	inv[14] = -m[ 0] * m[ 5] * m[14] + m[ 0] * m[ 6] * m[13] + m[ 4] * m[ 1] * m[14] - m[ 4] * m[ 2] * m[13] - m[12] * m[ 1] * m[ 6] + m[12] * m[ 2] * m[ 5];   
	inv[ 3] = -m[ 1] * m[ 6] * m[11] + m[ 1] * m[ 7] * m[10] + m[ 5] * m[ 2] * m[11] - m[ 5] * m[ 3] * m[10] - m[ 9] * m[ 2] * m[ 7] + m[ 9] * m[ 3] * m[ 6];  
	inv[ 7] =  m[ 0] * m[ 6] * m[11] - m[ 0] * m[ 7] * m[10] - m[ 4] * m[ 2] * m[11] + m[ 4] * m[ 3] * m[10] + m[ 8] * m[ 2] * m[ 7] - m[ 8] * m[ 3] * m[ 6];  
	inv[11] = -m[ 0] * m[ 5] * m[11] + m[ 0] * m[ 7] * m[ 9] + m[ 4] * m[ 1] * m[11] - m[ 4] * m[ 3] * m[ 9] - m[ 8] * m[ 1] * m[ 7] + m[ 8] * m[ 3] * m[ 5];   
	inv[15] =  m[ 0] * m[ 5] * m[10] - m[ 0] * m[ 6] * m[ 9] - m[ 4] * m[ 1] * m[10] + m[ 4] * m[ 2] * m[ 9] + m[ 8] * m[ 1] * m[ 6] - m[ 8] * m[ 2] * m[ 5];

#ifndef USE_SUM_EPSS
	*det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
#else
	*det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12] + epss;
#endif
}
static
inline void ComputeLikelihood4(double inv[16], double c1, double c2, double ym[4], double* like) {
	double ySIy;

	ySIy = ym[0] * (inv[ 0] * ym[0] + inv[ 1] * ym[1] + inv[ 2] * ym[2] + inv[ 3] * ym[3]) +
	       ym[1] * (inv[ 4] * ym[0] + inv[ 5] * ym[1] + inv[ 6] * ym[2] + inv[ 7] * ym[3]) +
	       ym[2] * (inv[ 8] * ym[0] + inv[ 9] * ym[1] + inv[10] * ym[2] + inv[11] * ym[3]) +
	       ym[3] * (inv[12] * ym[0] + inv[13] * ym[1] + inv[14] * ym[2] + inv[15] * ym[3]);

	*like = c1 * exp (c2 * ySIy);
}
static
inline void ComputeLikelihood1(double m, double ym, double* like) {
	//*	
	double det;

#ifndef USE_SUM_EPSS
	det = m * m * m * m;
	if (det < eps) {
		*like = 0;
		return;
	}
#else
	det = m * m * m * m + epss;
#endif

	*like = 1.0 / (PI2M4 * vcl_sqrt(det)) * exp (-0.5 * ym * ym / m * 4);
	/*/
	if (m < eps) {
		*like = 0;
		return;
	}

	*like = 1.0 / (PI2M4 * vcl_sqrt(det)) * exp (-0.5 * ym * ym / m);
	//*/
}
#define ComputeMD(inv, ym, md)																	\
	{ md = ym[0] * (inv[ 0] * ym[0] + inv[ 1] * ym[1] + inv[ 2] * ym[2] + inv[ 3] * ym[3]) +	\
	       ym[1] * (inv[ 4] * ym[0] + inv[ 5] * ym[1] + inv[ 6] * ym[2] + inv[ 7] * ym[3]) +	\
	       ym[2] * (inv[ 8] * ym[0] + inv[ 9] * ym[1] + inv[10] * ym[2] + inv[11] * ym[3]) +	\
		   ym[3] * (inv[12] * ym[0] + inv[13] * ym[1] + inv[14] * ym[2] + inv[15] * ym[3]); }


void version()
{
	printf("==========================================================================\n");
	printf("EvaluateQ (GLISTR): Internal Procedure\n");
#ifdef SW_VER
	printf("  Version %s\n", SW_VER);
#endif
#ifdef SW_REV
	printf("  Revision %s\n", SW_REV);
#endif
	printf("Copyright (c) 2014 University of Pennsylvania. All rights reserved.\n");
	printf("See http://www.cbica.upenn.edu/sbia/software/license.html or COPYING file.\n");
	printf("==========================================================================\n");
}
void usage()
{
}

//#define TEST_NO_SIMUL
//#define TEST_NO_SIMUL_ALL
int main(int argc, char* argv[])
{
	// intput: atlas reg scan 0, atlas reg scan 2, simulator input, scales
	// output: score
    const char* scan_atlas_reg_image_list = NULL;
    const char* atlas_prior_list = NULL;
#ifdef USE_PROB_PRIOR
	const char* scan_atlas_reg_prob_list = NULL;
	double prob_weight = 0.0;
#endif
#ifdef USE_ED_NON_WM_PROB
	double ed_non_wm_prob = 0.0;
#endif
    const char* scan_means_file = NULL;
    const char* scan_variances_file = NULL;
    const char* scan_h_hdr_file = NULL;
    const char* atlas_label_map_s_img_file = NULL;
	const char* simulator_input_file = NULL;
	const char* scales_file = NULL;
	const char* out_folder = NULL;
	const char* tmp_folder = NULL;
	const char* input_file = NULL;
	const char* output_file = NULL;
	int tag = 0;
	const char* tag_F = NULL;
	BOOL sol = FALSE;
	int i;

	/////////////////////////////////////////////////////////////////////////////
	printf("argc = %d\n", argc);
	for (i = 0; i < argc; i++) {
		printf("argv[%d] = %s\n", i, argv[i]);
	}
	/////////////////////////////////////////////////////////////////////////////

#ifdef USE_PROB_PRIOR
#ifdef USE_ED_NON_WM_PROB
	if (argc != 18) {
#else
	if (argc != 17) {
#endif
#else
#ifdef USE_ED_NON_WM_PROB
	if (argc != 16) {
#else
	if (argc != 15) {
#endif
#endif
		version();
        usage();
        exit(EXIT_SUCCESS);
	}

#if (!defined(WIN32) && !defined(WIN64)) || !defined(_DEBUG)
	{
		char MODULE_PATH[1024];
		str_strip_file(argv[0], MODULE_PATH);
		//
#if defined(WIN32) || defined(WIN64)
		if (!FindExecutableInPath("ForwardSolverDiffusion.exe", MODULE_PATH, SIMULATOR_PATH)) { printf("error: ForwardSolverDiffusion.exe is not existing.\n"); exit(EXIT_FAILURE); }
		//
		sprintf(RESAMPLE_IMAGE_PATH, "%sResampleImage.exe", MODULE_PATH);
		if (!IsFileExist(RESAMPLE_IMAGE_PATH)) { TRACE("error: %s is not existing.\n", RESAMPLE_IMAGE_PATH); exit(EXIT_FAILURE); }
		sprintf(RESAMPLE_DEFORMATION_FIELD_PATH, "%sResampleDeformationField.exe", MODULE_PATH);
		if (!IsFileExist(RESAMPLE_DEFORMATION_FIELD_PATH)) { TRACE("error: %s is not existing.\n", RESAMPLE_DEFORMATION_FIELD_PATH); exit(EXIT_FAILURE); }
		sprintf(REVERSE_DEFORMATION_FIELD_PATH, "%sReverseDeformationField.exe", MODULE_PATH);
		if (!IsFileExist(REVERSE_DEFORMATION_FIELD_PATH)) { TRACE("error: %s is not existing.\n", REVERSE_DEFORMATION_FIELD_PATH); exit(EXIT_FAILURE); }
#else
		if (!FindExecutableInPath("ForwardSolverDiffusion", MODULE_PATH, SIMULATOR_PATH)) { printf("error: ForwardSolverDiffusion.exe is not existing.\n"); exit(EXIT_FAILURE); }
		//
		sprintf(RESAMPLE_IMAGE_PATH, "%sResampleImage", MODULE_PATH);
		if (!IsFileExist(RESAMPLE_IMAGE_PATH)) { TRACE("error: %s is not existing.\n", RESAMPLE_IMAGE_PATH); exit(EXIT_FAILURE); }
		sprintf(RESAMPLE_DEFORMATION_FIELD_PATH, "%sResampleDeformationField", MODULE_PATH);
		if (!IsFileExist(RESAMPLE_DEFORMATION_FIELD_PATH)) { TRACE("error: %s is not existing.\n", RESAMPLE_DEFORMATION_FIELD_PATH); exit(EXIT_FAILURE); }
		sprintf(REVERSE_DEFORMATION_FIELD_PATH, "%sReverseDeformationField", MODULE_PATH);
		if (!IsFileExist(REVERSE_DEFORMATION_FIELD_PATH)) { TRACE("error: %s is not existing.\n", REVERSE_DEFORMATION_FIELD_PATH); exit(EXIT_FAILURE); }
#endif
	}
#endif

	/////////////////////////////////////////////////////////////////////////////
	printf("SIMULATOR_PATH = %s\n", SIMULATOR_PATH);
	printf("RESAMPLE_IMAGE_PATH = %s\n", RESAMPLE_IMAGE_PATH);
	printf("RESAMPLE_DEFORMATION_FIELD_PATH = %s\n", RESAMPLE_DEFORMATION_FIELD_PATH);
	printf("REVERSE_DEFORMATION_FIELD_PATH = %s\n", REVERSE_DEFORMATION_FIELD_PATH);
	/////////////////////////////////////////////////////////////////////////////

  // input parameters provided
	i = 1;
	scan_atlas_reg_image_list = argv[i++];
	atlas_prior_list = argv[i++];
#ifdef USE_PROB_PRIOR
	scan_atlas_reg_prob_list = argv[i++];
	prob_weight = atof(argv[i++]);
#endif
#ifdef USE_ED_NON_WM_PROB
	ed_non_wm_prob = atof(argv[i++]);
#endif
	scan_means_file = argv[i++];
	scan_variances_file = argv[i++];
	scan_h_hdr_file = argv[i++];
	atlas_label_map_s_img_file = argv[i++];
	simulator_input_file = argv[i++];
	scales_file = argv[i++];
	out_folder = argv[i++];
	tmp_folder = argv[i++];
	input_file = argv[i++];
	output_file = argv[i++];
	tag = atoi(argv[i++]);
	tag_F = argv[i++];

	if (strcmp(tag_F, "SOL") == 0) {
		sol = TRUE;
	}

	char simulator_folder[1024];
	char scan_atlas_reg_image_files[NumberOfImageChannels][1024];
	char atlas_prior_files[NumberOfPriorChannels][1024];
	//char scan_atlas_reg_mass_prior_files[NumberOfPriorChannels][1024];
	//char scan_atlas_reg_mass_tu_prior_files[NumberOfPriorChannels][1024];
	//char scan_atlas_reg_warp_prior_files[NumberOfPriorChannels][1024];
#ifdef USE_PROB_PRIOR
	char scan_atlas_reg_prob_files[NumberOfPriorChannels][1024];
#endif
	//
	char tumor_density_file[1024];
	char tumor_deformation_field_hdr_file[1024];
	char parameter_cost_file[1024];
	//
	FVolume scan_atlas_reg_images[NumberOfImageChannels];
	FVolume atlas_priors[NumberOfPriorChannels];
	FVolume scan_atlas_reg_mass_priors[NumberOfPriorChannels];
	FVolume scan_atlas_reg_warp_priors[NumberOfPriorChannels];
#ifdef USE_PROB_PRIOR
	FVolume scan_atlas_reg_probs[NumberOfPriorChannels];
#endif
	//
	FVolume tumor_density;
	//
	char szCmdLine[1024];
	//
	double scales[10];
	double gxc[MaxNumberOFTumorSeeds], gyc[MaxNumberOFTumorSeeds], gzc[MaxNumberOFTumorSeeds], T[MaxNumberOFTumorSeeds];
	double gp1, gp2, grho, gdiffwm, gdiffgm;
#if defined(USE_ED_TU_DISP) || defined(USE_ED_TU_DENS)
	double ged;
#endif
	int tp_num = 1;
	
	TRACE("%d_%s: EvaluateQ...\n", tag, tag_F);

	// load files
	{
		char str_path[1024];
		char str_tmp[1024];
		FILE* fp;
		//
		sprintf(simulator_folder, "%s%s%d_%s", tmp_folder, DIR_SEP, tag, tag_F);
		//
		str_strip_file((char*)scan_atlas_reg_image_list, str_path);
		fp = fopen(scan_atlas_reg_image_list, "r");
		for (i = 0; i < NumberOfImageChannels; i++) {
			fscanf(fp, "%s", str_tmp);
			sprintf(scan_atlas_reg_image_files[i], "%s%s", str_path, str_tmp);
		}
		fclose(fp);
		//
		str_strip_file((char*)atlas_prior_list, str_path);
		fp = fopen(atlas_prior_list, "r");
		for (i = 0; i < NumberOfPriorChannels; i++) {
			fscanf(fp, "%s", str_tmp);
			sprintf(atlas_prior_files[i], "%s%s", str_path, str_tmp);
		}
		//
#ifdef USE_PROB_PRIOR
		str_strip_file((char*)scan_atlas_reg_prob_list, str_path);
		fp = fopen(scan_atlas_reg_prob_list, "r");
		for (i = 0; i < NumberOfPriorChannels; i++) {
			fscanf(fp, "%s", str_tmp);
			sprintf(scan_atlas_reg_prob_files[i], "%s%s", str_path, str_tmp);
		}
		fclose(fp);
#endif
		//
		/*
		for (i = 0; i < NumberOfPriorChannels; i++) {
			sprintf(scan_atlas_reg_mass_prior_files[i], "%s%sscan_atlas_reg_mass_prior_%d.nii.gz", simulator_folder, DIR_SEP, i);
			sprintf(scan_atlas_reg_mass_tu_prior_files[i], "%s%sscan_atlas_reg_mass_tu_prior_%d.nii.gz", simulator_folder, DIR_SEP, i);
			sprintf(scan_atlas_reg_warp_prior_files[i], "%s%sscan_atlas_reg_warp_prior_%d.nii.gz", simulator_folder, DIR_SEP, i);
		}
		strcpy(scan_atlas_reg_mass_tu_prior_files[VS], scan_atlas_reg_mass_tu_prior_files[CSF]);
		strcpy(scan_atlas_reg_mass_tu_prior_files[NCR], scan_atlas_reg_mass_tu_prior_files[TU]);
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
		strcpy(scan_atlas_reg_mass_tu_prior_files[NE], scan_atlas_reg_mass_tu_prior_files[TU]);
#endif
#ifdef USE_WARPED_BG
		strcpy(scan_atlas_reg_mass_tu_prior_files[BG], scan_atlas_reg_mass_tu_prior_files[BG]);
#else
		strcpy(scan_atlas_reg_mass_tu_prior_files[BG], atlas_prior_files[BG]);
#endif
		//*/
		//
		sprintf(tumor_density_file, "%s%stumor_density.nii.gz", simulator_folder, DIR_SEP);
		sprintf(tumor_deformation_field_hdr_file, "%s%stumor_deformation_field.mhd", simulator_folder, DIR_SEP);
		//
		sprintf(parameter_cost_file, "%s%sparameter_cost.txt", tmp_folder, DIR_SEP);
	}

	// read input file and get params
	{
		FILE* fp;
		char tmp[1024], tmp_num[1024];
		fp = fopen(input_file, "r");
		if (fp == NULL) {
			exit(EXIT_FAILURE);
		}
		fscanf(fp, "%s", tmp);
		//if (strcmp(tmp, "F") != 0) {
		//	exit(EXIT_FAILURE);
		//}
		fscanf(fp, "%s", tmp);
#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
		tp_num = (atoi(tmp) - 6) / 4;
		sprintf(tmp_num, "%d", 6 + 4*tp_num);
#else
		tp_num = (atoi(tmp) - 5) / 4;
		sprintf(tmp_num, "%d", 5 + 4*tp_num);
#endif
#else
#ifdef USE_OPTIM_ED
		tp_num = (atoi(tmp) - 5) / 4;
		sprintf(tmp_num, "%d", 5 + 4*tp_num);
#else
		tp_num = (atoi(tmp) - 4) / 4;
		sprintf(tmp_num, "%d", 4 + 4*tp_num);
#endif
#endif
		if (tp_num > MaxNumberOFTumorSeeds) {
			TRACE("scan_seeds_num = %d > %d\n", tp_num, MaxNumberOFTumorSeeds);
			exit(EXIT_FAILURE);
		}
		if (strcmp(tmp, tmp_num) != 0) {
			fclose(fp);
			exit(EXIT_FAILURE);
		}

		fscanf(fp, "%s", tmp); gp1 = atof(tmp);
		fscanf(fp, "%s", tmp); gp2 = atof(tmp);
		fscanf(fp, "%s", tmp); grho = atof(tmp);
		fscanf(fp, "%s", tmp); gdiffwm = atof(tmp);
#ifdef USE_OPTIM_DG
		fscanf(fp, "%s", tmp); gdiffgm = atof(tmp);
#endif
#ifdef USE_OPTIM_ED
		fscanf(fp, "%s", tmp); ged = atof(tmp);
#endif
		for (i = 0; i < tp_num; i++) {
			fscanf(fp, "%s", tmp); gxc[i] = atof(tmp);
			fscanf(fp, "%s", tmp); gyc[i] = atof(tmp);
			fscanf(fp, "%s", tmp); gzc[i] = atof(tmp);
			fscanf(fp, "%s", tmp); T[i]   = atof(tmp);
		}
		fclose(fp);
	}

	// read and apply scales
	{
		FILE* fp;
		char tmp[1024];
		fp = fopen(scales_file, "r");
		if (fp == NULL) {
			exit(EXIT_FAILURE);
		}
#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
		for (i = 0; i < 10; i++) {
#else
		for (i = 0; i < 9; i++) {
#endif
#else
#ifdef USE_OPTIM_ED
		for (i = 0; i < 9; i++) {
#else
		for (i = 0; i < 8; i++) {
#endif
#endif
			fscanf(fp, "%s", tmp); scales[i] = atof(tmp);
		}
		fclose(fp);
		//
		gp1 *= scales[0];
		gp2 *= scales[1];
		grho *= scales[2];
		gdiffwm *= scales[3];
#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
		gdiffgm *= scales[4];
		ged *= scales[5];
		for (i = 0; i < tp_num; i++) {
			gxc[i] *= scales[6];
			gyc[i] *= scales[7];
			gzc[i] *= scales[8];
			T[i]   *= scales[9];
		}
#else
#ifdef USE_ED_TU_DISP
		ged = 0.1;
#endif
#ifdef USE_ED_TU_DENS
		ged = 1e-8;
#endif
		gdiffgm *= scales[4];
		for (i = 0; i < tp_num; i++) {
			gxc[i] *= scales[5];
			gyc[i] *= scales[6];
			gzc[i] *= scales[7];
			T[i]   *= scales[8];
		}
#endif
#else
		// let's set dg = dw / 5
		gdiffgm = gdiffwm / 5;
#ifdef USE_OPTIM_ED
		ged *= scales[4];
		for (i = 0; i < tp_num; i++) {
			gxc[i] *= scales[5];
			gyc[i] *= scales[6];
			gzc[i] *= scales[7];
			T[i]   *= scales[8];
		}
#else
		ged = 0.1;
		for (i = 0; i < tp_num; i++) {
			gxc[i] *= scales[4];
			gyc[i] *= scales[5];
			gzc[i] *= scales[6];
			T[i]   *= scales[7];
		}
#endif
#endif
	}

#ifdef TEST_NO_SIMUL_ALL
	if (sol) {
#endif	
	// run simulator with params and get density and deform field
	{
		FVolume td[MaxNumberOFTumorSeeds];
		FVolume df_x[MaxNumberOFTumorSeeds], df_y[MaxNumberOFTumorSeeds], df_z[MaxNumberOFTumorSeeds];
		float vd_ox, vd_oy, vd_oz;
		int j, k, l;

		vd_ox = vd_oy = vd_oz = 0;

		CreateDirectory(simulator_folder, NULL);
		//
		for (l = 0; l < tp_num; l++) {
			char simulator_sub_folder[1024];
			char simulator_input_copy_file[1024];
			char tumor_density_sub_file[1024];
			char tumor_deformation_field_sub_hdr_file[1024];
			char str_tmp[1024];
			BOOL bResize = TRUE;

			sprintf(simulator_sub_folder, "%s%s%d", simulator_folder, DIR_SEP, l); 
			str_strip_path((char*)simulator_input_file, str_tmp);
			sprintf(simulator_input_copy_file, "%s%s%s", simulator_sub_folder, DIR_SEP, str_tmp);
			//
			sprintf(tumor_density_sub_file, "%s%stumor_density.nii.gz", simulator_sub_folder, DIR_SEP);
			sprintf(tumor_deformation_field_sub_hdr_file, "%s%stumor_deformation_field.mhd", simulator_sub_folder, DIR_SEP);

			CreateDirectory(simulator_sub_folder, NULL);
			SetCurrentDirectory(simulator_sub_folder);
			CopyFile(simulator_input_file, simulator_input_copy_file, FALSE);

#ifndef TEST_NO_SIMUL
			if (T[l] >= 25) {
				sprintf(szCmdLine, "%s -gfileInput %s -T %.15g -gdiffgm %.15g -gdiffwm %.15g -grho %.15g -gp2 %.15g -gp1 %.15g -gzc %.15g -gyc %.15g -gxc %.15g",
					SIMULATOR_PATH, atlas_label_map_s_img_file, T[l], gdiffgm, gdiffwm, grho, gp2, gp1, gzc[l], gyc[l], gxc[l]);
			} else {
				int ntimesteps = 5;
				int nstore = 5;
				double gcinit = 1.0;
				double gsigsq = 2.25e-5;
				//
				ntimesteps = max((int)(T[l] / 5), 1);
				nstore = ntimesteps;
				gcinit = 1.0;
				gsigsq = 2.25e-5 / 2;
				//
				sprintf(szCmdLine, "%s -gfileInput %s -T %.15g -gdiffgm %.15g -gdiffwm %.15g -grho %.15g -gp2 %.15g -gp1 %.15g -gzc %.15g -gyc %.15g -gxc %.15g -ntimesteps %d -nstore %d -gcinit %.15g -gsigsq %.15g",
					SIMULATOR_PATH, atlas_label_map_s_img_file, T[l], gdiffgm, gdiffwm, grho, gp2, gp1, gzc[l], gyc[l], gxc[l], ntimesteps, nstore, gcinit, gsigsq);
			}
			//
			TRACE("%d_%s: %s\n", tag, tag_F, szCmdLine);
			//
			if (!ExecuteProcess(szCmdLine)) {
				exit(EXIT_FAILURE);
			}
#else
			{
				double vd_dx, vd_dy, vd_dz;
				double x, y, z;
				double cx, cy, cz;
				double d2, d, p;
				double seeds_d_TU;
				// sigmoid function
				// assume y = 1 / (1 + exp(a * (x - b)))
				double seeds_a_TU;
				int vd_x, vd_y, vd_z;
				float tp_r = pow(T[l] / (0.004 * 4.0), 1.0/3.0) * 2;
				//
				FVolume tumor_deformation_field;
				//
				seeds_d_TU = tp_r / 2;
				seeds_a_TU = 0.8;

				scan_atlas_reg_images[0].load(scan_atlas_reg_image_files[0], 1);

				vd_x = scan_atlas_reg_images[0].m_vd_x;
				vd_y = scan_atlas_reg_images[0].m_vd_y;
				vd_z = scan_atlas_reg_images[0].m_vd_z;

				vd_dx = scan_atlas_reg_images[0].m_vd_dx;
				vd_dy = scan_atlas_reg_images[0].m_vd_dy;
				vd_dz = scan_atlas_reg_images[0].m_vd_dz;

				scan_atlas_reg_images[0].clear();

				tumor_density.allocate(vd_x, vd_y, vd_z);
				tumor_deformation_field.allocate(vd_x, vd_y, vd_z, 3);

				cx = gxc[l] * 1000;
				cy = gyc[l] * 1000;
				cz = gzc[l] * 1000;

				z = 0;
				for (k = 0; k < vd_z; k++) {
					y = 0;
					for (j = 0; j < vd_y; j++) {
						x = 0;
						for (i = 0; i < vd_x; i++) {
							d2 = (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz);
							d =	sqrt(d2);
							//
							p = 1.0 / (1.0 + exp(seeds_a_TU * (d - seeds_d_TU)));
							//
							tumor_density.m_pData[k][j][i][0] = p;
							//
							tumor_deformation_field.m_pData[k][j][i][0] = 0;
							tumor_deformation_field.m_pData[k][j][i][1] = 0;
							tumor_deformation_field.m_pData[k][j][i][2] = 0;
							//
							x += vd_dx;
						}
						y += vd_dy;
					}
					z += vd_dz;
				}

				if (!tumor_density.save(tumor_density_sub_file, 1)) {
					exit(EXIT_FAILURE);
				}
				if (!SaveMHDData((LPCTSTR)NULL, tumor_deformation_field_sub_hdr_file, tumor_deformation_field.m_pData, vd_x, vd_y, vd_z, 3, vd_dx, vd_dy, vd_dz, 0, 0, 0)) {
					exit(EXIT_FAILURE);
				}

				tumor_density.clear();
				tumor_deformation_field.clear();

				bResize = FALSE;
			}
#endif
			//
			if (bResize) 
			{
				// resample density and deform field
				double ratio_x = 1.0;
				double ratio_y = 1.0;
				double ratio_z = 1.0;
				char tumor_density_sub_s_file[1024];
				char tumor_deformation_field_sub_s_file[1024];
				char tumor_deformation_field_sub_s_r_file[1024];

				sprintf(tumor_density_sub_s_file, "%s%sTumorDensity.001.mhd", simulator_sub_folder, DIR_SEP);
				sprintf(tumor_deformation_field_sub_s_file, "%s%sDeformationField.001.mhd", simulator_sub_folder, DIR_SEP);
				sprintf(tumor_deformation_field_sub_s_r_file, "%s%sReverse_DeformationField.001.mhd", simulator_sub_folder, DIR_SEP);

				sprintf(szCmdLine, "%s -i %s -o %s -r %s -x %.15f -y %.15f -z %.15f", RESAMPLE_IMAGE_PATH, tumor_density_sub_s_file, tumor_density_sub_file, scan_atlas_reg_image_files[0], ratio_x, ratio_y, ratio_z);
				//
				TRACE("%d_%s: %s\n", tag, tag_F, szCmdLine);
				//
				if (!ExecuteProcess(szCmdLine)) {
					exit(EXIT_FAILURE);
				}

				if (!sol) {
					sprintf(szCmdLine, "%s -i %s -o %s -n %d -s %e", REVERSE_DEFORMATION_FIELD_PATH, tumor_deformation_field_sub_s_file, tumor_deformation_field_sub_s_r_file, 15, 1e-6);
				} else {
					sprintf(szCmdLine, "%s -i %s -o %s -n %d -s %e", REVERSE_DEFORMATION_FIELD_PATH, tumor_deformation_field_sub_s_file, tumor_deformation_field_sub_s_r_file, 30, 1e-6);
				}
				//
				TRACE("%d_%s: %s\n", tag, tag_F, szCmdLine);
				//
				if (!ExecuteProcess(szCmdLine)) {
					exit(EXIT_FAILURE);
				}

				sprintf(szCmdLine, "%s -i %s -o %s -r %s -x %.15f -y %.15f -z %.15f", RESAMPLE_DEFORMATION_FIELD_PATH, tumor_deformation_field_sub_s_r_file, tumor_deformation_field_sub_hdr_file, scan_atlas_reg_image_files[0], ratio_x, ratio_y, ratio_z);
				//
				TRACE("%d_%s: %s\n", tag, tag_F, szCmdLine);
				//
				if (!ExecuteProcess(szCmdLine)) {
					exit(EXIT_FAILURE);
				}
			}
			//
			if (!td[l].load(tumor_density_sub_file, 1)) {
				exit(EXIT_FAILURE);
			} else {
				/*
				// make peak of td as 1
				float td_max = 0, td_sc;
				for (k = 0; k < td[l].m_vd_z; k++) {
					for (j = 0; j < td[l].m_vd_y; j++) {
						for (i = 0; i < td[l].m_vd_x; i++) {
							if (td[l].m_pData[k][j][i][0] > td_max) {
								td_max = td[l].m_pData[k][j][i][0];
							}
						}
					}
				}
				td_sc = 1.0 / td_max;
				for (k = 0; k < td[l].m_vd_z; k++) {
					for (j = 0; j < td[l].m_vd_y; j++) {
						for (i = 0; i < td[l].m_vd_x; i++) {
							td[l].m_pData[k][j][i][0] *= td_sc;
						}
					}
				}
				//*/
			}
			if (!LoadMHDData((LPCTSTR)NULL, tumor_deformation_field_sub_hdr_file, &df_x[l].m_pData, &df_y[l].m_pData, &df_z[l].m_pData, df_x[l].m_vd_x, df_x[l].m_vd_y, df_x[l].m_vd_z, df_x[l].m_vd_dx, df_x[l].m_vd_dy, df_x[l].m_vd_dz, vd_ox, vd_oy, vd_oz)) {
				exit(EXIT_FAILURE);
			} else {
				df_x[l].m_vd_s = 1;
				df_y[l].m_vd_s = 1;
				df_z[l].m_vd_s = 1;
				df_y[l].m_vd_x = df_x[l].m_vd_x; df_y[l].m_vd_y = df_x[l].m_vd_y; df_y[l].m_vd_z = df_x[l].m_vd_z;
				df_z[l].m_vd_x = df_x[l].m_vd_x; df_z[l].m_vd_y = df_x[l].m_vd_y; df_z[l].m_vd_z = df_x[l].m_vd_z;
				df_y[l].m_vd_dx = df_x[l].m_vd_dx; df_y[l].m_vd_dy = df_x[l].m_vd_dy; df_y[l].m_vd_dz = df_x[l].m_vd_dz;
				df_z[l].m_vd_dx = df_x[l].m_vd_dx; df_z[l].m_vd_dy = df_x[l].m_vd_dy; df_z[l].m_vd_dz = df_x[l].m_vd_dz;
			}
		}
		// combine density and deform field
		{
			FVolume td_sum;
			FVolume df_x_sum, df_y_sum, df_z_sum;
			int vd_x, vd_y, vd_z;
			float vd_dx, vd_dy, vd_dz;

			vd_x = td[0].m_vd_x;
			vd_y = td[0].m_vd_y;
			vd_z = td[0].m_vd_z;
			vd_dx = td[0].m_vd_dx;
			vd_dy = td[0].m_vd_dy;
			vd_dz = td[0].m_vd_dz;

			td_sum.copy(td[0]); td[0].clear();
			df_x_sum.copy(df_x[0]); df_x[0].clear();
			df_y_sum.copy(df_y[0]); df_y[0].clear();
			df_z_sum.copy(df_z[0]); df_z[0].clear();
			for (l = 1; l < tp_num; l++) {
				/*
				for (k = 0; k < vd_z; k++) {
					for (j = 0; j < vd_y; j++) {
						for (i = 0; i < vd_x; i++) {
							td_sum.m_pData[k][j][i][0] += td[l].m_pData[k][j][i][0];
						}
					}
				}

				{
					FVolume df_x_tmp, df_y_tmp, df_z_tmp;

					df_x_tmp.copy(df_x_sum);
					df_y_tmp.copy(df_y_sum);
					df_z_tmp.copy(df_z_sum);

					ConcatenateFields(df_x_tmp, df_y_tmp, df_z_tmp, df_x[l], df_y[l], df_z[l], df_x_sum, df_y_sum, df_z_sum);

					df_x_tmp.clear();
					df_y_tmp.clear();
					df_z_tmp.clear();
				}
				/*/
				for (k = 0; k < vd_z; k++) {
					for (j = 0; j < vd_y; j++) {
						for (i = 0; i < vd_x; i++) {
#ifdef USE_TU_MULTI_MAX
							if (td[l].m_pData[k][j][i][0] > td_sum.m_pData[k][j][i][0]) {
								td_sum.m_pData[k][j][i][0] = td[l].m_pData[k][j][i][0];
							}
#endif
#ifdef USE_TU_MULTI_ADD
							td_sum.m_pData[k][j][i][0] += td[l].m_pData[k][j][i][0];
#endif
							df_x_sum.m_pData[k][j][i][0] += df_x[l].m_pData[k][j][i][0];
							df_y_sum.m_pData[k][j][i][0] += df_y[l].m_pData[k][j][i][0];
							df_z_sum.m_pData[k][j][i][0] += df_z[l].m_pData[k][j][i][0];
						}
					}
				}
				//*/

				td[l].clear();
				df_x[l].clear();
				df_y[l].clear();
				df_z[l].clear();
			}
			//*
			for (k = 0; k < vd_z; k++) {
				for (j = 0; j < vd_y; j++) {
					for (i = 0; i < vd_x; i++) {
						if (td_sum.m_pData[k][j][i][0] > 1) {
							td_sum.m_pData[k][j][i][0] = 1;
						}
					}
				}
			}
			//*/
			if (tp_num > 1) {
				for (k = 0; k < vd_z; k++) {
					for (j = 0; j < vd_y; j++) {
						for (i = 0; i < vd_x; i++) {
							df_x_sum.m_pData[k][j][i][0] /= tp_num;
							df_y_sum.m_pData[k][j][i][0] /= tp_num;
							df_z_sum.m_pData[k][j][i][0] /= tp_num;
						}
					}
				}
			}

			td_sum.save(tumor_density_file, 1);
			ChangeNIIHeader(tumor_density_file, scan_atlas_reg_image_files[0]);
			td_sum.clear();
			//
			if (!SaveMHDData((LPCTSTR)NULL, tumor_deformation_field_hdr_file, df_x_sum.m_pData, df_y_sum.m_pData, df_z_sum.m_pData, df_x_sum.m_vd_x, df_x_sum.m_vd_y, df_x_sum.m_vd_z, vd_dx, vd_dy, vd_dz, 0, 0, 0)) {
				exit(EXIT_FAILURE);
			}
			df_x_sum.clear();
			df_y_sum.clear();
			df_z_sum.clear();
		}
		//
		SetCurrentDirectory(tmp_folder);
	}

	// deform scan 2
	{
		int vd_x, vd_y, vd_z;
		float vd_ox, vd_oy, vd_oz;
		FVolume v3;
			
		if (!LoadMHDData(NULL, tumor_deformation_field_hdr_file, &v3.m_pData, v3.m_vd_x, v3.m_vd_y, v3.m_vd_z, v3.m_vd_s, v3.m_vd_dx, v3.m_vd_dy, v3.m_vd_dz, vd_ox, vd_oy, vd_oz)) {
			TRACE("%d_%s: Loading %s failed..\n", tag, tag_F, tumor_deformation_field_hdr_file);
		} else  {
			v3.computeDimension();
		}

		// apply mass
		atlas_priors[CSF].load(atlas_prior_files[CSF], 1);
		//
		vd_x = atlas_priors[CSF].m_vd_x;
		vd_y = atlas_priors[CSF].m_vd_y;
		vd_z = atlas_priors[CSF].m_vd_z;
		//
		scan_atlas_reg_mass_priors[CSF].allocate(vd_x, vd_y, vd_z);
		GenerateBackwardWarpVolume(scan_atlas_reg_mass_priors[CSF], atlas_priors[CSF], v3, 0.0f, false);
		atlas_priors[CSF].clear();
		//
#if defined(USE_9_PRIORS) || defined(USE_10_PRIORS) || defined(USE_11_PRIORS)
		atlas_priors[VT ].load(atlas_prior_files[VT ], 1);
		scan_atlas_reg_mass_priors[VT ].allocate(vd_x, vd_y, vd_z);
		GenerateBackwardWarpVolume(scan_atlas_reg_mass_priors[VT ], atlas_priors[VT ], v3, 0.0f, false);
		atlas_priors[VT ].clear();
#endif
		//
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
		atlas_priors[CB ].load(atlas_prior_files[CB ], 1);
		scan_atlas_reg_mass_priors[CB ].allocate(vd_x, vd_y, vd_z);
		GenerateBackwardWarpVolume(scan_atlas_reg_mass_priors[CB ], atlas_priors[CB ], v3, 0.0f, false);
		atlas_priors[CB ].clear();
#endif
		//
		atlas_priors[GM ].load(atlas_prior_files[GM ], 1);
		scan_atlas_reg_mass_priors[GM ].allocate(vd_x, vd_y, vd_z);
		GenerateBackwardWarpVolume(scan_atlas_reg_mass_priors[GM ], atlas_priors[GM ], v3, 0.0f, false);
		atlas_priors[GM ].clear();
		//
		atlas_priors[WM ].load(atlas_prior_files[WM ], 1);
		scan_atlas_reg_mass_priors[WM ].allocate(vd_x, vd_y, vd_z);
		GenerateBackwardWarpVolume(scan_atlas_reg_mass_priors[WM ], atlas_priors[WM ], v3, 0.0f, false);
		atlas_priors[WM ].clear();
		//
#ifdef USE_WARPED_BG
		atlas_priors[BG ].load(atlas_prior_files[BG ], 1);
		scan_atlas_reg_mass_priors[BG ].allocate(vd_x, vd_y, vd_z);
		GenerateBackwardWarpVolume(scan_atlas_reg_mass_priors[BG ], atlas_priors[BG ], v3, 1.0f, false);
		atlas_priors[BG ].clear();
#endif

		/*
		scan_atlas_reg_mass_priors[CSF].save(scan_atlas_reg_mass_prior_files[CSF], 1); ChangeNIIHeader(scan_atlas_reg_mass_prior_files[CSF], scan_atlas_reg_image_files[0]);
		scan_atlas_reg_mass_priors[GM ].save(scan_atlas_reg_mass_prior_files[GM ], 1); ChangeNIIHeader(scan_atlas_reg_mass_prior_files[GM ], scan_atlas_reg_image_files[0]);
		scan_atlas_reg_mass_priors[WM ].save(scan_atlas_reg_mass_prior_files[WM ], 1); ChangeNIIHeader(scan_atlas_reg_mass_prior_files[WM ], scan_atlas_reg_image_files[0]);
#ifdef USE_WARPED_BG
		scan_atlas_reg_mass_priors[BG ].save(scan_atlas_reg_mass_prior_files[BG ], 1); ChangeNIIHeader(scan_atlas_reg_mass_prior_files[BG ], scan_atlas_reg_image_files[0]);
#endif
		//*/
			
		// combine tumor and edema prior
		tumor_density.load(tumor_density_file, 1);
		//
		// estimate optimal edema range
#ifdef USE_ED_FIND_RANGE
		{
			FVolume atlas_reg_mass_prior_ED;
			FVolume atlas_reg_mass_prior_WM;
			FVolume atlas_reg_warp_prior_ED;
			FVolume atlas_reg_warp_prior_WM;
			FVolume h_v3;
			double mv_WM[NumberOfImageChannels];
			double mv_ED[NumberOfImageChannels];
#ifdef USE_ED_FIND_RANGE_MD
			double vv_WM[16], vv_inv_WM[16], vv_det_WM;
			double vv_ED[16], vv_inv_ED[16], vv_det_ED;
#endif
			int i, j, k, l, n;
			double dist_sum_min = DBL_MAX;
#ifdef USE_ED_TU_DISP
			double ged_min = 0.01;
			double ged_arr[10] = { 0.0001, 0.0005, 0.0010, 0.0050, 0.0100, 0.0250, 0.0500, 0.1000, 0.5000, 1.0000 };
			int ged_arr_num = 10;
#endif
#ifdef USE_ED_TU_DENS
			double ged_min = 1e-8;
			double ged_arr[12] = { 1e-24, 1e-22, 1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2 };
			int ged_arr_num = 12;
#endif

			MeanVectorType scan_MeanVector;
			VarianceVectorType scan_VarianceVector;
	
			if (!LoadMeansAndVariances(scan_means_file, scan_variances_file, &scan_MeanVector, &scan_VarianceVector)) {
				return FALSE;
			}
			for (i = 0; i < NumberOfImageChannels; i++) {
				mv_WM[i] = scan_MeanVector[WM](i);
				mv_ED[i] = scan_MeanVector[ED](i);
			}
#ifdef USE_ED_FIND_RANGE_MD
			for (i = 0; i < 4; i++) {
				for (j = 0; j < 4; j++) {
					vv_WM[i*4+j] = scan_VarianceVector[WM](i, j);
					vv_ED[i*4+j] = scan_VarianceVector[ED](i, j);
				}
			}
			GetInv4(vv_WM, vv_inv_WM, &vv_det_WM);
			GetInv4(vv_ED, vv_inv_ED, &vv_det_ED);
#endif

			for (i = 0; i < NumberOfImageChannels; i++) {
				scan_atlas_reg_images[i].load(scan_atlas_reg_image_files[i], 1);
			}
			atlas_reg_mass_prior_ED.allocate(vd_x, vd_y, vd_z);
			atlas_reg_mass_prior_WM.allocate(vd_x, vd_y, vd_z);
			atlas_reg_warp_prior_ED.allocate(vd_x, vd_y, vd_z);
			atlas_reg_warp_prior_WM.allocate(vd_x, vd_y, vd_z);
			if (!LoadMHDData(NULL, (char*)scan_h_hdr_file, &h_v3.m_pData, h_v3.m_vd_x, h_v3.m_vd_y, h_v3.m_vd_z, h_v3.m_vd_s, h_v3.m_vd_dx, h_v3.m_vd_dy, h_v3.m_vd_dz, vd_ox, vd_oy, vd_oz)) {
				TRACE("%d_%s: Loading %s failed..\n", tag, tag_F, scan_h_hdr_file);
			} else  {
				h_v3.computeDimension();
			}

			for (n = 0; n < ged_arr_num; n++) {
				atlas_reg_mass_prior_WM = scan_atlas_reg_mass_priors[WM];

				for (k = 0; k < vd_z; k++) {
					for (j = 0; j < vd_y; j++) {
						for (i = 0; i < vd_x; i++) {
							float c, c_1, wm;
#ifdef USE_ED_TU_DISP
							float vx, vy, vz, v2;
							vx = v3.m_pData[k][j][i][0];
							vy = v3.m_pData[k][j][i][1];
							vz = v3.m_pData[k][j][i][2];
							v2 = vx*vx + vy*vy + vz*vz;
#endif
							//
							c = tumor_density.m_pData[k][j][i][0];
							c_1 = 1.0 - c;
							//
							wm = atlas_reg_mass_prior_WM.m_pData[k][j][i][0] * c_1;
							atlas_reg_mass_prior_WM.m_pData[k][j][i][0] = wm;
							atlas_reg_mass_prior_ED.m_pData[k][j][i][0] = 0;
#ifdef USE_ED_TU_DISP
							if (v2 > ged_arr[n] && c < 1.0) {
#endif
#ifdef USE_ED_TU_DENS
							if (c > ged_arr[n] && c < 1.0) {
#endif
								/*
								atlas_reg_mass_prior_WM.m_pData[k][j][i][0] = 0.5 * wm;
								atlas_reg_mass_prior_ED.m_pData[k][j][i][0] = 0.5 * wm;
								/*/
								atlas_reg_mass_prior_WM.m_pData[k][j][i][0] = 0;
								atlas_reg_mass_prior_ED.m_pData[k][j][i][0] = wm;
								//*/
							}
						}
					}
				}

				GenerateBackwardWarpVolume(atlas_reg_warp_prior_WM, atlas_reg_mass_prior_WM, h_v3, 0.0f, false);
				GenerateBackwardWarpVolume(atlas_reg_warp_prior_ED, atlas_reg_mass_prior_ED, h_v3, 0.0f, false);

				double dist_sum;
				double dist_wm_sum = 0;
				double dist_ed_sum = 0;
				double wm_sum = 0;
				double ed_sum = 0;
				for (k = 0; k < vd_z; k++) {
					for (j = 0; j < vd_y; j++) {
						for (i = 0; i < vd_x; i++) {
							float wm, ed;
							float y[NumberOfImageChannels];
							wm = atlas_reg_warp_prior_WM.m_pData[k][j][i][0];
							ed = atlas_reg_warp_prior_ED.m_pData[k][j][i][0];
							if (wm + ed > 0) {
								double dist_wm, dist_ed;
#ifndef USE_DISCARD_ZERO_AREA
								double ys = 0;
#else
								double ys = 1;
#endif
								for (l = 0; l < NumberOfImageChannels; l++) {
									y[l] = scan_atlas_reg_images[l].m_pData[k][j][i][0];
#ifndef USE_DISCARD_ZERO_AREA
									ys += y[l];
#else
									if (y[l] <= 0) {
										ys = 0;
									}
#endif
								}
								if (ys > 0) {
									dist_wm = 0;
									dist_ed = 0;
									for (l = 0; l < NumberOfImageChannels; l++) {
#ifdef USE_ED_FIND_RANGE_MD
										double ym[4], md;
										ym[0] = y[0] - mv_WM[0];
										ym[1] = y[1] - mv_WM[1];
										ym[2] = y[2] - mv_WM[2];
										ym[3] = y[3] - mv_WM[3];
										ComputeMD(vv_inv_WM, ym, md);
										dist_wm += md;
										ym[0] = y[0] - mv_ED[0];
										ym[1] = y[1] - mv_ED[1];
										ym[2] = y[2] - mv_ED[2];
										ym[3] = y[3] - mv_ED[3];
										ComputeMD(vv_inv_ED, ym, md);
										dist_ed += md;
#else
										dist_wm += (mv_WM[l] - y[l]) * (mv_WM[l] - y[l]);
										dist_ed += (mv_ED[l] - y[l]) * (mv_ED[l] - y[l]);
#endif
									}
									dist_wm_sum += wm * dist_wm;
									dist_ed_sum += ed * dist_ed;
									wm_sum += wm;
									ed_sum += ed;
								}
							}
						}
					}
				}
				if (wm_sum != 0) {
					dist_wm_sum /= wm_sum;
				} else {
					continue;
				}
				if (ed_sum != 0) {
					dist_ed_sum /= ed_sum;
				} else {
					continue;
				}
					
				TRACE("%d_%s: ged = %g, dist_wm_avg = %g, wm_sum = %g, dist_ed_avg = %g, ed_sum = %g\n", tag, tag_F, ged_arr[n], dist_wm_sum, wm_sum, dist_ed_sum, ed_sum);

				//dist_sum = dist_wm_sum + dist_ed_sum;
				dist_sum = dist_wm_sum;

				if (dist_sum_min >= dist_sum) {
					dist_sum_min = dist_sum;
					if (n != 0) {
						ged_min = ged_arr[n-1];
					} else {
						ged_min = ged_arr[0];
					}
				}
			}

			ged = ged_min;

			atlas_reg_mass_prior_ED.clear();
			atlas_reg_mass_prior_WM.clear();
			atlas_reg_warp_prior_ED.clear();
			atlas_reg_warp_prior_WM.clear();
			h_v3.clear();
			for (i = 0; i < NumberOfImageChannels; i++) {
				scan_atlas_reg_images[i].clear();
			}
		}
#endif
		//
		scan_atlas_reg_mass_priors[TU].allocate(vd_x, vd_y, vd_z);
		scan_atlas_reg_mass_priors[ED].allocate(vd_x, vd_y, vd_z);
		//
		MakeProbForTumorAndEdema(
			scan_atlas_reg_mass_priors[TU], scan_atlas_reg_mass_priors[ED], scan_atlas_reg_mass_priors[CSF], 
#if defined(USE_9_PRIORS) || defined(USE_10_PRIORS) || defined(USE_11_PRIORS)
			scan_atlas_reg_mass_priors[VT],
#endif
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
			scan_atlas_reg_mass_priors[CB],
#endif
			scan_atlas_reg_mass_priors[GM], scan_atlas_reg_mass_priors[WM], 
#ifdef USE_WARPED_BG
			scan_atlas_reg_mass_priors[BG], 
#endif
			tumor_density, v3
#ifdef USE_ED_NON_WM_PROB
			, ed_non_wm_prob
#endif
#if defined(USE_ED_TU_DISP) || defined(USE_ED_TU_DENS)
			, ged
#endif
#if defined(USE_ED_TU_DENS_NONE_MIX)
			, sol
#endif
#if defined(USE_ED_TU_DW)
			, gdiffwm
#endif
			);
		//
		tumor_density.clear();
		v3.clear();

		/*
		scan_atlas_reg_mass_priors[CSF].save(scan_atlas_reg_mass_tu_prior_files[CSF], 1); ChangeNIIHeader(scan_atlas_reg_mass_tu_prior_files[CSF], scan_atlas_reg_image_files[0]);
#ifndef USE_8_PRIORS
		scan_atlas_reg_mass_priors[VT ].save(scan_atlas_reg_mass_tu_prior_files[VT ], 1); ChangeNIIHeader(scan_atlas_reg_mass_tu_prior_files[VT ], scan_atlas_reg_image_files[0]);
#endif
		scan_atlas_reg_mass_priors[GM ].save(scan_atlas_reg_mass_tu_prior_files[GM ], 1); ChangeNIIHeader(scan_atlas_reg_mass_tu_prior_files[GM ], scan_atlas_reg_image_files[0]);
		scan_atlas_reg_mass_priors[WM ].save(scan_atlas_reg_mass_tu_prior_files[WM ], 1); ChangeNIIHeader(scan_atlas_reg_mass_tu_prior_files[WM ], scan_atlas_reg_image_files[0]);
		scan_atlas_reg_mass_priors[TU ].save(scan_atlas_reg_mass_tu_prior_files[TU ], 1); ChangeNIIHeader(scan_atlas_reg_mass_tu_prior_files[TU ], scan_atlas_reg_image_files[0]);
		scan_atlas_reg_mass_priors[ED ].save(scan_atlas_reg_mass_tu_prior_files[ED ], 1); ChangeNIIHeader(scan_atlas_reg_mass_tu_prior_files[ED ], scan_atlas_reg_image_files[0]);
#ifdef USE_WARPED_BG
		scan_atlas_reg_mass_priors[BG ].save(scan_atlas_reg_mass_tu_prior_files[BG ], 1); ChangeNIIHeader(scan_atlas_reg_mass_tu_prior_files[BG ], scan_atlas_reg_image_files[0]);
#endif
		//*/
	}
#ifdef TEST_NO_SIMUL_ALL
	}
#endif

	// compute matching cost regarding density information
	{
		if (!sol) {
			double score_sum = 0.0;
			double score2_sum = 0.0;

#if defined(TEST_NO_SIMUL_ALL)
			MySleep(5000);
			score_sum = rand();
#else
			// calcluate posterior * log (prior * image)
			{
#if 1
				int vd_x, vd_y, vd_z;
				float vd_ox, vd_oy, vd_oz;
				FVolume v3;
				
				if (!LoadMHDData(NULL, (char*)scan_h_hdr_file, &v3.m_pData, v3.m_vd_x, v3.m_vd_y, v3.m_vd_z, v3.m_vd_s, v3.m_vd_dx, v3.m_vd_dy, v3.m_vd_dz, vd_ox, vd_oy, vd_oz)) {
					TRACE("%d_%s: Loading %s failed..\n", tag, tag_F, scan_h_hdr_file);
				} else  {
					v3.computeDimension();
				}

				vd_x = scan_atlas_reg_mass_priors[CSF].m_vd_x;
				vd_y = scan_atlas_reg_mass_priors[CSF].m_vd_y;
				vd_z = scan_atlas_reg_mass_priors[CSF].m_vd_z;

				// apply warp
				scan_atlas_reg_warp_priors[CSF].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[CSF], scan_atlas_reg_mass_priors[CSF], v3, 0.0f, false);
				scan_atlas_reg_warp_priors[VS ].copy(scan_atlas_reg_warp_priors[CSF]);
				scan_atlas_reg_mass_priors[CSF].clear();
				//
#if defined(USE_9_PRIORS) || defined(USE_10_PRIORS) || defined(USE_11_PRIORS)
				scan_atlas_reg_warp_priors[VT ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[VT ], scan_atlas_reg_mass_priors[VT ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[VT ].clear();
#endif
				//
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				scan_atlas_reg_warp_priors[CB ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[CB ], scan_atlas_reg_mass_priors[CB ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[CB ].clear();
#endif
				//
				scan_atlas_reg_warp_priors[GM ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[GM ], scan_atlas_reg_mass_priors[GM ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[GM ].clear();
				//
				scan_atlas_reg_warp_priors[WM ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[WM ], scan_atlas_reg_mass_priors[WM ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[WM ].clear();
				//
				scan_atlas_reg_warp_priors[TU ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[TU ], scan_atlas_reg_mass_priors[TU ], v3, 0.0f, false);
				scan_atlas_reg_warp_priors[NCR].copy(scan_atlas_reg_warp_priors[TU ]);
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				scan_atlas_reg_warp_priors[NE ].copy(scan_atlas_reg_warp_priors[TU ]);
#endif
				scan_atlas_reg_mass_priors[TU ].clear();
				//
				scan_atlas_reg_warp_priors[ED ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[ED ], scan_atlas_reg_mass_priors[ED ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[ED ].clear();
				//
#ifdef USE_WARPED_BG
				scan_atlas_reg_warp_priors[BG ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[BG ], scan_atlas_reg_mass_priors[BG ], v3, 1.0f, false);
				scan_atlas_reg_mass_priors[BG ].clear();
#else
				scan_atlas_reg_warp_priors[BG ].load(atlas_prior_files[BG], 1);
#endif

				v3.clear();

				for (i = 0; i < NumberOfImageChannels; i++) {
					scan_atlas_reg_images[i].load(scan_atlas_reg_image_files[i], 1);
				}
#ifdef USE_PROB_PRIOR
#ifdef PROB_LOAD_TU_ONLY
				scan_atlas_reg_probs[TU].load(scan_atlas_reg_prob_files[TU], 1);
#else
				for (i = 0; i < NumberOfPriorChannels; i++) {
					scan_atlas_reg_probs[i].load(scan_atlas_reg_prob_files[i], 1);
				}
#endif
#endif

				ComputeQ(scan_atlas_reg_images, scan_atlas_reg_warp_priors, 
#ifdef USE_PROB_PRIOR
					scan_atlas_reg_probs, prob_weight, 
#endif
					scan_means_file, scan_variances_file, &score_sum, &score2_sum, tag, (char*)tag_F);

				for (i = 0; i < NumberOfPriorChannels; i++) {
					scan_atlas_reg_warp_priors[i].clear();
				}
				for (i = 0; i < NumberOfImageChannels; i++) {
					scan_atlas_reg_images[i].clear();
				}
#ifdef USE_PROB_PRIOR
#ifdef PROB_LOAD_TU_ONLY
				scan_atlas_reg_probs[TU].clear();
#else
				for (i = 0; i < NumberOfPriorChannels; i++) {
					scan_atlas_reg_probs[i].clear();
				}
#endif
#endif
#if 0
				// apply warp
				for (i = 0; i < NumberOfPriorChannels; i++) {
					sprintf(szCmdLine, "%s -i %s -r %s -o %s -d %s", WARP_IMAGE_PATH, scan_atlas_reg_mass_tu_prior_files[i], scan_atlas_reg_image_files[0], scan_atlas_reg_warp_prior_files[i], scan_h_hdr_file);
					//
					TRACE("%d_%s: %s\n", tag, tag_F, szCmdLine);
					//
					if (!ExecuteProcess(szCmdLine)) {
						exit(EXIT_FAILURE);
					}
				}

				for (i = 0; i < NumberOfImageChannels; i++) {
					scan_atlas_reg_images[i].load(scan_atlas_reg_image_files[i], 1);
				}
				for (i = 0; i < NumberOfPriorChannels; i++) {
					scan_atlas_reg_warp_priors[i].load(scan_atlas_reg_warp_prior_files[i], 1);
				}

				ComputeQ(scan_atlas_reg_images, scan_atlas_reg_warp_priors, scan_means_file, scan_variances_file, &score_sum, tag, (char*)tag_F);
#endif
#endif
			}
#endif

			{
				while (TRUE) {
					FILE* fp;
#if defined(WIN32) || defined(WIN64)
					fp = _fsopen(parameter_cost_file, "a", _SH_DENYWR);
#else
					fp = fopen(parameter_cost_file, "a");
#endif
					if (fp == NULL) {
						MySleep(100);
						continue;
					}
					//
#if defined(USE_ED_TU_DISP) || defined(USE_ED_TU_DENS)
					fprintf(fp, "no:%04d  p1:%10g  p2:%10g  rho:%10g  dw:%15g  dg:%15g  ged:%15g", tag, gp1, gp2, grho, gdiffwm, gdiffgm, ged);
#else
					fprintf(fp, "no:%04d  p1:%10g  p2:%10g  rho:%10g  dw:%15g  dg:%15g", tag, gp1, gp2, grho, gdiffwm, gdiffgm);
#endif
					for (i = 0; i < tp_num; i++) {
						fprintf(fp, "  gxc:%8.6g  gyc:%8.6g  gzc:%8.6g  T:%8g", gxc[i], gyc[i], gzc[i], T[i]);
					}
#if defined(USE_PROB_PRIOR)
					fprintf(fp, "  cost:%9.7f  cost1:%9.7f  cost2:%9.7f\n", score_sum, score_sum-score2_sum, score2_sum);
#else
					fprintf(fp, "  cost:%9.7f\n", score_sum);
#endif
					//
					fclose(fp);
					break;
				}
			}
		
			// store output file
			{
				FILE* fp;
				fp = fopen(output_file, "w");
				fprintf (fp, "1\n");
				fprintf(fp, "%23.15e", score_sum);
				fclose(fp);
			}

			SetCurrentDirectory(tmp_folder);

			// remove working directory
			DeleteAll(simulator_folder, TRUE);
		} else {
			// copy mass prior and deformation field to the tmp_folder
			// tag is the iteration number
			char u_hdr_file[1024];
			char atlas_reg_mass_prior_files[NumberOfPriorChannels][1024];
			
			sprintf(u_hdr_file, "%s%ss_u_jsr_%d.mhd", tmp_folder, DIR_SEP, tag);
			for (i = 0; i < NumberOfPriorChannels; i++) {
				sprintf(atlas_reg_mass_prior_files[i], "%s%ss_atlas_reg_mass_prior_jsr_%d_%d.nii.gz", tmp_folder, DIR_SEP, tag, i);
			}
			//
			CopyMHDData(NULL, tumor_deformation_field_hdr_file, NULL, u_hdr_file);
			//
			scan_atlas_reg_mass_priors[CSF].save(atlas_reg_mass_prior_files[CSF], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[CSF], scan_atlas_reg_image_files[0]);
			scan_atlas_reg_mass_priors[CSF].save(atlas_reg_mass_prior_files[VS ], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[VS ], scan_atlas_reg_image_files[0]);
			scan_atlas_reg_mass_priors[CSF].clear();
			//
#if defined(USE_9_PRIORS) || defined(USE_10_PRIORS) || defined(USE_11_PRIORS)
			scan_atlas_reg_mass_priors[VT ].save(atlas_reg_mass_prior_files[VT ], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[VT ], scan_atlas_reg_image_files[0]);
			scan_atlas_reg_mass_priors[VT ].clear();
#endif
			//
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
			scan_atlas_reg_mass_priors[CB ].save(atlas_reg_mass_prior_files[CB ], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[CB ], scan_atlas_reg_image_files[0]);
			scan_atlas_reg_mass_priors[CB ].clear();
#endif
			//
			scan_atlas_reg_mass_priors[GM ].save(atlas_reg_mass_prior_files[GM ], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[GM ], scan_atlas_reg_image_files[0]);
			scan_atlas_reg_mass_priors[GM ].clear();
			//
			scan_atlas_reg_mass_priors[WM ].save(atlas_reg_mass_prior_files[WM ], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[WM ], scan_atlas_reg_image_files[0]);
			scan_atlas_reg_mass_priors[WM ].clear();
			//
			scan_atlas_reg_mass_priors[TU ].save(atlas_reg_mass_prior_files[TU ], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[TU ], scan_atlas_reg_image_files[0]);
			scan_atlas_reg_mass_priors[TU ].save(atlas_reg_mass_prior_files[NCR], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[NCR], scan_atlas_reg_image_files[0]);
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
			scan_atlas_reg_mass_priors[TU ].save(atlas_reg_mass_prior_files[NE ], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[NE ], scan_atlas_reg_image_files[0]);
#endif
			scan_atlas_reg_mass_priors[TU ].clear();
			//
			scan_atlas_reg_mass_priors[ED ].save(atlas_reg_mass_prior_files[ED ], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[ED ], scan_atlas_reg_image_files[0]);
			scan_atlas_reg_mass_priors[ED ].clear();
			//
#ifndef USE_WARPED_BG
			scan_atlas_reg_mass_priors[BG ].load(atlas_prior_files[BG], 1);
#endif
			scan_atlas_reg_mass_priors[BG ].save(atlas_reg_mass_prior_files[BG ], 1); ChangeNIIHeader(atlas_reg_mass_prior_files[BG ], scan_atlas_reg_image_files[0]);
			scan_atlas_reg_mass_priors[BG ].clear();

			{
				while (TRUE) {
					FILE* fp;
#if defined(WIN32) || defined(WIN64)
					fp = _fsopen(parameter_cost_file, "a", _SH_DENYWR);
#else
					fp = fopen(parameter_cost_file, "a");
#endif
					if (fp == NULL) {
						MySleep(100);
						continue;
					}
					//
#if defined(USE_ED_TU_DISP) || defined(USE_ED_TU_DENS)
					fprintf(fp, "no:%d_%s p1:%10g  p2:%10g  rho:%10g  dw:%15g  dg:%15g  ged:%15g", tag, tag_F, gp1, gp2, grho, gdiffwm, gdiffgm, ged);
#else
					fprintf(fp, "no:%d_%s p1:%10g  p2:%10g  rho:%10g  dw:%15g  dg:%15g", tag, tag_F, gp1, gp2, grho, gdiffwm, gdiffgm);
#endif
					for (i = 0; i < tp_num; i++) {
						fprintf(fp, "  gxc:%8.6g  gyc:%8.6g  gzc:%8.6g  T:%8g", gxc[i], gyc[i], gzc[i], T[i]);
					}
					fprintf(fp, "\n");
					//
					fclose(fp);
					break;
				}
				//
				char params_file[1024];
				sprintf(params_file, "%s%ss_params_%d.txt", tmp_folder, DIR_SEP, tag);
				while (TRUE) {
					FILE* fp;
					fp = fopen(params_file, "w");
					if (fp == NULL) {
						continue;
					}
					//
#if defined(USE_ED_TU_DISP) || defined(USE_ED_TU_DENS)
					fprintf(fp, "p1:\t%g\np2:\t%g\nrho:\t%g\ndw:\t%g\ndg:\t%g\nged:\t%g", gp1, gp2, grho, gdiffwm, gdiffgm, ged);
#else
					fprintf(fp, "p1:\t%g\np2:\t%g\nrho:\t%g\ndw:\t%g\ndg:\t%g", gp1, gp2, grho, gdiffwm, gdiffgm);
#endif
					for (i = 0; i < tp_num; i++) {
						fprintf(fp, "\nxc:\t%g\nyc:\t%g\nzc:\t%g\nT:\t%g", gxc[i], gyc[i], gzc[i], T[i]);
					}
					//
					fclose(fp);
					break;
				}
			}
			
			// remove input file
			DeleteFile(input_file);

			// remove all remaining files
			DeleteFiles((char*)tmp_folder, "input.*_F", FALSE);
			DeleteFiles((char*)tmp_folder, "output.*_F", FALSE);
			DeleteSubDirs((char*)tmp_folder, "*_F");

			SetCurrentDirectory(tmp_folder);
		}
	}

	TRACE("%d_%s: EvaluateQ... - done\n", tag, tag_F);

	exit(EXIT_SUCCESS);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// modify priors regarding tumor and edema
//#define edema_eps 1e-8
//#define edema_eps_v2 1e-3
BOOL MakeProbForTumorAndEdema(
	FVolume& priors_TU, FVolume& priors_ED, FVolume& priors_CSF, 
#if defined(USE_9_PRIORS) || defined(USE_10_PRIORS) || defined(USE_11_PRIORS)
	FVolume& priors_VT, 
#endif
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	FVolume& priors_CB, 
#endif
	FVolume& priors_GM, FVolume& priors_WM, 
#ifdef USE_WARPED_BG
	FVolume& priors_BG, 
#endif
	FVolume& tumor_density, FVolume& tumor_disp
#ifdef USE_ED_NON_WM_PROB
	, double ed_non_wm_prob
#endif
#if defined(USE_ED_TU_DISP) || defined(USE_ED_TU_DENS)
	, double ged
#endif
#if defined(USE_ED_TU_DENS_NONE_MIX)
	, BOOL sol
#endif
#if defined(USE_ED_TU_DW)
	, double gdiffwm
#endif
	)
{
	int vd_x, vd_y, vd_z;
	int i, j, k;

	vd_x = priors_TU.m_vd_x;
	vd_y = priors_TU.m_vd_y;
	vd_z = priors_TU.m_vd_z;

	for (k = 0; k < vd_z; k++) {
		for (j = 0; j < vd_y; j++) {
			for (i = 0; i < vd_x; i++) {
				float c, c_1, wm, csf, gm;
#if defined(USE_9_PRIORS) || defined(USE_10_PRIORS) || defined(USE_11_PRIORS)
				float vt;
#endif
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				float cb;
#endif
#ifdef USE_WARPED_BG
				float bg;
#endif
#if defined(USE_ED_TU_DISP) || defined(USE_ED_TU_DISP_NONE)
				float vx, vy, vz, v2;
				vx = tumor_disp.m_pData[k][j][i][0];
				vy = tumor_disp.m_pData[k][j][i][1];
				vz = tumor_disp.m_pData[k][j][i][2];
				v2 = vx*vx + vy*vy + vz*vz;
#endif
				//
				c = tumor_density.m_pData[k][j][i][0];
				c_1 = 1.0 - c;
				//
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				priors_TU.m_pData[k][j][i][0] = c / 3;
#else
				priors_TU.m_pData[k][j][i][0] = 0.5 * c;
#endif
				//
				csf = priors_CSF.m_pData[k][j][i][0] * c_1;
				priors_CSF.m_pData[k][j][i][0] = 0.5 * csf;
				//
#if defined(USE_9_PRIORS) || defined(USE_10_PRIORS) || defined(USE_11_PRIORS)
				vt = priors_VT.m_pData[k][j][i][0] * c_1;
				priors_VT.m_pData[k][j][i][0] = vt;
#endif
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				cb = priors_CB.m_pData[k][j][i][0] * c_1;
				priors_CB.m_pData[k][j][i][0] = cb;
#endif
				//
				gm = priors_GM.m_pData[k][j][i][0] * c_1;
				priors_GM.m_pData[k][j][i][0] = gm;
				//
#ifdef USE_WARPED_BG
				bg = priors_BG.m_pData[k][j][i][0] * c_1;
				priors_BG.m_pData[k][j][i][0] = bg;
#endif
				//
				wm = priors_WM.m_pData[k][j][i][0] * c_1;
				priors_WM.m_pData[k][j][i][0] = wm;
				//
				priors_ED.m_pData[k][j][i][0] = 0;
#ifdef USE_ED_TU_DISP
				if (v2 > ged && c < 1.0) {
					double wed = 0.5;
#endif
#ifdef USE_ED_TU_DENS
				if (c > ged && c < 1.0) {
					double wed = 0.5;
#endif
#ifdef USE_ED_TU_NONE
				if (c < 1.0) {
					double wed = 0.5;
#endif
#ifdef USE_ED_TU_DISP_NONE
				double wed;
				if (c < 1.0) {
					if (v2 >= ED_TU_DISP_TH) {
						wed = 0.5;
					} else {
						wed = 0.5 * v2 / ED_TU_DISP_TH;
					}
#endif
#ifdef USE_ED_TU_DENS_NONE
				double wed;
				if (c < 1.0) {
					if (c > ED_TU_DENS_TH) {
						wed = 0.5;
					} else {
						if (c > 0) {
							wed = 0.5 / pow(1 - log10(c / ED_TU_DENS_TH), 0.5);
						} else {
							wed = 0;
						}
					}
#endif
#ifdef USE_ED_TU_DENS_NONE_MIX
				double wed;
				if (c < 1.0) {
					if (c > ED_TU_DENS_TH) {
						wed = 0.5;
					} else {
						if (sol) {
							if (c > 0) {
								wed = 0.5 / pow(1 - log10(c / ED_TU_DENS_TH), 0.5);
							} else {
								wed = 0;
							}
						} else {
							wed = 0;
						}
					}
#endif
#ifdef USE_ED_TU_DW
				double wed;
				if (c < 1.0) {
					double ed_tu_dens_th = 1.0 / (gdiffwm * 1e15);
					if (c > ed_tu_dens_th) {
						wed = 0.5;
					} else {
						if (c > 0) {
							wed = 0.5 / pow(1 - log10(c / ed_tu_dens_th), 0.5);
						} else {
							wed = 0;
						}
					}
#endif
#ifdef USE_ED_NON_WM_PROB
					priors_WM.m_pData[k][j][i][0]  = (1.0 - wed) * wm;
					priors_GM.m_pData[k][j][i][0]  = (1.0 - wed * ed_non_wm_prob) * gm;
					priors_CSF.m_pData[k][j][i][0] = (1.0 - wed * ed_non_wm_prob) * 0.5 * csf;
					priors_ED.m_pData[k][j][i][0]  = wed * wm + wed * ed_non_wm_prob * gm + wed * ed_non_wm_prob * csf;
#else
					priors_WM.m_pData[k][j][i][0] = (1.0 - wed) * wm;
					priors_ED.m_pData[k][j][i][0] = wed * wm;
#endif
				}
			}
		}
	}

	return TRUE;
}

BOOL LoadMeansAndVariances(const char* means_file, const char* variances_file, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector)
{
	FILE* fp;
	int i, j, k;

	////////////////////////////////////////////////////////////////////////////////
	// initialize
	pMeanVector->clear();
	for (i = 0; i < NumberOfPriorChannels; i++) {
		MeanType v(0.0);
		pMeanVector->push_back(v); 
	}
	pVarianceVector->clear();
	for(i = 0; i < NumberOfPriorChannels; i++) {
		VarianceType s;
		s.set_identity();
		pVarianceVector->push_back(s);
	}
	////////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////////
	// read means
	fp = fopen(means_file, "r");
	if (fp == NULL) {
		TRACE("Failed to open file: '%s'\n", means_file);
		return FALSE;
	}
	for(i = 0; i < NumberOfPriorChannels; i++) {
		char class_id[1024];
		MeanType mean;
		float meanj;
		//
		fscanf(fp, "%s", class_id);
		for (j = 0; j < NumberOfImageChannels; j++) {
			fscanf(fp, "%f", &meanj);
			mean(j) = meanj;
		}
		for (j = 0; j < NumberOfPriorChannels; j++) {
#if defined(USE_8_PRIORS)
			if ((strcmp(class_id, label[j]) == 0) ||
				(strcmp(class_id, label2[j]) == 0) ||
				(strcmp(class_id, label3[j]) == 0)) {
#else
			if (strcmp(class_id, label[j]) == 0) {
#endif
				pMeanVector->at(j) = mean;
				break;
			}
		}
		if (j == NumberOfPriorChannels) {
			TRACE("class id is wrong");
			fclose(fp);
			return FALSE;
		}
	}
	fclose(fp);
	////////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////////
	// read variances
	fp = fopen(variances_file, "r");
	if (fp == NULL) {
		TRACE("Failed to open file: '%s'\n", variances_file);
		return FALSE;
	}
	for(i = 0; i < NumberOfPriorChannels; i++) {
		char class_id[1024];
		VarianceType var;
		float varjk;
		//
		fscanf(fp, "%s", class_id);
		for (j = 0; j < NumberOfImageChannels; j++) {
			for (k = 0; k < NumberOfImageChannels; k++) {
				fscanf(fp, "%f", &varjk);
				var(j, k) = varjk;
			}
		}
		for (j = 0; j < NumberOfPriorChannels; j++) {
#if defined(USE_8_PRIORS)
			if ((strcmp(class_id, label[j]) == 0) ||
				(strcmp(class_id, label2[j]) == 0) ||
				(strcmp(class_id, label3[j]) == 0)) {
#else
			if (strcmp(class_id, label[j]) == 0) {
#endif
				pVarianceVector->at(j) = var;
				break;
			}
		}
		if (j == NumberOfPriorChannels) {
			TRACE("class id is wrong");
			fclose(fp);
			return FALSE;
		}
	}
	fclose(fp);
	////////////////////////////////////////////////////////////////////////////////

	return TRUE;
}

BOOL SaveMeansAndVariances(const char* means_file, const char* variances_file, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector)
{
	FILE* fp;
	int i, j, k;

	////////////////////////////////////////////////////////////////////////////////
	// save means
	fp = fopen(means_file, "w");
	if (fp == NULL) {
		TRACE("Failed to open file: '%s'\n", means_file);
		return FALSE;
	}
	for(i = 0; i < NumberOfPriorChannels; i++) {
		MeanType mean;
		//
#if defined(USE_8_PRIORS)
		mean = pMeanVector->at(labeln[i]);
		//
		fprintf(fp, "%s\n", label[labeln[i]]);
#else
		mean = pMeanVector->at(i);
		//
		fprintf(fp, "%s\n", label[i]);
#endif
		//
		for (j = 0; j < NumberOfImageChannels; j++) {
			fprintf(fp, "%f", mean(j));
			if (j == NumberOfImageChannels-1) {
				fprintf(fp, "\n");
			} else {
				fprintf(fp, " ");
			}
		}
	}
	fclose(fp);
	////////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////////
	// save variances
	fp = fopen(variances_file, "w");
	if (fp == NULL) {
		TRACE("Failed to open file: '%s'\n", variances_file);
		return FALSE;
	}
	for(i = 0; i < NumberOfPriorChannels; i++) {
		VarianceType var;
		//
#if defined(USE_8_PRIORS)
		var = pVarianceVector->at(labeln[i]);
		//
		fprintf(fp, "%s\n", label[labeln[i]]);
#else
		var = pVarianceVector->at(i);
		//
		fprintf(fp, "%s\n", label[i]);
#endif
		//
		for (j = 0; j < NumberOfImageChannels; j++) {
			for (k = 0; k < NumberOfImageChannels; k++) {
				fprintf(fp, "%f", var(j, k));
				if (k == NumberOfImageChannels-1) {
					fprintf(fp, "\n");
				} else {
					fprintf(fp, " ");
				}
			}
		}
	}
	fclose(fp);
	////////////////////////////////////////////////////////////////////////////////

	return TRUE;
}

BOOL ComputeQ(FVolume* scan_atlas_reg_images, FVolume* scan_atlas_reg_warp_prior, 
#ifdef USE_PROB_PRIOR
	FVolume* scan_atlas_reg_probs, double prob_weight, 
#endif
	const char* scan_means_file, const char* scan_variances_file, double* pScore, double* pScoreEx, int tag, char* tag_F)
{
	MeanVectorType scan_MeanVector;
	VarianceVectorType scan_VarianceVector;
	int vd_x, vd_y, vd_z;
	int i, j, k, l, m, n;
    double cost = 0.0;
    double number_of_pixels = 0.0;
#ifdef USE_PROB_PRIOR
	double cost_prior = 0.0;
#endif
#ifdef USE_FAST_LIKELIHOOD
	double vv[NumberOfPriorChannels][16];
	double vv_inv[NumberOfPriorChannels][16];
	double vv_det[NumberOfPriorChannels], vv_c1[NumberOfPriorChannels], vv_c2[NumberOfPriorChannels];
	double mv[NumberOfPriorChannels][4];

	// assume 4 channels for image
	if (NumberOfImageChannels != 4) {
		return FALSE;
	}
#endif

	TRACE("%d_%s: ComputeQ...\n", tag, tag_F);

	vd_x = scan_atlas_reg_images[0].m_vd_x;
	vd_y = scan_atlas_reg_images[0].m_vd_y;
	vd_z = scan_atlas_reg_images[0].m_vd_z;

	if (!LoadMeansAndVariances(scan_means_file, scan_variances_file, &scan_MeanVector, &scan_VarianceVector)) {
		return FALSE;
	}

#ifdef USE_COMPUTE_Q_ESTIMATE_VARIANCES
	// Update means and variances
	//if (scan_VarianceVector.at(WM)(0,1) == 0 && scan_VarianceVector.at(WM)(0,2) == 0 && scan_VarianceVector.at(WM)(0,3) == 0 &&
	//	scan_VarianceVector.at(WM)(1,2) == 0 && scan_VarianceVector.at(WM)(1,3) == 0 && scan_VarianceVector.at(WM)(2,3) == 0)
	{
		TRACE("%d_%s: Estimate means and variances...\n", tag, tag_F);

		double mv_sum[NumberOfPriorChannels][4];
		double vv_sum[NumberOfPriorChannels][16];
		double w_sum[NumberOfPriorChannels];

		for (k = 0; k < NumberOfPriorChannels; k++) {
			w_sum[k] = eps;
			for (j = 0; j < 4; j++) {
				mv_sum[k][j] = eps;
				for (i = 0; i < 4; i++) {
					if (i == j) {
						vv_sum[k][4*j+i] = eps;
					} else {
						vv_sum[k][4*j+i] = 0;
					}
				}
			}
		}

		for (n = 0; n < vd_z; n++) {
			for (m = 0; m < vd_y; m++) {
				for (l = 0; l < vd_x; l++) {
					double y[4];
#ifndef USE_DISCARD_ZERO_AREA
					double ys = 0;
#else
					double ys = 1;
#endif
					//
					for (i = 0; i < 4; i++) {
						y[i] = scan_atlas_reg_images[i].m_pData[n][m][l][0];
#ifndef USE_DISCARD_ZERO_AREA
						ys += y[i];
#else
						if (y[i] <= 0) {
							ys = 0;
						}
#endif
					}
					if (ys > 0) {
						for (k = 0; k < NumberOfPriorChannels; k++) {
							double p = (double)scan_atlas_reg_warp_prior[k].m_pData[n][m][l][0];
							for (i = 0; i < 4; i++) {
								mv_sum[k][i] += p * y[i];
							}
							w_sum[k] += p;
						}
					}
				}
			}
		}

		for (k = 0; k < NumberOfPriorChannels; k++) {
			double w_1 = 1.0 / (w_sum[k] + eps);
			for (i = 0; i < 4; i++) {
				mv[k][i] = w_1 * mv_sum[k][i] + eps;
				//
				//scan_MeanVector.at(k)(i) = mv[k][i];
				mv[k][i] = scan_MeanVector[k](i);
			}
		}

		for (n = 0; n < vd_z; n++) {
			for (m = 0; m < vd_y; m++) {
				for (l = 0; l < vd_x; l++) {
					double y[4];
					double ym[4];
#ifndef USE_DISCARD_ZERO_AREA
					double ys = 0;
#else
					double ys = 1;
#endif
					//
					for (i = 0; i < 4; i++) {
						y[i] = scan_atlas_reg_images[i].m_pData[n][m][l][0];
#ifndef USE_DISCARD_ZERO_AREA
						ys += y[i];
#else
						if (y[i] <= 0) {
							ys = 0;
						}
#endif
					}
					if (ys > 0) {
						for (k = 0; k < NumberOfPriorChannels; k++) {
							double p = (double)scan_atlas_reg_warp_prior[k].m_pData[n][m][l][0];
							for (i = 0; i < 4; i++) {
								ym[i] = y[i] - mv[k][i];
							}
							for (j = 0; j < 4; j++) {
								for (i = 0; i < 4; i++) {
									vv_sum[k][j*4+i] += p * ym[j] * ym[i];
								}
							}
						}
					}
				}
			}
		}

		// computing updated variances to the number of classes
		for (k = 0; k < NumberOfPriorChannels; k++) {
			double w_1 = 1.0 / (w_sum[k] + eps);
			for (j = 0; j < 4; j++) {
				for (i = 0; i < 4; i++) {
					vv[k][j*4+i] = w_1 * vv_sum[k][j*4+i];
					scan_VarianceVector[k](j, i) = vv[k][j*4+i];
				}
			}
		}

		/*
		{
			char means_file[1024];
			char variances_file[1024];
			sprintf(means_file, "s_means_%d_%s.txt", tag, tag_F);
			sprintf(variances_file, "s_variances_%d_%s.txt", tag, tag_F);

			if (!SaveMeansAndVariances(means_file, variances_file, &scan_MeanVector, &scan_VarianceVector)) {
				return FALSE;
			}
		}
		//*/

		for (i = 0; i < NumberOfPriorChannels; i++) {
			GetInv4(vv[i], vv_inv[i], &vv_det[i]);
			vv_c1[i] = 1.0 / (PI2M4 * vcl_sqrt(vv_det[i]));
#ifndef USE_SUM_EPSS
			if (vv_det[i] < eps) {
				vv_c2[i] = 0;
			} else {
				vv_c2[i] = -0.5 / vv_det[i];
			}
#else
			vv_c2[i] = -0.5 / vv_det[i];
#endif
		}
	}
	/*
	else {
		TRACE("%d_%s: Load means and variances...\n", tag, tag_F);
#ifdef USE_FAST_LIKELIHOOD
		for (i = 0; i < NumberOfPriorChannels; i++) {
			for (j = 0; j < 4; j++) {
				for (k = 0; k < 4; k++) {
					vv[i][j*4+k] = scan_VarianceVector.at(i)(j, k);
				}
				mv[i][j] = scan_MeanVector.at(i)(j);
			}
		}
		for (i = 0; i < NumberOfPriorChannels; i++) {
			GetInv4(vv[i], vv_inv[i], &vv_det[i]);
			vv_c1[i] = 1.0 / (PI2M4 * vcl_sqrt(vv_det[i]));
#ifndef USE_SUM_EPSS
			if (vv_det[i] < eps) {
				vv_c2[i] = 0;
			} else {
				vv_c2[i] = -0.5 / vv_det[i];
			}
#else
			vv_c2[i] = -0.5 / vv_det[i];
#endif
		}
#endif
	}
	//*/
#else
#ifdef USE_FAST_LIKELIHOOD
	for (i = 0; i < NumberOfPriorChannels; i++) {
		for (j = 0; j < 4; j++) {
			for (k = 0; k < 4; k++) {
				vv[i][j*4+k] = scan_VarianceVector[i](j, k);
			}
			mv[i][j] = scan_MeanVector[i](j);
		}
	}
	for (i = 0; i < NumberOfPriorChannels; i++) {
		GetInv4(vv[i], vv_inv[i], &vv_det[i]);
		vv_c1[i] = 1.0 / (PI2M4 * vcl_sqrt(vv_det[i]));
#ifndef USE_SUM_EPSS
		if (vv_det[i] < eps) {
			vv_c2[i] = 0;
		} else {
			vv_c2[i] = -0.5 / vv_det[i];
		}
#else
		vv_c2[i] = -0.5 / vv_det[i];
#endif
	}
#endif
#endif
	//
#ifdef USE_LIKE_TU_MAX
	bool bValidNCR = true;
	bool bValidNE = true;
	if (mv[TU][0] == mv[NCR][0] && mv[TU][1] == mv[NCR][1] && mv[TU][2] == mv[NCR][2] && mv[TU][3] == mv[NCR][3]) {
		bValidNCR = false;
	}
	if (mv[NE][0] == mv[NCR][0] && mv[NE][1] == mv[NCR][1] && mv[NE][2] == mv[NCR][2] && mv[NE][3] == mv[NCR][3]) {
		bValidNE = false;
	}
#endif

	for (n = 0; n < vd_z; n++) {
		for (m = 0; m < vd_y; m++) {
			for (l = 0; l < vd_x; l++) {
				double prior[NumberOfPriorChannels];
				double post[NumberOfPriorChannels];
				double like[NumberOfPriorChannels];
#ifdef USE_PROB_PRIOR
				double prob[NumberOfPriorChannels];
#endif
#ifdef USE_FAST_LIKELIHOOD
				double y[4], ym[4];
#else
				MeanType y;
#endif
#ifndef USE_DISCARD_ZERO_AREA
				double ys = 0;
#else
				double ys = 1;
#endif

				number_of_pixels++;

				for (i = 0; i < NumberOfPriorChannels; i++) {
					//scan_atlas_reg_warp_prior[i].GetAt(l, m, n, &prior[i]);
					prior[i] = scan_atlas_reg_warp_prior[i].m_pData[n][m][l][0];
#if defined(USE_PROB_PRIOR) && !defined(PROB_LOAD_TU_ONLY)
					prob[i] = scan_atlas_reg_probs[i].m_pData[n][m][l][0];
#endif
				}
#if defined(USE_PROB_PRIOR) && defined(PROB_LOAD_TU_ONLY)
				prob[TU] = scan_atlas_reg_probs[TU].m_pData[n][m][l][0];
#endif
				//
				// making a vnl vector from fixed images
				for (i = 0; i < NumberOfImageChannels; i++) {
#ifdef USE_FAST_LIKELIHOOD
					y[i] = scan_atlas_reg_images[i].m_pData[n][m][l][0];
#ifndef USE_DISCARD_ZERO_AREA
					ys += y[i];
#else
					if (y[i] <= 0) {
						ys = 0;
						break;
					}
#endif
#else
					//float dv;
					//scan_atlas_reg_images[i].GetAt(l, m, n, &dv);
					//y(i) = dv;
					y(i) = scan_atlas_reg_images[i].m_pData[n][m][l][0];
#ifndef USE_DISCARD_ZERO_AREA
					ys += y(i);
#else
					if (y(i) <= 0) {
						ys = 0;
					}
#endif
#endif
				}

				// if foreground...
				// consider also if tumor and ncr priors have resonable value
#ifndef USE_COST_BG
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				if (ys > 0 || (prior[TU]+prior[NCR]+prior[NE]) > 0.01) {
#else
				if (ys > 0 || (prior[TU]+prior[NCR]) > 0.01) {
#endif
#else
				if (1) {
#endif
					// to the number of classes
					for (i = 0; i < NumberOfPriorChannels; i++) {
#ifdef USE_FAST_LIKELIHOOD
						ym[0] = y[0] - mv[i][0]; 
						ym[1] = y[1] - mv[i][1]; 
						ym[2] = y[2] - mv[i][2]; 
						ym[3] = y[3] - mv[i][3];
#ifdef USE_LIKE_TU_T1CE
						if (i == TU) {
							ComputeLikelihood1(vv[i][5], ym[1], &like[i]);
						} else {
							//ComputeLikelihood4(vv[i], ym, &like[i]);
							ComputeLikelihood4(vv_inv[i], c1[i], c2[i], ym, &like[i]);
						}
#else
						//ComputeLikelihood4(vv[i], ym, &like[i]);
						ComputeLikelihood4(vv_inv[i], vv_c1[i], vv_c2[i], ym, &like[i]);
#endif
#else
						MeanType SIy = vnl_qr<double>(scan_VarianceVector.at(i)).solve(y - scan_MeanVector.at(i));
						double ySIy = dot_product(y - scan_MeanVector.at(i), SIy);
						double detS = vnl_determinant(scan_VarianceVector.at(i)) + epss;
						like[i] = 1.0 / (PI2M4 * vcl_sqrt(detS)) * exp(-0.5 * ySIy);
#endif
					}
#ifndef USE_LIKE_TU_MAX
					// compute the denum
#ifndef USE_SUM_EPSS2
					double denum = 0;
#else
					double denum = epss;
#endif
					for (i = 0; i < NumberOfPriorChannels; i++) {
						denum +=  prior[i] * like[i];
					}
#ifndef USE_SUM_EPSS2
					if (denum < eps) {
						denum = eps;
					}
#endif
					// compute the posterior
					for (i = 0; i < NumberOfPriorChannels; i++) {
						post[i] = prior[i] * like[i] / denum;
					}
#else
					// compute the denum
#ifndef USE_SUM_EPSS2
					double denum = 0.0;
#else
					double denum = epss;
#endif
					double like_tu_max = max(like[TU], max(like[NCR], like[NE]));
					double post_tu_max, denum_tu;
					for (i = 0; i < NumberOfPriorChannels; i++) {
						if (i != TU && i != NCR && i != NE) {
							denum += prior[i] * like[i];
						}
					}
					denum += prior[TU] * 3 * like_tu_max;
					// compute the posterior
					if (denum != 0) {
						for (i = 0; i < NumberOfPriorChannels; i++) {
							if (i != TU && i != NCR && i != NE) {
								post[i] = prior[i] * like[i] / denum;
							}
						}
						post_tu_max = prior[TU] * 3 * like_tu_max / denum;
						//
						denum_tu = like[TU];
						if (bValidNCR) {
							denum_tu += like[NCR];
						}
						if (bValidNE) {
							denum_tu += like[NE];
						}
						if (denum_tu != 0) {
							post[TU] = post_tu_max * like[TU] / denum_tu;
							if (bValidNCR) {
								post[NCR] = post_tu_max * like[NCR] / denum_tu;
							} else {
								post[NCR] = post[TU];
							}
							if (bValidNE) {
								post[NE] = post_tu_max * like[NE] / denum_tu;
							} else {
								post[NE] = post[NCR];
							}
						} else {
							post[TU] = 0;
							post[NCR] = 0;
							post[NE] = 0;
						}
					} else {
						for (i = 0; i < NumberOfPriorChannels; i++) {
							post[i] = 0.0;
						}
					}
#endif
					//
				// if background...
				} else {
					// compute the posterior
					for (i = 0; i < NumberOfPriorChannels; i++) {
						post[i] = 0.0;
						like[i] = 1.0;
					}
#ifdef USE_WARPED_BG
					post[BG] = 1.0;
#endif
				}

				// to the number of classes
				for (i = 0; i < NumberOfPriorChannels; i++) {
					// we have to take care of negative values
					double val = like[i] * prior[i];
#ifndef USE_SUM_EPSS2
					if (val < epss) val = epss;
#else
					if (val <= 0.0) val = epss;
#endif
					cost -= post[i] * log(val);
				}
#ifdef USE_PROB_PRIOR
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				cost_prior += prob_weight * 9 * (prior[TU] - prob[TU]) * (prior[TU] - prob[TU]);
#else
				// multiplying 4 is considering tu_prior = prior[TU] + prior[NCR]
				cost_prior += prob_weight * 4 * (prior[TU] - prob[TU]) * (prior[TU] - prob[TU]);
				//cost_prior += prob_weight * (post[TU] + post[NCR]) * ((prior[TU] - prob[TU]) * (prior[TU] - prob[TU]));
#endif
#endif
			} // l
		} // m
	} // n

	cost /= number_of_pixels;
#ifdef USE_PROB_PRIOR
	cost_prior /= number_of_pixels;
#endif

#ifdef USE_PROB_PRIOR
	*pScore = cost + cost_prior;
	*pScoreEx = cost_prior;
	TRACE("%d_%s: cost = %f, cost_prior = %f, sum = %f, number_of_pixels = %f\n", tag, tag_F, cost, cost_prior, *pScore, number_of_pixels);
#else
	*pScore = cost;
	*pScoreEx = 0;
	TRACE("%d_%s: cost = %f, number_of_pixels = %f\n", tag, tag_F, *pScore, number_of_pixels);
#endif

	TRACE("%d_%s: ComputeQ... - done\n", tag, tag_F);
	
	return TRUE;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
