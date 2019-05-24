///////////////////////////////////////////////////////////////////////////////////////
// BrainTumorSegmentation.cpp
// Developed by Dongjin Kwon
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014 University of Pennsylvania. All rights reserved.
// See http://www.cbica.upenn.edu/sbia/software/license.html or COYPING file.
//
// Contact: SBIA Group <sbia-software at uphs.upenn.edu>
///////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "MyUtils.h"
#include "Volume.h"
//
#include "itkImageFunction.h"
#include "vnl/algo/vnl_qr.h"
#include "vnl/vnl_math.h"
//
#if defined(WIN32) || defined(WIN64)
#include <io.h>
#else
#define _dup dup
#define _dup2 dup2
#endif
//
#include <omp.h>
//
#ifdef USE_JSR_GLISTR
#include <itkImageFileReader.h>
#include "itkMultiResolutionJointSegmentationRegistration.h"
#include "itkJointSegmentationRegistrationFunction.h"
#endif

#include <numeric> // for std::accumulate; mean calculation
#include <cmath>
#include <time.h>


#if (!defined(WIN32) && !defined(WIN64)) || !defined(_DEBUG)
char EVALUATE_Q_PATH[1024];
char HOPSPACK_PATH[1024];
char FLIRT_PATH[1024];
char CONVERT_XFM_PATH[1024];
char SIMULATOR_PATH[1024];
char RESAMPLE_IMAGE_PATH[1024];
char RESAMPLE_DEFORMATION_FIELD_PATH[1024];
char REVERSE_DEFORMATION_FIELD_PATH[1024];
char WARP_IMAGE_PATH[1024];
char CONCATENATE_FIELD_PATH[1024];
#endif


#define PI						3.1415926535897932384626433832795
#define PIM2					6.283185307179586476925286766559
#define PI2						9.8696044010893586188344909998762
#define PI2M4					39.478417604357434475337963999505
#define eps						1e-8
#define epss					1e-32
#define MaxNumberOFTumorSeeds	10
#define MaxNumberOFPoints		300 // initialization points from BrainTumorViewer

typedef vnl_matrix_fixed<double, NumberOfImageChannels, NumberOfImageChannels> VarianceType;
typedef vnl_vector_fixed<double, NumberOfImageChannels> MeanType;
typedef std::vector<VarianceType> VarianceVectorType;
typedef std::vector<MeanType> MeanVectorType;

#ifdef USE_JSR_GLISTR
const unsigned int ImageDimension = 3;

typedef float FixedPixelType;
typedef float MovingPixelType;
typedef itk::Vector<float, ImageDimension> VectorType;

typedef itk::Image<VectorType, ImageDimension> DeformationFieldType;
typedef itk::Image<FixedPixelType, ImageDimension> FixedImageType; 
typedef itk::Image<MovingPixelType, ImageDimension> MovingImageType;

typedef itk::ImageFileReader<FixedImageType> FixedImageReaderType;
typedef std::vector<FixedImageReaderType::Pointer> FixedImageReaderVectorType;

typedef itk::ImageFileReader<MovingImageType> MovingImageReaderType;
typedef std::vector<MovingImageReaderType::Pointer> MovingImageReaderVectorType;

typedef itk::ImageFileWriter<MovingImageType> MovingImageFileWriterType;
typedef itk::ImageFileWriter<DeformationFieldType> DeformationFieldWriterType;

typedef itk::MultiResolutionJointSegmentationRegistration<FixedImageType, MovingImageType, DeformationFieldType, MovingPixelType, NumberOfImageChannels, NumberOfPriorChannels>
        JointSegmentationRegistrationType;

typedef JointSegmentationRegistrationType::SegmentationRegistrationType
        JointSegmentationRegistrationFilteType;

typedef JointSegmentationRegistrationFilteType::JointSegmentationRegistrationFunctionType
        JointSegmentationRegistrationFunctionType;
#endif
BOOL UpdateMeansAndVariances(FVolume* vd, BVolume& mask, FVolume* posteriors, double (*mv_)[4], double (*vv_)[16], double (*mv_init_)[4], int NumberOfElapsedIterations, bool bMeanShiftUpdate, bool mean_shift_update, bool ms_tumor, bool ms_edema);


BOOL TransformPoints(char* scan_image_file, char* scan_atlas_reg_image_file, char* scan_to_atlas_mat_file, double (*scan_seeds_info)[4], double (*scan_atlas_reg_seeds_info)[4], int scan_seeds_num, int check_orientation);
BOOL GetPoints(char* scan_image_file, double (*scan_points_info)[4], double (*scan_points_info_out)[4], int scan_points_num, int check_orientation);

BOOL MakeScalesFile(const char* scales_file);
BOOL MakeSimulatorInputFile(const char* simulator_input_file, double o_dx, double o_dy, double o_dz, int x, int y, int z, double dx, double dy, double dz);
BOOL MakeHOPSFile(const char* hops_file, double (*tp_c)[3], double* tp_T, int tp_num, const char* scan_atlas_reg_image_list, const char* atlas_prior_list, 
#ifdef USE_PROB_PRIOR
	const char* atlas_reg_prob_list, double prob_weight, 
#endif
#ifdef USE_ED_NON_WM_PROB
	double ed_non_wm_prob,
#endif
	const char* scan_means_file, const char* scan_variances_file, const char* scan_h_hdr_file, const char* atlas_label_map_s_img_file, 
	const char* simulator_input_file, const char* scales_file, const char* out_folder, const char* tmp_folder, char* solution_file, int max_eval, double* s_val, int k, bool hop_sync_eval, bool hop_random_order, int num_hop_threads);


BOOL LoadMeansAndVariances(const char* means_file, const char* variances_file, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector);
BOOL LoadMeansAndVariances(const char* means_file, const char* variances_file, double (*mv)[4], double (*vv)[16]);
BOOL SaveMeansAndVariances(const char* means_file, const char* variances_file, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector);
BOOL SaveMeansAndVariances(const char* means_file, const char* variances_file, double (*mv)[4], double (*vv)[16]);
BOOL InitializeMeansAndVariances(const char* means_file, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector);
BOOL InitializeMeansAndVariancesFromPoints(double (*scan_points_info)[4], int scan_points_num, char (*image_files)[1024], MeanVectorType &pMeanVector, VarianceVectorType &pVarianceVector, int check_orientation, int* b_valid_label);
BOOL ComputePosteriors(FVolume* vd, FVolume* priors, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector, 	FVolume* posteriors
#ifdef USE_TU_PRIOR_CUTOFF
	, float tu_prior_cutoff_tu_th, float tu_prior_cutoff_ncr_th
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	, float tu_prior_cutoff_ne_th
#endif
	, float tu_prior_cutoff_ed_th
#endif
#ifdef USE_PL_COST
	, FVolume* pl_costs
#endif
	, int* b_valid_label
	);
BOOL ComputeLabelMap(FVolume* posteriors, BVolume& label_map, int* b_valid_label);

#ifdef USE_PROB_PRIOR
BOOL ComputeTumorProb(FVolume* vd, int vd_n, 
	int (*tp_c)[3], float* tp_r, int tp_num, int tp_r_init, 
	int (*pt_c)[3], int* pt_t, int pt_num, int pt_r_init,
	FVolume* ref_tu, FVolume* ref_bg, float* wt, FVolume* prob, int prob_n, FVolume* tu_prob, float re_m = 2.0f);
void EstimateTumorSize(FVolume& tu_prob, int tp_num, int (*tp_c)[3], float* tp_r_est);
#endif

#ifdef USE_JSR_GLISTR
#include <itkImageRegionConstIterator.h>
typedef itk::ImageRegionConstIterator<const FixedImageType> FixedImageConstIteratorType;
typedef itk::ImageRegionConstIterator<const MovingImageType> MovingImageConstIteratorType;
#endif

static
inline void GetInv4(double m[16], double inv[16], double* det);
static
inline void ComputeLikelihood4(double inv[16], double c1, double c2, double ym[4], double* like);
static
inline void ComputeLikelihood1(double m, double ym, double* like);

#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
extern BOOL ComputePosteriors(FVolume* vd, BVolume& mask, FVolume* priors, double (*mv)[4], double (*vv)[16], 
#ifdef USE_PROB_PRIOR
	FVolume* probs, double prob_weight, 
#endif
	FVolume* likelihoods, FVolume* posteriors);
#endif

#ifdef UPDATE_DEFORMATION_FIELD
extern BOOL UpdateDeformationFieldSyM(FVolume* vd1, FVolume* vd2, int d_num, FVolume* tu1, FVolume* tu2, FVolume& ab1, FVolume& ab2, 
	FVolume& h_x, FVolume& h_y, FVolume& h_z, FVolume& h_r_x, FVolume& h_r_y, FVolume& h_r_z,
	char* name, int dmode, int mmode, int m_reg_iter = 2, int m_sub_reg_iter = 3, int m_sub_reg_iter_rep = 1, bool mesh_iter = false,
	bool save_int = true, bool save_pyrd = true, float lambda_D = 1.0, float lambda_P = 0.2,
	float _gamma = -1, float _alpha_O1 = -1, float _d_O1 = -1, float _alpha_O2 = -1, float _d_O2 = -1);
extern BOOL SmoothField(FVolume& vx, FVolume& vy, FVolume& vz, FVolume& vx_g, FVolume& vy_g, FVolume& vz_g, float sig);
extern BOOL ReverseDeformationField(FVolume& vx, FVolume& vy, FVolume& vz, FVolume& vxr, FVolume& vyr, FVolume& vzr);
extern BOOL ReverseField(FVolume& vx, FVolume& vy, FVolume& vz, FVolume& vxr, FVolume& vyr, FVolume& vzr);
#endif


void version()
{
  time_t now = time(0);

  // convert now to string form
  char* dt = ctime(&now);
	printf("==========================================================================\n");
  printf("Time:%s\n", dt);
	printf("GLISTR: Glioma Image Segmentation and Registration\n");
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
	printf("\n");
	printf("Parameters:\n\n");
	printf("  --featureslist, -fl <file>        Text file listing the patient scans, one per line.\n");
	printf("                                    It is recommended to use T1, T1-CE, T2, and FLAIR\n");
	printf("                                    patient scans as input.\n");
	printf("  --seed_info, -si <file>           Tumor seed information. This option overrides -x, -y, -z, and -T.\n");
	printf("                                    The format of this file is:\n");
	printf("                                    \n");
	printf("                                      <n: number of seeds>\n");
	printf("                                      <x1> <y1> <z1> <d1>\n");
	printf("                                      ...\n");
	printf("                                      <xn> <yn> <zn> <dn>\n");
	printf("                                    \n");
	printf("                                    where <xi> <yi> <zi> is the RAS coordinate of ith seed and\n");
	printf("                                    <di> is the approximated diameter of ith tumor.\n");
	printf("                                    \n");
	printf("  --point_info, -pi <file>          One sample point for each class. This option overrides -im.\n");
	printf("                                    The format of this file is:\n");
	printf("                                    \n");
	printf("                                      <Class 1>\n");
	printf("                                      <x1> <y1> <z1>\n");
	printf("                                      ...\n");
	printf("                                      <Class n>\n");
	printf("                                      <xn> <yn> <zn>\n");
	printf("                                    \n");
	printf("                                    where <Class i> is the name of ith tissue class and\n");
	printf("                                    <xi> <yi> <zi> is the RAS coordinate of ith tissue class.\n");
	printf("                                    \n");
	printf("  --atlas_folder, -af <dir>         The directory path for the atlas.\n");
	printf("  --outputdir, -d <dir>             The directory path for output files.\n");
	printf("\n");
	printf("Optional Parameters:\n\n");
	printf("  --workingdir, -w <dir>            The directory path for intermediate files.\n");
	printf("  --atlas_template, -at <file>      The atlas template file.\n");
	printf("  --atlas_prior, -ap <file>         The list file for atlas priors.\n");
	printf("  --atlas_label, -al <file>         The atlas label map (.nii.gz).\n");
	printf("  --atlas_label_img, -ali <file>    The atlas label map (.img).\n");
	//printf("  --use_default_atlas, -ad <int>    1 if use Jakob, 0 if use others.\n");
#ifdef UPDATE_DEFORMATION_FIELD
	printf("  --update_deform, -ud <int>        1 if update deformation field, 0 otherwise.\n");
#endif
	printf("  --num_omp_threads, -not <int>     The number of OpemMP threads.\n");
	printf("  --num_itk_threads, -nit <int>     The number of ITK threads.\n");
	printf("  --num_hop_threads, -nht <int>     The number of HOPSPACK threads.\n");
	printf("\n");
	printf("  --verbose, -v <int>               if greater than 0, print log messages: 1 (default), 2 (detailed).\n");
	printf("  --help, -h                        Print help and exit.\n");
	printf("  --usage, -u                       Print help and exit.\n");
	printf("  --version, -V                     Print version information and exit.\n");
	printf("\n");
	printf("Advanced Parameters:\n\n");
#ifdef USE_MASK_WEIGHT
	printf("  --atlas_mask_weight, -amw <float> The threshold value for forground mask.\n");
#else
	printf("  --atlas_mask, -am <file>          The atlas mask file.\n");
#endif
	printf("  --erode_bound_num, -ebn <int>     The number of erosion on the brain boundary.\n");
	printf("  --apply_atlas_mask, -aa <int>     1 if apply atlas mask, 0 otherwise.\n");
#ifdef USE_PROB_PRIOR
	printf("  --prob_weight, -pw <float>        The weight value for prob prior.\n");
	printf("  --prob_wt_t1ce_only, -pwt <int>   1 if apply t1ce only for prob prior, 0 otherwise.\n");
#endif
#ifdef USE_ED_NON_WM_PROB
	printf("  --ed_non_wm_prob, -enw <float>    The prob value for edema on the non wm region.\n");
#endif
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE) || defined(USE_MEAN_SHIFT_UPDATE_STEP)
	printf("  --mean_shift_update, -msu <int>   1 if use mean shift update, 0 otherwise.\n");
#endif
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	printf("  --use_ne_as_cavity, -unc <int>    1 if use the NE as the cavity, 0 otherwise.\n");
	printf("  --cavity_label_val, -clv <int>    The label value of the cavity (default: 15).\n");
#endif
	printf("  --check_orientation, -co <int>    1 if check orientation, 0 otherwise.\n");
	printf("  --hop_sync_eval, -hse <int>       1 if use sync evaluations, 0 if use async evaluations.\n");
	printf("  --hop_random_order, -hro <int>    1 if use random order, 0 otherwise.\n");
	printf("\n");
	printf("Obsolete Parameters:\n\n");
	printf("  --xc, -x <float>                  Initial tumor center in sagital direction [mm].\n");
	printf("  --yc, -y <float>                  Initial tumor center in coronal direction [mm].\n");
	printf("  --zc, -z <float>                  Initial tumor center in axial direction [mm].\n");
	printf("  --growthtime, -T <int>            Initial tumor growth length. Should be larger than 25.\n");
	printf("                                    For huge tumors, set it to about 120.\n");
	printf("  --initialmeans, -im <file>        Text file specifying the initial mean values for\n");
	printf("                                    the different classes. The format of this file\n");
	printf("                                    \n");
	printf("                                      Class 1\n");
	printf("                                      <mean of feature1> ... <mean of featureN>\n");
	printf("                                      ...\n");
	printf("                                      Class n\n");
	printf("                                      <mean of feature1> ... <mean of featureN>\n");
	printf("                                    \n");
	printf("                                    where <*> are the mean values of the corresponding\n");
	printf("                                    tissue classes in the corresponding patient scans\n");
	printf("                                    separated by spaces.\n");
	printf("                                    \n");
	printf("                                    Note that the order of these values must correspond\n");
	printf("                                    to the order in which the corresponding feature\n");
	printf("                                    images are specified in the --featureslist file.\n");
	printf("\n\n");
}


int main(int argc, char* argv[])
{
  clock_t clock_end;
  const clock_t clock_start = clock();
	// intput: scan, tumor center, T (time or tumor volume)
	// output: deform field, posteriors, means and variances
	char scan_image_list[1024] = {0,};
	char scan_init_mean[1024] = {0,};
	char scan_init_seed[1024] = {0,};
	char scan_init_point[1024] = {0,};	
	char atlas_template_file[1024] = {0,};
	char atlas_prior_list[1024] = {0,};
#ifdef USE_MASK_WEIGHT
	float atlas_mask_weight = 0.8; // default value
#endif
	char atlas_mask_file[1024] = {0,};
	char atlas_label_map_s_file[1024] = {0,};
	char atlas_label_map_s_img_file[1024] = {0,};
	char out_folder[1024] = {0,};
	char tmp_folder[1024] = {0,};
	char atlas_folder[1024] = {0,};
	//
	bool b_seed_x = false;
	bool b_seed_y = false;
	bool b_seed_z = false;
	bool b_seed_T = false;
	double seed_x, seed_y, seed_z, seed_T;
	bool b_seed_info = false;
	bool b_point_info = false;
	//
	double scan_seeds_info[MaxNumberOFTumorSeeds][4];
	int scan_seeds_num = 0;
	//
	double scan_points_info[MaxNumberOFPoints][4];
	int scan_points_num = 0;
	//
	bool delete_tmp_folder = true;
	//
	int use_default_label_map_s = -1;
	bool apply_mask = true;
	//bool apply_mask = false;
	bool use_masked_images_for_posteriors = true;
	//bool use_masked_images_for_posteriors = false;
#ifdef USE_PROB_PRIOR
	float prob_weight = 40.0; // default value
	//float prob_weight = 30.0;
	//float prob_weight = 15.0;
	bool prob_wt_t1ce_only = false;
#endif
#ifdef USE_ED_NON_WM_PROB
	double ed_non_wm_prob = 0.05; // default value
#endif
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE) || defined(USE_MEAN_SHIFT_UPDATE_STEP)
	bool mean_shift_update = true;
#ifdef USE_MS_TU_NCR_NE
	bool ms_tumor = true;
#else
	bool ms_tumor = false;
#endif
#ifdef USE_MS_WM_ED
	bool ms_edema = true;
#else
	bool ms_edema = false;
#endif
#endif
#ifdef UPDATE_DEFORMATION_FIELD 
	bool update_deform = false;
	float lambda_D = 1.0;
#endif
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	bool use_ne_as_cavity = false;
	int cavity_label_val = 15;
#endif
#ifdef USE_HOPS_NO_SYNC_EVAL
	bool hop_sync_eval = false;
#else
	bool hop_sync_eval = true;
#endif
#ifdef USE_HOPS_RANDOM_ORDER
	bool hop_random_order = true;
#else
	bool hop_random_order = false;
#endif
	int align_image_index = 0;
	int check_orientation = 0;
	int jsr_iter = 3;
	int erode_bound_num = 0;
	//
	int b_valid_label[NumberOfPriorChannels];
	//
#if defined(WIN32) || defined(WIN64)
	int num_omp_threads = 3;
	int num_itk_threads = 3;
	int num_hop_threads = 3;
#else
	int num_omp_threads = 3;
	int num_itk_threads = 3;
	// consider memory limitations
	int num_hop_threads = 3;
#endif
	//
	bool provided_tumor_size = true;
	//
#ifdef USE_LABEL_NOISE_REDUCTION
	bool l_noise_reduction = true;
#endif
#ifdef USE_LABEL_CORRECTION
	bool lc_edema = true;
	bool lc_tumor = true;
#endif
	bool output_tmp_label = false;
	int flirt_search_angle = 180;
	//
	bool b_run_test = false;
	int verbose = 1;

	// parse command line
	{
		int i;
		if (argc == 1) {
			printf("use option -h or --help for help\n");
			exit(EXIT_FAILURE);
		}
		if (argc == 2) {
			if        (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
				version();
				usage();
				exit(EXIT_FAILURE);
			} else if (strcmp(argv[1], "-u") == 0 || strcmp(argv[1], "--usage") == 0) {
				version();
				usage();
				exit(EXIT_FAILURE);
			} else if (strcmp(argv[1], "-V") == 0 || strcmp(argv[1], "--version") == 0) {
				version();
				exit(EXIT_FAILURE);
			} else {
				printf("error: wrong arguments\n");
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
		}
		for (i = 1; i < argc; i++) {
			if (argv[i] == NULL || argv[i+1] == NULL) {
				printf("error: not specified argument\n");
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
			if        (strcmp(argv[i], "-fl" ) == 0 || strcmp(argv[i], "--featureslist"      ) == 0) { sprintf(scan_image_list    , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-x"  ) == 0 || strcmp(argv[i], "--xc"                ) == 0) { seed_x = (double)atof(argv[i+1]); b_seed_x = true; i++;
			} else if (strcmp(argv[i], "-y"  ) == 0 || strcmp(argv[i], "--yc"                ) == 0) { seed_y = (double)atof(argv[i+1]); b_seed_y = true; i++;
			} else if (strcmp(argv[i], "-z"  ) == 0 || strcmp(argv[i], "--zc"                ) == 0) { seed_z = (double)atof(argv[i+1]); b_seed_z = true; i++;
			} else if (strcmp(argv[i], "-T"  ) == 0 || strcmp(argv[i], "--growthtime"        ) == 0) { seed_T = (double)atof(argv[i+1]); b_seed_T = true; i++;
			} else if (strcmp(argv[i], "-im" ) == 0 || strcmp(argv[i], "--initialmeans"      ) == 0) { sprintf(scan_init_mean     , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-si" ) == 0 || strcmp(argv[i], "--seed_info"         ) == 0) { sprintf(scan_init_seed, "%s", argv[i+1]); b_seed_info = true; i++;
			} else if (strcmp(argv[i], "-pi" ) == 0 || strcmp(argv[i], "--point_info"        ) == 0) { sprintf(scan_init_point, "%s", argv[i+1]); b_point_info = true; i++;
			} else if (strcmp(argv[i], "-af" ) == 0 || strcmp(argv[i], "--atlas_folder"      ) == 0) { sprintf(atlas_folder       , "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-at" ) == 0 || strcmp(argv[i], "--atlas_template"    ) == 0) { sprintf(atlas_template_file, "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-ap" ) == 0 || strcmp(argv[i], "--atlas_prior"       ) == 0) { sprintf(atlas_prior_list   , "%s", argv[i+1]); i++;
#ifdef USE_MASK_WEIGHT
			} else if (strcmp(argv[i], "-amw") == 0 || strcmp(argv[i], "--atlas_mask_weight" ) == 0) { atlas_mask_weight = atof(argv[i+1]); i++;
#else
			} else if (strcmp(argv[i], "-am" ) == 0 || strcmp(argv[i], "--atlas_mask"        ) == 0) { sprintf(atlas_mask_file    , "%s", argv[i+1]); i++;
#endif
			} else if (strcmp(argv[i], "-ebn") == 0 || strcmp(argv[i], "--erode_bound_num"   ) == 0) { erode_bound_num = atoi(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-al" ) == 0 || strcmp(argv[i], "--atlas_label"       ) == 0) { sprintf(atlas_label_map_s_file, "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-ali") == 0 || strcmp(argv[i], "--atlas_label_img"   ) == 0) { sprintf(atlas_label_map_s_img_file, "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-ad" ) == 0 || strcmp(argv[i], "--use_default_atlas" ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					use_default_label_map_s = 0;
				} else {
					use_default_label_map_s = 1;
				}
				i++;
			} else if (strcmp(argv[i], "-aa" ) == 0 || strcmp(argv[i], "--apply_atlas_mask"  ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					apply_mask = false;
				} else {
					apply_mask = true;
				}
				i++;
#ifdef USE_PROB_PRIOR
			} else if (strcmp(argv[i], "-pw" ) == 0 || strcmp(argv[i], "--prob_weight"       ) == 0) { prob_weight = atof(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-pwt") == 0 || strcmp(argv[i], "--prob_wt_t1ce_only" ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					prob_wt_t1ce_only = false;
				} else {
					prob_wt_t1ce_only = true;
				}
				i++;
#endif
#ifdef USE_ED_NON_WM_PROB
			} else if (strcmp(argv[i], "-enw") == 0 || strcmp(argv[i], "--ed_non_wm_prob"    ) == 0) { ed_non_wm_prob = atof(argv[i+1]); i++;
#endif
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE) || defined(USE_MEAN_SHIFT_UPDATE_STEP)
			} else if (strcmp(argv[i], "-msu") == 0 || strcmp(argv[i], "--mean_shift_update" ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					mean_shift_update = false;
				} else {
					mean_shift_update = true;
				}
				i++;
			} else if (strcmp(argv[i], "-mst") == 0 || strcmp(argv[i], "--ms_tumor"          ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					ms_tumor = false;
				} else {
					ms_tumor = true;
				}
				i++;
			} else if (strcmp(argv[i], "-mse") == 0 || strcmp(argv[i], "--ms_edema"          ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					ms_edema = false;
				} else {
					ms_edema = true;
				}
				i++;
#endif
#ifdef UPDATE_DEFORMATION_FIELD
			} else if (strcmp(argv[i], "-ud" ) == 0 || strcmp(argv[i], "--update_deform"     ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					update_deform = false;
				} else {
					update_deform = true;
				}
				i++;
			} else if (strcmp(argv[i], "-lD" ) == 0 || strcmp(argv[i], "--lambda_D"          ) == 0) { lambda_D = (float)atof(argv[i+1]); i++;
#endif
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
			} else if (strcmp(argv[i], "-unc") == 0 || strcmp(argv[i], "--use_ne_as_cavity"  ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					use_ne_as_cavity = false;
				} else {
					use_ne_as_cavity = true;
				}
				i++;
			} else if (strcmp(argv[i], "-clv") == 0 || strcmp(argv[i], "--cavity_label_val"  ) == 0) { cavity_label_val = atoi(argv[i+1]); i++;
#endif
			} else if (strcmp(argv[i], "-hse") == 0 || strcmp(argv[i], "--hop_sync_eval"     ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					hop_sync_eval = false;
				} else {
					hop_sync_eval = true;
				}
				i++;
			} else if (strcmp(argv[i], "-hro") == 0 || strcmp(argv[i], "--hop_random_order"  ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					hop_random_order = false;
				} else {
					hop_random_order = true;
				}
				i++;
#ifdef USE_LABEL_NOISE_REDUCTION
			} else if (strcmp(argv[i], "-lnr") == 0 || strcmp(argv[i], "--l_noise_reduction" ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					l_noise_reduction = false;
				} else {
					l_noise_reduction = true;
				}
				i++;
#endif
#ifdef USE_LABEL_CORRECTION
			} else if (strcmp(argv[i], "-lce") == 0 || strcmp(argv[i], "--lc_edema"          ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					lc_edema = false;
				} else {
					lc_edema = true;
				}
				i++;
			} else if (strcmp(argv[i], "-lct") == 0 || strcmp(argv[i], "--lc_tumor"          ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					lc_tumor = false;
				} else {
					lc_tumor = true;
				}
				i++;
#endif
			} else if (strcmp(argv[i], "-aii") == 0 || strcmp(argv[i], "--align_image_index" ) == 0) { 
				align_image_index = atoi(argv[i+1]); 
				if (align_image_index < 0) {
					align_image_index = 0;
				}
				if (align_image_index >= NumberOfImageChannels) {
					align_image_index = NumberOfImageChannels-1;
				}
				i++;
			} else if (strcmp(argv[i], "-otl") == 0 || strcmp(argv[i], "--output_tmp_label"  ) == 0) {
				if (atoi(argv[i+1]) == 0) {
					output_tmp_label = false;
				} else {
					output_tmp_label = true;
				}
				i++;
			} else if (strcmp(argv[i], "-fsa") == 0 || strcmp(argv[i], "--flirt_search_angle") == 0) { flirt_search_angle = atoi(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-co" ) == 0 || strcmp(argv[i], "--check_orientation" ) == 0) { check_orientation = atoi(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-rt" ) == 0 || strcmp(argv[i], "--run_test") == 0) { 
				if ((int)atoi(argv[i+1]) == 0) {
					b_run_test = false;
				} else {
					b_run_test = true;
				}
				i++;
			} else if (strcmp(argv[i], "-not") == 0 || strcmp(argv[i], "--num_omp_threads"   ) == 0) { num_omp_threads = atoi(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-nit") == 0 || strcmp(argv[i], "--num_itk_threads"   ) == 0) { num_itk_threads = atoi(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-nht") == 0 || strcmp(argv[i], "--num_hop_threads"   ) == 0) { num_hop_threads = atoi(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-ji" ) == 0 || strcmp(argv[i], "--jsr_iter"          ) == 0) { jsr_iter = atoi(argv[i+1]); i++;
			} else if (strcmp(argv[i], "-d"  ) == 0 || strcmp(argv[i], "--outputdir"         ) == 0) { sprintf(out_folder, "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-w"  ) == 0 || strcmp(argv[i], "--workingdir"        ) == 0) { sprintf(tmp_folder, "%s", argv[i+1]); i++;
			} else if (strcmp(argv[i], "-v"  ) == 0 || strcmp(argv[i], "--verbose"           ) == 0) { verbose = atoi(argv[i+1]); i++;
			} else {
				printf("error: %s is not recognized\n", argv[i]);
				printf("use option -h or --help for help\n");
				exit(EXIT_FAILURE);
			}
		}
		//
		if (atlas_folder[0] != 0) {
			char atlas_folder_full[1024] = {0,};
			//
			if (strrchr(atlas_folder, '/') == NULL && strrchr(atlas_folder, '\\') == NULL) {
				char sPath[1024];
				if (!GetModulePath(sPath)) {
					printf("GetModulePath failed\n");
					exit(EXIT_FAILURE);
				}
				str_divide_footer(sPath, sPath, NULL, DIR_SEP_C);
				str_divide_footer(sPath, sPath, NULL, DIR_SEP_C);
				//
				sprintf(atlas_folder_full, "%s%s%s", sPath, DIR_SEP, atlas_folder);
			} else {
				strcpy(atlas_folder_full, atlas_folder);
			}
			//
			sprintf(atlas_template_file, "%s%sjakob_stripped_with_cere_lps_256256128.nii.gz", atlas_folder_full, DIR_SEP);
			if (!IsFileExist(atlas_template_file)) {
				sprintf(atlas_template_file, "%s%sJHU_1m_lps_T1.nii.gz", atlas_folder_full, DIR_SEP);
				if (!IsFileExist(atlas_template_file)) {
					printf("error: couldn't find atlas_template_file\n");
					exit(EXIT_FAILURE);
				}
			}
			//printf("atlas_template_file found: %s\n", atlas_template_file);
#if defined(USE_8_PRIORS)
			sprintf(atlas_prior_list, "%s%satlas_priors8.lst", atlas_folder_full, DIR_SEP);
#elif defined(USE_9_PRIORS)
			sprintf(atlas_prior_list, "%s%satlas_priors9.lst", atlas_folder_full, DIR_SEP);
#elif defined(USE_10_PRIORS)
			sprintf(atlas_prior_list, "%s%satlas_priors10.lst", atlas_folder_full, DIR_SEP);
#elif defined(USE_10A_PRIORS)
			sprintf(atlas_prior_list, "%s%satlas_priors10a.lst", atlas_folder_full, DIR_SEP);
#elif defined(USE_11_PRIORS)
			sprintf(atlas_prior_list, "%s%satlas_priors11.lst", atlas_folder_full, DIR_SEP);
#endif
#ifndef USE_MASK_WEIGHT
			sprintf(atlas_mask_file, "%s%smask_256256128.nii.gz", atlas_folder_full, DIR_SEP);
#endif
			sprintf(atlas_label_map_s_file, "%s%slabel_646464.nii.gz", atlas_folder_full, DIR_SEP);
			sprintf(atlas_label_map_s_img_file, "%s%slabel_646464.img", atlas_folder_full, DIR_SEP);
		}
		if (atlas_folder[0] == 0 && (atlas_template_file[0] == 0 || atlas_prior_list[0] == 0 
#ifndef USE_MASK_WEIGHT
			|| atlas_mask_file[0] == 0 
#endif
			|| atlas_label_map_s_file[0] == 0 || atlas_label_map_s_img_file[0] == 0)) {
			char sPath[1024];
			if (!GetModulePath(sPath)) {
				printf("GetModulePath failed\n");
				exit(EXIT_FAILURE);
			}
			str_divide_footer(sPath, sPath, NULL, DIR_SEP_C);
			str_divide_footer(sPath, sPath, NULL, DIR_SEP_C);
			//
			sprintf(atlas_template_file, "%s%s%s%sjakob_stripped_with_cere_lps_256256128.nii.gz", sPath, DIR_SEP, "atlas_jakob_with_cere_type", DIR_SEP);
			if (!IsFileExist(atlas_template_file)) {
				printf("error: couldn't find atlas_template_file (%s)\n", atlas_template_file);
				exit(EXIT_FAILURE);
			}
#if defined(USE_8_PRIORS)
			sprintf(atlas_prior_list, "%s%s%s%satlas_priors8.lst", sPath, DIR_SEP, "atlas_jakob_with_cere_type", DIR_SEP);
#elif defined(USE_9_PRIORS)
			sprintf(atlas_prior_list, "%s%s%s%satlas_priors9.lst", sPath, DIR_SEP, "atlas_jakob_with_cere_type", DIR_SEP);
#elif defined(USE_10_PRIORS)
			sprintf(atlas_prior_list, "%s%s%s%satlas_priors10.lst", sPath, DIR_SEP, "atlas_jakob_with_cere_type", DIR_SEP);
#elif defined(USE_10A_PRIORS)
			sprintf(atlas_prior_list, "%s%s%s%satlas_priors10a.lst", sPath, DIR_SEP, "atlas_jakob_with_cere_type", DIR_SEP);
#elif defined(USE_11_PRIORS)
			sprintf(atlas_prior_list, "%s%s%s%satlas_priors11.lst", sPath, DIR_SEP, "atlas_jakob_with_cere_type", DIR_SEP);
#endif
#ifndef USE_MASK_WEIGHT
			sprintf(atlas_mask_file, "%s%s%s%smask_256256128.nii.gz", sPath, DIR_SEP, "atlas_jakob_with_cere_type", DIR_SEP);
#endif
			sprintf(atlas_label_map_s_file, "%s%s%s%slabel_646464.nii.gz", sPath, DIR_SEP, "atlas_jakob_with_cere_type", DIR_SEP);
			sprintf(atlas_label_map_s_img_file, "%s%s%s%slabel_646464.img", sPath, DIR_SEP, "atlas_jakob_with_cere_type", DIR_SEP);
		}
		if (scan_image_list[0] == 0 || (scan_init_mean[0] == 0 && b_point_info == false) || (b_seed_info == false && (b_seed_x == false || b_seed_y == false || b_seed_z == false || b_seed_T == false))
			|| atlas_template_file[0] == 0 || atlas_prior_list[0] == 0
#ifndef USE_MASK_WEIGHT
			|| atlas_mask_file[0] == 0 
#endif
			|| atlas_label_map_s_file[0] == 0 || atlas_label_map_s_img_file[0] == 0 || out_folder[0] == 0)
		{
			printf("error: essential arguments are not specified\n");
			printf("use option -h or --help for help\n");
			exit(EXIT_FAILURE);
		}
		if (tmp_folder[0] == 0) {
			char tmpn[1024] = {0,};
#if defined(WIN32) || defined(WIN64)
			sprintf(tmpn, "tmp_XXXXXX");
			if (mktemp(tmpn) == NULL) {
				sprintf(tmpn, "tmp");
			}			
			sprintf(tmp_folder, "%s%s%s", out_folder, DIR_SEP, tmpn);
#else
			char *tmpn_res = NULL;
			if (getenv("SBIA_TMPDIR") != NULL) {
				sprintf(tmp_folder, "%s", getenv("SBIA_TMPDIR"));
				CreateDirectory(tmp_folder, NULL);
				//
				sprintf(tmpn, "%s%stmp_XXXXXX", getenv("SBIA_TMPDIR"), DIR_SEP);
				tmpn_res = mkdtemp(tmpn);
				if (tmpn_res != NULL) {
					sprintf(tmp_folder, "%s", tmpn_res);
				} else {
					// making temporary directory failed: let's use subdir of out_folder
					sprintf(tmp_folder, "%s%stmp", out_folder, DIR_SEP);
				}
			} else if (getenv("USER") != NULL) {
				sprintf(tmp_folder, "%stmp%s%s", DIR_SEP, DIR_SEP, getenv("USER"));
				CreateDirectory(tmp_folder, NULL);
				//
				sprintf(tmpn, "%stmp%s%s%stmp_XXXXXX", DIR_SEP, DIR_SEP, getenv("USER"), DIR_SEP);
				tmpn_res = mkdtemp(tmpn);
				if (tmpn_res != NULL) {
					sprintf(tmp_folder, "%s", tmpn_res);
				} else {
					// making temporary directory failed: let's use subdir of out_folder
					sprintf(tmp_folder, "%s%stmp", out_folder, DIR_SEP);
				}
			} else {
				// making temporary directory failed: let's use subdir of out_folder
				sprintf(tmp_folder, "%s%stmp", out_folder, DIR_SEP);
			}
#endif
		} else {
			delete_tmp_folder = false;
		}
		//
		if (!b_seed_info) {
			scan_seeds_num = 1;
			scan_seeds_info[0][0] = seed_x;
			scan_seeds_info[0][1] = seed_y;
			scan_seeds_info[0][2] = seed_z;
			// tumor days
			//T = 0.004 * pow(scan_atlas_reg_seeds_info[0][3] / 2, 3) * 4;
			scan_seeds_info[0][3] = pow(seed_T / (0.004 * 4.0), 1.0/3.0) * 2;
		}
		if (!b_point_info) {
			for (i = 0; i < NumberOfPriorChannels; i++) {
				b_valid_label[i] = 1;
			}
		} else {
			for (i = 0; i < NumberOfPriorChannels; i++) {
				b_valid_label[i] = 0;
			}
			b_valid_label[BG] = 1;
		}
		//
		if (use_default_label_map_s < 0) {
			char sName[1024];
			str_divide_footer(atlas_template_file, NULL, sName, DIR_SEP_C);
			if (strcmp(sName, "jakob_stripped_with_cere_lps_256256128.nii.gz") == 0) {
				use_default_label_map_s = 1;
			} else {
				use_default_label_map_s = 0;
			}
		}
	}

#if (!defined(WIN32) && !defined(WIN64)) || !defined(_DEBUG)
	{
		char MODULE_PATH[1024];
#if 0
		str_strip_file(argv[0], MODULE_PATH);
#else
		if (!GetModulePath(MODULE_PATH)) {
			printf("GetModulePath failed\n");
			exit(EXIT_FAILURE);
		}
#endif
		//
#if defined(WIN32) || defined(WIN64)
		sprintf(EVALUATE_Q_PATH, "%sEvaluateQ.exe", MODULE_PATH);
		//
		if (!b_run_test) {
			if (!FindExecutableInPath("HOPSPACK_main_threaded.exe", MODULE_PATH, HOPSPACK_PATH))  { printf("error: HOPSPACK_main_threaded.exe is not existing.\n"); exit(EXIT_FAILURE); }
			if (!FindExecutableInPath("flirt.exe", MODULE_PATH, FLIRT_PATH))					  { printf("error: flirt.exe is not existing.\n"); exit(EXIT_FAILURE); }
			if (!FindExecutableInPath("convert_xfm.exe", MODULE_PATH, CONVERT_XFM_PATH))		  { printf("error: convert_xfm.exe is not existing.\n"); exit(EXIT_FAILURE); }
			if (!FindExecutableInPath("ForwardSolverDiffusion.exe", MODULE_PATH, SIMULATOR_PATH)) { printf("error: ForwardSolverDiffusion.exe is not existing.\n"); exit(EXIT_FAILURE); }
		} else {
			sprintf(HOPSPACK_PATH, "%sHOPSPACK_main_threaded.exe", MODULE_PATH);
			sprintf(FLIRT_PATH, "%sflirt.exe", MODULE_PATH);
			sprintf(CONVERT_XFM_PATH, "%sconvert_xfm.exe", MODULE_PATH);
			sprintf(SIMULATOR_PATH, "%sForwardSolverDiffusion.exe", MODULE_PATH);
		}
		//
		sprintf(RESAMPLE_IMAGE_PATH, "%sResampleImage.exe", MODULE_PATH);
		sprintf(RESAMPLE_DEFORMATION_FIELD_PATH, "%sResampleDeformationField.exe", MODULE_PATH);
		sprintf(REVERSE_DEFORMATION_FIELD_PATH, "%sReverseDeformationField.exe", MODULE_PATH);
		sprintf(WARP_IMAGE_PATH, "%sWarpImage.exe", MODULE_PATH);
		sprintf(CONCATENATE_FIELD_PATH, "%sConcatenateFields.exe", MODULE_PATH);
#else
		sprintf(EVALUATE_Q_PATH, "%sEvaluateQ", MODULE_PATH);
		//
		if (!b_run_test) {
			if (!FindExecutableInPath("HOPSPACK_main_threaded", MODULE_PATH, HOPSPACK_PATH))  { printf("error: HOPSPACK_main_threaded is not existing.\n"); exit(EXIT_FAILURE); }
			if (!FindExecutableInPath("flirt", MODULE_PATH, FLIRT_PATH))					  { printf("error: flirt is not existing.\n"); exit(EXIT_FAILURE); }
			if (!FindExecutableInPath("convert_xfm", MODULE_PATH, CONVERT_XFM_PATH))		  { printf("error: convert_xfm is not existing.\n"); exit(EXIT_FAILURE); }
			if (!FindExecutableInPath("ForwardSolverDiffusion", MODULE_PATH, SIMULATOR_PATH)) { printf("error: ForwardSolverDiffusion is not existing.\n"); exit(EXIT_FAILURE); }
		} else {
			sprintf(HOPSPACK_PATH, "%sHOPSPACK_main_threaded", MODULE_PATH);
			sprintf(FLIRT_PATH, "%sflirt", MODULE_PATH);
			sprintf(CONVERT_XFM_PATH, "%sconvert_xfm", MODULE_PATH);
			sprintf(SIMULATOR_PATH, "%sForwardSolverDiffusion", MODULE_PATH);
		}
		//
		sprintf(RESAMPLE_IMAGE_PATH, "%sResampleImage", MODULE_PATH);
		sprintf(RESAMPLE_DEFORMATION_FIELD_PATH, "%sResampleDeformationField", MODULE_PATH);
		sprintf(REVERSE_DEFORMATION_FIELD_PATH, "%sReverseDeformationField", MODULE_PATH);
		sprintf(WARP_IMAGE_PATH, "%sWarpImage", MODULE_PATH);
		sprintf(CONCATENATE_FIELD_PATH, "%sConcatenateFields", MODULE_PATH);
#endif
	}
#endif

	char scan_image_files[NumberOfImageChannels][1024];
	char scan_image_masked_files[NumberOfImageChannels][1024];
	//
	char atlas_prior_files[NumberOfPriorChannels][1024];
	char scan_to_atlas_mat_file[1024];
	char atlas_to_scan_mat_file[1024];
	//
	char scan_atlas_reg_image_list[1024];
	char scan_atlas_reg_image_masked_list[1024];
#ifdef USE_MASK_G
	char scan_atlas_reg_image_masked_g_list[1024];
#endif
	char scan_atlas_reg_image_files[NumberOfImageChannels][1024];
	char scan_atlas_reg_image_masked_files[NumberOfImageChannels][1024];
#ifdef USE_MASK_G
	char scan_atlas_reg_image_masked_g_files[NumberOfImageChannels][1024];
#endif
	char scan_atlas_reg_label_map_file[1024];
	//
	char scan_init_means_file[1024];
	char scan_init_variances_file[1024];
	char scan_means_file[1024];
	char scan_variances_file[1024];
	char scan_atlas_reg_mass_prior_list[1024];
	char scan_atlas_reg_mass_prior_files[NumberOfPriorChannels][1024];
	char scan_atlas_reg_warp_prior_list[1024];
	char scan_atlas_reg_warp_prior_files[NumberOfPriorChannels][1024];
	char scan_atlas_reg_posterior_list[1024];
	char scan_atlas_reg_posterior_files[NumberOfPriorChannels][1024];
#ifdef USE_PROB_PRIOR
	char scan_atlas_reg_prob_list[1024];
	char scan_atlas_reg_prob_files[NumberOfPriorChannels][1024];
#endif
	//
	char scan_u_hdr_file[1024];
	char scan_h_hdr_file[1024];
	char scan_h0_hdr_file[1024];
	//
	char scan_prior_list[1024];
	char scan_prior_files[NumberOfPriorChannels][1024];
	char scan_posterior_list[1024];
	char scan_posterior_files[NumberOfPriorChannels][1024];
	char scan_label_map_file[1024];
	//
	char scales_file[1024];
	char simulator_input_file[1024];
	char hops_file[1024];
	//
	char scan_mask_file[1024];
	char scan_atlas_reg_mask_file[1024];
	//
	double scan_atlas_reg_seeds_info[MaxNumberOFTumorSeeds][4];
	double scan_atlas_reg_points_info[MaxNumberOFPoints][4];
	//
	BVolume atlas_mask;
	BVolume atlas_label_map_s;
	//
	FVolume scan_atlas_reg_images[NumberOfImageChannels];
	FVolume scan_atlas_reg_images_masked[NumberOfImageChannels];
	BVolume scan_atlas_reg_label_map;
	//
	//FVolume scan_atlas_reg_priors[NumberOfPriorChannels];
	FVolume scan_atlas_reg_mass_priors[NumberOfPriorChannels];
	FVolume scan_atlas_reg_warp_priors[NumberOfPriorChannels];
	FVolume scan_atlas_reg_posteriors[NumberOfPriorChannels];
#ifdef USE_PROB_PRIOR
	FVolume scan_atlas_reg_probs[NumberOfPriorChannels];
#endif
	//
	FVolume scan_posteriors[NumberOfPriorChannels];
	BVolume scan_label_map;
	//
	MeanVectorType scan_InitMeanVector;
	MeanVectorType scan_MeanVector;
	VarianceVectorType scan_VarianceVector;
	//
	RVolume scan_u_x, scan_u_y, scan_u_z;
	RVolume scan_h_x, scan_h_y, scan_h_z;
	//
	FVolume scan_mask;
	FVolume scan_atlas_reg_mask;
	//
	double tp_c[MaxNumberOFTumorSeeds][3], tp_r[MaxNumberOFTumorSeeds], tp_T[MaxNumberOFTumorSeeds];
	double s_val[5+6+4*MaxNumberOFTumorSeeds] = {0,};
	//
	char szCmdLine[2048];
	int i, j, k, l, m, n;
	//
	int scan_jsr_max_iter = jsr_iter;

	// set intermediate folders & files
	CreateDirectory(out_folder, NULL);
	CreateDirectory(tmp_folder, NULL);
	//
	SetCurrentDirectory(tmp_folder);

	/////////////////////////////////////////////////////////////////////////////
	sprintf(g_trace_file, "%s%slog.txt", out_folder, DIR_SEP);
	if (g_fp_trace != NULL) {
		fclose(g_fp_trace);
		g_fp_trace = NULL;
	}
	g_bTrace = FALSE;
	g_bTraceStdOut = (verbose>=1)?TRUE:FALSE;
	g_verbose = verbose;
	/////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////
	TRACE("==========================================================================\n");
	TRACE("GLISTR: Glioma Image Segmentation and Registration\n");
#ifdef SW_VER
	TRACE("  Version %s\n", SW_VER);
#endif
#ifdef SW_REV
	TRACE("  Revision %s\n", SW_REV);
#endif
	TRACE("Copyright (c) 2014 University of Pennsylvania. All rights reserved.\n");
	TRACE("See http://www.cbica.upenn.edu/sbia/software/license.html or COPYING file.\n");
	TRACE("==========================================================================\n");
	/////////////////////////////////////////////////////////////////////////////
	TRACE("\n");
	/////////////////////////////////////////////////////////////////////////////
	if (!b_run_test) {
		TRACE("EVALUATE_Q_PATH = %s\n", EVALUATE_Q_PATH);
		//
		TRACE("HOPSPACK_PATH = %s\n", HOPSPACK_PATH);
		TRACE("FLIRT_PATH = %s\n", FLIRT_PATH);
		TRACE("CONVERT_XFM_PATH = %s\n", CONVERT_XFM_PATH);
		TRACE("SIMULATOR_PATH = %s\n", SIMULATOR_PATH);
		//
		TRACE("RESAMPLE_IMAGE_PATH = %s\n", RESAMPLE_IMAGE_PATH);
		TRACE("RESAMPLE_DEFORMATION_FIELD_PATH = %s\n", RESAMPLE_DEFORMATION_FIELD_PATH);
		TRACE("REVERSE_DEFORMATION_FIELD_PATH = %s\n", REVERSE_DEFORMATION_FIELD_PATH);
		TRACE("WARP_IMAGE_PATH = %s\n", WARP_IMAGE_PATH);
		TRACE("CONCATENATE_FIELD_PATH = %s\n", CONCATENATE_FIELD_PATH);
		//
		if (!IsFileExist(EVALUATE_Q_PATH))					{ TRACE("error: %s is not existing.\n", EVALUATE_Q_PATH); exit(EXIT_FAILURE); }
		//
		if (!IsFileExist(HOPSPACK_PATH))					{ TRACE("error: %s is not existing.\n", HOPSPACK_PATH); exit(EXIT_FAILURE); }
		if (!IsFileExist(FLIRT_PATH))						{ TRACE("error: %s is not existing.\n", FLIRT_PATH); exit(EXIT_FAILURE); }
		if (!IsFileExist(CONVERT_XFM_PATH))					{ TRACE("error: %s is not existing.\n", CONVERT_XFM_PATH); exit(EXIT_FAILURE); }
		if (!IsFileExist(SIMULATOR_PATH))					{ TRACE("error: %s is not existing.\n", SIMULATOR_PATH); exit(EXIT_FAILURE); }
		//
		if (!IsFileExist(RESAMPLE_IMAGE_PATH))				{ TRACE("error: %s is not existing.\n", RESAMPLE_IMAGE_PATH); exit(EXIT_FAILURE); }
		if (!IsFileExist(RESAMPLE_DEFORMATION_FIELD_PATH))	{ TRACE("error: %s is not existing.\n", RESAMPLE_DEFORMATION_FIELD_PATH); exit(EXIT_FAILURE); }
		if (!IsFileExist(REVERSE_DEFORMATION_FIELD_PATH))	{ TRACE("error: %s is not existing.\n", REVERSE_DEFORMATION_FIELD_PATH); exit(EXIT_FAILURE); }
		if (!IsFileExist(WARP_IMAGE_PATH))					{ TRACE("error: %s is not existing.\n", WARP_IMAGE_PATH); exit(EXIT_FAILURE); }
		if (!IsFileExist(CONCATENATE_FIELD_PATH))			{ TRACE("error: %s is not existing.\n", CONCATENATE_FIELD_PATH); exit(EXIT_FAILURE); }
	}
	/////////////////////////////////////////////////////////////////////////////
	TRACE("\n");
	/////////////////////////////////////////////////////////////////////////////
	TRACE("out_folder = %s\n", out_folder);
	TRACE("tmp_folder = %s\n", tmp_folder);
	TRACE2("delete_tmp_folder = %d\n", (delete_tmp_folder)?1:0);
	/////////////////////////////////////////////////////////////////////////////
	TRACE("\n");
	/////////////////////////////////////////////////////////////////////////////
#ifdef _OPENMP
	if (num_omp_threads > 0) {
		omp_set_num_threads(num_omp_threads);
	}
	TRACE("num_omp_threads = %d\n", num_omp_threads);
#else
	TRACE("OpenMP is not supported.\n");
#endif
	TRACE("num_itk_threads = %d\n", num_itk_threads);
	TRACE("num_hop_threads = %d\n", num_hop_threads);
	/////////////////////////////////////////////////////////////////////////////
	TRACE("\n");
	/////////////////////////////////////////////////////////////////////////////
	TRACE("atlas_template_file = %s\n", atlas_template_file);
	TRACE("atlas_prior_list = %s\n", atlas_prior_list);
#ifdef USE_MASK_WEIGHT
	TRACE2("atlas_mask_weight = %f\n", atlas_mask_weight);
#else
	TRACE("atlas_mask_file = %s\n", atlas_mask_file);
#endif
	TRACE("atlas_label_map_s_file = %s\n", atlas_label_map_s_file);
	TRACE("atlas_label_map_s_img_file = %s\n", atlas_label_map_s_img_file);
	//
	TRACE2("use_default_label_map_s = %d\n", use_default_label_map_s);
	TRACE2("apply_mask = %d\n", (apply_mask)?1:0);
	TRACE2("use_masked_images_for_posteriors = %d\n", (use_masked_images_for_posteriors)?1:0);
#ifdef USE_PROB_PRIOR
	TRACE2("prob_weight = %f\n", prob_weight);
	TRACE2("prob_wt_t1ce_only = %d\n", (prob_wt_t1ce_only)?1:0);
#endif
#ifdef USE_ED_NON_WM_PROB
	TRACE2("ed_non_wm_prob = %f\n", ed_non_wm_prob);	
#endif
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE) || defined(USE_MEAN_SHIFT_UPDATE_STEP)
	TRACE2("mean_shift_update = %d\n", (mean_shift_update)?1:0);	
	TRACE2("ms_tumor = %d\n", (ms_tumor)?1:0);	
	TRACE2("ms_edema = %d\n", (ms_edema)?1:0);	
#endif
#ifdef UPDATE_DEFORMATION_FIELD
	TRACE2("update_deform = %d\n", (update_deform)?1:0);
	TRACE2("lambda_D = %f\n", lambda_D);
#endif
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	TRACE2("use_ne_as_cavity = %d\n", (use_ne_as_cavity)?1:0);
	TRACE2("cavity_label_val = %d\n", cavity_label_val);
#endif
	TRACE2("hop_sync_eval = %d\n", (hop_sync_eval)?1:0);
	TRACE2("hop_random_order = %d\n", (hop_random_order)?1:0);
	TRACE2("align_image_index = %d\n", align_image_index);
	TRACE2("check_orientation = %d\n", check_orientation);
	TRACE2("jsr_iter = %d\n", jsr_iter);	
	TRACE2("erode_bound_num = %d\n", erode_bound_num);
#ifdef USE_LABEL_NOISE_REDUCTION
	TRACE2("l_noise_reduction = %d\n", (l_noise_reduction)?1:0);
#endif
#ifdef USE_LABEL_CORRECTION
	TRACE2("lc_edema = %d\n", (lc_edema)?1:0);
	TRACE2("lc_tumor = %d\n", (lc_tumor)?1:0);
#endif
	/////////////////////////////////////////////////////////////////////////////
	TRACE("\n");
	/////////////////////////////////////////////////////////////////////////////
#ifdef USE_8_PRIORS
	TRACE2("USE_8_PRIORS\n");
#endif
#ifdef USE_9_PRIORS
	TRACE2("USE_9_PRIORS\n");
#endif
#ifdef USE_10_PRIORS
	TRACE2("USE_10_PRIORS\n");
#endif
#ifdef USE_10A_PRIORS
	TRACE2("USE_10A_PRIORS\n");
#endif
#ifdef USE_11_PRIORS
	TRACE2("USE_11_PRIORS\n");
#endif
#ifdef USE_3_IMAGES
	TRACE2("USE_3_IMAGES\n");
#endif
#ifdef USE_4_IMAGES
	TRACE2("USE_4_IMAGES\n");
#endif
#ifdef USE_5_IMAGES
	TRACE2("USE_5_IMAGES\n");
#endif
#ifdef USE_OPTIM_DG
	TRACE2("USE_OPTIM_DG\n");
#endif
#ifdef USE_OPTIM_LIMIT_XYZ_RANGES
	TRACE2("USE_OPTIM_LIMIT_XYZ_RANGES\n");
#endif
#ifdef USE_OPTIM_UPDATE_C
	TRACE2("USE_OPTIM_UPDATE_C\n");
#endif
#ifdef USE_OPTIM_UPDATE_T
	TRACE2("USE_OPTIM_UPDATE_T\n");
#endif
	TRACE2("OPTIM_DW_MIN = %g\n", OPTIM_DW_MIN * 1e-6);
	TRACE2("OPTIM_DW_MAX = %g\n", OPTIM_DW_MAX * 1e-6);
#ifdef USE_HOPS_INITIAL_X
	TRACE2("USE_HOPS_INITIAL_X\n");
#endif
#ifdef USE_HOPS_NO_SYNC_EVAL
	TRACE2("USE_HOPS_NO_SYNC_EVAL\n");
#ifdef USE_HOPS_MAX_RETURN
	TRACE2("USE_HOPS_MAX_RETURN\n");
#endif
#endif
#ifdef USE_HOPS_RANDOM_ORDER
	TRACE2("USE_HOPS_RANDOM_ORDER\n");
#endif
#ifdef USE_DISCARD_ZERO_AREA
	TRACE2("USE_DISCARD_ZERO_AREA\n");
#endif
#ifdef USE_WARPED_BG
	TRACE2("USE_WARPED_BG\n");
#endif
#ifdef USE_COST_BG
	TRACE2("USE_COST_BG\n");
#endif
#ifdef USE_MASK_WEIGHT
	TRACE2("USE_MASK_WEIGHT\n");
#endif
#ifdef USE_MASK_G
	TRACE2("USE_MASK_G\n");
#endif
#ifdef USE_PROB_PRIOR
	TRACE2("USE_PROB_PRIOR\n");
#endif
#ifdef PROB_ESTIMATE_T
	TRACE2("PROB_ESTIMATE_T\n");
#endif
#ifdef PROB_PRIOR_USE_PREV_REGION
	TRACE2("PROB_PRIOR_USE_PREV_REGION\n");
#endif
#ifdef PROB_SMOOTHING
	TRACE2("PROB_SMOOTHING\n");
#endif
#ifdef PROB_LOAD_TU_ONLY
	TRACE2("PROB_LOAD_TU_ONLY\n");
#endif
#ifdef USE_ED_TU_DISP
	TRACE2("USE_ED_TU_DISP\n");
#endif
#ifdef USE_ED_TU_DENS
	TRACE2("USE_ED_TU_DENS\n");
#endif
#ifdef USE_OPTIM_ED
	TRACE2("USE_OPTIM_ED\n");
#endif
#ifdef USE_ED_FIND_RANGE
	TRACE2("USE_ED_FIND_RANGE\n");
#endif
#ifdef USE_ED_FIND_RANGE_MD
	TRACE2("USE_ED_FIND_RANGE_MD\n");
#endif
#ifdef USE_ED_TU_NONE
	TRACE2("USE_ED_TU_NONE\n");
#endif
#ifdef USE_ED_TU_DISP_NONE
	TRACE2("USE_ED_TU_DISP_NONE\n");
#endif
#ifdef USE_ED_TU_DENS_NONE
	TRACE2("USE_ED_TU_DENS_NONE\n");
#endif
#ifdef USE_ED_TU_DENS_NONE_MIX
	TRACE2("USE_ED_TU_DENS_NONE_MIX\n");
#endif
#if defined(USE_ED_TU_DENS_NONE) || defined(USE_ED_TU_DENS_NONE_MIX)
	TRACE2("ED_TU_DENS_TH = %g\n", ED_TU_DENS_TH);
#endif
#ifdef USE_ED_TU_DW
	TRACE2("USE_ED_TU_DW\n");
#endif
#ifdef USE_ED_NON_WM_PROB
	TRACE2("USE_ED_NON_WM_PROB\n");
#endif
#ifdef USE_LIKE_TU_T1CE
	TRACE2("USE_LIKE_TU_T1CE\n");
#endif
#ifdef USE_OUTLIER_REJECTION_INIT_MEANS
	TRACE2("USE_OUTLIER_REJECTION_INIT_MEANS\n");
#endif
#ifdef USE_INIT_MEANS_CSF
	TRACE2("USE_INIT_MEANS_CSF\n");
#endif
#ifdef USE_INIT_MEANS_VS
	TRACE2("USE_INIT_MEANS_VS\n");
#endif
#ifdef USE_INIT_MEANS_VT
	TRACE2("USE_INIT_MEANS_VT\n");
#endif
#ifdef USE_INIT_MEANS_ED
	TRACE2("USE_INIT_MEANS_ED\n");
#endif
#ifdef USE_INIT_MEANS_GM
	TRACE2("USE_INIT_MEANS_GM\n");
#endif
#ifdef USE_INIT_MEANS_WM
	TRACE2("USE_INIT_MEANS_WM\n");
#endif
#ifdef USE_WM_UPDATE_SKIP_ED_REGION
	TRACE2("USE_WM_UPDATE_SKIP_ED_REGION\n");
#endif
#ifdef USE_OUTLIER_REJECTION_LAMBDA
	TRACE2("USE_OUTLIER_REJECTION_LAMBDA\n");
#endif
#ifdef USE_MEAN_SHIFT_UPDATE
	TRACE2("USE_MEAN_SHIFT_UPDATE\n");
#endif
#ifdef USE_MEAN_SHIFT_UPDATE_ONCE
	TRACE2("USE_MEAN_SHIFT_UPDATE_ONCE\n");
#endif
#ifdef USE_MEAN_SHIFT_UPDATE_STEP
	TRACE2("USE_MEAN_SHIFT_UPDATE_STEP\n");
#endif
#ifdef USE_TU_PRIOR_CUTOFF
	TRACE2("USE_TU_PRIOR_CUTOFF\n");
	TRACE2("TU_PRIOR_CUTOFF_TU_TH = %f\n", TU_PRIOR_CUTOFF_TU_TH);
	TRACE2("TU_PRIOR_CUTOFF_NCR_TH = %f\n", TU_PRIOR_CUTOFF_NCR_TH);
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	TRACE2("TU_PRIOR_CUTOFF_NE_TH = %f\n", TU_PRIOR_CUTOFF_NE_TH);
#endif
	TRACE2("TU_PRIOR_CUTOFF_ED_TH = %f\n", TU_PRIOR_CUTOFF_ED_TH);
#endif
#ifdef USE_ED_EXPAND
	TRACE2("USE_ED_EXPAND\n");
#endif
#ifdef USE_PL_COST
	TRACE2("USE_PL_COST\n");
#endif
#ifdef USE_TU_SIMUL_H0
	TRACE2("USE_TU_SIMUL_H0\n");
#endif
#ifdef USE_LIKE_TU_MAX
	TRACE2("USE_LIKE_TU_MAX\n");
#endif	
#ifdef USE_TU_MULTI_MAX
	TRACE2("USE_TU_MULTI_MAX\n");
#endif	
#ifdef USE_TU_MULTI_ADD
	TRACE2("USE_TU_MULTI_ADD\n");
#endif	
#ifdef USE_COMPUTE_Q_ESTIMATE_VARIANCES
	TRACE2("USE_COMPUTE_Q_ESTIMATE_VARIANCES\n");
#endif	
#ifdef RWR_BG_PROB
	TRACE2("RWR_BG_PROB\n");
#ifdef RWR_USE_POST_TH
	TRACE2("RWR_USE_POST_TH\n");
	TRACE2("RWR_POST_TH = %f\n", RWR_POST_TH);
#endif
#ifdef RWR_USE_POST_COMP
	TRACE2("RWR_USE_POST_COMP\n");
#endif
#endif
#ifdef RWR_ESTIMATE_RE_M
	TRACE2("RWR_ESTIMATE_RE_M\n");
#endif
	TRACE2("SIMUL_ITER_STEP_0 = %d\n", SIMUL_ITER_STEP_0);
	TRACE2("SIMUL_ITER_STEP_1 = %d\n", SIMUL_ITER_STEP_1);
	TRACE2("SIMUL_ITER_STEP_N = %d\n", SIMUL_ITER_STEP_N);
#ifdef USE_JSR_GLISTR
	TRACE2("USE_JSR_GLISTR\n");
	TRACE2("JSR_ITER_STEP_0 = %d\n", JSR_ITER_STEP_0);
	TRACE2("JSR_ITER_STEP_N = %d\n", JSR_ITER_STEP_N);
#ifdef USE_JSR_ELASTIC
	TRACE2("USE_JSR_ELASTIC\n");
#endif
#ifdef USE_JSR_FLUID
	TRACE2("USE_JSR_FLUID\n");
#endif
#ifdef USE_JSR_MULTI_LEVEL
	TRACE2("USE_JSR_MULTI_LEVEL\n");
#endif
#ifdef USE_JSR_NORMALIZE_UPDATE_FIELD
	TRACE2("USE_JSR_NORMALIZE_UPDATE_FIELD\n");
#endif
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	TRACE2("USE_JSR_WEGHT_IMAGE_PYRAMID\n");
#endif
#ifdef USE_JSR_CHECK_PRIOR_MAG
	TRACE2("USE_JSR_CHECK_PRIOR_MAG\n");
#endif
#ifdef USE_JSR_PROB_WEIGHT_OLD
	TRACE2("USE_JSR_PROB_WEIGHT_OLD\n");
#endif
#ifdef USE_JSR_NO_WARPED_BG
	TRACE2("USE_JSR_NO_WARPED_BG\n");
#endif
#endif
	//
#ifdef USE_SUM_EPSS
	TRACE2("USE_SUM_EPSS\n");
#endif
#ifdef USE_SUM_EPSS2
	TRACE2("USE_SUM_EPSS2\n");
#endif
	//
#ifdef USE_LABEL_NOISE_REDUCTION
	TRACE2("USE_LABEL_NOISE_REDUCTION\n");
	TRACE2("CC_NUM_TH = %d\n", CC_NUM_TH);
#endif
#ifdef USE_LABEL_CORRECTION
	TRACE2("USE_LABEL_CORRECTION\n");
#endif	
	//
#ifdef UPDATE_DEFORMATION_FIELD
	TRACE2("UPDATE_DEFORMATION_FIELD\n");
#ifdef USE_SYMM_REG
	TRACE2("USE_SYMM_REG\n");
#endif	
#ifdef USE_CC_NCC
	TRACE2("USE_CC_NCC\n");
#endif	
#ifdef USE_DISCRETE_OPTIMIZATION
	TRACE2("USE_DISCRETE_OPTIMIZATION\n");
#endif	
#ifdef USE_CONTINUOUS_OPTIMIZATION
	TRACE2("USE_CONTINUOUS_OPTIMIZATION\n");
	TRACE2("CONT_OPT_ITER_LEVEL_N = %d\n", CONT_OPT_ITER_LEVEL_N);
	TRACE2("CONT_OPT_ITER_LEVEL_0 = %d\n", CONT_OPT_ITER_LEVEL_0);
#endif	
#ifdef USE_INIT_FIELD
	TRACE2("USE_INIT_FIELD\n");
#endif	
#ifdef USE_INIT_FIELD_WEIGHT_ABNORMAL_REGION
	TRACE2("USE_INIT_FIELD_WEIGHT_ABNORMAL_REGION\n");
#endif
#endif
	/////////////////////////////////////////////////////////////////////////////
	TRACE2("\n");
	/////////////////////////////////////////////////////////////////////////////
  clock_end = clock();
  //std::cout << "===Time stamp 00: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
  std::cout << "===Percentage Done : 00.00\n";
	/////////////////////////////////////////////////////////////////////////////
	// load files
	/////////////////////////////////////////////////////////////////////////////
	{
		char str_path[1024];
		char str_tmp[1024];
		FILE* fp;
		//
		str_strip_file((char*)scan_image_list, str_path);
		fp = fopen(scan_image_list, "r");
		if (fp == NULL) {
			TRACE("Cannot open %s\n", scan_image_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfImageChannels; i++) {
			fscanf(fp, "%s", str_tmp);
			sprintf(scan_image_files[i], "%s%s", str_path, str_tmp);
			{
				FILE *fp_test;
				fp_test = fopen(scan_image_files[i], "r");
				if (fp_test == NULL) {
					TRACE("Cannot open %s\n", scan_image_files[i]);
					exit(EXIT_FAILURE);
				}
				fclose(fp_test);
			}
		}
		fclose(fp);
		str_strip_file((char*)atlas_prior_list, str_path);
		fp = fopen(atlas_prior_list, "r");
		if (fp == NULL) {
			TRACE("Cannot open %s\n", atlas_prior_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfPriorChannels; i++) {
			fscanf(fp, "%s", str_tmp);
			sprintf(atlas_prior_files[i], "%s%s", str_path, str_tmp);
			{
				FILE *fp_test;
				fp_test = fopen(atlas_prior_files[i], "r");
				if (fp_test == NULL) {
					TRACE("Cannot open %s\n", atlas_prior_files[i]);
					exit(EXIT_FAILURE);
				}
				fclose(fp_test);
			}
		}
		fclose(fp);
		//
#ifdef USE_MASK_WEIGHT
		sprintf(atlas_mask_file, "%s%satlas_mask.nii.gz", tmp_folder, DIR_SEP);
#endif
		//
		if (b_seed_info) {
			fp = fopen(scan_init_seed, "r");
			if (fp == NULL) {
				TRACE("Cannot open %s\n", scan_init_seed);
				exit(EXIT_FAILURE);
			}
			fscanf(fp, "%s", str_tmp);
			scan_seeds_num = atoi(str_tmp);
			if (scan_seeds_num > MaxNumberOFTumorSeeds) {
				TRACE("scan_seeds_num = %d > %d\n", scan_seeds_num, MaxNumberOFTumorSeeds);
				exit(EXIT_FAILURE);
			}
			TRACE("scan_seeds_num = %d\n", scan_seeds_num);
			for (i = 0; i < scan_seeds_num; i++) {
				fscanf(fp, "%lf %lf %lf %lf", &scan_seeds_info[i][0], &scan_seeds_info[i][1], &scan_seeds_info[i][2], &scan_seeds_info[i][3]);
				TRACE("seed %d: %f %f %f %f\n", i, scan_seeds_info[i][0], scan_seeds_info[i][1], scan_seeds_info[i][2], scan_seeds_info[i][3]);
			}
			fclose(fp);
		}
		//
		if (b_point_info) {
			fp = fopen(scan_init_point, "r");
			if (fp == NULL) {
				TRACE("Cannot open %s\n", scan_init_point);
				exit(EXIT_FAILURE);
			}
			scan_points_num = 0;
			while (TRUE) {
				int res;
				char class_id[1024];
				double fx, fy, fz;
				//
				res = fscanf(fp, "%s %lf %lf %lf", class_id, &fx, &fy, &fz);
				if (res == 0 || res == EOF) { break; }
				//
				for (j = 0; j < NumberOfPriorChannels; j++) {
#if defined(USE_8_PRIORS)
					if ((strcmp(class_id, label[j]) == 0) ||
						(strcmp(class_id, label2[j]) == 0) ||
						(strcmp(class_id, label3[j]) == 0)) {
#else
					if (strcmp(class_id, label[j]) == 0) {
#endif
						scan_points_info[scan_points_num][0] = fx;
						scan_points_info[scan_points_num][1] = fy;
						scan_points_info[scan_points_num][2] = fz;
						scan_points_info[scan_points_num][3] = j;
						scan_points_num++;
						if (scan_points_num > MaxNumberOFPoints) {
							TRACE("scan_points_num = %d > %d\n", scan_points_num, MaxNumberOFPoints);
							exit(EXIT_FAILURE);
						}
						//
						b_valid_label[j] = 1;
						//
						break;
					}
				}
				if (j == NumberOfPriorChannels) {
					TRACE("class id is wrong");
					exit(EXIT_FAILURE);
				}
			}
			for (i = 0; i < scan_points_num; i++) {
				TRACE("point %d: %s %f %f %f\n", i, label[(int)(scan_points_info[i][3] + 0.1)], scan_points_info[i][0], scan_points_info[i][1], scan_points_info[i][2]);
			}
			fclose(fp);
		}
		for (i = 0; i < NumberOfPriorChannels; i++) {
			TRACE2("b_valid_label[%s] = %d\n", label[i], b_valid_label[i]);
		}
		//
		sprintf(scan_to_atlas_mat_file, "%s%sscan_to_atlas.mat", tmp_folder, DIR_SEP);
		sprintf(atlas_to_scan_mat_file, "%s%satlas_to_scan.mat", tmp_folder, DIR_SEP);
		//
		for (i = 0; i < NumberOfImageChannels; i++) {
			sprintf(scan_image_masked_files[i], "%s%sscan_image_masked_%d.nii.gz", tmp_folder, DIR_SEP, i);
			//
			sprintf(scan_atlas_reg_image_files[i], "%s%sscan_atlas_reg_image_%d.nii.gz", tmp_folder, DIR_SEP, i);
			sprintf(scan_atlas_reg_image_masked_files[i], "%s%sscan_atlas_reg_image_masked_%d.nii.gz", tmp_folder, DIR_SEP, i);
#ifdef USE_MASK_G
			sprintf(scan_atlas_reg_image_masked_g_files[i], "%s%sscan_atlas_reg_image_masked_g_%d.nii.gz", tmp_folder, DIR_SEP, i);
#endif
		}
		sprintf(scan_atlas_reg_image_list, "%s%sscan_atlas_reg_image.lst", tmp_folder, DIR_SEP);
		fp = fopen(scan_atlas_reg_image_list, "w");
		if (fp == NULL) {
			TRACE("Cannot make %s\n", scan_atlas_reg_image_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfImageChannels; i++) {
			fprintf(fp, "scan_atlas_reg_image_%d.nii.gz\n", i);
		}
		fclose(fp);
		sprintf(scan_atlas_reg_image_masked_list, "%s%sscan_atlas_reg_image_masked.lst", tmp_folder, DIR_SEP);
		fp = fopen(scan_atlas_reg_image_masked_list, "w");
		if (fp == NULL) {
			TRACE("Cannot make %s\n", scan_atlas_reg_image_masked_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfImageChannels; i++) {
			fprintf(fp, "scan_atlas_reg_image_masked_%d.nii.gz\n", i);
		}
		fclose(fp);
#ifdef USE_MASK_G
		sprintf(scan_atlas_reg_image_masked_g_list, "%s%sscan_atlas_reg_image_masked_g.lst", tmp_folder, DIR_SEP);
		fp = fopen(scan_atlas_reg_image_masked_g_list, "w");
		if (fp == NULL) {
			TRACE("Cannot make %s\n", scan_atlas_reg_image_masked_g_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfImageChannels; i++) {
			fprintf(fp, "scan_atlas_reg_image_masked_g_%d.nii.gz\n", i);
		}
		fclose(fp);
#endif
		//
		sprintf(scan_atlas_reg_label_map_file, "%s%sscan_atlas_reg_label_map.nii.gz", tmp_folder, DIR_SEP);
		//
		for (i = 0; i < NumberOfPriorChannels; i++) {
			sprintf(scan_atlas_reg_mass_prior_files[i], "%s%sscan_atlas_reg_mass_prior_%d.nii.gz", tmp_folder, DIR_SEP, i);
			sprintf(scan_atlas_reg_warp_prior_files[i], "%s%sscan_atlas_reg_warp_prior_%d.nii.gz", tmp_folder, DIR_SEP, i);
			sprintf(scan_atlas_reg_posterior_files[i], "%s%sscan_atlas_reg_posterior_%d.nii.gz", tmp_folder, DIR_SEP, i);
#ifdef USE_PROB_PRIOR
			sprintf(scan_atlas_reg_prob_files[i], "%s%sscan_atlas_reg_prob_%d.nii.gz", tmp_folder, DIR_SEP, i);
#endif
		}
		sprintf(scan_atlas_reg_mass_prior_list, "%s%sscan_atlas_reg_mass_prior.lst", tmp_folder, DIR_SEP);
		fp = fopen(scan_atlas_reg_mass_prior_list, "w");
		if (fp == NULL) {
			TRACE("Cannot make %s\n", scan_atlas_reg_mass_prior_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfPriorChannels; i++) {
			fprintf(fp, "scan_atlas_reg_mass_prior_%d.nii.gz\n", i);
		}
		fclose(fp);
		sprintf(scan_atlas_reg_warp_prior_list, "%s%sscan_atlas_reg_warp_prior.lst", tmp_folder, DIR_SEP);
		fp = fopen(scan_atlas_reg_warp_prior_list, "w");
		if (fp == NULL) {
			TRACE("Cannot make %s\n", scan_atlas_reg_warp_prior_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfPriorChannels; i++) {
			fprintf(fp, "scan_atlas_reg_warp_prior_%d.nii.gz\n", i);
		}
		fclose(fp);
		sprintf(scan_atlas_reg_posterior_list, "%s%sscan_atlas_reg_posterior.lst", tmp_folder, DIR_SEP);
		fp = fopen(scan_atlas_reg_posterior_list, "w");
		if (fp == NULL) {
			TRACE("Cannot make %s\n", scan_atlas_reg_posterior_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfPriorChannels; i++) {
			fprintf(fp, "scan_atlas_reg_posterior_%d.nii.gz\n", i);
		}
		fclose(fp);
#ifdef USE_PROB_PRIOR
		sprintf(scan_atlas_reg_prob_list, "%s%sscan_atlas_reg_prob.lst", tmp_folder, DIR_SEP);
		fp = fopen(scan_atlas_reg_prob_list, "w");
		if (fp == NULL) {
			TRACE("Cannot make %s\n", scan_atlas_reg_prob_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfPriorChannels; i++) {
			fprintf(fp, "scan_atlas_reg_prob_%d.nii.gz\n", i);
		}
		fclose(fp);
#endif
		sprintf(scan_init_means_file, "%s%sscan_init_means.txt", tmp_folder, DIR_SEP);
		sprintf(scan_init_variances_file, "%s%sscan_init_variances.txt", tmp_folder, DIR_SEP);
		sprintf(scan_means_file, "%s%sscan_means.txt", tmp_folder, DIR_SEP);
		sprintf(scan_variances_file, "%s%sscan_variances.txt", tmp_folder, DIR_SEP);
		//
		sprintf(scan_u_hdr_file, "%s%sscan_u.mhd", tmp_folder, DIR_SEP);
		sprintf(scan_h_hdr_file, "%s%sscan_h.mhd", tmp_folder, DIR_SEP);
		sprintf(scan_h0_hdr_file, "%s%sscan_h0.mhd", tmp_folder, DIR_SEP);
		//
		for (i = 0; i < NumberOfPriorChannels; i++) {
			sprintf(scan_prior_files[i], "%s%sscan_prior_%d.nii.gz", out_folder, DIR_SEP, i);
			sprintf(scan_posterior_files[i], "%s%sscan_posterior_%d.nii.gz", out_folder, DIR_SEP, i);
		}
		sprintf(scan_prior_list, "%s%sscan_prior.lst", out_folder, DIR_SEP);
		fp = fopen(scan_prior_list, "w");
		if (fp == NULL) {
			TRACE("Cannot make %s\n", scan_prior_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfPriorChannels; i++) {
			fprintf(fp, "scan_prior_%d.nii.gz\n", i);
		}
		fclose(fp);
		sprintf(scan_posterior_list, "%s%sscan_posterior.lst", out_folder, DIR_SEP);
		fp = fopen(scan_posterior_list, "w");
		if (fp == NULL) {
			TRACE("Cannot make %s\n", scan_posterior_list);
			exit(EXIT_FAILURE);
		}
		for (i = 0; i < NumberOfPriorChannels; i++) {
			fprintf(fp, "scan_posterior_%d.nii.gz\n", i);
		}
		fclose(fp);
		//
		sprintf(scan_label_map_file, "%s%sscan_label_map.nii.gz", out_folder, DIR_SEP);
		//
		sprintf(scales_file, "%s%sscales.txt", tmp_folder, DIR_SEP);
		sprintf(simulator_input_file, "%s%sForwardSolver.in", tmp_folder, DIR_SEP);
		sprintf(hops_file, "%s%sOptim.txt", tmp_folder, DIR_SEP);
		//
		sprintf(scan_mask_file, "%s%sscan_mask.nii.gz", tmp_folder, DIR_SEP);
		sprintf(scan_atlas_reg_mask_file, "%s%sscan_atlas_reg_mask.nii.gz", tmp_folder, DIR_SEP);
	}
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////


	if (b_run_test) {
		/////////////////////////////////////////////////////////////////////////////
#ifdef USE_TRACE
		if (g_fp_trace != NULL) {
			fclose(g_fp_trace);
			g_fp_trace = NULL;
		}
		g_bTrace = FALSE;
		g_bTraceStdOut = FALSE;
		g_verbose = 0;
#endif
		/////////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////////
		//DeleteAll(tmp_folder, TRUE);
		//DeleteAll(out_folder, TRUE);
		/////////////////////////////////////////////////////////////////////////////

		exit(EXIT_SUCCESS);
	}

  clock_end = clock();
  //std::cout << "===Time stamp 01: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
	/////////////////////////////////////////////////////////////////////////////
	// erode brain boundary & adjust the maximum intensity value to 255
	/////////////////////////////////////////////////////////////////////////////
	if (!IsFileExist(scan_image_masked_files[0])) {
		FVolume scan_images[NumberOfImageChannels];

		for (i = 0; i < NumberOfImageChannels; i++) {
			scan_images[i].load(scan_image_files[i], 1);
		}

		// erode brain boundary
		if (erode_bound_num > 0) {
			FVolume img, tmp;
			int vd_x, vd_y, vd_z, vd_x_1, vd_y_1, vd_z_1;
			int iter;
			int* hist;
			int hist_num, hist_sum, hist_sum_th, hist_acc;
			float val, val_max, val_th;

			// assuming flair image
			img.load(scan_image_files[NumberOfImageChannels-1], 1);

			vd_x = img.m_vd_x;
			vd_y = img.m_vd_y;
			vd_z = img.m_vd_z;
			vd_x_1 = vd_x - 1;
			vd_y_1 = vd_y - 1;
			vd_z_1 = vd_z - 1;

			val_max = 0;
			for (k = 0; k < vd_z; k++) {
				for (j = 0; j < vd_y; j++) {
					for (i = 0; i < vd_x; i++) {
						val = img.m_pData[k][j][i][0];
						if (val > val_max) {
							val_max = val;
						}
					}
				}
			}

			hist_num = (int)val_max + 1;
			hist = (int*)malloc(hist_num * sizeof(int));
			for (i = 0; i < hist_num; i++) {
				hist[i] = 0;
			}
			hist_sum = 0;
			for (k = 0; k < vd_z; k++) {
				for (j = 0; j < vd_y; j++) {
					for (i = 0; i < vd_x; i++) {
						val = img.m_pData[k][j][i][0];
						if (val <= 0) {
							continue;
						}
						//
						hist[(int)val]++;
						hist_sum++;
					}
				}
			}
			hist_sum_th = (int)(0.995 * hist_sum);
			hist_acc = 0;
			val_th = val_max;
			for (i = 0; i < hist_num; i++) {
				hist_acc += hist[i];
				if (hist_acc > hist_sum_th) {
					val_th = i;
					break;
				}
			}
			
			tmp.allocate(vd_x, vd_y, vd_z);

			for (iter = 0; iter < erode_bound_num; iter++) {
				tmp.copy(img);
				for (k = 1; k < vd_z_1; k++) {
					for (j = 1; j < vd_y_1; j++) {
						for (i = 1; i < vd_x_1; i++) {
							if (tmp.m_pData[k][j][i][0] > val_th) {
								if (tmp.m_pData[k  ][j  ][i-1][0] > 0 &&
									tmp.m_pData[k  ][j  ][i+1][0] > 0 &&
									tmp.m_pData[k  ][j-1][i  ][0] > 0 &&
									tmp.m_pData[k  ][j+1][i  ][0] > 0 &&
									tmp.m_pData[k-1][j  ][i  ][0] > 0 &&
									tmp.m_pData[k+1][j  ][i  ][0] > 0) {
								} else {
									img.m_pData[k][j][i][0] = 0;
								}
							}
						}
					}
				}
			}

			for (k = 0; k < vd_z; k++) {
				for (j = 0; j < vd_y; j++) {
					for (i = 0; i < vd_x; i++) {
						if (img.m_pData[k][j][i][0] == 0) {
							for (l = 0; l < NumberOfImageChannels; l++) {
								scan_images[l].m_pData[k][j][i][0] = 0;
							}
						}
					}
				}
			}

			free(hist);
		}

		// adjust the maximum value of images to 255
		{
			int vd_x, vd_y, vd_z;
			
			vd_x = scan_images[0].m_vd_x;
			vd_y = scan_images[0].m_vd_y;
			vd_z = scan_images[0].m_vd_z;
		
			for (l = 0; l < NumberOfImageChannels; l++) {
				float scale_min = 0;
				float scale_max = 0;
				float scale, val;

				for (k = 0; k < vd_z; k++) {
					for (j = 0; j < vd_y; j++) {
						for (i = 0; i < vd_x; i++) {
							val = scan_images[l].m_pData[k][j][i][0];
							if (val > scale_max) {
								scale_max = val;
							}
							if (val < scale_min) {
								scale_min = val;
							}
						}
					}
				}

				TRACE2("%s: scale_min = %f, scale_max = %f\n", scan_image_files[l], scale_min, scale_max);
				if (scale_min < 0) {
					TRACE("%s has minus values: check value ranges\n", scan_image_files[l]);
					exit(EXIT_FAILURE);
				}
				if (scale_max > 255.0) {
					TRACE("%s: adjust scale_max to 255\n", scan_image_files[l]);

					scale = 255.0 / scale_max;

					for (k = 0; k < vd_z; k++) {
						for (j = 0; j < vd_y; j++) {
							for (i = 0; i < vd_x; i++) {
								val = scan_images[l].m_pData[k][j][i][0] * scale;
								scan_images[l].m_pData[k][j][i][0] = val;
							}
						}
					}
				}
			}
		}

		for (i = 0; i < NumberOfImageChannels; i++) {
			scan_images[i].save(scan_image_masked_files[i], 1);
			ChangeNIIHeader(scan_image_masked_files[i], scan_image_files[0]);
			//
			scan_images[i].clear();
		}
	}
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////

  clock_end = clock();
  //std::cout << "===Time stamp 02: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
  std::cout << "===Percentage Done : 00.10\n";
	/////////////////////////////////////////////////////////////////////////////
	// affine registration to atlas
	/////////////////////////////////////////////////////////////////////////////
	if (!IsFileExist(scan_to_atlas_mat_file)) {
		putenv((char*)"FSLOUTPUTTYPE=NIFTI_GZ");
		//
		// register scan to atlas template
		{
			//sprintf(szCmdLine, "%s -in %s -ref %s -out %s -omat %s -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -datatype float",
			sprintf(szCmdLine, "%s -in %s -ref %s -out %s -omat %s -cost mutualinfo -searchcost mutualinfo -searchrx -%d %d -searchry -%d %d -searchrz -%d %d -datatype float",
				FLIRT_PATH, scan_image_masked_files[align_image_index], atlas_template_file, scan_atlas_reg_image_files[align_image_index], scan_to_atlas_mat_file, 
				flirt_search_angle, flirt_search_angle, flirt_search_angle, flirt_search_angle, flirt_search_angle, flirt_search_angle);
			TRACE("%s\n", szCmdLine);
			//
			if (!ExecuteProcess(szCmdLine)) {
				TRACE("failed\n");
				exit(EXIT_FAILURE);
			}
		}
		//
		// generate atlas reg images for scan
		for (i = 0; i < NumberOfImageChannels; i++) {
			sprintf(szCmdLine, "%s -in %s -ref %s -out %s -init %s -datatype float -applyxfm", 
				FLIRT_PATH, scan_image_masked_files[i], atlas_template_file, scan_atlas_reg_image_files[i], scan_to_atlas_mat_file);
			TRACE("%s\n", szCmdLine);
			//
			if (!ExecuteProcess(szCmdLine)) {
				TRACE("failed\n");
				exit(EXIT_FAILURE);
			}
		}
	}
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////

  clock_end = clock();
  //std::cout << "===Time stamp 03: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
	/////////////////////////////////////////////////////////////////////////////
	// apply brain mask
	/////////////////////////////////////////////////////////////////////////////
	if (!IsFileExist(scan_atlas_reg_image_masked_files[0])) {
		if (apply_mask) {
#ifdef USE_MASK_WEIGHT
			// create mask using atlas_mask_weight
			{
				FVolume atlas_prior_BG;
				int vd_x, vd_y, vd_z;

				atlas_prior_BG.load(atlas_prior_files[BG], 1);

				vd_x = atlas_prior_BG.m_vd_x;
				vd_y = atlas_prior_BG.m_vd_y;
				vd_z = atlas_prior_BG.m_vd_z;

				atlas_mask.allocate(vd_x, vd_y, vd_z);

				for (k = 0; k < vd_z; k++) {
					for (j = 0; j < vd_y; j++) {
						for (i = 0; i < vd_x; i++) {
							if (atlas_prior_BG.m_pData[k][j][i][0] < atlas_mask_weight) {
								atlas_mask.m_pData[k][j][i][0] = 1;
							} else {
								atlas_mask.m_pData[k][j][i][0] = 0;
							}
						}
					}
				}

				atlas_mask.save(atlas_mask_file, 1);
				ChangeNIIHeader(atlas_mask_file, atlas_prior_files[BG]);
			}
#else
			// load mask
			atlas_mask.load((char*)atlas_mask_file, 1);
#endif
			// apply brain mask
			for (i = 0; i < NumberOfImageChannels; i++) {
				if (!scan_atlas_reg_images[i].load(scan_atlas_reg_image_files[i], 1)) {
					TRACE("Cannot load %s\n", scan_atlas_reg_image_files[i]);
					exit(EXIT_FAILURE);
				}
				scan_atlas_reg_images_masked[i].allocate(scan_atlas_reg_images[i].m_vd_x, scan_atlas_reg_images[i].m_vd_y, scan_atlas_reg_images[i].m_vd_z);
				//
				for (n = 0; n < scan_atlas_reg_images[i].m_vd_z; n++) {
					for (m = 0; m < scan_atlas_reg_images[i].m_vd_y; m++) {
						for (l = 0; l < scan_atlas_reg_images[i].m_vd_x; l++) {
							if (atlas_mask.m_pData[n][m][l][0] != 0) {
								scan_atlas_reg_images_masked[i].m_pData[n][m][l][0] = scan_atlas_reg_images[i].m_pData[n][m][l][0];
							} else {
								scan_atlas_reg_images_masked[i].m_pData[n][m][l][0] = 0;
							}
						}
					}
				}
				//
				scan_atlas_reg_images_masked[i].save(scan_atlas_reg_image_masked_files[i], 1);
				ChangeNIIHeader(scan_atlas_reg_image_masked_files[i], scan_atlas_reg_image_files[i]);
				//
				scan_atlas_reg_images[i].clear();
				scan_atlas_reg_images_masked[i].clear();
			}
			//
			atlas_mask.clear();
		} else {
			for (i = 0; i < NumberOfImageChannels; i++) {
				CopyFile(scan_atlas_reg_image_files[i], scan_atlas_reg_image_masked_files[i], FALSE);
			}
		}
	}
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////

  clock_end = clock();
  //std::cout << "===Time stamp 04: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
  std::cout << "===Percentage Done : 00.22\n";
#ifdef USE_MASK_G
	/////////////////////////////////////////////////////////////////////////////
	// smoothing input files
	/////////////////////////////////////////////////////////////////////////////
	if (!IsFileExist(scan_atlas_reg_image_masked_g_files[0])) {
		for (i = 0; i < NumberOfImageChannels; i++) {
			FVolume vd, vd_g;
			
			vd.load(scan_atlas_reg_image_masked_files[i], 1);
			vd.GaussianSmoothing(vd_g, 0.125, 5);

			vd_g.save(scan_atlas_reg_image_masked_g_files[i], 1);
			ChangeNIIHeader(scan_atlas_reg_image_masked_g_files[i], scan_atlas_reg_image_files[i]);

			vd.clear();
			vd_g.clear();
		}
	}
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
#endif

  clock_end = clock();
  //std::cout << "===Time stamp 05: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
  std::cout << "===Percentage Done : 00.50\n";
	/////////////////////////////////////////////////////////////////////////////
	// transform seeds info
	/////////////////////////////////////////////////////////////////////////////
	{
		if (!TransformPoints(scan_image_masked_files[0], scan_atlas_reg_image_files[0], scan_to_atlas_mat_file, scan_seeds_info, scan_atlas_reg_seeds_info, scan_seeds_num, check_orientation)) {
			TRACE("TransformPoints - failed\n");
			exit(EXIT_FAILURE);
		}
  }
  clock_end = clock();
  //std::cout << "===Time stamp 06: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
  std::cout << "===Percentage Done : 00.51\n";
	/////////////////////////////////////////////////////////////////////////////
	// transform points info
	/////////////////////////////////////////////////////////////////////////////
	if (b_point_info) {
		if (!TransformPoints(scan_image_masked_files[0], scan_atlas_reg_image_files[0], scan_to_atlas_mat_file, scan_points_info, scan_atlas_reg_points_info, scan_points_num, check_orientation)) {
			TRACE("TransformPoints - failed\n");
			exit(EXIT_FAILURE);
		}
	}
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////

  clock_end = clock();
  //std::cout << "===Time stamp 07: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
	/////////////////////////////////////////////////////////////////////////////
	// joint segmentation & registration
	/////////////////////////////////////////////////////////////////////////////
	{
		int vd_x, vd_y, vd_z;
		float vd_dx, vd_dy, vd_dz;
		float vd_ox, vd_oy, vd_oz;

		{
			scan_atlas_reg_images[0].load(scan_atlas_reg_image_files[0], 1);
			//
			vd_x = scan_atlas_reg_images[0].m_vd_x;
			vd_y = scan_atlas_reg_images[0].m_vd_y;
			vd_z = scan_atlas_reg_images[0].m_vd_z;
			vd_dx = scan_atlas_reg_images[0].m_vd_dx;
			vd_dy = scan_atlas_reg_images[0].m_vd_dy;
			vd_dz = scan_atlas_reg_images[0].m_vd_dz;
			vd_ox = scan_atlas_reg_images[0].m_vd_ox;
			vd_oy = scan_atlas_reg_images[0].m_vd_oy;
			vd_oz = scan_atlas_reg_images[0].m_vd_oz;
			//
			scan_atlas_reg_images[0].clear();
		}

    std::cout << "===Percentage Done : 00.88\n";
		for (k = 0; k < scan_jsr_max_iter; k++) {
			char hops_file[1024];
			char solution_file[1024];
			//
			char cost_file[1024];
			char u_hdr_file[1024];
			char h_hdr_file[1024];
			char means_file[1024];
			char variances_file[1024];
#ifdef USE_MEAN_SHIFT_UPDATE_STEP
			char means_ms_file[1024];
			char variances_ms_file[1024];
#endif
			char atlas_reg_mass_prior_files[NumberOfPriorChannels][1024];
			char atlas_reg_warp_prior_files[NumberOfPriorChannels][1024];
			char atlas_reg_posterior_files[NumberOfPriorChannels][1024];
			char atlas_reg_label_map_file[1024];
#ifdef USE_PROB_PRIOR
			char atlas_reg_prob_list[1024];
			char atlas_reg_prob_files[NumberOfPriorChannels][1024];
#endif

			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			sprintf(hops_file, "%s%ss_Optim_%d.txt", tmp_folder, DIR_SEP, k);
			sprintf(solution_file, "%s%ss_solution_%d.txt", tmp_folder, DIR_SEP, k);
			//
			sprintf(cost_file, "%s%ss_cost_jsr_%d.txt", tmp_folder, DIR_SEP, k);
			sprintf(u_hdr_file, "%s%ss_u_jsr_%d.mhd", tmp_folder, DIR_SEP, k);
			sprintf(h_hdr_file, "%s%ss_h_jsr_%d.mhd", tmp_folder, DIR_SEP, k);
			sprintf(means_file, "%s%ss_means_jsr_%d.txt", tmp_folder, DIR_SEP, k);
			sprintf(variances_file, "%s%ss_variances_jsr_%d.txt", tmp_folder, DIR_SEP, k);
#ifdef USE_MEAN_SHIFT_UPDATE_STEP
			sprintf(means_ms_file, "%s%ss_means_ms_jsr_%d.txt", tmp_folder, DIR_SEP, k);
			sprintf(variances_ms_file, "%s%ss_variances_ms_jsr_%d.txt", tmp_folder, DIR_SEP, k);
#endif
			for (i = 0; i < NumberOfPriorChannels; i++) {
				sprintf(atlas_reg_mass_prior_files[i], "%s%ss_atlas_reg_mass_prior_jsr_%d_%d.nii.gz", tmp_folder, DIR_SEP, k, i);
				sprintf(atlas_reg_warp_prior_files[i], "%s%ss_atlas_reg_warp_prior_jsr_%d_%d.nii.gz", tmp_folder, DIR_SEP, k, i);
				sprintf(atlas_reg_posterior_files[i], "%s%ss_atlas_reg_posterior_jsr_%d_%d.nii.gz", tmp_folder, DIR_SEP, k, i);
#ifdef USE_PROB_PRIOR
				sprintf(atlas_reg_prob_files[i], "%s%ss_atlas_reg_prob_jsr_%d_%d.nii.gz", tmp_folder, DIR_SEP, k, i);
#endif
			}
			sprintf(atlas_reg_label_map_file, "%s%ss_atlas_reg_label_map_jsr_%d.nii.gz", tmp_folder, DIR_SEP, k);
#ifdef USE_PROB_PRIOR
			{
				FILE* fp;
				sprintf(atlas_reg_prob_list, "%s%ss_atlas_reg_prob_jsr_%d.lst", tmp_folder, DIR_SEP, k);
				fp = fopen(atlas_reg_prob_list, "w");
				if (fp == NULL) {
					TRACE("Cannot make %s\n", atlas_reg_prob_list);
					exit(EXIT_FAILURE);
				}
				for (i = 0; i < NumberOfPriorChannels; i++) {
					fprintf(fp, "s_atlas_reg_prob_jsr_%d_%d.nii.gz\n", k, i);
				}
				fclose(fp);
			}
#endif
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////


      clock_end = clock();
      //std::cout << "===Time stamp 08: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";

			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (k == 0) {
				// init means and variances
				if (b_point_info) {
					if (!InitializeMeansAndVariancesFromPoints(scan_points_info, scan_points_num, scan_image_masked_files, scan_MeanVector, scan_VarianceVector, check_orientation, b_valid_label)) {
						TRACE("InitializeMeansAndVariancesFromPoints - failed\n");
						exit(EXIT_FAILURE);
					}
					SaveMeansAndVariances(scan_init_means_file, scan_init_variances_file, &scan_MeanVector, &scan_VarianceVector);
				} else {
					if (!InitializeMeansAndVariances(scan_init_mean, &scan_MeanVector, &scan_VarianceVector)) {
						TRACE("InitializeMeansAndVariances - failed\n");
						exit(EXIT_FAILURE);
					}
				}
				SaveMeansAndVariances(scan_means_file, scan_variances_file, &scan_MeanVector, &scan_VarianceVector);
				//
				scan_InitMeanVector.clear();
				for (i = 0; i < NumberOfPriorChannels; i++) {
					scan_InitMeanVector.push_back(scan_MeanVector.at(i));
				}

				// make init deformation field
				{
					FVolume v0;
					v0.allocate(vd_x, vd_y, vd_z, 3);
					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								v0.m_pData[n][m][l][0] = 0;
								v0.m_pData[n][m][l][1] = 0;
								v0.m_pData[n][m][l][2] = 0;
							}
						}
					}
					SaveMHDData(NULL, scan_h_hdr_file, v0.m_pData, vd_x, vd_y, vd_z, 3, vd_dx, vd_dy, vd_dz, 0, 0, 0);
					SaveMHDData(NULL, scan_h0_hdr_file, v0.m_pData, vd_x, vd_y, vd_z, 3, vd_dx, vd_dy, vd_dz, 0, 0, 0);
				}

				for (i = 0; i < scan_seeds_num; i++) {
					tp_c[i][0] = scan_atlas_reg_seeds_info[i][0];
					tp_c[i][1] = scan_atlas_reg_seeds_info[i][1];
					tp_c[i][2] = scan_atlas_reg_seeds_info[i][2];
					tp_r[i] = scan_atlas_reg_seeds_info[i][3];
					// tumor days
					tp_T[i] = 0.004 * pow(scan_atlas_reg_seeds_info[i][3] / 2, 3) * 4;
					TRACE2("tp[%d]: xc = %f, yc = %f, zc = %f, T = %f, r = %f\n", i, tp_c[i][0], tp_c[i][1], tp_c[i][2], tp_T[i], tp_r[i]);
					//
					if (tp_r[i] <= 0) {
						provided_tumor_size = false;
					}
				}
				TRACE2("provided_tumor_size = %d\n", (provided_tumor_size)?1:0);
			} else {
				/*
				if (b_point_info) {
					if (!InitializeMeansAndVariancesFromPoints(scan_points_info, scan_points_num, scan_image_masked_files, &scan_MeanVector, &scan_VarianceVector, b_valid_label)) {
						TRACE("InitializeMeansAndVariancesFromPoints - failed\n");
						exit(EXIT_FAILURE);
					}
				} else {
					if (!InitializeMeansAndVariances(scan_init_mean, &scan_MeanVector, &scan_VarianceVector)) {
						TRACE("InitializeMeansAndVariances - failed\n");
						exit(EXIT_FAILURE);
					}
				}
				SaveMeansAndVariances(scan_means_file, scan_variances_file, &scan_MeanVector, &scan_VarianceVector);
				//
				scan_InitMeanVector.clear();
				for (i = 0; i < NumberOfPriorChannels; i++) {
					scan_InitMeanVector.push_back(scan_MeanVector.at(i));
				}
				//*/
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////

      clock_end = clock();
      //std::cout << "===Time stamp 09: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			{
#ifdef USE_PROB_PRIOR
				if (!IsFileExist(atlas_reg_prob_files[0]) || !provided_tumor_size) {
					int tp_c_t[MaxNumberOFTumorSeeds][3];
					float tp_r_t[MaxNumberOFTumorSeeds];
					int pt_c[MaxNumberOFPoints][3];
					int pt_t[MaxNumberOFPoints];
					//
					float wt[4];
					char str_tmp[1024];
          
					for (i = 0; i < NumberOfImageChannels; i++) {
						scan_atlas_reg_images_masked[i].load(scan_atlas_reg_image_masked_files[i], 1);
					}
					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_probs[i].allocate(vd_x, vd_y, vd_z);
					}

					for (i = 0; i < scan_seeds_num; i++) {
						// mm -> vox
						tp_c_t[i][0] = (int)(tp_c[i][0] / scan_atlas_reg_images_masked[0].m_vd_dx + 0.5);
						tp_c_t[i][1] = (int)(tp_c[i][1] / scan_atlas_reg_images_masked[0].m_vd_dy + 0.5);
						tp_c_t[i][2] = (int)(tp_c[i][2] / scan_atlas_reg_images_masked[0].m_vd_dz + 0.5);
						tp_r_t[i] = tp_r[i] / scan_atlas_reg_images_masked[0].m_vd_dx;
						//
						TRACE2("tp_t[%d]: xc = %d, yc = %d, zc = %d, r = %f\n", i, tp_c_t[i][0], tp_c_t[i][1], tp_c_t[i][2], tp_r_t[i]);
					}

					for (i = 0; i < scan_points_num; i++) {
						// mm -> vox
						pt_c[i][0] = (int)(scan_atlas_reg_points_info[i][0] / scan_atlas_reg_images_masked[0].m_vd_dx + 0.1);
						pt_c[i][1] = (int)(scan_atlas_reg_points_info[i][1] / scan_atlas_reg_images_masked[0].m_vd_dy + 0.1);
						pt_c[i][2] = (int)(scan_atlas_reg_points_info[i][2] / scan_atlas_reg_images_masked[0].m_vd_dz + 0.1);
						pt_t[i]    = (int)(scan_atlas_reg_points_info[i][3] + 0.1);
						//
						TRACE2("pt_c[%d]: xc = %d, yc = %d, zc = %d, t = %d\n", i, pt_c[i][0], pt_c[i][1], pt_c[i][2], pt_t[i]);
					}

					{
						FVolume tu_prob1, tu_prob2;
#ifdef PROB_SMOOTHING
						FVolume tu_prob1_g, tu_prob2_g;
#endif
						FVolume tu_prob;

						tu_prob1.allocate(vd_x, vd_y, vd_z);
						tu_prob2.allocate(vd_x, vd_y, vd_z);
						tu_prob.allocate(vd_x, vd_y, vd_z);

						if (k == 0) {
							if (!provided_tumor_size) {
								sprintf(str_tmp, "%s%ss_atlas_reg_tu_prob_%d_re_m.nii.gz", tmp_folder, DIR_SEP, k);
								if (!IsFileExist(str_tmp)) {
									wt[0] = 0.25; wt[1] = 0.25; wt[2] = 0.25; wt[3] = 0.25;
									//wt[0] = 0.0;  wt[1] = 1.0;  wt[2] = 0.0;  wt[3] = 0.0;
									//
									ComputeTumorProb(scan_atlas_reg_images_masked, NumberOfImageChannels, 
										tp_c_t, tp_r_t, scan_seeds_num, 5, 
										pt_c, pt_t, scan_points_num, 1,
										NULL, NULL, wt, NULL, 0, &tu_prob1, 4.0f);
									//
#ifdef PROB_SMOOTHING
									tu_prob1.GaussianSmoothing(tu_prob1_g, 1.0, 5);
									tu_prob1 = tu_prob1_g;
#endif
									//
									tu_prob1.save(str_tmp, 1);
									ChangeNIIHeader(str_tmp, scan_atlas_reg_image_files[0]);
								} else {
									tu_prob1.load(str_tmp, 1);
								}
								//
                EstimateTumorSize(tu_prob1, scan_seeds_num, tp_c_t, tp_r_t);
							}
							//
              clock_end = clock();
              //std::cout << "===Time stamp 10: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

							sprintf(str_tmp, "%s%ss_atlas_reg_tu_prob1_%d.nii.gz", tmp_folder, DIR_SEP, k);
							if (!IsFileExist(str_tmp)) {
								wt[0] = 0.25; wt[1] = 0.25; wt[2] = 0.25; wt[3] = 0.25;
								//
								ComputeTumorProb(scan_atlas_reg_images_masked, NumberOfImageChannels, 
									tp_c_t, tp_r_t, scan_seeds_num, 5, 
									pt_c, pt_t, scan_points_num, 1,
									NULL, NULL, wt, NULL, 0, &tu_prob1);
								//
#ifdef PROB_SMOOTHING
                
								tu_prob1.GaussianSmoothing(tu_prob1_g, 1.0, 5);
								tu_prob1 = tu_prob1_g;
#endif
								//
                tu_prob1.save(str_tmp, 1);
                
								ChangeNIIHeader(str_tmp, scan_atlas_reg_image_files[0]);
							} else {
								tu_prob1.load(str_tmp, 1);
							}

              clock_end = clock();
              //std::cout << "===Time stamp 11: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

							sprintf(str_tmp, "%s%ss_atlas_reg_tu_prob2_%d.nii.gz", tmp_folder, DIR_SEP, k);
							if (!IsFileExist(str_tmp)) {
								wt[0] = 0.0;  wt[1] = 1.0;  wt[2] = 0.0;  wt[3] = 0.0;
								//
								ComputeTumorProb(scan_atlas_reg_images_masked, NumberOfImageChannels, 
									tp_c_t, tp_r_t, scan_seeds_num, 5, 
									pt_c, pt_t, scan_points_num, 1,
#if 1
									NULL, 
#else
									&tu_prob1,
#endif
									NULL, wt, NULL, 0, &tu_prob2);
								//

#ifdef PROB_SMOOTHING
								tu_prob2.GaussianSmoothing(tu_prob2_g, 1.0, 5);
                tu_prob2 = tu_prob2_g;

#endif
								//
								tu_prob2.save(str_tmp, 1);
                ChangeNIIHeader(str_tmp, scan_atlas_reg_image_files[0]);

							} else {
								tu_prob2.load(str_tmp, 1);
							}
						} else {
							sprintf(str_tmp, "%s%ss_atlas_reg_tu_prob1_%d.nii.gz", tmp_folder, DIR_SEP, 0);
							tu_prob1.load(str_tmp, 1);
							sprintf(str_tmp, "%s%ss_atlas_reg_tu_prob2_%d.nii.gz", tmp_folder, DIR_SEP, 0);
							tu_prob2.load(str_tmp, 1);
						}

            clock_end = clock();
            //std::cout << "===Time stamp 12: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
            std::cout << "===Percentage Done : "<< 10.01 * (k + 1) << "\n";

						for (n = 0; n < vd_z; n++) {
							for (m = 0; m < vd_y; m++) {
								for (l = 0; l < vd_x; l++) {
									if (scan_atlas_reg_images_masked[0].m_pData[n][m][l][0] == 0) {
										for (i = 0; i < NumberOfPriorChannels; i++) {
											scan_atlas_reg_probs[i].m_pData[n][m][l][0] = 0;
										}
										scan_atlas_reg_probs[BG].m_pData[n][m][l][0] = 1;
									} else {
										float p, p_1_n;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
										float p_3;
#else
										float p_2;
#endif
										p = (tu_prob1.m_pData[n][m][l][0] + tu_prob2.m_pData[n][m][l][0]) * 0.5;
										tu_prob.m_pData[n][m][l][0] = p;

#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
										p_3 = p / 3;
										p_1_n = (1.0 - p) / (NumberOfPriorChannels-4);
#else
										p_2 = p * 0.5;
										p_1_n = (1.0 - p) / (NumberOfPriorChannels-3);
#endif

										for (i = 0; i < NumberOfPriorChannels; i++) {
											scan_atlas_reg_probs[i].m_pData[n][m][l][0] = p_1_n;
										}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
										scan_atlas_reg_probs[TU ].m_pData[n][m][l][0] = p_3;
										scan_atlas_reg_probs[NCR].m_pData[n][m][l][0] = p_3;
										scan_atlas_reg_probs[NE ].m_pData[n][m][l][0] = p_3;
#else
										scan_atlas_reg_probs[TU ].m_pData[n][m][l][0] = p_2;
										scan_atlas_reg_probs[NCR].m_pData[n][m][l][0] = p_2;
#endif
										scan_atlas_reg_probs[BG].m_pData[n][m][l][0] = 0;
									}
								}
							}
						}

            clock_end = clock();
            //std::cout << "===Time stamp 13: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

						if (k == 0) {
							if (!provided_tumor_size) {
                EstimateTumorSize(tu_prob, scan_seeds_num, tp_c_t, tp_r_t);

								//
								for (i = 0; i < scan_seeds_num; i++) {
									if (tp_r[i] > 0) continue;
									//
									tp_r[i] = tp_r_t[i] * scan_atlas_reg_images_masked[0].m_vd_dx;
									// tumor days
									tp_T[i] = 0.004 * pow(tp_r[i] / 2, 3) * 4;
									//
									TRACE2("estimated tp[%d]: T = %f, r = %f\n", i, tp_T[i], tp_r[i]);
								}

							}
						}
						
						tu_prob1.clear();
						tu_prob2.clear();
#ifdef PROB_SMOOTHING
						tu_prob1_g.clear();
						tu_prob2_g.clear();
#endif
						tu_prob.clear();
					}

          clock_end = clock();
          //std::cout << "===Time stamp 14: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

					for (i = 0; i < NumberOfImageChannels; i++) {
						scan_atlas_reg_images_masked[i].clear();
					}
					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_probs[i].save(atlas_reg_prob_files[i], 1);
						ChangeNIIHeader(atlas_reg_prob_files[i], scan_atlas_reg_image_files[0]);
						//
						scan_atlas_reg_probs[i].clear();
          }

				}
#endif
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////

      clock_end = clock();
      //std::cout << "===Time stamp 15: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple	
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			// get optimal tumor parameters
			if (!IsFileExist(atlas_reg_mass_prior_files[0])) {
				MakeScalesFile(scales_file);
				//
				{
					if (use_default_label_map_s) {
						MakeSimulatorInputFile(simulator_input_file, 
							vd_dx, vd_dy, vd_dz, 
							64, 64, 64,
							3.75, 3.75, 3.0
							);
					} else {
						atlas_label_map_s.load((char*)atlas_label_map_s_file, 1);
						//
						MakeSimulatorInputFile(simulator_input_file, 
							vd_dx, vd_dy, vd_dz, 
							atlas_label_map_s.m_vd_x, atlas_label_map_s.m_vd_y, atlas_label_map_s.m_vd_z, 
							atlas_label_map_s.m_vd_dx, atlas_label_map_s.m_vd_dy, atlas_label_map_s.m_vd_dz
							);
						//
						atlas_label_map_s.clear();
					}
				}
				//
        clock_end = clock();
        //std::cout << "===Time stamp 16: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

				{
					int max_eval;
					if (k == 0) {
						max_eval = SIMUL_ITER_STEP_0;
					} else if (k == 1) {
						max_eval = SIMUL_ITER_STEP_1;
					} else {
						max_eval = SIMUL_ITER_STEP_N;
					}
					//
					MakeHOPSFile(hops_file, tp_c, tp_T, scan_seeds_num, 
#ifdef USE_MASK_G
						scan_atlas_reg_image_masked_g_list, 
#else
						scan_atlas_reg_image_masked_list, 
#endif
						atlas_prior_list, 
#ifdef USE_PROB_PRIOR
						atlas_reg_prob_list, prob_weight, 
#endif
#ifdef USE_ED_NON_WM_PROB
						ed_non_wm_prob,
#endif
						scan_means_file, scan_variances_file, 
#ifdef USE_TU_SIMUL_H0
						scan_h0_hdr_file, 
#else
						scan_h_hdr_file, 
#endif
						atlas_label_map_s_img_file, simulator_input_file, scales_file, out_folder, tmp_folder, solution_file, max_eval, s_val, k, hop_sync_eval, hop_random_order, num_hop_threads);
				}

        clock_end = clock();
        //std::cout << "===Time stamp 17: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

				if (!IsFileExist(solution_file)) {
					sprintf(szCmdLine, "%s %s", HOPSPACK_PATH, hops_file);
					//
					TRACE("%s\n", szCmdLine);
					//
					/////////////////////////////////////////////////////////////////////////////
					char trace_file_out[1024];
					fflush(stdout);
					sprintf(trace_file_out, "%s%sOptim_%d_log.txt", tmp_folder, DIR_SEP, k);
#if defined(WIN32) || defined(WIN64)
					FILE *fp_trace_out;
					fp_trace_out = freopen(trace_file_out, "w", stdout);
#else
					int fd_trace_out;
					fpos_t pos_out; 
					fgetpos(stdout, &pos_out);
					fd_trace_out = _dup(fileno(stdout));
					freopen(trace_file_out, "w", stdout);
#endif
					/////////////////////////////////////////////////////////////////////////////
					//
					if (!ExecuteProcess(szCmdLine)) {
						TRACE("failed\n");
						exit(EXIT_FAILURE);
					}
					//
					/////////////////////////////////////////////////////////////////////////////
					fflush(stdout);
#if defined(WIN32) || defined(WIN64)
					fp_trace_out = freopen("CON", "w", stdout);
#else
					_dup2(fd_trace_out, fileno(stdout));
					close(fd_trace_out);
					clearerr(stdout);
					fsetpos(stdout, &pos_out);
#endif
					/////////////////////////////////////////////////////////////////////////////
				}
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////

      clock_end = clock();
      //std::cout << "===Time stamp 18: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple


			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			// get optimal params from solution file and run simulator
			// and update tumor parameters
			{
				char tmp[5+6+4*MaxNumberOFTumorSeeds][1024];
				int p_num;
				FILE* fp;

#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
				p_num = 6 + 4 * scan_seeds_num;
#else
				p_num = 5 + 4 * scan_seeds_num;
#endif
#else
#ifdef USE_OPTIM_ED
				p_num = 5 + 4 * scan_seeds_num;
#else
				p_num = 4 + 4 * scan_seeds_num;
#endif
#endif
				fp = fopen(solution_file, "r");
				if (fp == NULL) {
					TRACE("Cannot open %s\n", solution_file);
					exit(EXIT_FAILURE);
				}
				for (i = 0; i < p_num+5; i++) {
					fscanf(fp, "%s", (char*)&tmp[i]);
					if (i >= 4 && i < p_num+5) {
						s_val[i-4] = atof(tmp[i]);
					}
				}
				fclose(fp);
				//
				for (i = 0; i < scan_seeds_num; i++) {
#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
#ifdef USE_OPTIM_UPDATE_C
					tp_c[i][0] = s_val[6+i*4] * 1000;
					tp_c[i][1] = s_val[7+i*4] * 1000;
					tp_c[i][2] = s_val[8+i*4] * 1000;
#endif
#ifdef USE_OPTIM_UPDATE_T
					tp_T[i] = s_val[9+i*4] * 1000;
#endif
#else
#ifdef USE_OPTIM_UPDATE_C
					tp_c[i][0] = s_val[5+i*4] * 1000;
					tp_c[i][1] = s_val[6+i*4] * 1000;
					tp_c[i][2] = s_val[7+i*4] * 1000;
#endif
#ifdef USE_OPTIM_UPDATE_T
					tp_T[i] = s_val[8+i*4] * 1000;
#endif
#endif
#else
#ifdef USE_OPTIM_ED
#ifdef USE_OPTIM_UPDATE_C
					tp_c[i][0] = s_val[5+i*4] * 1000;
					tp_c[i][1] = s_val[6+i*4] * 1000;
					tp_c[i][2] = s_val[7+i*4] * 1000;
#endif
#ifdef USE_OPTIM_UPDATE_T
					tp_T[i] = s_val[8+i*4] * 1000;
#endif
#else
#ifdef USE_OPTIM_UPDATE_C
					tp_c[i][0] = s_val[4+i*4] * 1000;
					tp_c[i][1] = s_val[5+i*4] * 1000;
					tp_c[i][2] = s_val[6+i*4] * 1000;
#endif
#ifdef USE_OPTIM_UPDATE_T
					tp_T[i] = s_val[7+i*4] * 1000;
#endif
#endif
#endif
				}
				// tumor days
				for (i = 0; i < scan_seeds_num; i++) {
					//T = 0.004 * pow(scan_atlas_reg_seeds_info[0][3] / 2, 3) * 4;
					tp_r[i] = pow(tp_T[i] / (0.004 * 4.0), 1.0/3.0) * 2;
					TRACE2("tp[%d]: xc = %f, yc = %f, zc = %f, T = %f, r = %f\n", i, tp_c[i][0], tp_c[i][1], tp_c[i][2], tp_T[i], tp_r[i]);
				}
				//
				if (!IsFileExist(u_hdr_file)) {
					char input_file[1024];
					char output_file[1024] = "SOL";
					int tag = k;
					char tag_F[1024] = "SOL";

					sprintf(input_file, "%s%sinput.%d_%s.txt", tmp_folder, DIR_SEP, tag, tag_F);

					fp = fopen(input_file, "w");
					if (fp == NULL) {
						TRACE("Cannot make %s\n", input_file);
						exit(EXIT_FAILURE);
					}
#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
					fprintf(fp, "F\n%d\n%.15g\n%.15g\n%.15g\n%.15g\n%.15g\n%.15g", p_num, s_val[0], s_val[1], s_val[2], s_val[3], s_val[4], s_val[5]);
					for (i = 0; i < scan_seeds_num; i++) {
						fprintf(fp, "\n%.15g\n%.15g\n%.15g\n%.15g", s_val[6+i*4], s_val[7+i*4], s_val[8+i*4], s_val[9+i*4]);
					}
#else
					fprintf(fp, "F\n%d\n%.15g\n%.15g\n%.15g\n%.15g\n%.15g", p_num, s_val[0], s_val[1], s_val[2], s_val[3], s_val[4]);
					for (i = 0; i < scan_seeds_num; i++) {
						fprintf(fp, "\n%.15g\n%.15g\n%.15g\n%.15g", s_val[5+i*4], s_val[6+i*4], s_val[7+i*4], s_val[8+i*4]);
					}
#endif
#else
#ifdef USE_OPTIM_ED
					fprintf(fp, "F\n%d\n%.15g\n%.15g\n%.15g\n%.15g\n%.15g", p_num, s_val[0], s_val[1], s_val[2], s_val[3], s_val[4]);
					for (i = 0; i < scan_seeds_num; i++) {
						fprintf(fp, "\n%.15g\n%.15g\n%.15g\n%.15g", s_val[5+i*4], s_val[6+i*4], s_val[7+i*4], s_val[8+i*4]);
					}
#else
#endif
					fprintf(fp, "F\n%d\n%.15g\n%.15g\n%.15g\n%.15g", p_num, s_val[0], s_val[1], s_val[2], s_val[3]);
					for (i = 0; i < scan_seeds_num; i++) {
						fprintf(fp, "\n%.15g\n%.15g\n%.15g\n%.15g", s_val[4+i*4], s_val[5+i*4], s_val[6+i*4], s_val[7+i*4]);
					}
#endif
					fclose(fp);
					//
#ifdef USE_PROB_PRIOR
#ifdef USE_ED_NON_WM_PROB
					sprintf(szCmdLine, "%s %s %s %s %f %f %s %s %s %s %s %s %s %s %s %s %d %s",
#else
					sprintf(szCmdLine, "%s %s %s %s %f %s %s %s %s %s %s %s %s %s %s %d %s",
#endif
#else
#ifdef USE_ED_NON_WM_PROB
					sprintf(szCmdLine, "%s %s %s %f %s %s %s %s %s %s %s %s %s %s %d %s",
#else
					sprintf(szCmdLine, "%s %s %s %s %s %s %s %s %s %s %s %s %s %d %s",
#endif
#endif
						EVALUATE_Q_PATH, scan_atlas_reg_image_masked_list, atlas_prior_list, 
#ifdef USE_PROB_PRIOR
						atlas_reg_prob_list, prob_weight, 
#endif
#ifdef USE_ED_NON_WM_PROB
						ed_non_wm_prob,
#endif
						scan_means_file, scan_variances_file, 
#ifdef USE_TU_SIMUL_H0
						scan_h0_hdr_file, 
#else
						scan_h_hdr_file, 
#endif
						atlas_label_map_s_img_file, 
						simulator_input_file, scales_file, out_folder, tmp_folder, input_file, output_file, tag, tag_F);
					//
					TRACE("%s\n", szCmdLine);
					//
					if (!ExecuteProcess(szCmdLine)) {
						TRACE("failed\n");
						exit(EXIT_FAILURE);
					}
				}

				// refresh tumor deformation field
				CopyMHDData(NULL, u_hdr_file, NULL, scan_u_hdr_file, true);
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////

      clock_end = clock();
      //std::cout << "===Time stamp 19: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
      std::cout << "===Percentage Done : " << 15.13 * (k + 1) << "\n";
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			// update deformation, means, variances
			if (!IsFileExist(h_hdr_file)) {
				if (num_itk_threads > 0) {	
					char str_num_itk_threads[1024];
					sprintf(str_num_itk_threads, "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d", num_itk_threads);
					TRACE2("%s\n", str_num_itk_threads);
					putenv(str_num_itk_threads);
				}
				//
				TRACE("Running JSR glistr...\n");
				//
				/////////////////////////////////////////////////////////////////////////////
				char trace_file_out[1024];
				fflush(stdout);
				sprintf(trace_file_out, "%s%ss_jsr_%d_log.txt", tmp_folder, DIR_SEP, k);
#if defined(WIN32) || defined(WIN64)
				FILE *fp_trace_out;
				fp_trace_out = freopen(trace_file_out, "w", stdout);
#else
				int fd_trace_out;
				fpos_t pos_out; 
				fgetpos(stdout, &pos_out);
				fd_trace_out = _dup(fileno(stdout));
				freopen(trace_file_out, "w", stdout);
#endif
				/////////////////////////////////////////////////////////////////////////////
				//
#ifdef USE_JSR_GLISTR
#if defined(USE_JSR_MULTI_LEVEL) || defined(USE_JSR_WEGHT_IMAGE_PYRAMID)
				double mv[NumberOfPriorChannels][NumberOfImageChannels];
				double vv[NumberOfPriorChannels][NumberOfImageChannels*NumberOfImageChannels];
				double mv_init[NumberOfPriorChannels][NumberOfImageChannels];
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
				char atlas_reg_post_tmp_files[NumberOfPriorChannels][1024];
				for (i = 0; i < NumberOfPriorChannels; i++) {
					sprintf(atlas_reg_post_tmp_files[i], "%s%ss_atlas_reg_post_tmp_jsr_%d_%d.nii.gz", tmp_folder, DIR_SEP, k, i);
				}
#endif
				//
				{
					BVolume mask;
					//
					for (i = 0; i < NumberOfImageChannels; i++) {
						scan_atlas_reg_images[i].load(scan_atlas_reg_image_masked_files[i], 1);
					}
					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_mass_priors[i].load(atlas_reg_mass_prior_files[i], 1);
#ifdef USE_PROB_PRIOR
						scan_atlas_reg_probs[i].load(atlas_reg_prob_files[i], 1);
#endif
					}
					//
					mask.allocate(vd_x, vd_y, vd_z, 1, vd_dx, vd_dy, vd_dz);
					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								double y[4];
#ifndef USE_DISCARD_ZERO_AREA
								double ys = 0;
#else
								double ys = 1;
#endif

								for (i = 0; i < NumberOfImageChannels; i++) {
									y[i] = scan_atlas_reg_images[i].m_pData[n][m][l][0];
#ifndef USE_DISCARD_ZERO_AREA
									ys += y[i];
#else
									if (y[i] <= 0) {
										ys = 0;
									}
#endif
								}

#ifndef USE_COST_BG
								if (ys > 0) {
#else
								if (1) {
#endif
									mask.m_pData[n][m][l][0] = 1;
								} else {
									mask.m_pData[n][m][l][0] = 0;
								}
							}
						}
					}
					//
          clock_end = clock();
          std::cout << "===Time stamp 20: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

					LoadMeansAndVariances(scan_means_file, scan_variances_file, mv, vv);
					for (n = 0; n < NumberOfPriorChannels; n++) {
						for (m = 0; m < NumberOfImageChannels; m++) {
							mv_init[n][m] = scan_InitMeanVector.at(n)(m);
						}
					}
					//
					if (k == 0) {
						UpdateMeansAndVariances(scan_atlas_reg_images, mask, scan_atlas_reg_mass_priors, mv, vv, mv_init, 0, false, mean_shift_update, ms_tumor, ms_edema);
						//
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
						for (i = 0; i < NumberOfPriorChannels; i++) {
							scan_atlas_reg_posteriors[i].allocate(vd_x, vd_y, vd_z, 1, vd_dx, vd_dy, vd_dz);
						}
						//
						ComputePosteriors(scan_atlas_reg_images, mask, scan_atlas_reg_mass_priors, mv, vv, 
#ifdef USE_PROB_PRIOR
							scan_atlas_reg_probs, prob_weight, 
#endif
							NULL, scan_atlas_reg_posteriors);
#endif
					} else {
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
						for (i = 0; i < NumberOfPriorChannels; i++) {
							char posterior_file[1024];
							sprintf(posterior_file, "%s%ss_atlas_reg_posterior_jsr_%d_%d.nii.gz", tmp_folder, DIR_SEP, k-1, i);
							scan_atlas_reg_posteriors[i].load(posterior_file, 1);

						}
#endif
					}
					//
					mask.clear();
					for (i = 0; i < NumberOfImageChannels; i++) {
						scan_atlas_reg_images[i].clear();
					}
					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_mass_priors[i].clear();
#ifdef USE_PROB_PRIOR
						scan_atlas_reg_probs[i].clear();
#endif
					}
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_posteriors[i].save(atlas_reg_post_tmp_files[i], 1);
						ChangeNIIHeader(atlas_reg_post_tmp_files[i], scan_atlas_reg_image_files[0]);
						//
						scan_atlas_reg_posteriors[i].clear();
					}
#endif
				}
#endif
				//
				// read fixed images
				FixedImageReaderVectorType FixedReaderVector(NumberOfImageChannels);
				for (i = 0; i < NumberOfImageChannels; i++) {
					FixedReaderVector.at(i) = FixedImageReaderType::New();
					FixedReaderVector.at(i)->SetFileName(scan_atlas_reg_image_masked_files[i]);
					try {
						FixedReaderVector.at(i)->UpdateLargestPossibleRegion();
					} catch (itk::ExceptionObject& e) {
						std::cerr << e << std::endl;
						exit(EXIT_FAILURE);
					}
				}

				// read moving images (atlas priors)
				MovingImageReaderVectorType MovingReaderVector(NumberOfPriorChannels);
				for (i = 0; i < NumberOfPriorChannels; i++) {
					MovingReaderVector.at(i) = MovingImageReaderType::New();
					MovingReaderVector.at(i)->SetFileName(atlas_reg_mass_prior_files[i]);
					try {
						MovingReaderVector.at(i)->UpdateLargestPossibleRegion();
					} catch (itk::ExceptionObject& e) {
						std::cerr << e << std::endl;
						exit(EXIT_FAILURE);
					}
				}

#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
				// read moving images (atlas priors)
				MovingImageReaderVectorType WeightReaderVector(NumberOfPriorChannels);
				for (i = 0; i < NumberOfPriorChannels; i++) {
					WeightReaderVector.at(i) = MovingImageReaderType::New();
					WeightReaderVector.at(i)->SetFileName(atlas_reg_post_tmp_files[i]);
					try {
						WeightReaderVector.at(i)->UpdateLargestPossibleRegion();
					} catch (itk::ExceptionObject& e) {
						std::cerr << e << std::endl;
						exit(EXIT_FAILURE);
					}
				}
#endif

#ifdef USE_PROB_PRIOR
				// read prob images
				MovingImageReaderVectorType ProbReaderVector(NumberOfPriorChannels);
				for (i = 0; i < NumberOfPriorChannels; i++) {
					ProbReaderVector.at(i) = MovingImageReaderType::New();
					ProbReaderVector.at(i)->SetFileName(atlas_reg_prob_files[i]);
					try {
						ProbReaderVector.at(i)->UpdateLargestPossibleRegion();
					} catch (itk::ExceptionObject& e) {
						std::cerr << e << std::endl;
						exit(EXIT_FAILURE);
					}
				}
#endif

        clock_end = clock();
        std::cout << "===Time stamp 21: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

				// perform joint segmentation & registration
				JointSegmentationRegistrationType::Pointer j_seg_reg_filter = JointSegmentationRegistrationType::New();
				JointSegmentationRegistrationFunctionType* drfp = dynamic_cast<JointSegmentationRegistrationFunctionType*>(j_seg_reg_filter->GetSegmentationRegistrationFilter()->GetDifferenceFunction().GetPointer());
        
				// set input images
				for (i = 1; i <= NumberOfImageChannels; i++) {
					j_seg_reg_filter->SetNthFixedImage(FixedReaderVector.at(i-1)->GetOutput(), i);
				}
				for (i = 1; i <= NumberOfPriorChannels; i++) {
					j_seg_reg_filter->SetNthMovingImage(MovingReaderVector.at(i-1)->GetOutput(), i);
				}
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
				for (i = 1; i <= NumberOfPriorChannels; i++) {
					j_seg_reg_filter->SetNthWeightImage(WeightReaderVector.at(i-1)->GetOutput(), i);
				}
#endif
#ifdef USE_PROB_PRIOR
				for (i = 1; i <= NumberOfPriorChannels; i++) {
					j_seg_reg_filter->SetNthProbImage(ProbReaderVector.at(i-1)->GetOutput(), i);
				}
#endif

#if 0
				{
					itk::ImageFileWriter<FixedImageType>::Pointer writer_test = itk::ImageFileWriter<FixedImageType>::New();
					itk::NiftiImageIO::Pointer imageIO = itk::NiftiImageIO::New();
					writer_test->SetImageIO(imageIO);

					for (i = 1; i <= NumberOfPriorChannels; i++) {
						char name[1024];
						sprintf(name,"jsr_init_moving_image_%d.nii.gz", i-1);
						writer_test->SetFileName(name);
						writer_test->SetInput(j_seg_reg_filter->GetNthMovingImage(i));
						writer_test->Update();
					}
				}
#endif

        clock_end = clock();
        std::cout << "===Time stamp 22: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

				// set parameters
				unsigned int NumberOfLevels;
				unsigned int num_of_iter[10];
				double sigma2 = 0.2;
				if (k == scan_jsr_max_iter-1) {
#ifdef USE_JSR_MULTI_LEVEL
					NumberOfLevels = 3;
					num_of_iter[0] = 0;
					num_of_iter[1] = 50;
					num_of_iter[2] = JSR_ITER_STEP_N;
#else
					NumberOfLevels = 3;
					num_of_iter[0] = 0;
					num_of_iter[1] = 0;
					num_of_iter[2] = JSR_ITER_STEP_N;
#endif
				} else {
					NumberOfLevels = 3;
					num_of_iter[0] = 0;
					num_of_iter[1] = 0;
					num_of_iter[2] = JSR_ITER_STEP_0;
				}
				//
				j_seg_reg_filter->SetNumberOfLevels(NumberOfLevels);
				j_seg_reg_filter->SetNumberOfIterations(num_of_iter);
				drfp->SetSigma2(sigma2);
#ifdef USE_PROB_PRIOR
				drfp->SetProbWeight(prob_weight);
#endif
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
				drfp->mean_shift_update = mean_shift_update;
				drfp->ms_tumor = ms_tumor;
				drfp->ms_edema = ms_edema;
#endif

#ifdef USE_JSR_ELASTIC
				// setting for a elastic registration performance
				j_seg_reg_filter->GetSegmentationRegistrationFilter()->SmoothUpdateFieldOff();
#if ITK_VERSION_MAJOR >= 4
				j_seg_reg_filter->GetSegmentationRegistrationFilter()->SmoothDisplacementFieldOn();
#else
				j_seg_reg_filter->GetSegmentationRegistrationFilter()->SmoothDeformationFieldOn();
#endif
				j_seg_reg_filter->GetSegmentationRegistrationFilter()->SetStandardDeviations(2.0);
#endif
#ifdef USE_JSR_FLUID
				// setting for a viscos registration performance
				j_seg_reg_filter->GetSegmentationRegistrationFilter()->SmoothUpdateFieldOn();
				j_seg_reg_filter->GetSegmentationRegistrationFilter()->SetUpdateFieldStandardDeviations(3.0);
#if ITK_VERSION_MAJOR >= 4
				j_seg_reg_filter->GetSegmentationRegistrationFilter()->SmoothDisplacementFieldOn();
#else
				j_seg_reg_filter->GetSegmentationRegistrationFilter()->SmoothDeformationFieldOn();
#endif
				j_seg_reg_filter->GetSegmentationRegistrationFilter()->SetStandardDeviations(0.5);
#endif

#if 0
				// set initial mean values
				if (scan_means_file != NULL) {
					FILE* fp;
					int i, j;
					fp = fopen(scan_means_file, "r");
					if (fp == NULL) {
						TRACE("Failed to open file: '%s'\n", scan_means_file);
						exit(EXIT_FAILURE);
					}
					for (i = 0; i < NumberOfPriorChannels; i++) {
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
								drfp->GetMeanVector()->at(j) = mean;
								break;
							}
						}
						if (j == NumberOfPriorChannels) {
							TRACE("class id is wrong");
							fclose(fp);
							exit(EXIT_FAILURE);
						}
					}
					fclose(fp);
				}
#endif

        clock_end = clock();
        std::cout << "===Time stamp 23: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

				for (i = 0; i < NumberOfPriorChannels; i++) {
					drfp->GetMeanVector()->at(i) = scan_MeanVector.at(i);
					drfp->GetInitMeanVector()->at(i) = scan_InitMeanVector.at(i);
					//
#ifdef USE_JSR_MULTI_LEVEL
					for (n = 0; n < NumberOfImageChannels; n++) {
						for (m = 0; m < NumberOfImageChannels; m++) {
							drfp->GetVarianceVector()->at(i)[n][m] = vv[i][n*NumberOfImageChannels+m];
						}
					}
#endif
				}

				// update filter
				try {
					j_seg_reg_filter->Update();
				} catch (itk::ExceptionObject& e) {
					std::cerr << e << std::endl;
					exit(EXIT_FAILURE);
				}

        clock_end = clock();
        std::cout << "===Time stamp 24: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

				// write output deformation field
				DeformationFieldWriterType::Pointer field_writer = DeformationFieldWriterType::New();
				field_writer->SetInput(j_seg_reg_filter->GetOutput());
				field_writer->SetFileName(h_hdr_file);
				field_writer->SetUseCompression(true);
				try {
					field_writer->Update();
				} catch (itk::ExceptionObject& e) {
					std::cerr << e << std::endl;
					exit(EXIT_FAILURE);
				}

				// write means and variances
				if (!SaveMeansAndVariances(means_file, variances_file, drfp->GetMeanVector(), drfp->GetVarianceVector())) {
					TRACE("SaveMeansAndVariances failed\n");
					exit(EXIT_FAILURE);
				}

				// write cost to file
				std::ofstream fcost(cost_file);
				if (!fcost) {
					TRACE("Failed to open file %s for writing\n", cost_file);
					exit(EXIT_FAILURE);
				} else {
					fcost << j_seg_reg_filter->GetMetric();
					fcost.close();
				}
#endif

        //
				/////////////////////////////////////////////////////////////////////////////
				fflush(stdout);
#if defined(WIN32) || defined(WIN64)
				fp_trace_out = freopen("CON", "w", stdout);
#else
				_dup2(fd_trace_out, fileno(stdout));
				close(fd_trace_out);
				clearerr(stdout);
				fsetpos(stdout, &pos_out);
#endif
				/////////////////////////////////////////////////////////////////////////////
				//
				TRACE("Running JSR glistr...done\n");

				// refresh deformation field
				CopyMHDData(NULL, h_hdr_file, NULL, scan_h_hdr_file, true);
				// refresh means and variances
				LoadMeansAndVariances(means_file, variances_file, &scan_MeanVector, &scan_VarianceVector);
				SaveMeansAndVariances(scan_means_file, scan_variances_file, &scan_MeanVector, &scan_VarianceVector);

			} else {
				CopyMHDData(NULL, h_hdr_file, NULL, scan_h_hdr_file, true);
				//
				LoadMeansAndVariances(means_file, variances_file, &scan_MeanVector, &scan_VarianceVector);
				SaveMeansAndVariances(scan_means_file, scan_variances_file, &scan_MeanVector, &scan_VarianceVector);
			}
			//exit(0);
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////

      clock_end = clock();
      //std::cout << "===Time stamp 25: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
      std::cout << "===Percentage Done : "<< 31.84 * (k + 1) << "\n";


#ifdef USE_MEAN_SHIFT_UPDATE_STEP
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Update means and variances using mean shift
			if (k > 0) {
				if (!IsFileExist(means_ms_file)) {
					TRACE("Update means and variances using mean shift\n");
					//
					/////////////////////////////////////////////////////////////////////////////
					char trace_file_out[1024];
					fflush(stdout);
					sprintf(trace_file_out, "%s%ss_jsr_%d_log.txt", tmp_folder, DIR_SEP, k);
#if defined(WIN32) || defined(WIN64)
					FILE *fp_trace_out;
					fp_trace_out = freopen(trace_file_out, "a", stdout);
#else
					int fd_trace_out;
					fpos_t pos_out; 
					fgetpos(stdout, &pos_out);
					fd_trace_out = _dup(fileno(stdout));
					freopen(trace_file_out, "a", stdout);
#endif
					/////////////////////////////////////////////////////////////////////////////
					//
					BVolume mask;
					double mv[NumberOfPriorChannels][NumberOfImageChannels];
					double vv[NumberOfPriorChannels][NumberOfImageChannels*NumberOfImageChannels];
					double mv_init[NumberOfPriorChannels][NumberOfImageChannels];

					for (i = 0; i < NumberOfImageChannels; i++) {
						scan_atlas_reg_images[i].load(scan_atlas_reg_image_masked_files[i], 1);
					}
					for (i = 0; i < NumberOfPriorChannels; i++) {
						char posterior_file[1024];
						sprintf(posterior_file, "%s%ss_atlas_reg_posterior_jsr_%d_%d.nii.gz", tmp_folder, DIR_SEP, k-1, i);
						scan_atlas_reg_posteriors[i].load(posterior_file, 1);
					}
					//
					mask.allocate(vd_x, vd_y, vd_z);
					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								double y[4];
#ifndef USE_DISCARD_ZERO_AREA
								double ys = 0;
#else
								double ys = 1;
#endif

								for (i = 0; i < NumberOfImageChannels; i++) {
									y[i] = scan_atlas_reg_images[i].m_pData[n][m][l][0];
#ifndef USE_DISCARD_ZERO_AREA
									ys += y[i];
#else
									if (y[i] <= 0) {
										ys = 0;
									}
#endif
								}

#ifndef USE_COST_BG
								if (ys > 0) {
#else
								if (1) {
#endif
									mask.m_pData[n][m][l][0] = 1;
								} else {
									mask.m_pData[n][m][l][0] = 0;
								}
							}
						}
					}
					//
					LoadMeansAndVariances(scan_means_file, scan_variances_file, mv, vv);
					for (n = 0; n < NumberOfPriorChannels; n++) {
						for (m = 0; m < NumberOfImageChannels; m++) {
							mv_init[n][m] = scan_InitMeanVector.at(n)(m);
						}
					}
					//
					UpdateMeansAndVariances(scan_atlas_reg_images, mask, scan_atlas_reg_posteriors, mv, vv, mv_init, 1, true, mean_shift_update, ms_tumor, ms_edema);
					//
          clock_end = clock();
          std::cout << "===Time stamp 26: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

					// write means and variances
					if (!SaveMeansAndVariances(means_ms_file, variances_ms_file, mv, vv)) {
						TRACE("SaveMeansAndVariances failed\n");
						exit(EXIT_FAILURE);
					}
					//
					mask.clear();
					for (i = 0; i < NumberOfImageChannels; i++) {
						scan_atlas_reg_images[i].clear();
					}
					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_posteriors[i].clear();
					}
					//
					/////////////////////////////////////////////////////////////////////////////
					fflush(stdout);
#if defined(WIN32) || defined(WIN64)
					fp_trace_out = freopen("CON", "w", stdout);
#else
					_dup2(fd_trace_out, fileno(stdout));
					close(fd_trace_out);
					clearerr(stdout);
					fsetpos(stdout, &pos_out);
#endif
					/////////////////////////////////////////////////////////////////////////////

					// refresh means and variances
					LoadMeansAndVariances(means_ms_file, variances_ms_file, &scan_MeanVector, &scan_VarianceVector);
					SaveMeansAndVariances(scan_means_file, scan_variances_file, &scan_MeanVector, &scan_VarianceVector);
				} else {
					LoadMeansAndVariances(means_ms_file, variances_ms_file, &scan_MeanVector, &scan_VarianceVector);
					SaveMeansAndVariances(scan_means_file, scan_variances_file, &scan_MeanVector, &scan_VarianceVector);
				}
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

      clock_end = clock();
      //std::cout << "===Time stamp 27: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			// get posteriors
			if (!IsFileExist(atlas_reg_posterior_files[0])) {
			//if (1) {
				float vd_ox, vd_oy, vd_oz;
				FVolume v3;
#ifdef USE_PL_COST
				FVolume atlas_reg_pl_costs[NumberOfPriorChannels];
				char atlas_reg_pl_cost_files[NumberOfPriorChannels][1024];
				//
				for (i = 0; i < NumberOfPriorChannels; i++) {
					sprintf(atlas_reg_pl_cost_files[i], "%s%ss_atlas_reg_pl_costs_jsr_%d_%d.nii.gz", tmp_folder, DIR_SEP, k, i);
				}
#endif

				if (!LoadMHDData(NULL, (char*)h_hdr_file, &v3.m_pData, v3.m_vd_x, v3.m_vd_y, v3.m_vd_z, v3.m_vd_s, v3.m_vd_dx, v3.m_vd_dy, v3.m_vd_dz, vd_ox, vd_oy, vd_oz)) {
					TRACE("Loading %s failed..\n", h_hdr_file);
				} else  {
					v3.computeDimension();
				}

				// apply warp
				scan_atlas_reg_mass_priors[CSF].load(atlas_reg_mass_prior_files[CSF], 1);
				scan_atlas_reg_warp_priors[CSF].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[CSF], scan_atlas_reg_mass_priors[CSF], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[CSF].clear();
				//
#if defined(USE_9_PRIORS) || defined(USE_10_PRIORS) || defined(USE_11_PRIORS)
				scan_atlas_reg_mass_priors[VT ].load(atlas_reg_mass_prior_files[VT ], 1);
				scan_atlas_reg_warp_priors[VT ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[VT ], scan_atlas_reg_mass_priors[VT ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[VT ].clear();
#endif
				//
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				scan_atlas_reg_mass_priors[CB ].load(atlas_reg_mass_prior_files[CB ], 1);
				scan_atlas_reg_warp_priors[CB ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[CB ], scan_atlas_reg_mass_priors[CB ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[CB ].clear();
#endif
				//
				scan_atlas_reg_mass_priors[GM ].load(atlas_reg_mass_prior_files[GM ], 1);
				scan_atlas_reg_warp_priors[GM ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[GM ], scan_atlas_reg_mass_priors[GM ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[GM ].clear();
				//
				scan_atlas_reg_mass_priors[WM ].load(atlas_reg_mass_prior_files[WM ], 1);
				scan_atlas_reg_warp_priors[WM ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[WM ], scan_atlas_reg_mass_priors[WM ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[WM ].clear();
				//
				scan_atlas_reg_mass_priors[TU ].load(atlas_reg_mass_prior_files[TU ], 1);
				scan_atlas_reg_warp_priors[TU ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[TU ], scan_atlas_reg_mass_priors[TU ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[TU ].clear();
				//
				scan_atlas_reg_mass_priors[ED ].load(atlas_reg_mass_prior_files[ED ], 1);
				scan_atlas_reg_warp_priors[ED ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[ED ], scan_atlas_reg_mass_priors[ED ], v3, 0.0f, false);
				scan_atlas_reg_mass_priors[ED ].clear();
				//
#ifdef USE_WARPED_BG
				scan_atlas_reg_mass_priors[BG ].load(atlas_reg_mass_prior_files[BG ], 1);
				scan_atlas_reg_warp_priors[BG ].allocate(vd_x, vd_y, vd_z);
				GenerateBackwardWarpVolume(scan_atlas_reg_warp_priors[BG ], scan_atlas_reg_mass_priors[BG ], v3, 1.0f, false);
				scan_atlas_reg_mass_priors[BG ].clear();
#else
				scan_atlas_reg_warp_priors[BG ].load(atlas_prior_files[BG], 1);
#endif

				v3.clear();

				scan_atlas_reg_warp_priors[VS ].copy(scan_atlas_reg_warp_priors[CSF]);
				scan_atlas_reg_warp_priors[NCR].copy(scan_atlas_reg_warp_priors[TU ]);
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				scan_atlas_reg_warp_priors[NE ].copy(scan_atlas_reg_warp_priors[TU ]);
#endif

        clock_end = clock();
        //std::cout << "===Time stamp 28: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

				for (i = 0; i < NumberOfImageChannels; i++) {
					if (use_masked_images_for_posteriors) {
						scan_atlas_reg_images[i].load(scan_atlas_reg_image_masked_files[i], 1);
						//scan_atlas_reg_images[i].load(scan_atlas_reg_image_masked_g_files[i], 1);
					} else {
						scan_atlas_reg_images[i].load(scan_atlas_reg_image_files[i], 1);
					}
				}
				for (i = 0; i < NumberOfPriorChannels; i++) {
					scan_atlas_reg_posteriors[i].allocate(vd_x, vd_y, vd_z);
					//
#ifdef USE_PL_COST
					atlas_reg_pl_costs[i].allocate(vd_x, vd_y, vd_z);
#endif
				}

        clock_end = clock();
        //std::cout << "===Time stamp 29: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

#ifdef USE_TU_PRIOR_CUTOFF
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				float tu_prior_cutoff_tu_th = TU_PRIOR_CUTOFF_TU_TH / 3;
				float tu_prior_cutoff_ncr_th = TU_PRIOR_CUTOFF_NCR_TH / 3;
				float tu_prior_cutoff_ne_th = TU_PRIOR_CUTOFF_NE_TH / 3;
				float tu_prior_cutoff_ed_th = TU_PRIOR_CUTOFF_ED_TH / 3;
#else
				float tu_prior_cutoff_tu_th = TU_PRIOR_CUTOFF_TU_TH / 2;
				float tu_prior_cutoff_ncr_th = TU_PRIOR_CUTOFF_NCR_TH / 2;
				float tu_prior_cutoff_ed_th = TU_PRIOR_CUTOFF_ED_TH / 2;
#endif
#ifdef USE_PROB_PRIOR
				// adjust warp prior using prob prior
				{
					int vd_x_1, vd_y_1, vd_z_1;
					FVolume prob_ncr_th;
					FVolume prob_tu_th;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
					FVolume prob_ne_th;
#endif
					FVolume prob_ed_th;
					float tu_prior_cutoff_tu_sum = 0;
					float tu_prior_cutoff_ncr_sum = 0;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
					float tu_prior_cutoff_ne_sum = 0;
#endif
					float tu_prior_cutoff_ed_sum = 0;
					int tu_prior_cutoff_tu_num = 0;
					int tu_prior_cutoff_ncr_num = 0;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
					int tu_prior_cutoff_ne_num = 0;
#endif
					int tu_prior_cutoff_ed_num = 0;

					vd_x_1 = vd_x - 1;
					vd_y_1 = vd_y - 1;
					vd_z_1 = vd_z - 1;

					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_probs[i].load(atlas_reg_prob_files[i], 1);
					}
					prob_tu_th.allocate(vd_x, vd_y, vd_z);
					prob_ncr_th.allocate(vd_x, vd_y, vd_z);
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
					prob_ne_th.allocate(vd_x, vd_y, vd_z);
#endif
					prob_ed_th.allocate(vd_x, vd_y, vd_z);

					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								float val = scan_atlas_reg_probs[TU].m_pData[n][m][l][0];
								if (val > tu_prior_cutoff_tu_th) {
									prob_tu_th.m_pData[n][m][l][0] = 1;
								} else {
									prob_tu_th.m_pData[n][m][l][0] = 0;
								}
								if (val > tu_prior_cutoff_ncr_th) {
									prob_ncr_th.m_pData[n][m][l][0] = 1;
								} else {
									prob_ncr_th.m_pData[n][m][l][0] = 0;
								}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								if (val > tu_prior_cutoff_ne_th) {
									prob_ne_th.m_pData[n][m][l][0] = 1;
								} else {
									prob_ne_th.m_pData[n][m][l][0] = 0;
								}
#endif
								if (val > tu_prior_cutoff_ed_th) {
									prob_ed_th.m_pData[n][m][l][0] = 1;
								} else {
									prob_ed_th.m_pData[n][m][l][0] = 0;
								}
							}
						}
					}

          clock_end = clock();
          //std::cout << "===Time stamp 30: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

					for (n = 1; n < vd_z_1; n++) {
						for (m = 1; m < vd_y_1; m++) {
							for (l = 1; l < vd_x_1; l++) {
								if (prob_tu_th.m_pData[n][m][l][0] == 0) {
									if (prob_tu_th.m_pData[n  ][m  ][l-1][0] > 0 || 
									    prob_tu_th.m_pData[n  ][m  ][l+1][0] > 0 ||
									    prob_tu_th.m_pData[n  ][m-1][l  ][0] > 0 ||
									    prob_tu_th.m_pData[n  ][m+1][l  ][0] > 0 ||
									    prob_tu_th.m_pData[n-1][m  ][l  ][0] > 0 ||
									    prob_tu_th.m_pData[n+1][m  ][l  ][0] > 0) {
										tu_prior_cutoff_tu_sum += scan_atlas_reg_warp_priors[TU].m_pData[n][m][l][0];
										tu_prior_cutoff_tu_num++;
									}
								}
								if (prob_ncr_th.m_pData[n][m][l][0] == 0) {
									if (prob_ncr_th.m_pData[n  ][m  ][l-1][0] > 0 || 
									    prob_ncr_th.m_pData[n  ][m  ][l+1][0] > 0 ||
									    prob_ncr_th.m_pData[n  ][m-1][l  ][0] > 0 ||
									    prob_ncr_th.m_pData[n  ][m+1][l  ][0] > 0 ||
									    prob_ncr_th.m_pData[n-1][m  ][l  ][0] > 0 ||
									    prob_ncr_th.m_pData[n+1][m  ][l  ][0] > 0) {
										tu_prior_cutoff_ncr_sum += scan_atlas_reg_warp_priors[TU].m_pData[n][m][l][0];
										tu_prior_cutoff_ncr_num++;
									}
								}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								if (prob_ne_th.m_pData[n][m][l][0] == 0) {
									if (prob_ne_th.m_pData[n  ][m  ][l-1][0] > 0 || 
									    prob_ne_th.m_pData[n  ][m  ][l+1][0] > 0 ||
									    prob_ne_th.m_pData[n  ][m-1][l  ][0] > 0 ||
									    prob_ne_th.m_pData[n  ][m+1][l  ][0] > 0 ||
									    prob_ne_th.m_pData[n-1][m  ][l  ][0] > 0 ||
									    prob_ne_th.m_pData[n+1][m  ][l  ][0] > 0) {
										tu_prior_cutoff_ne_sum += scan_atlas_reg_warp_priors[TU].m_pData[n][m][l][0];
										tu_prior_cutoff_ne_num++;
									}
								}
#endif
								if (prob_ed_th.m_pData[n][m][l][0] == 0) {
									if (prob_ed_th.m_pData[n  ][m  ][l-1][0] > 0 || 
									    prob_ed_th.m_pData[n  ][m  ][l+1][0] > 0 ||
									    prob_ed_th.m_pData[n  ][m-1][l  ][0] > 0 ||
									    prob_ed_th.m_pData[n  ][m+1][l  ][0] > 0 ||
									    prob_ed_th.m_pData[n-1][m  ][l  ][0] > 0 ||
									    prob_ed_th.m_pData[n+1][m  ][l  ][0] > 0) {
										tu_prior_cutoff_ed_sum += scan_atlas_reg_warp_priors[TU].m_pData[n][m][l][0];
										tu_prior_cutoff_ed_num++;
									}
								}
							}
						}
					}


          clock_end = clock();
          //std::cout << "===Time stamp 31: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple


					if (tu_prior_cutoff_tu_num > 0) {
						tu_prior_cutoff_tu_th = tu_prior_cutoff_tu_sum / tu_prior_cutoff_tu_num;
						TRACE2("tu_prior_cutoff_tu_th = %f\n", tu_prior_cutoff_tu_th);
					}
					if (tu_prior_cutoff_ncr_num > 0) {
						tu_prior_cutoff_ncr_th = tu_prior_cutoff_ncr_sum / tu_prior_cutoff_ncr_num;
						TRACE2("tu_prior_cutoff_ncr_th = %f\n", tu_prior_cutoff_ncr_th);
					}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
					if (tu_prior_cutoff_ne_num > 0) {
						tu_prior_cutoff_ne_th = tu_prior_cutoff_ne_sum / tu_prior_cutoff_ne_num;
						TRACE2("tu_prior_cutoff_ne_th = %f\n", tu_prior_cutoff_ne_th);
					}
#endif
					if (tu_prior_cutoff_ed_num > 0) {
						tu_prior_cutoff_ed_th = tu_prior_cutoff_ed_sum / tu_prior_cutoff_ed_num;
						TRACE2("tu_prior_cutoff_ed_th = %f\n", tu_prior_cutoff_ed_th);
					}

					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_probs[i].clear();
					}
				}
#endif
#endif

				// store output posteriors for scan
				if (!ComputePosteriors(scan_atlas_reg_images, scan_atlas_reg_warp_priors, &scan_MeanVector, &scan_VarianceVector, scan_atlas_reg_posteriors
#ifdef USE_TU_PRIOR_CUTOFF
					, tu_prior_cutoff_tu_th, tu_prior_cutoff_ncr_th
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
					, tu_prior_cutoff_ne_th
#endif
					, tu_prior_cutoff_ed_th
#endif
#ifdef USE_PL_COST
					, atlas_reg_pl_costs
#endif
					, b_valid_label
					)) {
					TRACE("ComputePosteriors failed..\n");
					exit(EXIT_FAILURE);
				} else {
					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_warp_priors[i].save(atlas_reg_warp_prior_files[i], 1);
						ChangeNIIHeader(atlas_reg_warp_prior_files[i], scan_atlas_reg_image_files[0]);
						//
						scan_atlas_reg_posteriors[i].save(atlas_reg_posterior_files[i], 1);
						ChangeNIIHeader(atlas_reg_posterior_files[i], scan_atlas_reg_image_files[0]);
						//
#ifdef USE_PL_COST
						atlas_reg_pl_costs[i].save(atlas_reg_pl_cost_files[i], 1);
						ChangeNIIHeader(atlas_reg_pl_cost_files[i], scan_atlas_reg_image_files[0]);
#endif
					}
				}

        clock_end = clock();
        //std::cout << "===Time stamp 32: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

#ifdef USE_ED_EXPAND
				// expanding ED to NCR or NE
				{
					BVolume lm_tmp_o, lm_tmp_p, lm_tmp_n;

					lm_tmp_o.allocate(scan_atlas_reg_posteriors[0].m_vd_x, scan_atlas_reg_posteriors[0].m_vd_y, scan_atlas_reg_posteriors[0].m_vd_z);
					//
					ComputeLabelMap(scan_atlas_reg_posteriors, lm_tmp_o, b_valid_label);
					//
					{
						int it, it_max = 10;
						int vd_x_1, vd_y_1, vd_z_1;
					
						vd_x_1 = vd_x-1;
						vd_y_1 = vd_y-1;
						vd_z_1 = vd_z-1;
					
#if 1
						// remove noise of ED
						{
							unsigned char fore_label[1] = { label_idx[ED] };
							int fore_label_num = 1;
							SVolume cc;
							int cc_sum[MAXNCC];
							int cc_num;
							int cc_remove[MAXNCC];

							cc.allocate(lm_tmp_o.m_vd_x, lm_tmp_o.m_vd_y, lm_tmp_o.m_vd_z);
				
							GetConnectedComponents(lm_tmp_o, fore_label, fore_label_num, cc, cc_sum, &cc_num);

              clock_end = clock();
              std::cout << "===Time stamp 33: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

							//cc.save("D:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\data\\cc.nii.gz", 1);

							for (i = 0; i < cc_num; i++) {
								if (cc_sum[i] < 100) {
									cc_remove[i] = 1;
								} else {
									cc_remove[i] = 0;
								}
							}

							for (n = 0; n < vd_z; n++) {
								for (m = 0; m < vd_y; m++) {
									for (l = 0; l < vd_x; l++) {
										int cc_val = cc.m_pData[n][m][l][0];
										if (cc_val < cc_num) {
											if (cc_remove[cc_val]) {
												lm_tmp_o.m_pData[n][m][l][0] = label_idx[WM];
											}
										}
									}
								}
							}

              clock_end = clock();
              std::cout << "===Time stamp 34: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

							//lm_tmp_o.save("D:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\data\\lm_tmp_o.nii.gz", 1);
						}
#endif
						lm_tmp_n.copy(lm_tmp_o);
						//
						for (it = 0; it < it_max; it++) {
							int changed = 0;
							//
							lm_tmp_p.copy(lm_tmp_n);
							//
							for (n = 1; n < vd_z_1; n++) {
								for (m = 1; m < vd_y_1; m++) {
									for (l = 1; l < vd_x_1; l++) {
										if (
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
											(lm_tmp_p.m_pData[n][m][l][0] == label_idx[NCR] || lm_tmp_p.m_pData[n][m][l][0] == label_idx[NE]) && scan_atlas_reg_warp_priors[TU].m_pData[n][m][l][0]*3 < 0.5
#else
											lm_tmp_p.m_pData[n][m][l][0] == label_idx[NCR] && scan_atlas_reg_warp_priors[TU].m_pData[n][m][l][0]*2 < 0.5
#endif
										   )
										{
											if (lm_tmp_p.m_pData[n  ][m  ][l-1][0] == label_idx[ED] ||
												lm_tmp_p.m_pData[n  ][m  ][l+1][0] == label_idx[ED] ||
												lm_tmp_p.m_pData[n  ][m-1][l  ][0] == label_idx[ED] ||
												lm_tmp_p.m_pData[n  ][m+1][l  ][0] == label_idx[ED] ||
												lm_tmp_p.m_pData[n-1][m  ][l  ][0] == label_idx[ED] ||
												lm_tmp_p.m_pData[n+1][m  ][l  ][0] == label_idx[ED]) {
												lm_tmp_n.m_pData[n][m][l][0] = label_idx[ED];
												changed++;
											}
										}
									}
								}
							}
							//
              clock_end = clock();
              std::cout << "===Time stamp 35: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

							if (changed == 0) {
								break;
							}
						}

						//lm_tmp_n.save("D:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\data\\lm_tmp_n.nii.gz", 1);
						

#if 1
						// modify posteriors
						{
							char tmp_atlas_reg_posterior_files[NumberOfPriorChannels][1024];
							for (i = 0; i < NumberOfPriorChannels; i++) {
								sprintf(tmp_atlas_reg_posterior_files[i], "%s%ss_atlas_reg_posterior_jsr_%d_%d_tmp.nii.gz", tmp_folder, DIR_SEP, k, i);
							}
							for (n = 0; n < vd_z; n++) {
								for (m = 0; m < vd_y; m++) {
									for (l = 0; l < vd_x; l++) {
										if (lm_tmp_n.m_pData[n][m][l][0] == label_idx[ED]) {
											float p_tu = 0;
											p_tu += scan_atlas_reg_posteriors[TU].m_pData[n][m][l][0];
											scan_atlas_reg_posteriors[TU].m_pData[n][m][l][0] = 0;
											p_tu += scan_atlas_reg_posteriors[NCR].m_pData[n][m][l][0];
											scan_atlas_reg_posteriors[NCR].m_pData[n][m][l][0] = 0;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
											p_tu += scan_atlas_reg_posteriors[NE].m_pData[n][m][l][0];
											scan_atlas_reg_posteriors[NE].m_pData[n][m][l][0] = 0;
#endif
											scan_atlas_reg_posteriors[ED].m_pData[n][m][l][0] += p_tu;
										}
									}
								}
              }
              clock_end = clock();
              std::cout << "===Time stamp 36: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

							for (i = 0; i < NumberOfPriorChannels; i++) {
								//scan_atlas_reg_posteriors[i].save(tmp_atlas_reg_posterior_files[i], 1);
								//ChangeNIIHeader(tmp_atlas_reg_posterior_files[i], scan_atlas_reg_image_files[0]);
								scan_atlas_reg_posteriors[i].save(atlas_reg_posterior_files[i], 1);
								ChangeNIIHeader(atlas_reg_posterior_files[i], scan_atlas_reg_image_files[0]);
							}
						}
#endif
					}
				}
#endif

				if (k == scan_jsr_max_iter-1) {
					for (i = 0; i < NumberOfPriorChannels; i++) {
						CopyFile(atlas_reg_mass_prior_files[i], scan_atlas_reg_mass_prior_files[i], FALSE);
						//
						scan_atlas_reg_warp_priors[i].save(scan_atlas_reg_warp_prior_files[i], 1);
						ChangeNIIHeader(scan_atlas_reg_warp_prior_files[i], scan_atlas_reg_image_files[0]);
						//
						scan_atlas_reg_posteriors[i].save(scan_atlas_reg_posterior_files[i], 1);
						ChangeNIIHeader(scan_atlas_reg_posterior_files[i], scan_atlas_reg_image_files[0]);
					}
				}

				// free data
				for (i = 0; i < NumberOfPriorChannels; i++) {
					scan_atlas_reg_warp_priors[i].clear();
					scan_atlas_reg_posteriors[i].clear();
				}
				for (i = 0; i < NumberOfImageChannels; i++) {
					scan_atlas_reg_images[i].clear();
				}
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////

      clock_end = clock();
      //std::cout << "===Time stamp 37: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple;
      //////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			// generate atlas reg label map for scan
			if (!IsFileExist(atlas_reg_label_map_file)) {
			//{
				for (i = 0; i < NumberOfPriorChannels; i++) {
					scan_atlas_reg_posteriors[i].load(atlas_reg_posterior_files[i], 1);
				}
				//
				scan_atlas_reg_label_map.allocate(scan_atlas_reg_posteriors[0].m_vd_x, scan_atlas_reg_posteriors[0].m_vd_y, scan_atlas_reg_posteriors[0].m_vd_z);
				//
				ComputeLabelMap(scan_atlas_reg_posteriors, scan_atlas_reg_label_map, b_valid_label);
				//
				scan_atlas_reg_label_map.save(atlas_reg_label_map_file, 1);
				ChangeNIIHeader(atlas_reg_label_map_file, scan_atlas_reg_image_files[0]);

				if (k == scan_jsr_max_iter-1) {
					scan_atlas_reg_label_map.save(scan_atlas_reg_label_map_file, 1);
					ChangeNIIHeader(scan_atlas_reg_label_map_file, scan_atlas_reg_image_files[0]);
				}

				// free data
				{
					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_posteriors[i].clear();
					}
					//
					scan_atlas_reg_label_map.clear();
				}
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////////////////////////			
      std::cout << "===Percentage Done : " << 32 * (k + 1) << "\n";
		} // k

    clock_end = clock();
    //std::cout << "===Time stamp 38: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
		/////////////////////////////////////////////////////////////////////////////
		// Update deformation field
		/////////////////////////////////////////////////////////////////////////////
		{
			char h1u_hdr_file[1024];
			char h1u_r_hdr_file[1024];
			float _vd_dx, _vd_dy, _vd_dz;
			FVolume scan_hu_x, scan_hu_y, scan_hu_z;
			FVolume scan_hu_r_x, scan_hu_r_y, scan_hu_r_z;
			//
			sprintf(h1u_hdr_file  , "%s%ss_h1u.mhd"  , tmp_folder, DIR_SEP);
			sprintf(h1u_r_hdr_file, "%s%ss_h1u_r.mhd", tmp_folder, DIR_SEP);
			//
			_vd_dx = 1.0 / vd_dx;
			_vd_dy = 1.0 / vd_dy;
			_vd_dz = 1.0 / vd_dz;
			//
			{
				FVolume scan_u_v3;
				FVolume scan_h_v3;

				scan_hu_x.allocate(vd_x, vd_y, vd_z); scan_hu_x.m_vd_dx = vd_dx; scan_hu_x.m_vd_dy = vd_dy; scan_hu_x.m_vd_dz = vd_dz;
				scan_hu_y.allocate(vd_x, vd_y, vd_z); scan_hu_y.m_vd_dx = vd_dx; scan_hu_y.m_vd_dy = vd_dy; scan_hu_y.m_vd_dz = vd_dz;
				scan_hu_z.allocate(vd_x, vd_y, vd_z); scan_hu_z.m_vd_dx = vd_dx; scan_hu_z.m_vd_dy = vd_dy; scan_hu_z.m_vd_dz = vd_dz;
				scan_hu_r_x.allocate(vd_x, vd_y, vd_z); scan_hu_r_x.m_vd_dx = vd_dx; scan_hu_r_x.m_vd_dy = vd_dy; scan_hu_r_x.m_vd_dz = vd_dz;
				scan_hu_r_y.allocate(vd_x, vd_y, vd_z); scan_hu_r_y.m_vd_dx = vd_dx; scan_hu_r_y.m_vd_dy = vd_dy; scan_hu_r_y.m_vd_dz = vd_dz;
				scan_hu_r_z.allocate(vd_x, vd_y, vd_z); scan_hu_r_z.m_vd_dx = vd_dx; scan_hu_r_z.m_vd_dy = vd_dy; scan_hu_r_z.m_vd_dz = vd_dz;

				if (!LoadMHDData(NULL, scan_u_hdr_file, 
					&scan_u_v3.m_pData, scan_u_v3.m_vd_x, scan_u_v3.m_vd_y, scan_u_v3.m_vd_z, scan_u_v3.m_vd_s, scan_u_v3.m_vd_dx, scan_u_v3.m_vd_dy, scan_u_v3.m_vd_dz, vd_ox, vd_oy, vd_oz)) {
					TRACE("Loading %s failed..\n", scan_u_hdr_file);
				} else  {
					scan_u_v3.computeDimension();
					//
					{
						scan_u_x.allocate(vd_x, vd_y, vd_z); scan_u_x.m_vd_dx = vd_dx; scan_u_x.m_vd_dy = vd_dy; scan_u_x.m_vd_dz = vd_dz;
						scan_u_y.allocate(vd_x, vd_y, vd_z); scan_u_y.m_vd_dx = vd_dx; scan_u_y.m_vd_dy = vd_dy; scan_u_y.m_vd_dz = vd_dz;
						scan_u_z.allocate(vd_x, vd_y, vd_z); scan_u_z.m_vd_dx = vd_dx; scan_u_z.m_vd_dy = vd_dy; scan_u_z.m_vd_dz = vd_dz;
						for (n = 0; n < vd_z; n++) {
							for (m = 0; m < vd_y; m++) {
								for (l = 0; l < vd_x; l++) {
									// mm -> voxel
									scan_u_x.m_pData[n][m][l][0] = scan_u_v3.m_pData[n][m][l][0] * _vd_dx;
									scan_u_y.m_pData[n][m][l][0] = scan_u_v3.m_pData[n][m][l][1] * _vd_dy;
									scan_u_z.m_pData[n][m][l][0] = scan_u_v3.m_pData[n][m][l][2] * _vd_dz;
								}
							}
						}
					}
					//
          clock_end = clock();
          //std::cout << "===Time stamp 39: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

					scan_u_v3.clear();
				}
				if (!LoadMHDData(NULL, scan_h_hdr_file, 
					&scan_h_v3.m_pData, scan_h_v3.m_vd_x, scan_h_v3.m_vd_y, scan_h_v3.m_vd_z, scan_h_v3.m_vd_s, scan_h_v3.m_vd_dx, scan_h_v3.m_vd_dy, scan_h_v3.m_vd_dz, vd_ox, vd_oy, vd_oz)) {
					TRACE("Loading %s failed..\n", scan_h_hdr_file);
				} else  {
					scan_h_v3.computeDimension();
					//
					{
						scan_h_x.allocate(vd_x, vd_y, vd_z); scan_h_x.m_vd_dx = vd_dx; scan_h_x.m_vd_dy = vd_dy; scan_h_x.m_vd_dz = vd_dz;
						scan_h_y.allocate(vd_x, vd_y, vd_z); scan_h_y.m_vd_dx = vd_dx; scan_h_y.m_vd_dy = vd_dy; scan_h_y.m_vd_dz = vd_dz;
						scan_h_z.allocate(vd_x, vd_y, vd_z); scan_h_z.m_vd_dx = vd_dx; scan_h_z.m_vd_dy = vd_dy; scan_h_z.m_vd_dz = vd_dz;
						for (n = 0; n < vd_z; n++) {
							for (m = 0; m < vd_y; m++) {
								for (l = 0; l < vd_x; l++) {
									// mm -> voxel
									scan_h_x.m_pData[n][m][l][0] = scan_h_v3.m_pData[n][m][l][0] * _vd_dx;
									scan_h_y.m_pData[n][m][l][0] = scan_h_v3.m_pData[n][m][l][1] * _vd_dy;
									scan_h_z.m_pData[n][m][l][0] = scan_h_v3.m_pData[n][m][l][2] * _vd_dz;
								}
							}
						}
					}
					//
          clock_end = clock();
          //std::cout << "===Time stamp 40: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
          std::cout << "===Percentage Done : 99.00\n";

					scan_h_v3.clear();
				}

				ConcatenateFields(scan_h_x, scan_h_y, scan_h_z, scan_u_x, scan_u_y, scan_u_z, scan_hu_x, scan_hu_y, scan_hu_z);

				scan_h_x.clear(); scan_h_y.clear(); scan_h_z.clear();
				scan_u_x.clear(); scan_u_y.clear(); scan_u_z.clear();
			}
			//
#ifdef UPDATE_DEFORMATION_FIELD
			if (update_deform) {
				char scan_atlas_reg_tumor_mask_file[1024];
				char scan_atlas_reg_tumor_prob_file[1024];
				char scan_atlas_reg_abnor_mask_file[1024];
				char scan_atlas_reg_abnor_prob_file[1024];
				char h2u_hdr_file[1024];
				char h2u_r_hdr_file[1024];
				char h2u_log_name[1024];
				//
				FVolume atlas_template;
				FVolume scan_atlas_reg_image_0;
				FVolume atlas_template_abnor_mask;
				FVolume atlas_template_abnor_prob;
				FVolume scan_atlas_reg_tumor_mask;
				FVolume scan_atlas_reg_tumor_prob;
				FVolume scan_atlas_reg_abnor_mask;
				FVolume scan_atlas_reg_abnor_prob;
				//
				int c;
				bool save_int = false;
				bool save_pyrd = true;

				sprintf(scan_atlas_reg_tumor_mask_file, "%s%sscan_atlas_reg_tumor_mask.nii.gz", tmp_folder, DIR_SEP);
				sprintf(scan_atlas_reg_tumor_prob_file, "%s%sscan_atlas_reg_tumor_prob.nii.gz", tmp_folder, DIR_SEP);
				sprintf(scan_atlas_reg_abnor_mask_file, "%s%sscan_atlas_reg_abnor_mask.nii.gz", tmp_folder, DIR_SEP);
				sprintf(scan_atlas_reg_abnor_prob_file, "%s%sscan_atlas_reg_abnor_prob.nii.gz", tmp_folder, DIR_SEP);
				//
				sprintf(h2u_hdr_file  , "%s%ss_h2u.mhd"  , tmp_folder, DIR_SEP);
				sprintf(h2u_r_hdr_file, "%s%ss_h2u_r.mhd", tmp_folder, DIR_SEP);
				sprintf(h2u_log_name  , "%s%ss_h2u"      , tmp_folder, DIR_SEP);

				scan_atlas_reg_image_0.load(scan_atlas_reg_image_masked_files[0], 1);
				atlas_template.load(atlas_template_file, 1);
				//
				scan_atlas_reg_tumor_mask.allocate(vd_x, vd_y, vd_z);
				scan_atlas_reg_tumor_prob.allocate(vd_x, vd_y, vd_z);
				scan_atlas_reg_abnor_mask.allocate(vd_x, vd_y, vd_z);
				scan_atlas_reg_abnor_prob.allocate(vd_x, vd_y, vd_z);
				atlas_template_abnor_mask.allocate(vd_x, vd_y, vd_z);
				atlas_template_abnor_prob.allocate(vd_x, vd_y, vd_z);

				// get tumor mask for scan 0
				{
					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_posteriors[i].load(scan_atlas_reg_posterior_files[i], 1);
					}

					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								float max_l_val = scan_atlas_reg_posteriors[0].m_pData[n][m][l][0];
								int max_l_idx = 0;
								for (c = 1; c < NumberOfPriorChannels; c++) {
									// prefer higher labels
									if (max_l_val <= scan_atlas_reg_posteriors[c].m_pData[n][m][l][0]) {
										max_l_val = scan_atlas_reg_posteriors[c].m_pData[n][m][l][0];
										max_l_idx = c;
									}
								}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								if (max_l_idx == TU || max_l_idx == NCR || max_l_idx == NE) {
#else
								if (max_l_idx == TU || max_l_idx == NCR) {
#endif
									scan_atlas_reg_tumor_mask.m_pData[n][m][l][0] = 1;
								} else {
									scan_atlas_reg_tumor_mask.m_pData[n][m][l][0] = 0;
								}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								if (max_l_idx == TU || max_l_idx == NCR || max_l_idx == NE || max_l_idx == ED) {
#else
								if (max_l_idx == TU || max_l_idx == NCR || max_l_idx == ED) {
#endif
									scan_atlas_reg_abnor_mask.m_pData[n][m][l][0] = 1;
								} else {
									scan_atlas_reg_abnor_mask.m_pData[n][m][l][0] = 0;
								}
								//
								scan_atlas_reg_tumor_prob.m_pData[n][m][l][0] = 0;
								scan_atlas_reg_tumor_prob.m_pData[n][m][l][0] += scan_atlas_reg_posteriors[TU ].m_pData[n][m][l][0];
								scan_atlas_reg_tumor_prob.m_pData[n][m][l][0] += scan_atlas_reg_posteriors[NCR].m_pData[n][m][l][0];
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								scan_atlas_reg_tumor_prob.m_pData[n][m][l][0] += scan_atlas_reg_posteriors[NE ].m_pData[n][m][l][0];
#endif
								//
								if (scan_atlas_reg_tumor_prob.m_pData[n][m][l][0] > 1.0f) {
									scan_atlas_reg_tumor_prob.m_pData[n][m][l][0] = 1.0f;
								}
								//
								scan_atlas_reg_abnor_prob.m_pData[n][m][l][0] = 0;
								scan_atlas_reg_abnor_prob.m_pData[n][m][l][0] += scan_atlas_reg_posteriors[TU ].m_pData[n][m][l][0];
								scan_atlas_reg_abnor_prob.m_pData[n][m][l][0] += scan_atlas_reg_posteriors[NCR].m_pData[n][m][l][0];
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								scan_atlas_reg_abnor_prob.m_pData[n][m][l][0] += scan_atlas_reg_posteriors[NE ].m_pData[n][m][l][0];
#endif
								scan_atlas_reg_abnor_prob.m_pData[n][m][l][0] += scan_atlas_reg_posteriors[ED ].m_pData[n][m][l][0];
								//
								if (scan_atlas_reg_abnor_prob.m_pData[n][m][l][0] > 1.0f) {
									scan_atlas_reg_abnor_prob.m_pData[n][m][l][0] = 1.0f;
								}
								//
								atlas_template_abnor_mask.m_pData[n][m][l][0] = 0;
								atlas_template_abnor_prob.m_pData[n][m][l][0] = 0;
							}
						}
					}

          clock_end = clock();
          //std::cout << "===Time stamp 41: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

					scan_atlas_reg_tumor_mask.save(scan_atlas_reg_tumor_mask_file, 1);
					ChangeNIIHeader(scan_atlas_reg_tumor_mask_file, scan_atlas_reg_image_files[0]);
					scan_atlas_reg_tumor_prob.save(scan_atlas_reg_tumor_prob_file, 1);
					ChangeNIIHeader(scan_atlas_reg_tumor_prob_file, scan_atlas_reg_image_files[0]);
					scan_atlas_reg_abnor_mask.save(scan_atlas_reg_abnor_mask_file, 1);
					ChangeNIIHeader(scan_atlas_reg_abnor_mask_file, scan_atlas_reg_image_files[0]);
					scan_atlas_reg_abnor_prob.save(scan_atlas_reg_abnor_prob_file, 1);
					ChangeNIIHeader(scan_atlas_reg_abnor_prob_file, scan_atlas_reg_image_files[0]);

					for (i = 0; i < NumberOfPriorChannels; i++) {
						scan_atlas_reg_posteriors[i].clear();
					}
				}

				// fill abnormal region with average wm value
				{
					float scan_mean_wm[NumberOfImageChannels];

					LoadMeansAndVariances(scan_means_file, scan_variances_file, &scan_MeanVector, &scan_VarianceVector);

					for (i = 0; i < NumberOfImageChannels; i++) {
						scan_mean_wm[i] = scan_MeanVector.at(WM)[i];
					}

					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								double alpha, v0;

								alpha = scan_atlas_reg_abnor_prob.m_pData[n][m][l][0];
								if (alpha > 0) {
									v0 = scan_atlas_reg_image_0.m_pData[n][m][l][0];
									scan_atlas_reg_image_0.m_pData[n][m][l][0] = alpha * scan_mean_wm[0] + (1.0-alpha) * v0;
								}
							}
						}
					}
				}

        clock_end = clock();
        //std::cout << "===Time stamp 42: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

#ifdef USE_INIT_FIELD
				// weight abnormal region
#ifdef USE_INIT_FIELD_WEIGHT_ABNORMAL_REGION
				{
					FVolume prob_g;
					prob_g.allocate(vd_x, vd_y, vd_z);

					//scan_atlas_reg_abnor_prob.GaussianSmoothing(prob_g, 3.0, 9);
					scan_atlas_reg_tumor_prob.GaussianSmoothing(prob_g, 3.0, 9);
					/*
					{
						char str_tmp[1024];
						sprintf(str_tmp, "%s%stest_prob_g.nii.gz", tmp_folder, DIR_SEP);
						scan_atlas_reg_tumor_prob.save(str_tmp, 1);
					}
					//*/

					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								double abnor_prob;

								abnor_prob  = prob_g.m_pData[n][m][l][0];

								scan_hu_x.m_pData[n][m][l][0] = scan_hu_x.m_pData[n][m][l][0] * abnor_prob;
								scan_hu_y.m_pData[n][m][l][0] = scan_hu_y.m_pData[n][m][l][0] * abnor_prob;
								scan_hu_z.m_pData[n][m][l][0] = scan_hu_z.m_pData[n][m][l][0] * abnor_prob;
							}
						}
					}
					/*
					{
						char str_tmp[1024];
						sprintf(str_tmp, "%s%stest_h1u.mhd", tmp_folder, DIR_SEP);
						SaveMHDData(NULL, str_tmp, scan_hu_x.m_pData, scan_hu_y.m_pData, scan_hu_z.m_pData, scan_hu_x.m_vd_x, scan_hu_x.m_vd_y, scan_hu_x.m_vd_z, scan_hu_x.m_vd_dx, scan_hu_x.m_vd_dy, scan_hu_x.m_vd_dz, 0, 0, 0);
					}
					//*/

					prob_g.clear();

					SmoothField(scan_hu_x, scan_hu_y, scan_hu_z, scan_hu_x, scan_hu_y, scan_hu_z, 3.0);
					/*
					{
						char str_tmp[1024];
						sprintf(str_tmp, "%s%stest_h1u_s.mhd", tmp_folder, DIR_SEP);
						SaveMHDData(NULL, str_tmp, scan_hu_x.m_pData, scan_hu_y.m_pData, scan_hu_z.m_pData, scan_hu_x.m_vd_x, scan_hu_x.m_vd_y, scan_hu_x.m_vd_z, scan_hu_x.m_vd_dx, scan_hu_x.m_vd_dy, scan_hu_x.m_vd_dz, 0, 0, 0);
					}
					//*/
				}
#endif
        clock_end = clock();
        //std::cout << "===Time stamp 43: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

				/*
				// caution: this function performs bad
				ReverseDeformationField(scan_hu_x, scan_hu_y, scan_hu_z, scan_hu_r_x, scan_hu_r_y, scan_hu_r_z);
				ReverseDeformationField(scan_hu_r_x, scan_hu_r_y, scan_hu_r_z, scan_hu_x, scan_hu_y, scan_hu_z);
				/*/
				ReverseField(scan_hu_x, scan_hu_y, scan_hu_z, scan_hu_r_x, scan_hu_r_y, scan_hu_r_z);
				ReverseField(scan_hu_r_x, scan_hu_r_y, scan_hu_r_z, scan_hu_x, scan_hu_y, scan_hu_z);
				//*/
				//
				SaveMHDData(NULL, h1u_r_hdr_file, scan_hu_r_x.m_pData, scan_hu_r_y.m_pData, scan_hu_r_z.m_pData, scan_hu_r_x.m_vd_x, scan_hu_r_x.m_vd_y, scan_hu_r_x.m_vd_z, scan_hu_r_x.m_vd_dx, scan_hu_r_x.m_vd_dy, scan_hu_r_x.m_vd_dz, 0, 0, 0);
				SaveMHDData(NULL, h1u_hdr_file, scan_hu_x.m_pData, scan_hu_y.m_pData, scan_hu_z.m_pData, scan_hu_x.m_vd_x, scan_hu_x.m_vd_y, scan_hu_x.m_vd_z, scan_hu_x.m_vd_dx, scan_hu_x.m_vd_dy, scan_hu_x.m_vd_dz, 0, 0, 0);
#else
				{
					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								scan_hu_x.m_pData[n][m][l][0] = 0;
								scan_hu_y.m_pData[n][m][l][0] = 0;
								scan_hu_z.m_pData[n][m][l][0] = 0;
							}
						}
					}
				}
#endif

				{
					UpdateDeformationFieldSyM(&scan_atlas_reg_image_0, &atlas_template, 1, 
						NULL, NULL, scan_atlas_reg_abnor_prob, atlas_template_abnor_prob, 
						scan_hu_x, scan_hu_y, scan_hu_z, scan_hu_r_x, scan_hu_r_y, scan_hu_r_z,
						h2u_log_name, 20, 11, 3, 5, 1, false, save_int, save_pyrd, lambda_D);
				}

				SaveMHDData(NULL, h2u_hdr_file, scan_hu_x.m_pData, scan_hu_y.m_pData, scan_hu_z.m_pData, scan_hu_x.m_vd_x, scan_hu_x.m_vd_y, scan_hu_x.m_vd_z, scan_hu_x.m_vd_dx, scan_hu_x.m_vd_dy, scan_hu_x.m_vd_dz, 0, 0, 0);
				SaveMHDData(NULL, h2u_r_hdr_file, scan_hu_r_x.m_pData, scan_hu_r_y.m_pData, scan_hu_r_z.m_pData, scan_hu_r_x.m_vd_x, scan_hu_r_x.m_vd_y, scan_hu_r_x.m_vd_z, scan_hu_r_x.m_vd_dx, scan_hu_r_x.m_vd_dy, scan_hu_r_x.m_vd_dz, 0, 0, 0);

				scan_atlas_reg_image_0.clear();
				atlas_template.clear();
				//
				scan_atlas_reg_tumor_mask.clear();
				scan_atlas_reg_tumor_prob.clear();
				scan_atlas_reg_abnor_mask.clear();
				scan_atlas_reg_abnor_prob.clear();
				atlas_template_abnor_mask.clear();
				atlas_template_abnor_prob.clear();
			} else {
				SaveMHDData(NULL, h1u_hdr_file, scan_hu_x.m_pData, scan_hu_y.m_pData, scan_hu_z.m_pData, scan_hu_x.m_vd_x, scan_hu_x.m_vd_y, scan_hu_x.m_vd_z, scan_hu_x.m_vd_dx, scan_hu_x.m_vd_dy, scan_hu_x.m_vd_dz, 0, 0, 0);
			}
#else
			SaveMHDData(NULL, h1u_hdr_file, scan_hu_x.m_pData, scan_hu_y.m_pData, scan_hu_z.m_pData, scan_hu_x.m_vd_x, scan_hu_x.m_vd_y, scan_hu_x.m_vd_z, scan_hu_x.m_vd_dx, scan_hu_x.m_vd_dy, scan_hu_x.m_vd_dz, 0, 0, 0);
#endif
			//
			scan_hu_x.clear(); scan_hu_y.clear(); scan_hu_z.clear();
			scan_hu_r_x.clear(); scan_hu_r_y.clear(); scan_hu_r_z.clear();
		}
		/////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////
	}
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////

  clock_end = clock();
  //std::cout << "===Time stamp 44: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
  std::cout << "===Percentage Done : 99.10\n";
	/////////////////////////////////////////////////////////////////////////////
	// transform results
	/////////////////////////////////////////////////////////////////////////////
	{
		putenv((char*)"FSLOUTPUTTYPE=NIFTI_GZ");
		//
		// get reverse mapping
		{
			sprintf(szCmdLine, "%s -omat %s -inverse %s", 
				CONVERT_XFM_PATH, atlas_to_scan_mat_file, scan_to_atlas_mat_file);
			TRACE("%s\n", szCmdLine);
			//
			if (!ExecuteProcess(szCmdLine)) {
				TRACE("failed\n");
				exit(EXIT_FAILURE);
			}
		}
		//
		// transform prior and posterior
		for (i = 0; i < NumberOfPriorChannels; i++) {
			sprintf(szCmdLine, "%s -in %s -ref %s -out %s -init %s -datatype float -applyxfm", 
				FLIRT_PATH, scan_atlas_reg_warp_prior_files[i], scan_image_masked_files[0], scan_prior_files[i], atlas_to_scan_mat_file);
			TRACE("%s\n", szCmdLine);
			//
			if (!ExecuteProcess(szCmdLine)) {
				TRACE("failed\n");
				exit(EXIT_FAILURE);
			}
		}
		for (i = 0; i < NumberOfPriorChannels; i++) {
			sprintf(szCmdLine, "%s -in %s -ref %s -out %s -init %s -datatype float -applyxfm", 
				FLIRT_PATH, scan_atlas_reg_posterior_files[i], scan_image_masked_files[0], scan_posterior_files[i], atlas_to_scan_mat_file);
			TRACE("%s\n", szCmdLine);
			//
			if (!ExecuteProcess(szCmdLine)) {
				TRACE("failed\n");
				exit(EXIT_FAILURE);
			}
		}
		//
    clock_end = clock();
    //std::cout << "===Time stamp 45: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
    std::cout << "===Percentage Done : 99.20\n";

		// make label map
		{
			int vd_x, vd_y, vd_z;

			for (i = 0; i < NumberOfPriorChannels; i++) {
				scan_posteriors[i].load(scan_posterior_files[i], 1);
			}
			//
			vd_x = scan_posteriors[0].m_vd_x;
			vd_y = scan_posteriors[0].m_vd_y;
			vd_z = scan_posteriors[0].m_vd_z;
			//
			scan_label_map.allocate(vd_x, vd_y, vd_z);
			//
			ComputeLabelMap(scan_posteriors, scan_label_map, b_valid_label);
			//
#ifdef USE_LABEL_NOISE_REDUCTION
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
			if (l_noise_reduction && b_valid_label[ED] > 0 && (b_valid_label[NCR] > 0 || b_valid_label[TU] > 0 || b_valid_label[NE] > 0)) 
#else
			if (l_noise_reduction && b_valid_label[ED] > 0 && (b_valid_label[NCR] > 0 || b_valid_label[TU] > 0)) 
#endif
			{
				int fore_num = 3;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				unsigned char fore_label[3][4] = { { label_idx[ED], label_idx[NCR], label_idx[TU], label_idx[NE] }, { label_idx[ED] }, { label_idx[NCR], label_idx[TU], label_idx[NE] } };
				int fore_label_num[3] = { 4, 1, 3 };
				int cc_num_th[3] = { CC_NUM_TH*4, CC_NUM_TH, CC_NUM_TH/4 };
#else
				unsigned char fore_label[3][3] = { { label_idx[ED], label_idx[NCR], label_idx[TU] }, { label_idx[ED] }, { label_idx[NCR], label_idx[TU] } };
				int fore_label_num[3] = { 3, 1, 2 };
				int cc_num_th[3] = { CC_NUM_TH*4, CC_NUM_TH, CC_NUM_TH/4 };
#endif
				for (k = 0; k < fore_num; k++) {
					SVolume cc;
					int cc_sum[MAXNCC];
					int cc_num;
					int cc_remove[MAXNCC];
					int cc_num_max, cc_num_max_idx;

					cc.allocate(vd_x, vd_y, vd_z);
				
					GetConnectedComponents(scan_label_map, fore_label[k], fore_label_num[k], cc, cc_sum, &cc_num);
					if (cc_num == 0) {
						continue;
					}

          clock_end = clock();
          //std::cout << "===Time stamp 46: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
          std::cout << "===Percentage Done : 99.30\n";

					/*
					{
						char str_tmp[1024];
						sprintf(str_tmp, "%s%scc_%d.nii.gz", tmp_folder, DIR_SEP, k);
						cc.save(str_tmp, 1);
					}
					//*/

					cc_num_max = 0;
					cc_num_max_idx = -1;
					for (i = 0; i < cc_num; i++) {
						if (cc_num_max < cc_sum[i]) {
							cc_num_max = cc_sum[i];
							cc_num_max_idx = i;
						}
						//
						if (cc_sum[i] < cc_num_th[k]) {
							cc_remove[i] = 1;
						} else {
							cc_remove[i] = 0;
						}
					}
					// keep maximum cc
					if (cc_num_max_idx != -1) {
						cc_remove[cc_num_max_idx] = 0;
					}

					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								int cc_val = cc.m_pData[n][m][l][0];
								if (cc_val < cc_num) {
									if (!cc_remove[cc_val]) {
										continue;
									}
									// change label
									float max_l_val = 0;
									int max_l_idx = -1;
									for (i = 0; i < NumberOfPriorChannels; i++) {
										if (b_valid_label[i] == 0) {
											continue;
										}
										//
										bool bMatched = false;
										for (j = 0; j < fore_label_num[k]; j++) {
											if (fore_label[k][j] == label_idx[i]) {
												bMatched = true;
											}
										}
										if (bMatched) {
											continue;
										}
										//
										if (max_l_val < scan_posteriors[i].m_pData[n][m][l][0]) {
											max_l_val = scan_posteriors[i].m_pData[n][m][l][0];
											max_l_idx = i;
										} else if (max_l_val == scan_posteriors[i].m_pData[n][m][l][0]) {
											// prefer TU or higher labels
											if ((max_l_idx == TU && i == NCR) || (max_l_idx == NCR && i == TU)) {
												if (b_valid_label[TU] > 0) {
													max_l_idx = TU;
												} else if (b_valid_label[NCR] > 0) {
													max_l_idx = NCR;
												} else {
													max_l_idx = TU;
												}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
											} else if ((max_l_idx == TU && i == NE) || (max_l_idx == NE && i == TU)) {
												if (b_valid_label[TU] > 0) {
													max_l_idx = TU;
												} else if (b_valid_label[NE] > 0) {
													max_l_idx = NE;
												} else {
													max_l_idx = TU;
												}
											} else if ((max_l_idx == NCR && i == NE) || (max_l_idx == NE && i == NCR)) {
												if (b_valid_label[NCR] > 0) {
													max_l_idx = NCR;
												} else if (b_valid_label[NE] > 0) {
													max_l_idx = NE;
												} else {
													max_l_idx = NCR;
												}
#endif
											} else {
												max_l_idx = i;
											}
										}
									}
									scan_label_map.m_pData[n][m][l][0] = label_idx[max_l_idx];
								}
							}
						}
					}
				}

        clock_end = clock();
        //std::cout << "===Time stamp 47: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
        std::cout << "===Percentage Done : 99.40\n";

				//*
				{
					char str_tmp[1024];
					if (output_tmp_label) {
						sprintf(str_tmp, "%s%sscan_label_map_cc.nii.gz", out_folder, DIR_SEP);
					} else {
						sprintf(str_tmp, "%s%sscan_label_map_cc.nii.gz", tmp_folder, DIR_SEP);
					}
					scan_label_map.save(str_tmp, 1);
					ChangeNIIHeader(str_tmp, scan_image_masked_files[0]);
				}
				//*/
			}
#endif
			//
#ifdef USE_LABEL_CORRECTION
			if (lc_edema || lc_tumor)
			{
				FVolume scan_images[NumberOfImageChannels];
				double vv[NumberOfPriorChannels][16];
				double vv_inv[NumberOfPriorChannels][16];
				double vv_det[NumberOfPriorChannels], vv_c1[NumberOfPriorChannels], vv_c2[NumberOfPriorChannels];
				double mv[NumberOfPriorChannels][4];

				for (i = 0; i < NumberOfImageChannels; i++) {
					scan_images[i].load(scan_image_masked_files[i], 1);
				}

				LoadMeansAndVariances(scan_means_file, scan_variances_file, &scan_MeanVector, &scan_VarianceVector);

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

				if (lc_edema && b_valid_label[ED] > 0) 
				{
					BVolume scan_label_map_tmp;
					int iter;
					int label_corr_num = 10;
					int vd_x_1, vd_y_1, vd_z_1;
#if 0
					int corr_set_a[2] = { label_idx[CSF], label_idx[VS] }; // labels to be corrected from
					bool corr_set_a_found[2];
					int corr_set_a_idx[2] = { CSF, VS };
					int corr_set_a_num = 2;
#else
					int corr_set_a[3] = { label_idx[CSF], label_idx[VS], label_idx[GM] }; // labels to be corrected from
					bool corr_set_a_found[3];
					int corr_set_a_idx[3] = { CSF, VS, GM };
					int corr_set_a_num = 3;
#endif
					int corr_set_b[1] = { label_idx[ED] }; // labels to be corrected to
					bool corr_set_b_found[1];
					int corr_set_b_idx[1] = { ED };
					int corr_set_b_num = 1;

					vd_x_1 = vd_x - 1;
					vd_y_1 = vd_y - 1;
					vd_z_1 = vd_z - 1;					

					for (iter = 0; iter < label_corr_num; iter++) {
						scan_label_map_tmp.copy(scan_label_map);
						for (k = 1; k < vd_z_1; k++) {
							for (j = 1; j < vd_y_1; j++) {
								for (i = 1; i < vd_x_1; i++) {
									bool bFound;
									//
									bFound = false;
									for (l = 0; l < corr_set_a_num; l++) {
										corr_set_a_found[l] = false;
										if (scan_label_map_tmp.m_pData[k][j][i][0] == corr_set_a[l]) {
											corr_set_a_found[l] = true;
											bFound = true;
										}
									}
									if (!bFound) { continue; }
									//
									bFound = false;
									for (l = 0; l < corr_set_b_num; l++) {
										corr_set_b_found[l] = false;
										if (   scan_label_map_tmp.m_pData[k  ][j  ][i-1][0] == corr_set_b[l]
											|| scan_label_map_tmp.m_pData[k  ][j  ][i+1][0] == corr_set_b[l]
											|| scan_label_map_tmp.m_pData[k  ][j-1][i  ][0] == corr_set_b[l]
											|| scan_label_map_tmp.m_pData[k  ][j+1][i  ][0] == corr_set_b[l]
											|| scan_label_map_tmp.m_pData[k-1][j  ][i  ][0] == corr_set_b[l]
											|| scan_label_map_tmp.m_pData[k+1][j  ][i  ][0] == corr_set_b[l]) {
											corr_set_b_found[l] = true;
											bFound = true;
										}
									}
									if (!bFound) { continue; }
									//
									double y[NumberOfImageChannels], ym[NumberOfImageChannels];
									double like, like_max;
									int like_max_idx;
									for (l = 0; l < NumberOfImageChannels; l++) {
										y[l] = scan_images[l].m_pData[k][j][i][0];
									}
									like_max = 0;
									like_max_idx = -1;
									for (l = 0; l < corr_set_a_num; l++) {
										if (!corr_set_a_found[l]) { continue; }
										int idx = corr_set_a_idx[l];
										ym[0] = y[0] - mv[idx][0]; 
										ym[1] = y[1] - mv[idx][1]; 
										ym[2] = y[2] - mv[idx][2]; 
										ym[3] = y[3] - mv[idx][3];
										ComputeLikelihood4(vv_inv[idx], vv_c1[idx], vv_c2[idx], ym, &like);
										if (like_max < like) {
											like_max = like;
											like_max_idx = corr_set_a_idx[l];
										}
									}
									for (l = 0; l < corr_set_b_num; l++) {
										if (!corr_set_b_found[l]) { continue; }
										int idx = corr_set_b_idx[l];
										ym[0] = y[0] - mv[idx][0]; 
										ym[1] = y[1] - mv[idx][1]; 
										ym[2] = y[2] - mv[idx][2]; 
										ym[3] = y[3] - mv[idx][3]; 
										ComputeLikelihood4(vv_inv[idx], vv_c1[idx], vv_c2[idx], ym, &like);
										if (like_max < like) {
											like_max = like;
											like_max_idx = corr_set_b_idx[l];
										}
									}
									if (like_max_idx != -1) {
										scan_label_map.m_pData[k][j][i][0] = label_idx[like_max_idx];
									}
								}
							}
						}
					}

          clock_end = clock();
          //std::cout << "===Time stamp 48: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
          std::cout << "===Percentage Done : 99.50\n";

					//*
					{
						char str_tmp[1024];
						if (output_tmp_label) {
							sprintf(str_tmp, "%s%sscan_label_map_corr_ed.nii.gz", out_folder, DIR_SEP);
						} else {
							sprintf(str_tmp, "%s%sscan_label_map_corr_ed.nii.gz", tmp_folder, DIR_SEP);
						}
						scan_label_map.save(str_tmp, 1);
						ChangeNIIHeader(str_tmp, scan_image_masked_files[0]);
					}
					//*/
				}
				//
				// change NE label to cavity
				if (use_ne_as_cavity) {
					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								if (scan_label_map.m_pData[n][m][l][0] == label_idx[NE]) {
									scan_label_map.m_pData[n][m][l][0] = cavity_label_val;
								}
							}
						}
					}

					//*
					{
						char str_tmp[1024];
						if (output_tmp_label) {
							sprintf(str_tmp, "%s%sscan_label_map_corr_nac.nii.gz", out_folder, DIR_SEP);
						} else {
							sprintf(str_tmp, "%s%sscan_label_map_corr_nac.nii.gz", tmp_folder, DIR_SEP);
						}
						scan_label_map.save(str_tmp, 1);
						ChangeNIIHeader(str_tmp, scan_image_masked_files[0]);
					}
					//*/
          clock_end = clock();
          //std::cout << "===Time stamp 49: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple

				}
				//
				// estimate enhancing regions using T1-CE only
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				if (lc_tumor && b_valid_label[TU] > 0 && (b_valid_label[NCR] > 0 || b_valid_label[NE] > 0)) 
#else
				if (lc_tumor && b_valid_label[TU] > 0 && b_valid_label[NCR] > 0) 
#endif
				{
					SVolume tu_fore1; // entire tumor region
					SVolume tu_fore2; // internal tumor region: inside -> tu, ncr, ne, outside -> tu, ne, ncr, ed
					SVolume tu_fore3; // internal tumor region
					double t_sum[NumberOfPriorChannels], t_sum2[NumberOfPriorChannels];
					double t_mean[NumberOfPriorChannels], t_var[NumberOfPriorChannels];
					int t_num[NumberOfPriorChannels];

					tu_fore1.allocate(vd_x, vd_y, vd_z);
					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								tu_fore1.m_pData[n][m][l][0] = 0;
								if (scan_label_map.m_pData[n][m][l][0] == label_idx[TU]) {
									tu_fore1.m_pData[n][m][l][0] = 1;
									continue;
								}
								if (scan_label_map.m_pData[n][m][l][0] == label_idx[NCR]) {
									tu_fore1.m_pData[n][m][l][0] = 1;
									continue;
								}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								if (scan_label_map.m_pData[n][m][l][0] == label_idx[NE]) {
									tu_fore1.m_pData[n][m][l][0] = 1;
									continue;
								}
#endif
							}
						}
					}
					/*
					{
						char str_tmp[1024];
						sprintf(str_tmp, "%s%sscan_label_map_corr_tu_fore1.nii.gz", tmp_folder, DIR_SEP);
						tu_fore1.save(str_tmp, 1);
						ChangeNIIHeader(str_tmp, scan_image_masked_files[0]);
					}
					//*/
					tu_fore2 = tu_fore1;
					ErodeVolume(tu_fore2, 1);
					tu_fore3 = tu_fore2;
					ErodeVolume(tu_fore3, 1);
					/*
					{
						char str_tmp[1024];
						sprintf(str_tmp, "%s%sscan_label_map_corr_tu_fore2.nii.gz", tmp_folder, DIR_SEP);
						tu_fore2.save(str_tmp, 1);
						ChangeNIIHeader(str_tmp, scan_image_masked_files[0]);
						sprintf(str_tmp, "%s%sscan_label_map_corr_tu_fore3.nii.gz", tmp_folder, DIR_SEP);
						tu_fore3.save(str_tmp, 1);
						ChangeNIIHeader(str_tmp, scan_image_masked_files[0]);
					}
					//*/

          clock_end = clock();
          std::cout << "===Time stamp 50: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
          std::cout << "===Percentage Done : 99.60\n";

					for (i = 0; i < NumberOfPriorChannels; i++) {
						t_sum[i] = t_sum2[i] = 0;
						t_mean[i] = t_var[i] = 0;
						t_num[i] = 0;
					}
					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								double t_val = scan_images[1].m_pData[n][m][l][0];
								if (scan_label_map.m_pData[n][m][l][0] == label_idx[TU]) {
									t_sum[TU] += t_val;
									t_sum2[TU] += t_val*t_val;
									t_num[TU]++;
								}
								if (scan_label_map.m_pData[n][m][l][0] == label_idx[NCR]) {
									t_sum[NCR] += t_val;
									t_sum2[NCR] += t_val*t_val;
									t_num[NCR]++;
								}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								if (scan_label_map.m_pData[n][m][l][0] == label_idx[NE]) {
									t_sum[NE] += t_val;
									t_sum2[NE] += t_val*t_val;
									t_num[NE]++;
								}
#endif
							}
						}
					}
					for (i = 0; i < NumberOfPriorChannels; i++) {
						if (t_num[i] > 0) {
							t_mean[i] = t_sum[i] / t_num[i];
							t_var[i] = t_sum2[i] / t_num[i] - t_mean[i] * t_mean[i];
							TRACE2("%s: mean %f, var %f\n", label[i], t_mean[i], t_var[i]);
						}
					}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
					if (b_valid_label[NCR] == 0) {
						t_mean[NCR] = t_mean[NE];
						t_var[NCR] = t_var[NE];
					}
					if (b_valid_label[NE] == 0) {
						t_mean[NE] = t_mean[NCR];
						t_var[NE] = t_var[NCR];
					}
#endif

					for (n = 0; n < vd_z; n++) {
						for (m = 0; m < vd_y; m++) {
							for (l = 0; l < vd_x; l++) {
								if (tu_fore1.m_pData[n][m][l][0] == 0) {
									continue;
								}

								double y[NumberOfImageChannels];
								double ym_tu[NumberOfImageChannels], like_tu;
								double ym_ncr[NumberOfImageChannels], like_ncr;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								double ym_ne[NumberOfImageChannels], like_ne;
#endif
								double ym_ed[NumberOfImageChannels], like_ed;
								for (i = 0; i < NumberOfImageChannels; i++) {
									y[i] = scan_images[i].m_pData[n][m][l][0];
									//
									ym_tu[i]  = y[i] - mv[TU ][i];
									ym_ncr[i] = y[i] - mv[NCR][i];
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
									ym_ne[i]  = y[i] - mv[NE ][i];
#endif
									ym_ed[i]  = y[i] - mv[ED ][i];
								}

#if 1
								// early return case
								if (y[1] >= t_mean[TU]) {
									scan_label_map.m_pData[n][m][l][0] = label_idx[TU];
									continue;
								}
								if (b_valid_label[NCR] > 0 && y[1] <= t_mean[NCR]) {
									if (tu_fore3.m_pData[n][m][l][0] > 0) {
										scan_label_map.m_pData[n][m][l][0] = label_idx[NCR];
										continue;
									}
								}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								if (b_valid_label[NCR] == 0 && b_valid_label[NE] > 0 && y[1] <= t_mean[NE]) {
									if (!use_ne_as_cavity) {
										scan_label_map.m_pData[n][m][l][0] = label_idx[NE];
									} else {
										scan_label_map.m_pData[n][m][l][0] = cavity_label_val;
									}
									continue;
								}
#endif
								like_tu  = exp(-0.5 * (y[1]-t_mean[TU ])  * (y[1]-t_mean[TU ]) / t_var[TU ]) / sqrt(t_var[TU ] * PIM2);
								like_ncr = exp(-0.5 * (y[1]-t_mean[NCR])  * (y[1]-t_mean[NCR]) / t_var[NCR]) / sqrt(t_var[NCR] * PIM2);
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								like_ne  = exp(-0.5 * (y[1]-t_mean[NE ])  * (y[1]-t_mean[NE ]) / t_var[NE ]) / sqrt(t_var[NE ] * PIM2);
#endif
#endif
#if 0
								ComputeLikelihood1(vv[TU ][5], ym_tu[1],  &like_tu);
								ComputeLikelihood1(vv[NCR][5], ym_ncr[1], &like_ncr);
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								ComputeLikelihood1(vv[NE][5], ym_ne[1], &like_ne);
#endif
#endif
#if 0
								like_tu  = exp(-ym_tu[1]  * ym_tu[1]  / 1000);
								like_ncr = exp(-ym_ncr[1] * ym_ncr[1] / 1000);
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								like_ne  = exp(-ym_ne[1]  * ym_ne[1]  / 1000);
#endif
#endif
								if (   like_tu >= like_ncr
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
									&& like_tu >= like_ne
#endif
									) {
									scan_label_map.m_pData[n][m][l][0] = label_idx[TU];
								} else {
									if (b_valid_label[ED] > 0) {
										ComputeLikelihood4(vv_inv[ED ], vv_c1[ED ], vv_c2[ED ], ym_ed , &like_ed );
									} else {
										like_ed = 0;
									}
									if (b_valid_label[TU] > 0) {
										ComputeLikelihood4(vv_inv[TU ], vv_c1[TU ], vv_c2[TU ], ym_tu , &like_tu );
									} else {
										like_tu = 0;
									}
									if (b_valid_label[NCR] > 0) {
										ComputeLikelihood4(vv_inv[NCR], vv_c1[NCR], vv_c2[NCR], ym_ncr, &like_ncr);
									} else {
										like_ncr = 0;
									}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
									if (b_valid_label[NE] > 0) {
										ComputeLikelihood4(vv_inv[NE ], vv_c1[NE ], vv_c2[NE ], ym_ne , &like_ne );
									} else {
										like_ne = 0;
									}
#endif
									// early return for the boundary
									if (tu_fore3.m_pData[n][m][l][0] <= 0) {
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
										if (like_tu >= like_ed && like_tu >= like_ncr && like_tu >= like_ne) {
#else
										if (like_tu >= like_ed && like_tu >= like_ncr) {
#endif
											scan_label_map.m_pData[n][m][l][0] = label_idx[TU];
											continue;
										}
									}
									//
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
									if (b_valid_label[NCR] > 0 && b_valid_label[NE] > 0) {
										if (like_ncr >= like_ne) {
											if (tu_fore2.m_pData[n][m][l][0] > 0) {
												scan_label_map.m_pData[n][m][l][0] = label_idx[NCR];
											} else {
												if (like_ncr >= like_ed) {
													scan_label_map.m_pData[n][m][l][0] = label_idx[NCR];
												} else {
													scan_label_map.m_pData[n][m][l][0] = label_idx[ED];
												}
											}
										} else {
											if (tu_fore2.m_pData[n][m][l][0] > 0) {
												if (!use_ne_as_cavity) {
													scan_label_map.m_pData[n][m][l][0] = label_idx[NE];
												} else {
													scan_label_map.m_pData[n][m][l][0] = cavity_label_val;
												}
											} else {
												if (like_ne >= like_ed) {
													if (!use_ne_as_cavity) {
														scan_label_map.m_pData[n][m][l][0] = label_idx[NE];
													} else {
														scan_label_map.m_pData[n][m][l][0] = cavity_label_val;
													}
												} else {
													scan_label_map.m_pData[n][m][l][0] = label_idx[ED];
												}
											}
										}
									} else if (b_valid_label[NCR] > 0 && b_valid_label[NE] == 0) {
										if (tu_fore3.m_pData[n][m][l][0] > 0) {
											scan_label_map.m_pData[n][m][l][0] = label_idx[NCR];
										} else if (tu_fore2.m_pData[n][m][l][0] > 0) {
											scan_label_map.m_pData[n][m][l][0] = label_idx[NCR];
											//scan_label_map.m_pData[n][m][l][0] = label_idx[NE];
										} else {
											if (like_ncr >= like_ed) {
												scan_label_map.m_pData[n][m][l][0] = label_idx[NCR];
												//scan_label_map.m_pData[n][m][l][0] = label_idx[NE];
											} else {
												scan_label_map.m_pData[n][m][l][0] = label_idx[ED];
											}
										}
									} else if (b_valid_label[NCR] == 0 && b_valid_label[NE] > 0) {
										if (tu_fore2.m_pData[n][m][l][0] > 0) {
											if (!use_ne_as_cavity) {
												scan_label_map.m_pData[n][m][l][0] = label_idx[NE];
											} else {
												scan_label_map.m_pData[n][m][l][0] = cavity_label_val;
											}
										} else {
											if (like_ne >= like_ed) {
												if (!use_ne_as_cavity) {
													scan_label_map.m_pData[n][m][l][0] = label_idx[NE];
												} else {
													scan_label_map.m_pData[n][m][l][0] = cavity_label_val;
												}
											} else {
												scan_label_map.m_pData[n][m][l][0] = label_idx[ED];
											}
										}
									} else {
										if (tu_fore2.m_pData[n][m][l][0] > 0) {
											scan_label_map.m_pData[n][m][l][0] = label_idx[TU];
										} else {
											if (like_tu >= like_ed) {
												scan_label_map.m_pData[n][m][l][0] = label_idx[TU];
											} else {
												scan_label_map.m_pData[n][m][l][0] = label_idx[ED];
											}
										}
									}
#else
									if (tu_fore2.m_pData[n][m][l][0] > 0) {
										scan_label_map.m_pData[n][m][l][0] = label_idx[NCR];
									} else {
										if (like_ncr >= like_ed) {
											scan_label_map.m_pData[n][m][l][0] = label_idx[NCR];
										} else {
											scan_label_map.m_pData[n][m][l][0] = label_idx[ED];
										}
									}
#endif
								}
							}
						}
					}

          clock_end = clock();
          //std::cout << "===Time stamp 51: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n"; // multiple
          std::cout << "===Percentage Done : 99.70\n";

					//*
					{
						char str_tmp[1024];
						if (output_tmp_label) {
							sprintf(str_tmp, "%s%sscan_label_map_corr_tu.nii.gz", out_folder, DIR_SEP);
						} else {
							sprintf(str_tmp, "%s%sscan_label_map_corr_tu.nii.gz", tmp_folder, DIR_SEP);
						}
						scan_label_map.save(str_tmp, 1);
						ChangeNIIHeader(str_tmp, scan_image_masked_files[0]);
					}
					//*/
				}
#endif
				//
				for (i = 0; i < NumberOfImageChannels; i++) {
					scan_images[i].clear();
				}
			}
			//
			scan_label_map.save(scan_label_map_file, 1);
			// To do: this works only for the LPS header
			ChangeNIIHeader(scan_label_map_file, scan_image_masked_files[0]);

			// free data
			{
				for (i = 0; i < NumberOfPriorChannels; i++) {
					scan_posteriors[i].clear();
				}
				//
				scan_label_map.clear();
			}
		}
		//
		// copy results
		{
			char str_in[1024];
			char str_out[1024];
			//
			sprintf(str_out, "%s%sscan_to_atlas.mat", out_folder, DIR_SEP);
			CopyFile(scan_to_atlas_mat_file, str_out, FALSE);
			sprintf(str_out, "%s%satlas_to_scan.mat", out_folder, DIR_SEP);
			CopyFile(atlas_to_scan_mat_file, str_out, FALSE);
#ifdef UPDATE_DEFORMATION_FIELD
			if (update_deform) {
				sprintf(str_in, "%s%ss_h2u.mhd", tmp_folder, DIR_SEP);
			} else {
				sprintf(str_in, "%s%ss_h1u.mhd", tmp_folder, DIR_SEP);
			}
#else
			sprintf(str_in, "%s%ss_h1u.mhd", tmp_folder, DIR_SEP);
#endif
			sprintf(str_out, "%s%sscan_u_h_field.mhd", out_folder, DIR_SEP);
			CopyMHDData(NULL, str_in, NULL, str_out, true);
			/*
			sprintf(str_out, "%s%sscan_u.mhd", out_folder, DIR_SEP);
			CopyMHDData(NULL, scan_u_hdr_file, NULL, str_out, true);
			sprintf(str_out, "%s%sscan_h.mhd", out_folder, DIR_SEP);
			CopyMHDData(NULL, scan_h_hdr_file, NULL, str_out, true);
			//*/
			//
			sprintf(str_out, "%s%sscan_init_means.txt", out_folder, DIR_SEP);
			CopyFile(scan_init_means_file, str_out, FALSE);
			sprintf(str_out, "%s%sscan_means.txt", out_folder, DIR_SEP);
			CopyFile(scan_means_file, str_out, FALSE);
			sprintf(str_out, "%s%sscan_variances.txt", out_folder, DIR_SEP);
			CopyFile(scan_variances_file, str_out, FALSE);
			//
			sprintf(str_in, "%s%ss_params_%d.txt", tmp_folder, DIR_SEP, scan_jsr_max_iter-1);
			sprintf(str_out, "%s%stumor_params.txt", out_folder, DIR_SEP);
			CopyFile(str_in, str_out, FALSE);
		}
	}
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////

  clock_end = clock();
  //std::cout << "===Time stamp 52: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
  std::cout << "===Percentage Done : 99.80\n";
	/////////////////////////////////////////////////////////////////////////////
	SetCurrentDirectory((char*)out_folder);
	//
	if (delete_tmp_folder) {
		DeleteAll(tmp_folder, TRUE);
	}
	/////////////////////////////////////////////////////////////////////////////

  TRACE("finished\n");
  std::cout << "===Percentage Done : 99.99\n";
	/////////////////////////////////////////////////////////////////////////////
#ifdef USE_TRACE
	if (g_fp_trace != NULL) {
		fclose(g_fp_trace);
		g_fp_trace = NULL;
	}
	g_bTrace = FALSE;
	g_bTraceStdOut = FALSE;
	g_verbose = 0;
#endif
	/////////////////////////////////////////////////////////////////////////////

  clock_end = clock();
  //std::cout << "===Time stamp 53: " << static_cast<float>(clock_end - clock_start) / CLOCKS_PER_SEC << "\n";
  std::cout << "===Percentage Done : 100.0\n";

	exit(EXIT_SUCCESS);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_PROB_PRIOR
void QuickSortA(float* values, int* labels, const int lo, const int hi)
{
	int i = lo, j = hi;
	float v; int l;
	float x = values[(lo+hi)/2];

	do {    
		while (values[i] < x) i++; 
		while (values[j] > x) j--;
		if (i <= j) {
			v = values[i]; values[i] = values[j]; values[j] = v;
			l = labels[i]; labels[i] = labels[j]; labels[j] = l;
			i++; j--;
		}
	} while (i <= j);

	if (lo < j ) QuickSortA(values, labels, lo, j );
	if (i  < hi) QuickSortA(values, labels, i,  hi);
}
void QuickSortD(float* values, int* labels, const int lo, const int hi)
{
	int i = lo, j = hi;
	float v; int l;
	float x = values[(lo+hi)/2];

	do {    
		while (values[i] > x) i++; 
		while (values[j] < x) j--;
		if (i <= j) {
			v = values[i]; values[i] = values[j]; values[j] = v;
			l = labels[i]; labels[i] = labels[j]; labels[j] = l;
			i++; j--;
		}
	} while (i <= j);

	if (lo < j ) QuickSortD(values, labels, lo, j );
	if (i  < hi) QuickSortD(values, labels, i,  hi);
}
//
#include "SMatrix.h"
#define RWR_PHI_NO
//#define RWR_PHI_UN
BOOL rwr(FVolume* vd, int vd_n, FVolume& mask, float* wt, int (*tp_c)[3], int* tp_r, int tp_num, FVolume& result) 
{
	int vd_x, vd_y, vd_z;
	int i, j, k, l, ni, nj, nk, x, y, z, nx, ny, nz, nn, c;
	int nn_x[26] = {  0,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, +1,  0, +1, +1, +1,  0, +1, +1, +1, +1, +1 };
	int nn_y[26] = {  0, -1,  0, -1,  0, -1, -1, +1, +1, +1, +1,  0, -1,  0, +1,  0, +1,  0, +1, +1, -1, -1, -1, -1,  0, +1 };
	int nn_z[26] = { -1,  0,  0, -1, -1,  0, -1, -1, -1,  0, +1, +1, +1, +1,  0,  0, +1, +1,  0, +1, +1, +1,  0, -1, -1, -1 };
	double nn_r[26] = { 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0 };
	double nn_r_sq[26];
	int nn_num = 26;
	float max_diff;
	//
	int *edges_a, *edges_b;
	double *weight_g, *weight_v, *weight;
	int edges_num;
	int x_a, x_b, y_a, y_b, z_a, z_b, xw, yw, zw, xw_1, yw_1, zw_1;
	int bx_a, bx_b, by_a, by_b, bz_a, bz_b;
	double wg_sc, wv_sc;
	int N;
	//
	CDVector lines, likes;
#ifdef RWR_BG_PROB
	CDVector lines_bg, likes_bg;
#endif

	wg_sc = 0;
	wv_sc = 60;

	vd_x = vd[0].m_vd_x;
	vd_y = vd[0].m_vd_y;
	vd_z = vd[0].m_vd_z;
	max_diff = 255 * 255;

	// estimate image boundary
	bx_a = vd_x-1; bx_b = 0;
	by_a = vd_y-1; by_b = 0;
	bz_a = vd_z-1; bz_b = 0;
	for (k = 0; k < vd_z; k++) {
		for (j = 0; j < vd_y; j++) {
			for (i = 0; i < vd_x; i++) {
				int ys = 0;
				for (c = 0; c < vd_n; c++) {
					if (vd[c].m_pData[k][j][i][0] > 0) {
						ys = 1;
						break;
					}
				}
				if (ys <= 0) {
					continue;
				}
				//
				if (bx_a > i) bx_a = i;
				if (bx_b < i) bx_b = i;
				if (by_a > j) by_a = j;
				if (by_b < j) by_b = j;
				if (bz_a > k) bz_a = k;
				if (bz_b < k) bz_b = k;
			}
		}
	}
	bx_a = max(bx_a-1, 1);
	by_a = max(by_a-1, 1);
	bz_a = max(bz_a-1, 1);
	bx_b = min(bx_b+1, vd_x-2);
	by_b = min(by_b+1, vd_y-2);
	bz_b = min(bz_b+1, vd_z-2);
	TRACE2("bx_a = %d, bx_b = %d, by_a = %d, by_b = %d, bz_a = %d, bz_b = %d\n", bx_a, bx_b, by_a, by_b, bz_a, bz_b);

	x_a = tp_c[0][0]; x_b = tp_c[0][0];
	y_a = tp_c[0][1]; y_b = tp_c[0][1];
	z_a = tp_c[0][2]; z_b = tp_c[0][2];
	for (l = 0; l < tp_num; l++) {
		if (x_a > tp_c[l][0] - tp_r[l]) { x_a = tp_c[l][0] - tp_r[l]; }
		if (x_b < tp_c[l][0] + tp_r[l]) { x_b = tp_c[l][0] + tp_r[l]; }
		if (y_a > tp_c[l][1] - tp_r[l]) { y_a = tp_c[l][1] - tp_r[l]; }
		if (y_b < tp_c[l][1] + tp_r[l]) { y_b = tp_c[l][1] + tp_r[l]; }
		if (z_a > tp_c[l][2] - tp_r[l]) { z_a = tp_c[l][2] - tp_r[l]; }
		if (z_b < tp_c[l][2] + tp_r[l]) { z_b = tp_c[l][2] + tp_r[l]; }
	}
	/*
	x_a = max(x_a, 1);
	x_b = min(x_b, vd_x-2);
	y_a = max(y_a, 1);
	y_b = min(y_b, vd_y-2);
	z_a = max(z_a, 1);
	z_b = min(z_b, vd_z-2);
	/*/
	x_a = max(x_a, bx_a);
	x_b = min(x_b, bx_b);
	y_a = max(y_a, by_a);
	y_b = min(y_b, by_b);
	z_a = max(z_a, bz_a);
	z_b = min(z_b, bz_b);
	//*/
	TRACE2("x_a = %d, x_b = %d, y_a = %d, y_b = %d, z_a = %d, z_b = %d, vol = %d\n", x_a, x_b, y_a, y_b, z_a, z_b, (x_b-x_a)*(y_b-y_a)*(z_b-z_a));

	xw = x_b - x_a + 1;
	yw = y_b - y_a + 1;
	zw = z_b - z_a + 1;
	N = xw * yw * zw;
	xw_1 = xw - 1;
	yw_1 = yw - 1;
	zw_1 = zw - 1;

	for (nn = 0; nn < nn_num; nn++) {
		nn_r_sq[nn] = sqrt(nn_r[nn]);
	}

	edges_a  = (int*)MyAlloc(N * nn_num * sizeof(int));
	edges_b  = (int*)MyAlloc(N * nn_num * sizeof(int));
	weight   = (double*)MyAlloc(N * nn_num * sizeof(double));
	weight_g = (double*)MyAlloc(N * nn_num * sizeof(double));
	weight_v = (double*)MyAlloc(N * nn_num * sizeof(double));
	//
	lines.Init(N);
	likes.Init(N);
#ifdef RWR_BG_PROB
	lines_bg.Init(N);
	likes_bg.Init(N);
#endif

	edges_num = 0;
	for (k = 1; k < zw_1; k++) {
		for (j = 1; j < yw_1; j++) {
			for (i = 1; i < xw_1; i++) {
				double diff, val, valref;
				int idx, idxn;
#ifndef USE_DISCARD_ZERO_AREA
				double ys = 0;
#else
				double ys = 1;
#endif

				x = i + x_a;
				y = j + y_a;
				z = k + z_a;
				idx  = (k * yw + j) * xw + i;

				for (c = 0; c < vd_n; c++) {
#ifndef USE_DISCARD_ZERO_AREA
					ys += vd[c].m_pData[z][y][x][0];
#else
					if (vd[c].m_pData[z][y][x][0] <= 0) {
						ys = 0;
					}
#endif
				}
				if (ys <= 0) {
					continue;
				}
				if (mask.m_pData[z][y][x][0] == 1) {
					lines.m_dv[idx] = 1;
				}

				valref = max_diff;
				/*
				for (nn = 0; nn < nref; nn++) {
					val = 0;
					for (c = 0; c < vd_n; c++) {
						diff = vd[c].m_pData[z][y][x][0] - cref[nn * vd_n + c];
						val += wt[c] * diff * diff;
					}

					if (valref > val) {
						valref = val;
					}
				}
				if (mask.m_pData[z][y][x][0] == 2) {
					valref = 0;
				}
				//*/

				for (nn = 0; nn < nn_num; nn++) {
					ni = i + nn_x[nn];
					nj = j + nn_y[nn];
					nk = k + nn_z[nn];
					nx = ni + x_a;
					ny = nj + y_a;
					nz = nk + z_a;

					idxn = (nk * yw + nj) * xw + ni;

					val = 0;
					for (c = 0; c < vd_n; c++) {
						diff = vd[c].m_pData[z][y][x][0] - vd[c].m_pData[nz][ny][nx][0];
						val += wt[c] * diff * diff;
					}

					edges_a[edges_num] = idx;
					edges_b[edges_num] = idxn;
					weight_g[edges_num] = nn_r_sq[nn];
					weight_v[edges_num] = sqrt(min(val, valref));
					edges_num++;
				}
			}
		}
	}
	TRACE2("edges_num = %d\n", edges_num);

#ifdef RWR_BG_PROB
	{
		int ia[6] = { 0,    xw_1, 0,    0,    0,    0    };
		int ib[6] = { 0,    xw_1, xw_1, xw_1, xw_1, xw_1 };
		int ja[6] = { 0,    0,    0,    yw_1, 0,    0    };
		int jb[6] = { yw_1, yw_1, 0,    yw_1, yw_1, yw_1 };
		int ka[6] = { 0,    0,    0,    0,    0,    zw_1 };
		int kb[6] = { zw_1, zw_1, zw_1, zw_1, 0,    zw_1 };
		for (l = 0; l < 6; l++) {
			for (k = ka[l]; k <= kb[l]; k++) {
				for (j = ja[l]; j <= jb[l]; j++) {
					for (i = ia[l]; i <= ib[l]; i++) {
						int idx;
#ifndef USE_DISCARD_ZERO_AREA
						double ys = 0;
#else
						double ys = 1;
#endif

						x = i + x_a;
						y = j + y_a;
						z = k + z_a;
						idx  = (k * yw + j) * xw + i;

						for (c = 0; c < vd_n; c++) {
#ifndef USE_DISCARD_ZERO_AREA
							ys += vd[c].m_pData[z][y][x][0];
#else
							if (vd[c].m_pData[z][y][x][0] <= 0) {
								ys = 0;
							}
#endif
						}
						if (ys <= 0) {
							continue;
						}
						if (mask.m_pData[z][y][x][0] != 1) {
							lines_bg.m_dv[idx] = 1;
						}
					}
				}
			}
		}
		for (k = 1; k < zw_1; k++) {
			for (j = 1; j < yw_1; j++) {
				for (i = 1; i < xw_1; i++) {
					int idx;
#ifndef USE_DISCARD_ZERO_AREA
					double ys = 0;
#else
					double ys = 1;
#endif

					x = i + x_a;
					y = j + y_a;
					z = k + z_a;
					idx  = (k * yw + j) * xw + i;

					for (c = 0; c < vd_n; c++) {
#ifndef USE_DISCARD_ZERO_AREA
						ys += vd[c].m_pData[z][y][x][0];
#else
						if (vd[c].m_pData[z][y][x][0] <= 0) {
							ys = 0;
						}
#endif
					}
					if (ys <= 0) {
						continue;
					}
					if (mask.m_pData[z][y][x][0] == 2) {
						lines_bg.m_dv[idx] = 1;
					}
				}
			}
		}
	}
#endif

	// normalize weight
	{
		double wg_min, wg_max, wg_max_1, wg;
		double wv_min, wv_max, wv_max_1, wv;

		wg_min = DBL_MAX;
		wv_min = DBL_MAX;
		for (i = 0; i < edges_num; i++) {
			wg = weight_g[i];
			wv = weight_v[i];
			if (wg < wg_min) { wg_min = wg; }
			if (wv < wv_min) { wv_min = wv; }
		}
		//
		wg_max = 0;
		wv_max = 0;
		for (i = 0; i < edges_num; i++) {
			wg = weight_g[i] - wg_min;
			wv = weight_v[i] - wv_min;
			if (wg > wg_max) { wg_max = wg; }
			if (wv > wv_max) { wv_max = wv; }
			weight_g[i] = wg;
			weight_v[i] = wv;
		}
		if (wg_max > 0) {
			wg_max_1 = 1.0f / wg_max;
		} else {
			wg_max_1 = 0;
		}
		if (wv_max > 0) {
			wv_max_1 = 1.0f / wv_max;
		} else {
			wv_max_1 = 0;
		}
		for (i = 0; i < edges_num; i++) {
			weight_g[i] *= wg_max_1;
			weight_v[i] *= wv_max_1;
		}
		//
		for (i = 0; i < edges_num; i++) {
			weight[i] = exp(-(wg_sc * weight_g[i] + wv_sc * weight_v[i]));
		}
	}

	MyFree(weight_g);
	MyFree(weight_v);

	CSMatrix E(N, N, N);
	CSMatrix P(N, N, edges_num);
	CSMatrix EP(N, N, edges_num+N);

	// E = sparse(1:N,1:N,ones(N,1));
	// iD = sparse(1:N,1:N,1./sum(W));
	// P = iD*W; 
	// R = c*(E-(1-c)*P)\lines;
	for (i = 0; i < edges_num; i++) {
		P.Add(edges_a[i], edges_b[i], weight[i]);
	}

	MyFree(edges_a);
	MyFree(edges_b);
	MyFree(weight);

	P.GenerateCompInfo();

#ifdef RWR_PHI_NO
	//*
	double cf = 0.0004;
	double cf1 = -cf * (1.0 - cf);
	for (i = 0; i < N; i++) {
		EP.Add(i, i, cf);
	}
	for (j = 0; j < N; j++) {
		double dval, dval_1;
		int cp1, cp2;
		cp1 = P.m_cr_rowptr[j];
		cp2 = P.m_cr_rowptr[j+1];
		if (cp1 != cp2) {
			dval = 0;
			for (i = cp1; i < cp2; i++) {
				dval += P.m_cr_val[i];
			}
			if (dval > 0) {
				dval_1 = 1.0f / dval;
			} else {
				dval_1 = 0;
			}
			for (i = cp1; i < cp2; i++) {
				EP.Add(j, P.m_cr_colind[i], cf1 * P.m_cr_val[i] * dval_1);
			}
		}
	}
	/*/
	double cf = 0.00001;
	double cf1 = -cf * (1.0 - cf);
	double* d_diag;
	d_diag = (double*)malloc(N * sizeof(double));
	//
	for (i = 0; i < N; i++) {
		EP.Add(i, i, cf);
	}
	for (j = 0; j < N; j++) {
		double dval;
		int cp1, cp2;
		cp1 = P.m_cr_rowptr[j];
		cp2 = P.m_cr_rowptr[j+1];
		if (cp1 != cp2) {
			dval = 0;
			for (i = cp1; i < cp2; i++) {
				dval += P.m_cr_val[i];
			}
			if (dval > 0) {
				d_diag[j] = 1.0f / sqrt(dval);
			} else {
				d_diag[j] = 0;
			}
		} else {
			d_diag[j] = 0;
		}
	}
	for (j = 0; j < N; j++) {
		int cp1, cp2;
		cp1 = P.m_cr_rowptr[j];
		cp2 = P.m_cr_rowptr[j+1];
		if (cp1 != cp2) {
			for (i = cp1; i < cp2; i++) {
				EP.Add(j, P.m_cr_colind[i], cf1 * P.m_cr_val[i] * d_diag[j] * d_diag[P.m_cr_colind[i]]);
			}
		}
	}
	//
	free(d_diag);
	//*/
#endif
#ifdef RWR_PHI_UN
	double cf = 0.00001;
	double cf1 = -cf * (1.0 - cf);
	for (j = 0; j < N; j++) {
		double dval;
		int cp1, cp2;
		cp1 = P.m_cr_rowptr[j];
		cp2 = P.m_cr_rowptr[j+1];
		if (cp1 != cp2) {
			dval = 0;
			for (i = cp1; i < cp2; i++) {
				dval += P.m_cr_val[i];
			}
			EP.Add(j, j, cf * dval);
		}
	}
	for (j = 0; j < N; j++) {
		int cp1, cp2;
		cp1 = P.m_cr_rowptr[j];
		cp2 = P.m_cr_rowptr[j+1];
		if (cp1 != cp2) {
			for (i = cp1; i < cp2; i++) {
				EP.Add(j, P.m_cr_colind[i], cf1 * P.m_cr_val[i]);
			}
		}
	}
#endif

	EP.GenerateCompInfo();
	//
#ifdef RWR_BG_PROB
	if (!EP.SolveLinearSystem2(lines, likes, lines_bg, likes_bg)) {
		return FALSE;
	}
#else
	if (!EP.SolveLinearSystem(lines, likes)) {
		return FALSE;
	}
#endif

	// normalize likes
	{
		double likes_max, likes_max_1, likes_sum;
#ifdef RWR_BG_PROB
		double likes_bg_max, likes_bg_max_1, likes_bg_sum;
#endif
		likes_max = 0;
		likes_sum = 0;
#ifdef RWR_BG_PROB
		likes_bg_max = 0;
		likes_bg_sum = 0;
#endif
		for (i = 0; i < N; i++) {
			likes_sum += likes.m_dv[i];
			if (likes_max < likes.m_dv[i]) {
				likes_max = likes.m_dv[i];
			}
#ifdef RWR_BG_PROB
			likes_bg_sum += likes_bg.m_dv[i];
			if (likes_bg_max < likes_bg.m_dv[i]) {
				likes_bg_max = likes_bg.m_dv[i];
			}
#endif
		}
		TRACE2("likes_sum = %f, likes_max = %f\n", likes_sum, likes_max);
#ifdef RWR_BG_PROB
		TRACE2("likes_bg_sum = %f, likes_bg_max = %f\n", likes_bg_sum, likes_bg_max);
#endif
		if (likes_max != 0) {
			likes_max_1 = 1.0 / likes_max;
		} else {
			likes_max_1 = 1.0;
		}
#ifdef RWR_BG_PROB
		if (likes_bg_max != 0) {
			likes_bg_max_1 = 1.0 / likes_bg_max;
		} else {
			likes_bg_max_1 = 1.0;
		}
#endif

		result.allocate(vd_x, vd_y, vd_z);

#ifdef RWR_BG_PROB
		FVolume post, post_bg;
		post.allocate(vd_x, vd_y, vd_z);
		post_bg.allocate(vd_x, vd_y, vd_z);
#endif

		for (k = 0; k < zw; k++) {
			for (j = 0; j < yw; j++) {
				for (i = 0; i < xw; i++) {
					int idx;

					x = i + x_a;
					y = j + y_a;
					z = k + z_a;
					idx  = (k * yw + j) * xw + i;

#ifdef RWR_BG_PROB
					double like, like_bg, norm;
					
					like = likes.m_dv[idx] / likes_sum;
					like_bg = likes_bg.m_dv[idx] / likes_bg_sum;
					norm = like + like_bg + eps;
					post.m_pData[z][y][x][0] = like / norm;
					post_bg.m_pData[z][y][x][0] = like_bg / norm;

#ifdef RWR_USE_POST_TH
					if (post.m_pData[z][y][x][0] > RWR_POST_TH) {
						result.m_pData[z][y][x][0] = post.m_pData[z][y][x][0];
					}
#endif
#ifdef RWR_USE_POST_COMP
					if (post.m_pData[z][y][x][0] > post_bg.m_pData[z][y][x][0]) {
						result.m_pData[z][y][x][0] = post.m_pData[z][y][x][0];
					}
#endif
#else
					result.m_pData[z][y][x][0] = likes.m_dv[idx] * likes_max_1;
#endif
				}
			}
		}

#ifdef RWR_BG_PROB
		/*
		{
			char str_tmp[1024];
			sprintf(str_tmp, "D:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\data\\test_post.nii.gz");
			post.save(str_tmp, 1);
			sprintf(str_tmp, "D:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\data\\test_post_bg.nii.gz");
			post_bg.save(str_tmp, 1);
			sprintf(str_tmp, "D:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\data\\test.nii.gz");
			result.save(str_tmp, 1);
		}
		//*/
#endif
	}

	return TRUE;
}
//
// cref: vd_n * ncref
BOOL ComputeTumorProb(FVolume* vd, int vd_n, 
	int (*tp_c)[3], float* tp_r, int tp_num, int tp_r_init, 
	int (*pt_c)[3], int* pt_t, int pt_num, int pt_r_init,
	FVolume* ref_tu, FVolume* ref_bg, float* wt, FVolume* prob, int prob_n, FVolume* tu_prob, float re_m)
{
	FVolume mask;
	FVolume dist;
	int vd_x, vd_y, vd_z;
	int i, j, k, l;
	float re_m_1;

	vd_x = vd[0].m_vd_x;
	vd_y = vd[0].m_vd_y;
	vd_z = vd[0].m_vd_z;

	mask.allocate(vd_x, vd_y, vd_z);
	dist.allocate(vd_x, vd_y, vd_z);

	re_m_1 = 1.0f / re_m;

	{
		float r2[MaxNumberOFTumorSeeds], d2;

		for (l = 0; l < tp_num; l++) {
			if (tp_r[l] <= 0) {
				r2[l] = tp_r_init * tp_r_init;
			} else {
				if (tp_r_init <= tp_r[l]/2) {
					r2[l] = tp_r_init * tp_r_init;
				} else {
					r2[l] = (tp_r[l]/2) * (tp_r[l]/2);
				}
			}
		}

		for (k = 0; k < vd_z; k++) {
			for (j = 0; j < vd_y; j++) {
				for (i = 0; i < vd_x; i++) {
					for (l = 0; l < tp_num; l++) {
						d2 = (i-tp_c[l][0])*(i-tp_c[l][0]) + (j-tp_c[l][1])*(j-tp_c[l][1]) + (k-tp_c[l][2])*(k-tp_c[l][2]);
						if (d2 <= r2[l]) {
							mask.m_pData[k][j][i][0] = 1;
						}
					}
				}
			}
		}
	}

	{
		float r2, d2;

		r2 = pt_r_init * pt_r_init;

		for (k = 0; k < vd_z; k++) {
			for (j = 0; j < vd_y; j++) {
				for (i = 0; i < vd_x; i++) {
					for (l = 0; l < pt_num; l++) {
						d2 = (i-pt_c[l][0])*(i-pt_c[l][0]) + (j-pt_c[l][1])*(j-pt_c[l][1]) + (k-pt_c[l][2])*(k-pt_c[l][2]);
						if (d2 <= r2) {
							if (   pt_t[l] == TU
								|| pt_t[l] == NCR 
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
								|| pt_t[l] == NE
#endif
							) {
								mask.m_pData[k][j][i][0] = 1;
							} else {
								mask.m_pData[k][j][i][0] = 2;
							}
						}
					}
				}
			}
		}

		for (l = 0; l < pt_num; l++) {
			if (pt_t[l] == TU || pt_t[l] == NCR 
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				|| pt_t[l] == NE
#endif
			) {
				mask.m_pData[pt_c[l][2]][pt_c[l][1]][pt_c[l][0]][0] = 1;
			} else {
				mask.m_pData[pt_c[l][2]][pt_c[l][1]][pt_c[l][0]][0] = 2;
			}
		}
	}

	if (ref_tu != NULL) {
		for (k = 0; k < vd_z; k++) {
			for (j = 0; j < vd_y; j++) {
				for (i = 0; i < vd_x; i++) {
					if (ref_tu->m_pData[k][j][i][0] > 0) {
						mask.m_pData[k][j][i][0] = 1;
					}
				}
			}
		}
	}
	if (ref_bg != NULL) {
		for (k = 0; k < vd_z; k++) {
			for (j = 0; j < vd_y; j++) {
				for (i = 0; i < vd_x; i++) {
					if (ref_bg->m_pData[k][j][i][0] > 0) {
						mask.m_pData[k][j][i][0] = 2;
					}
				}
			}
		}
	}

	// rwr
	{
		float dist_th_vol;
		float dist_th = 0.1f;

#if 0
		rwr(vd, vd_n, mask, cref, ncref, wt, xc, yc, zc, r_th, dist);
#endif
#if 1
		{
			FVolume mask_s;
			FVolume dist_s;
			FVolume* vd_s;
			int tp_c_s[MaxNumberOFTumorSeeds][3], tp_r_s[MaxNumberOFTumorSeeds];

			vd_s = new FVolume[vd_n];

			for (i = 0; i < vd_n; i++) {
				vd[i].GaussianSmoothing(vd_s[i], 0.67, 5);
				vd_s[i].imresize(re_m_1);
			}
			mask_s = mask;
			mask_s.imresize(re_m_1, true);

			for (l = 0; l < tp_num; l++) {
				tp_c_s[l][0] = (int)(tp_c[l][0] * re_m_1);
				tp_c_s[l][1] = (int)(tp_c[l][1] * re_m_1);
				tp_c_s[l][2] = (int)(tp_c[l][2] * re_m_1);
				//
				if (tp_r[l] <= 0) {
					if (l == 0) {
						tp_r_s[l] = (int)(60 * re_m_1);
					} else {
						tp_r_s[l] = (int)(30 * re_m_1);
					}
				} else {
					tp_r_s[l] = (int)((tp_r[l]*1.2) * re_m_1);
				}
			}

			rwr(vd_s, vd_n, mask_s, wt, tp_c_s, tp_r_s, tp_num, dist_s);

			dist_s.imresize(re_m);
			for (k = 0; k < dist_s.m_vd_z; k++) {
				for (j = 0; j < dist_s.m_vd_y; j++) {
					for (i = 0; i < dist_s.m_vd_x; i++) {
						dist.m_pData[k][j][i][0] = dist_s.m_pData[k][j][i][0];
					}
				}
			}

			//dist.save(name, 1);

			for (i = 0; i < vd_n; i++) {
				vd_s[i].clear();
			}
			mask_s.clear();
			dist_s.clear();
			delete [] vd_s;
		}

		if (tu_prob) {
			tu_prob->copy(dist);
		}

		dist_th_vol = 0;
		for (k = 0; k < vd_z; k++) {
			for (j = 0; j < vd_y; j++) {
				for (i = 0; i < vd_x; i++) {
					if (dist.m_pData[k][j][i][0] > dist_th) {
						dist_th_vol += 1;
					}
				}
			}
		}
		TRACE2("estimated tumor volume = %f\n", dist_th_vol);

		if (prob) {
			for (k = 0; k < vd_z; k++) {
				for (j = 0; j < vd_y; j++) {
					for (i = 0; i < vd_x; i++) {
						if (vd[0].m_pData[k][j][i][0] == 0) {
							for (l = 0; l < prob_n; l++) {
								prob[l].m_pData[k][j][i][0] = 0;
							}
							prob[BG].m_pData[k][j][i][0] = 1;
						} else {
							float p, p_1_n;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
							float p_3;
#else
							float p_2;
#endif
							if (dist.m_pData[k][j][i][0] > dist_th) {
								p = 1;
							} else {
								p = 0;
							}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
							p_3 = p / 3;
							p_1_n = (1.0 - p) / (prob_n-4);
#else
							p_2 = p * 0.5;
							p_1_n = (1.0 - p) / (prob_n-3);
#endif

							for (l = 0; l < prob_n; l++) {
								prob[l].m_pData[k][j][i][0] = p_1_n;
							}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
							prob[TU ].m_pData[k][j][i][0] = p_3;
							prob[NCR].m_pData[k][j][i][0] = p_3;
							prob[NE ].m_pData[k][j][i][0] = p_3;
#else
							prob[TU ].m_pData[k][j][i][0] = p_2;
							prob[NCR].m_pData[k][j][i][0] = p_2;
#endif
							prob[BG].m_pData[k][j][i][0] = 0;
						}
					}
				}
			}
		}
#endif
	}

	return TRUE;
}
void EstimateTumorSize(FVolume& tu_prob, int tp_num, int (*tp_c)[3], float* tp_r_est) 
{
	int vd_x, vd_y, vd_z;
	int i, l, m, n;
	//
	vd_x = tu_prob.m_vd_x;
	vd_y = tu_prob.m_vd_y;
	vd_z = tu_prob.m_vd_z;
	//
	for (i = 0; i < tp_num; i++) {
		TRACE2("estimating tumor size for tumor %d\n", i);
		//
		// estimating tumor size
		int rr, rr2, rr_step, rr_min, rr_max;
		int x_c, y_c, z_c;
		float fg_val_sel, fg_rr_sel;
		float fg_val_th = 0.8f;
		float fg_inc_val_th = 0.5f;
		int count, count_fg, count_prev, count_fg_prev;
		//
		rr_step = 1;
		rr_min = 5;
		if (i == 0) {
			rr_max = 60;
		} else {
			rr_max = 30;
		}
		//
		x_c = tp_c[i][0];
		y_c = tp_c[i][1];
		z_c = tp_c[i][2];
		//
		fg_rr_sel = rr_min;
		fg_val_sel = 0;
		count_prev = 0;
		count_fg_prev = 0;
		for (rr = rr_min; rr <= rr_max; rr += rr_step) {
			int x_a, x_b, y_a, y_b, z_a, z_b;
			int dx, dy, dz, dx2, dy2, dz2, d2;
			float fg_val;
			float fg_inc_val = 0;
			//
			x_a = x_c - rr; x_b = x_c + rr;
			y_a = y_c - rr; y_b = y_c + rr;
			z_a = z_c - rr; z_b = z_c + rr;
			x_a = max(x_a, 1); x_b = min(x_b, vd_x-2);
			y_a = max(y_a, 1); y_b = min(y_b, vd_y-2);
			z_a = max(z_a, 1); z_b = min(z_b, vd_z-2);
			//
			rr2 = rr * rr;
			count = count_fg = 0;
			//
			for (n = z_a; n <= z_b; n++) {
				dz = n - z_c;
				dz2 = dz * dz;
				for (m = y_a; m <= y_b; m++) {
					dy = m - y_c;
					dy2 = dy * dy;
					for (l = x_a; l <= x_b; l++) {
						dx = l - x_c;
						dx2 = dx * dx;
						d2 = dx2 + dy2 + dz2;
						if (d2 <= rr2) {
							if (tu_prob.m_pData[n][m][l][0] >= 0.5f) {
								count_fg++;
							}
							count++;
						}
					}
				}
			}
			//
			if (count > 0) {
				fg_val = (float)count_fg / count;
				fg_inc_val = (float)(count_fg - count_fg_prev) / (count - count_prev);
			} else {
				fg_val = 0;
			}
			TRACE2("rr = %d, fg_val = %f, fg_inc_val = %f, count_fg = %d, count = %d\n", rr, fg_val, fg_inc_val, count_fg, count);
			//
			count_prev = count;
			count_fg_prev = count_fg;
			//
			if (fg_val >= fg_val_th && fg_inc_val >= fg_inc_val_th) {
				fg_val_sel = fg_val;
				fg_rr_sel = rr;
			} else {
				break;
			}
		}
		//
		// major axis length
		tp_r_est[i] = fg_rr_sel * 2;
		TRACE2("tp_r_est[%d] = %f, rr_sel = %f\n", i, tp_r_est[i], fg_rr_sel);
	}
}
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOOL TransformPoints(char* scan_image_file, char* scan_atlas_reg_image_file, char* scan_to_atlas_mat_file, double (*scan_seeds_info)[4], double (*scan_atlas_reg_seeds_info)[4], int scan_seeds_num, int check_orientation)
{
	nifti_image* pNII = NULL;
	double cx, cy, cz;
	double gxc, gyc, gzc;
	double wx, wy, wz;
	int l;
	//
	pNII = nifti_image_read(scan_image_file, 1);
	if (pNII == NULL) {
		return FALSE;
	}
	wx = pNII->nx-1;
	wy = pNII->ny-1;
	wz = pNII->nz-1;
	//
	for (l = 0; l < scan_seeds_num; l++) {
		cx = scan_seeds_info[l][0];
		cy = scan_seeds_info[l][1];
		cz = scan_seeds_info[l][2];
		//
#if 1
		// points = physical coordinates of scan 2
		// 1. convert spatial input coordinates to voxel indices
		if (check_orientation == 0) {
			// We assume RAS header
			gxc = pNII->qto_ijk.m[0][0] * cx + pNII->qto_ijk.m[0][1] * cy + pNII->qto_ijk.m[0][2] * cz + pNII->qto_ijk.m[0][3];
			gyc = pNII->qto_ijk.m[1][0] * cx + pNII->qto_ijk.m[1][1] * cy + pNII->qto_ijk.m[1][2] * cz + pNII->qto_ijk.m[1][3];
			gzc = pNII->qto_ijk.m[2][0] * cx + pNII->qto_ijk.m[2][1] * cy + pNII->qto_ijk.m[2][2] * cz + pNII->qto_ijk.m[2][3];
		} else if (check_orientation == 1) {
			// For general header
			int icod, jcod, kcod;
			nifti_mat44_to_orientation(pNII->qto_ijk, &icod, &jcod, &kcod);
			if (icod != NIFTI_L2R) { cx = -cx; }
			if (jcod != NIFTI_P2A) { cy = -cy; }
			if (kcod != NIFTI_S2I) { cz = -cz; }
			gxc = pNII->qto_ijk.m[0][0] * cx + pNII->qto_ijk.m[0][1] * cy + pNII->qto_ijk.m[0][2] * cz + pNII->qto_ijk.m[0][3];
			gyc = pNII->qto_ijk.m[1][0] * cx + pNII->qto_ijk.m[1][1] * cy + pNII->qto_ijk.m[1][2] * cz + pNII->qto_ijk.m[1][3];
			gzc = pNII->qto_ijk.m[2][0] * cx + pNII->qto_ijk.m[2][1] * cy + pNII->qto_ijk.m[2][2] * cz + pNII->qto_ijk.m[2][3];
			if (icod != NIFTI_L2R) { gxc = -gxc; }
			if (jcod != NIFTI_P2A) { gyc = -gyc; }
			if (kcod != NIFTI_S2I) { gzc = -gzc; }
		} else {
			if (check_orientation == 211) {
				cx = -cx;
			}
			if (check_orientation == 121) {
				cy = -cy;
			}
			if (check_orientation == 112) {
				cz = -cz;
			}
			if (check_orientation == 221) {
				cx = -cx;
				cy = -cy;
			}
			if (check_orientation == 212) {
				cx = -cx;
				cz = -cz;
			}
			if (check_orientation == 122) {
				cy = -cy;
				cz = -cz;
			}
			if (check_orientation == 222) {
				cx = -cx;
				cy = -cy;
				cz = -cz;
			}
			gxc = pNII->qto_ijk.m[0][0] * cx + pNII->qto_ijk.m[0][1] * cy + pNII->qto_ijk.m[0][2] * cz + pNII->qto_ijk.m[0][3];
			gyc = pNII->qto_ijk.m[1][0] * cx + pNII->qto_ijk.m[1][1] * cy + pNII->qto_ijk.m[1][2] * cz + pNII->qto_ijk.m[1][3];
			gzc = pNII->qto_ijk.m[2][0] * cx + pNII->qto_ijk.m[2][1] * cy + pNII->qto_ijk.m[2][2] * cz + pNII->qto_ijk.m[2][3];
			if (check_orientation == 211) {
				gxc = -gxc;
			}
			if (check_orientation == 121) {
				gyc = -gyc;
			}
			if (check_orientation == 112) {
				gzc = -gzc;
			}
			if (check_orientation == 221) {
				gxc = -gxc;
				gyc = -gyc;
			}
			if (check_orientation == 212) {
				gxc = -gxc;
				gzc = -gzc;
			}
			if (check_orientation == 122) {
				gyc = -gyc;
				gzc = -gzc;
			}
			if (check_orientation == 222) {
				gxc = -gxc;
				gyc = -gyc;
				gzc = -gzc;
			}
		}
		//*/
#else
		// points = voxel indices of scan 0 or scan 2
		gxc = cx;
		gyc = cy;
		gzc = cz;
#endif
		// 2. multiply by voxel size and change units to meters
		gxc = gxc * pNII->dx;
		gyc = gyc * pNII->dy;
		gzc = gzc * pNII->dz;
		//
		// 3. transform coordinates to atlas space
		{
			nifti_image* pNIIa;
			float scan_to_atlas_mat[4][4];
			double xc, yc, zc;
			//
			pNIIa = nifti_image_read(scan_atlas_reg_image_file, 1);
			if (pNIIa == NULL) {
				return FALSE;
			}
			ReadXFMData(scan_to_atlas_mat_file, scan_to_atlas_mat);
			//
			gxc = (pNII->nx-1)*pNII->dx - gxc;
			// apply transform
			xc = scan_to_atlas_mat[0][0] * gxc + scan_to_atlas_mat[0][1] * gyc + scan_to_atlas_mat[0][2] * gzc + scan_to_atlas_mat[0][3];
			yc = scan_to_atlas_mat[1][0] * gxc + scan_to_atlas_mat[1][1] * gyc + scan_to_atlas_mat[1][2] * gzc + scan_to_atlas_mat[1][3];
			zc = scan_to_atlas_mat[2][0] * gxc + scan_to_atlas_mat[2][1] * gyc + scan_to_atlas_mat[2][2] * gzc + scan_to_atlas_mat[2][3];
			//
			// voxel indices on atlas reg scan 2
			gxc = (pNIIa->nx-1) - xc / pNIIa->dx;
			gyc = yc / pNIIa->dy;
			gzc = zc / pNIIa->dz;
			//
			// multiply by voxel size and change units to meters
			gxc = gxc * pNIIa->dx;
			gyc = gyc * pNIIa->dy;
			gzc = gzc * pNIIa->dz;
			//
			nifti_image_free(pNIIa);
		}
		//
		scan_atlas_reg_seeds_info[l][0] = gxc;
		scan_atlas_reg_seeds_info[l][1] = gyc;
		scan_atlas_reg_seeds_info[l][2] = gzc;
		//
		scan_atlas_reg_seeds_info[l][3] = scan_seeds_info[l][3];
	}
	//
	nifti_image_free(pNII);

	return TRUE;
}
BOOL GetPoints(char* scan_image_file, double (*scan_points_info)[4], double (*scan_points_info_out)[4], int scan_points_num, int check_orientation)
{
	nifti_image* pNII = NULL;
	double cx, cy, cz;
	double gxc, gyc, gzc;
	int l;
	//
	pNII = nifti_image_read(scan_image_file, 1);
	if (pNII == NULL) {
		return FALSE;
	}
	//
	for (l = 0; l < scan_points_num; l++) {
		cx = scan_points_info[l][0];
		cy = scan_points_info[l][1];
		cz = scan_points_info[l][2];
		//
		// convert spatial input coordinates to voxel indices
		if (check_orientation == 0) {
			// We assume RAS header
			gxc = pNII->qto_ijk.m[0][0] * cx + pNII->qto_ijk.m[0][1] * cy + pNII->qto_ijk.m[0][2] * cz + pNII->qto_ijk.m[0][3];
			gyc = pNII->qto_ijk.m[1][0] * cx + pNII->qto_ijk.m[1][1] * cy + pNII->qto_ijk.m[1][2] * cz + pNII->qto_ijk.m[1][3];
			gzc = pNII->qto_ijk.m[2][0] * cx + pNII->qto_ijk.m[2][1] * cy + pNII->qto_ijk.m[2][2] * cz + pNII->qto_ijk.m[2][3];
		} else if (check_orientation == 1) {
			// For general header
			int icod, jcod, kcod;
			nifti_mat44_to_orientation(pNII->qto_ijk, &icod, &jcod, &kcod);
			if (icod != NIFTI_L2R) { cx = -cx; }
			if (jcod != NIFTI_P2A) { cy = -cy; }
			if (kcod != NIFTI_S2I) { cz = -cz; }
			gxc = pNII->qto_ijk.m[0][0] * cx + pNII->qto_ijk.m[0][1] * cy + pNII->qto_ijk.m[0][2] * cz + pNII->qto_ijk.m[0][3];
			gyc = pNII->qto_ijk.m[1][0] * cx + pNII->qto_ijk.m[1][1] * cy + pNII->qto_ijk.m[1][2] * cz + pNII->qto_ijk.m[1][3];
			gzc = pNII->qto_ijk.m[2][0] * cx + pNII->qto_ijk.m[2][1] * cy + pNII->qto_ijk.m[2][2] * cz + pNII->qto_ijk.m[2][3];
			if (icod != NIFTI_L2R) { gxc = -gxc; }
			if (jcod != NIFTI_P2A) { gyc = -gyc; }
			if (kcod != NIFTI_S2I) { gzc = -gzc; }
		} else {
			if (check_orientation == 211) {
				cx = -cx;
			}
			if (check_orientation == 121) {
				cy = -cy;
			}
			if (check_orientation == 112) {
				cz = -cz;
			}
			if (check_orientation == 221) {
				cx = -cx;
				cy = -cy;
			}
			if (check_orientation == 212) {
				cx = -cx;
				cz = -cz;
			}
			if (check_orientation == 122) {
				cy = -cy;
				cz = -cz;
			}
			if (check_orientation == 222) {
				cx = -cx;
				cy = -cy;
				cz = -cz;
			}
			gxc = pNII->qto_ijk.m[0][0] * cx + pNII->qto_ijk.m[0][1] * cy + pNII->qto_ijk.m[0][2] * cz + pNII->qto_ijk.m[0][3];
			gyc = pNII->qto_ijk.m[1][0] * cx + pNII->qto_ijk.m[1][1] * cy + pNII->qto_ijk.m[1][2] * cz + pNII->qto_ijk.m[1][3];
			gzc = pNII->qto_ijk.m[2][0] * cx + pNII->qto_ijk.m[2][1] * cy + pNII->qto_ijk.m[2][2] * cz + pNII->qto_ijk.m[2][3];
			if (check_orientation == 211) {
				gxc = -gxc;
			}
			if (check_orientation == 121) {
				gyc = -gyc;
			}
			if (check_orientation == 112) {
				gzc = -gzc;
			}
			if (check_orientation == 221) {
				gxc = -gxc;
				gyc = -gyc;
			}
			if (check_orientation == 212) {
				gxc = -gxc;
				gzc = -gzc;
			}
			if (check_orientation == 122) {
				gyc = -gyc;
				gzc = -gzc;
			}
			if (check_orientation == 222) {
				gxc = -gxc;
				gyc = -gyc;
				gzc = -gzc;
			}
		}
		//*/
		//
		scan_points_info_out[l][0] = gxc;
		scan_points_info_out[l][1] = gyc;
		scan_points_info_out[l][2] = gzc;
		//
		scan_points_info_out[l][3] = scan_points_info[l][3];
	}
	//
	nifti_image_free(pNII);

	return TRUE;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOOL MakeScalesFile(const char* scales_file)
{
	FILE* fp;

	fp = fopen(scales_file, "w");
#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
	fprintf(fp, "1e2\n1e1\n1e0\n1e-6\n1e-7\n1e0\n1e0\n1e0\n1e0\n1e3\n");
#else
	fprintf(fp, "1e2\n1e1\n1e0\n1e-6\n1e-7\n1e0\n1e0\n1e0\n1e3\n");
#endif
#else
#ifdef USE_OPTIM_ED
	fprintf(fp, "1e2\n1e1\n1e0\n1e-6\n1e0\n1e0\n1e0\n1e0\n1e3\n");
#else
	fprintf(fp, "1e2\n1e1\n1e0\n1e-6\n1e0\n1e0\n1e0\n1e3\n");
#endif
#endif
	fclose(fp);

	return TRUE;
}
BOOL MakeSimulatorInputFile(const char* simulator_input_file, double o_dx, double o_dy, double o_dz, int x, int y, int z, double dx, double dy, double dz)
{
	FILE* fp;

	fp = fopen(simulator_input_file, "w");
	if (fp == NULL) {
		return FALSE;
	}

	fprintf(fp, "-T 100\n");
	fprintf(fp, "-ntimesteps 5\n");
	fprintf(fp, "-nstore 5\n");
	fprintf(fp, "-cfln 0.5\n");
	//
	fprintf(fp, "-gstiffwm 1.0\n");
	fprintf(fp, "-gstiffgm 1.0\n");
	fprintf(fp, "-gstiffvent 0.238\n");
#if defined(USE_9_PRIORS) || defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	fprintf(fp, "-gstiffcsf 0.238\n");
#else
	fprintf(fp, "-gstiffcsf 1.0\n");
#endif
	fprintf(fp, "-gdiffwm 1e-7\n");
	fprintf(fp, "-gdiffgm .2e-7\n");
	fprintf(fp, "-gdiffvent 0.0\n");
	fprintf(fp, "-gcompresbrain 0.45\n");
	//fprintf(fp, "-gcompresvent 0.1\n");
	fprintf(fp, "-gcompresvent 0.02\n");
	//
	fprintf(fp, "-gres_x %.15f\n", o_dx / 1000);
	fprintf(fp, "-gres_y %.15f\n", o_dy / 1000);
	fprintf(fp, "-gres_z %.15f\n", o_dz / 1000);
	//
	fprintf(fp, "-grho .25e-2\n");
	fprintf(fp, "-gp1 0.5\n");
	fprintf(fp, "-gse 2.0\n");
	fprintf(fp, "-gp2 0.0\n");
	//
	fprintf(fp, "-gcinit 1.0\n");
	fprintf(fp, "-gsigsq 2.25e-5\n");
	//
	fprintf(fp, "-nsd 3\n");
	fprintf(fp, "-ndimx %d\n", (x / 8) + 1);
	fprintf(fp, "-ndimy %d\n", (y / 8) + 1);
	fprintf(fp, "-ndimz %d\n", (z / 8) + 1);
	fprintf(fp, "-imgx %d\n", x);
	fprintf(fp, "-imgy %d\n", y);
	fprintf(fp, "-imgz %d\n", z);
	fprintf(fp, "-imgdx %.15f\n", dx);
	fprintf(fp, "-imgdy %.15f\n", dy);
	fprintf(fp, "-imgdz %.15f\n", dz);
	fprintf(fp, "-Lx %.15f\n", x * dx / 1000);
	fprintf(fp, "-Ly %.15f\n", y * dy / 1000);
	fprintf(fp, "-Lz %.15f\n", z * dz / 1000);
	//
	fprintf(fp, "-material_projection_required 1\n");
	fprintf(fp, "-steady 1\n");
	fprintf(fp, "-ksp_type cg\n");
	fprintf(fp, "-ksp_rtol 1.0e-1\n");
	fprintf(fp, "-ksp_max_it 30\n");
	fprintf(fp, "-ksp_monitor\n");
	fprintf(fp, "-youngs_global 2\n");
	fprintf(fp, "-matrixfree 1\n");
	fprintf(fp, "-youngs_min 1\n");
	fprintf(fp, "-mgnlevels 4\n");
	fprintf(fp, "-restype 2\n");
	fprintf(fp, "-pcShell 2\n");
	fprintf(fp, "-mg_levels_pc_type shell\n");
	fprintf(fp, "-mg_levels_ksp_type cg\n");
	fprintf(fp, "-mg_coarse_pc_type lu\n");
	//fprintf(fp, "-mg_coarse_ksp_monitor\n");
	fprintf(fp, "-mg_coarse_ksp_max_it 10\n");
	fprintf(fp, "-mg_coarse_ksp_rtol 1e-12\n");
	fprintf(fp, "-mg_levels_ksp_max_it 5\n");
	fprintf(fp, "-mg_levels_ksp_rtol 1.0e-1\n");

	fclose(fp);

	return TRUE;
}
BOOL MakeHOPSFile(const char* hops_file, double (*tp_c)[3], double* tp_T, int tp_num, const char* scan_atlas_reg_image_list, const char* atlas_prior_list, 
#ifdef USE_PROB_PRIOR
	const char* atlas_reg_prob_list, double prob_weight, 
#endif
#ifdef USE_ED_NON_WM_PROB
	double ed_non_wm_prob,
#endif
	const char* scan_means_file, const char* scan_variances_file, const char* scan_h_hdr_file, const char* atlas_label_map_s_img_file, 
	const char* simulator_input_file, const char* scales_file, const char* out_folder, const char* tmp_folder, char* solution_file, int max_eval, double* s_val, int k, bool hop_sync_eval, bool hop_random_order, int num_hop_threads)
{
	BOOL optim_xc = TRUE;
	BOOL optim_yc = TRUE;
	BOOL optim_zc = TRUE;
	BOOL optim_T = TRUE;
	BOOL optim_p1 = TRUE;   // mass effect
	BOOL optim_p2 = FALSE;  // mass effect
	BOOL optim_rho = FALSE; // tumor growth rate
	BOOL optim_dw = TRUE;   // tumor cell diffusity in the white matter
#ifdef USE_OPTIM_DG
	//BOOL optim_dg = TRUE;
	BOOL optim_dg = FALSE;  // tumor cell diffusity in the gray matter
#endif
#ifdef USE_OPTIM_ED
	BOOL optim_ed = TRUE;
#endif
	double minx[MaxNumberOFTumorSeeds], maxx[MaxNumberOFTumorSeeds], miny[MaxNumberOFTumorSeeds], maxy[MaxNumberOFTumorSeeds], minz[MaxNumberOFTumorSeeds], maxz[MaxNumberOFTumorSeeds];
	double min_growth_time[MaxNumberOFTumorSeeds], max_growth_time[MaxNumberOFTumorSeeds];
	double minp1, maxp1;
	double minp2, maxp2;
	double minrho, maxrho;
	double mindw, maxdw;
#ifdef USE_OPTIM_DG
	double mindg, maxdg;
#endif
#ifdef USE_OPTIM_ED
	double mined, maxed;
#endif
	double rw;
	FILE* fp;
	//
	double gxc[MaxNumberOFTumorSeeds], gyc[MaxNumberOFTumorSeeds], gzc[MaxNumberOFTumorSeeds], T[MaxNumberOFTumorSeeds];
	int i;

    // calcurate range weight for xc, yc, zc and growth_time
    if (max_eval < 1) {
		max_eval = 1;
	}
	if (max_eval <= 40) {
		optim_xc = FALSE;
		optim_yc = FALSE;
		optim_zc = FALSE;
	} else if (max_eval <= 70) {
		optim_xc = FALSE;
		optim_yc = FALSE;
		optim_zc = FALSE;
	} else {
		optim_xc = TRUE;
		optim_yc = TRUE;
		optim_zc = TRUE;
	}
    rw = (double)max_eval / 100;

	// convert units
	for (i = 0; i < tp_num; i++) {
		gxc[i] = tp_c[i][0] / 1000;
		gyc[i] = tp_c[i][1] / 1000;
		gzc[i] = tp_c[i][2] / 1000;
		T[i]   = tp_T[i]  / 1000;
	}

	// set valid ranges of parameters
	// assuming label image of atlas of size 64x64x64 and voxel size 3.5x3.5x3mm^3
	for (i = 0; i < tp_num; i++) {
		if (optim_xc) {					// scale: 1e0
#ifdef USE_OPTIM_LIMIT_XYZ_RANGES
			minx[i] = gxc[i] - (0.004 * rw);
			maxx[i] = gxc[i] + (0.004 * rw);
#else
			minx[i] = gxc[i] - (0.0105 * rw);	// -10.5 mm
			maxx[i] = gxc[i] + (0.0105 * rw);	// +10.5 mm
#endif
		} else {
			minx[i] = gxc[i];
			maxx[i] = gxc[i];
		}
		if (optim_yc) {					// scale: 1e0
#ifdef USE_OPTIM_LIMIT_XYZ_RANGES
			miny[i] = gyc[i] - (0.004 * rw);
			maxy[i] = gyc[i] + (0.004 * rw);
#else
			miny[i] = gyc[i] - (0.0105 * rw);	// -10.5 mm
			maxy[i] = gyc[i] + (0.0105 * rw);	// +10.5 mm
#endif
		} else {
			miny[i] = gyc[i];
			maxy[i] = gyc[i];
		}
		if (optim_zc) {					// scale: 1e0 
#ifdef USE_OPTIM_LIMIT_XYZ_RANGES
			minz[i] = gzc[i] - (0.004 * rw);
			maxz[i] = gzc[i] + (0.004 * rw);
#else
			minz[i] = gzc[i] - (0.0105 * rw);	// -10.5 mm
			maxz[i] = gzc[i] + (0.0105 * rw);	// +10.5 mm
#endif
		} else {
			minz[i] = gzc[i];
			maxz[i] = gzc[i];
		}
		if (optim_T) {				// scale: 1e3
			min_growth_time[i] = (1.0 - (0.5 * rw)) * T[i];
			max_growth_time[i] = (1.0 + (0.5 * rw)) * T[i];
		} else {
			min_growth_time[i] = T[i];
			max_growth_time[i] = T[i];
		}
	}
	if (optim_p1) {				// scale: 1e2
		minp1 = 0.01;
		//maxp1 = 0.20;
		maxp1 = 0.08;
	} else {
		minp1 = 0.03;
		maxp1 = 0.03;
	}
	if (optim_p2) {				// scale: 1e1
		minp2 = 0.00;
		maxp2 = 0.20;
	} else {
		minp2 = 0.00;
		maxp2 = 0.00;
	}
	if (optim_rho) {			// scale: 1e0
		minrho = 0.01;
		maxrho = 0.05;
	} else {
		minrho = 0.03;
		maxrho = 0.03;
	}
#ifdef USE_OPTIM_DG
	/*
	if (optim_dw) {				// scale: 1e-6
		mindw = 0.0001;
		//mindw = 0.00;
		maxdw = 0.80;
		mindg = 0.02;
		maxdg = 0.02;
	} else {
		mindw = 0.10;
		maxdw = 0.10;
		mindg = 0.02;
		maxdg = 0.02;
	}
	if (optim_dg) {				// scale: 1e-7
		mindg = 0.0001;
		//mindg = 0.00;
		maxdg = 0.15;
	}
	/*/
	if (optim_dw) {				
		mindw = OPTIM_DW_MIN;	// scale: 1e-6
		maxdw = OPTIM_DW_MAX;
		mindg = 0.001;			// scale: 1e-7
		maxdg = 0.001;
	} else {
		mindw = 0.15;
		maxdw = 0.15;
		mindg = 0.03;
		maxdg = 0.03;
	}
	if (optim_dg) {				// scale: 1e-7
		mindg = 0.001;
		maxdg = 0.03;
	}
	//*/
#else
	if (optim_dw) {				// scale: 1e-6
		//mindw = 0.075;
		mindw = 0.001;
		maxdw = 0.75;
	} else {
		// low diffusivity
		mindw = 0.075;
		maxdw = 0.075;
	}
#endif
#ifdef USE_OPTIM_ED
	if (optim_ed) {				// scale: 1e0
		mined = 0.001;
		maxed = 0.1;
	} else {
		mined = 0.01;
		maxed = 0.01;
	}
#endif
	
	// compose input for HOPSPACK
	fp = fopen(hops_file, "w");
	if (fp == NULL) {
		return FALSE;
	}

	fprintf(fp, "@ \"Problem Definition\"\n");
	fprintf(fp, "  \"Objective Type\" string \"Minimize\"\n");
	{
		int p_num;

#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
		p_num = 6 + 4 * tp_num;
#else
		p_num = 5 + 4 * tp_num;
#endif
#else
#ifdef USE_OPTIM_ED
		p_num = 5 + 4 * tp_num;
#else
		p_num = 4 + 4 * tp_num;
#endif
#endif
		fprintf(fp, "  \"Number Unknowns\" int %d\n", p_num);
#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
		fprintf(fp, "  \"Upper Bounds\" vector %d  %.15g %.15g %.15g %.15g %.15g %.15g", p_num, maxp1, maxp2, maxrho, maxdw, maxdg, maxed);
#else
		fprintf(fp, "  \"Upper Bounds\" vector %d  %.15g %.15g %.15g %.15g %.15g", p_num, maxp1, maxp2, maxrho, maxdw, maxdg);
#endif
#else
#ifdef USE_OPTIM_ED
		fprintf(fp, "  \"Upper Bounds\" vector %d  %.15g %.15g %.15g %.15g %.15g", p_num, maxp1, maxp2, maxrho, maxdw, maxed);
#else
		fprintf(fp, "  \"Upper Bounds\" vector %d  %.15g %.15g %.15g %.15g", p_num, maxp1, maxp2, maxrho, maxdw);
#endif
#endif
		for (i = 0; i < tp_num; i++) {
			fprintf(fp, " %.15g %.15g %.15g %.15g", maxx[i], maxy[i], maxz[i], max_growth_time[i]);
		}
		fprintf(fp, "\n");
#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
		fprintf(fp, "  \"Lower Bounds\" vector %d  %.15g %.15g %.15g %.15g %.15g %.15g", p_num, minp1, minp2, minrho, mindw, mindg, mined);
#else
		fprintf(fp, "  \"Lower Bounds\" vector %d  %.15g %.15g %.15g %.15g %.15g", p_num, minp1, minp2, minrho, mindw, mindg);
#endif
#else
#ifdef USE_OPTIM_ED
		fprintf(fp, "  \"Lower Bounds\" vector %d  %.15g %.15g %.15g %.15g %.15g", p_num, minp1, minp2, minrho, mindw, mined);
#else
		fprintf(fp, "  \"Lower Bounds\" vector %d  %.15g %.15g %.15g %.15g", p_num, minp1, minp2, minrho, mindw);
#endif
#endif
		for (i = 0; i < tp_num; i++) {
			fprintf(fp, " %.15g %.15g %.15g %.15g", minx[i], miny[i], minz[i], min_growth_time[i]);
		}
		fprintf(fp, "\n");
#ifdef USE_HOPS_INITIAL_X
		if (k > 0) {
			fprintf(fp, "  \"Initial X\" vector %d ", p_num);
			for (i = 0; i < p_num; i++) {
				fprintf(fp, " %.15g", s_val[i]);
			}
			fprintf(fp, "\n");
		}
#endif
#ifdef USE_OPTIM_DG
#ifdef USE_OPTIM_ED
		fprintf(fp, "  \"Scaling\" vector %d  1 1 1 1 1 1", p_num);
#else
		fprintf(fp, "  \"Scaling\" vector %d  1 1 1 1 1", p_num);
#endif
#else
#ifdef USE_OPTIM_ED
		fprintf(fp, "  \"Scaling\" vector %d  1 1 1 1 1", p_num);
#else
		fprintf(fp, "  \"Scaling\" vector %d  1 1 1 1", p_num);
#endif
#endif
		for (i = 0; i < tp_num; i++) {
			fprintf(fp, " 1 1 1 1");
		}
		fprintf(fp, "\n");
	}
	//fprintf(fp, "  \"Display\" int 0\n");
	fprintf(fp, "  \"Display\" int 2\n");
	fprintf(fp, "@@\n");

	fprintf(fp, "@ \"Evaluator\"\n");
	fprintf(fp, "  \"Evaluator Type\" string \"System Call\"\n");
#if defined(USE_PROB_PRIOR)
#ifdef USE_ED_NON_WM_PROB
	fprintf(fp, "  \"Executable Name\" string \"%s %s %s %s %f %f %s %s %s %s %s %s %s %s\"\n", 
#else
	fprintf(fp, "  \"Executable Name\" string \"%s %s %s %s %f %s %s %s %s %s %s %s %s\"\n", 
#endif
#else
#ifdef USE_ED_NON_WM_PROB
	fprintf(fp, "  \"Executable Name\" string \"%s %s %s %f %s %s %s %s %s %s %s %s\"\n", 
#else
	fprintf(fp, "  \"Executable Name\" string \"%s %s %s %s %s %s %s %s %s %s %s\"\n", 
#endif
#endif
		EVALUATE_Q_PATH, 
		scan_atlas_reg_image_list, atlas_prior_list, 
#ifdef USE_PROB_PRIOR
		atlas_reg_prob_list, prob_weight, 
#endif
#ifdef USE_ED_NON_WM_PROB
		ed_non_wm_prob,
#endif
		scan_means_file, scan_variances_file, scan_h_hdr_file, atlas_label_map_s_img_file, 
		simulator_input_file, scales_file, out_folder, tmp_folder);
	fprintf(fp, "  \"Save IO Files\" bool false\n");
	fprintf(fp, "@@\n");

	fprintf(fp, "@ \"Mediator\"\n");
	fprintf(fp, "  \"Citizen Count\" int 1\n");
	fprintf(fp, "  \"Number Processors\" int 1\n");
	fprintf(fp, "  \"Number Threads\" int %d\n", num_hop_threads+1);
	fprintf(fp, "  \"Maximum Evaluations\" int %d\n", max_eval);
	if (hop_sync_eval) {
		fprintf(fp, "  \"Synchronous Evaluations\" bool true\n");
	} else {
		fprintf(fp, "  \"Synchronous Evaluations\" bool false\n");
#ifdef USE_HOPS_MAX_RETURN
		fprintf(fp, "  \"Maximum Exchange Return\" int %d\n", num_hop_threads);
#endif
	}
	fprintf(fp, "  \"Solution File\" string \"%s\"\n", solution_file);
	//fprintf(fp, "  \"Display\" int 3\n");
	fprintf(fp, "  \"Display\" int 4\n");
	fprintf(fp, "@@\n");

	fprintf(fp, "@ \"Citizen 1\"\n");
	fprintf(fp, "  \"Type\" string \"GSS\"\n");
	fprintf(fp, "  \"Step Tolerance\" double 5.0e-3\n");
	if (hop_random_order) {
		fprintf(fp, "  \"Use Random Order\" bool true\n");
	} else {
		fprintf(fp, "  \"Use Random Order\" bool false\n");
	}
	//fprintf(fp, "  \"Display\" int 1\n");
	fprintf(fp, "  \"Display\" int 3\n");
	fprintf(fp, "@@\n");

	fclose(fp);

	return TRUE;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
	for (i = 0; i < NumberOfPriorChannels; i++) {
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
	for (i = 0; i < NumberOfPriorChannels; i++) {
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
			if ((strcmp(class_id, label[j] ) == 0) ||
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
	for (i = 0; i < NumberOfPriorChannels; i++) {
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
			if ((strcmp(class_id, label[j]) == 0 ) ||
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
BOOL LoadMeansAndVariances(const char* means_file, const char* variances_file, double (*mv)[4], double (*vv)[16])
{
	FILE* fp;
	int i, j, k, ii, jj;

	////////////////////////////////////////////////////////////////////////////////
	// initialize
	for (i = 0; i < NumberOfPriorChannels; i++) {
		for (j = 0; j < NumberOfImageChannels; j++) {
			mv[i][j] = 0;
		}
	}
	for (i = 0; i < NumberOfPriorChannels; i++) {
		for (j = 0; j < NumberOfImageChannels; j++) {
			for (k = 0; k < NumberOfImageChannels; k++) {
				vv[i][j*NumberOfImageChannels+k] = 0;
			}
		}
	}
	////////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////////
	// read means
	fp = fopen(means_file, "r");
	if (fp == NULL) {
		TRACE("Failed to open file: '%s'\n", means_file);
		return FALSE;
	}
	for (i = 0; i < NumberOfPriorChannels; i++) {
		char class_id[1024];
		float mean[NumberOfImageChannels];
		//
		fscanf(fp, "%s", class_id);
		for (ii = 0; ii < NumberOfImageChannels; ii++) {
			fscanf(fp, "%f", &mean[ii]);
		}
		for (j = 0; j < NumberOfPriorChannels; j++) {
#if defined(USE_8_PRIORS)
			if ((strcmp(class_id, label[j] ) == 0) ||
				(strcmp(class_id, label2[j]) == 0) ||
				(strcmp(class_id, label3[j]) == 0)) {
#else
			if (strcmp(class_id, label[j]) == 0) {
#endif
				for (ii = 0; ii < NumberOfImageChannels; ii++) {
					mv[j][ii] = mean[ii];
				}
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
	for (i = 0; i < NumberOfPriorChannels; i++) {
		char class_id[1024];
		float var[NumberOfImageChannels][NumberOfImageChannels];
		//
		fscanf(fp, "%s", class_id);
		for (ii = 0; ii < NumberOfImageChannels; ii++) {
			for (jj = 0; jj < NumberOfImageChannels; jj++) {
				fscanf(fp, "%f", &var[ii][jj]);
			}
		}
		for (j = 0; j < NumberOfPriorChannels; j++) {
#if defined(USE_8_PRIORS)
			if ((strcmp(class_id, label[j]) == 0 ) ||
				(strcmp(class_id, label2[j]) == 0) ||
				(strcmp(class_id, label3[j]) == 0)) {
#else
			if (strcmp(class_id, label[j]) == 0) {
#endif
				for (ii = 0; ii < NumberOfImageChannels; ii++) {
					for (jj = 0; jj < NumberOfImageChannels; jj++) {
						vv[j][ii*NumberOfImageChannels+jj] = var[ii][jj];
					}
				}
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
BOOL SaveMeansAndVariances(const char* means_file, const char* variances_file, double (*mv)[4], double (*vv)[16])
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
		double* mean;
		//
#if defined(USE_8_PRIORS)
		mean = mv[labeln[i]];
		//
		fprintf(fp, "%s\n", label[labeln[i]]);
#else
		mean = mv[i];
		//
		fprintf(fp, "%s\n", label[i]);
#endif
		//
		for (j = 0; j < NumberOfImageChannels; j++) {
			fprintf(fp, "%f", mean[j]);
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
		double* var;
		//
#if defined(USE_8_PRIORS)
		var = vv[labeln[i]];
		//
		fprintf(fp, "%s\n", label[labeln[i]]);
#else
		var = vv[i];
		//
		fprintf(fp, "%s\n", label[i]);
#endif
		//
		for (j = 0; j < NumberOfImageChannels; j++) {
			for (k = 0; k < NumberOfImageChannels; k++) {
				fprintf(fp, "%f", var[j * NumberOfImageChannels + k]);
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
/**
\brief

Input from Brain Tumor Viewer (points per tissue)
\return Estimates mean intensity and model the variance (currently, it just does +/- 10 to mean)

\param means_files File names of means
\param pMeanVector Vector that holds a mean value for each tissue type on each modality type in the order [T1 T1CE T2 FLAIR]
\param pVarianceVector Vector that holds a variance value for each tissue type on each modality type in the order [T1 T1CE T2 FLAIR] (currently, it just does +/- 10 to mean)
*/

BOOL InitializeMeansAndVariances(const char* means_file, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector)
{
	FILE* fp;
	int i, j;

	////////////////////////////////////////////////////////////////////////////////
	pMeanVector->clear();
	for(i = 0; i < NumberOfPriorChannels; i++) {
		MeanType v(0.0);
		pMeanVector->push_back(v); 
	}
	////////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////////
	// read means
	fp = fopen(means_file, "r");
	if (fp == NULL) {
		TRACE("Failed to open file: '%s'\n", means_file);
		return FALSE;
	}
	for (i = 0; i < NumberOfPriorChannels; i++) {
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
	pVarianceVector->clear();
	for(i = 0; i < NumberOfPriorChannels; i++) {
		VarianceType s;
		s.set_identity();
		s *= 10.0;
		pVarianceVector->push_back(s);
	}
	////////////////////////////////////////////////////////////////////////////////

	return TRUE;
}

/**
*/
void tempMeanAndVarIterator(std::vector< std::vector< double > > &ourMeansWrap, std::vector< std::vector< double > > &ourStDevWrap,
  const std::vector< std::vector< double > > &inputMasterVector, const std::string &ImageType )
{
  std:vector< double > tempMean, tempStD;
  tempMean.clear();
  tempStD.clear();

  // once passed through all points, show array of intensities
  //std::cout << "\n===intensity_values_:" << ImageType;

  for (size_t tissueTypes = 0; tissueTypes < inputMasterVector.size(); ++tissueTypes)
  {
    //std::cout << "\n--index:" << sarthak << "\n";
    double ourMean = 0, ourVariance = 0.0;
    ourMean = std::accumulate(inputMasterVector[tissueTypes].begin(), inputMasterVector[tissueTypes].end(), 0.0);
    if (inputMasterVector[tissueTypes].size() > 0)
    {
      ourMean /= inputMasterVector[tissueTypes].size();
    }
    for (size_t points = 0; points < inputMasterVector[tissueTypes].size(); points++)
    {
      //std::cout << inputMasterVector[sarthak][spyros] << " ";
      ourVariance += (inputMasterVector[tissueTypes][points] - ourMean) * (inputMasterVector[tissueTypes][points] - ourMean);
    }
    //std::cout << "\n";

    if (inputMasterVector[tissueTypes].size() > 1)
    {
      ourVariance = ourVariance / (inputMasterVector[tissueTypes].size() - 1);
    }
    double ourStDev;
    if (ourVariance != 0)
    {
      ourStDev = sqrt(ourVariance) /*+ 10sqrt(ourVariance)/2*/;
    }
    else
    {
      ourStDev = 10;
    }

    //std::cout << "||mean = " << ourMean << "\n";
    //std::cout << "||stDev = " << ourStDev << "\n";
    tempStD.push_back(ourStDev);
    tempMean.push_back(ourMean);
  }
  ourMeansWrap.push_back(tempMean);
  ourStDevWrap.push_back(tempStD);
}

/**
\brief 

Input from Brain Tumor Viewer (points per tissue)
\return Estimates mean intensity and model the variance (currently, it just does +/- 10 to mean)

\param scan_points_info Coordinates of each initialization points (x,y,z)
\param scan_points_num Number of points per tissue
\param image_files File names of images 
\param pMeanVector Vector that holds a mean value for each tissue type on each modality type in the order [T1 T1CE T2 FLAIR]
\param pVarianceVector 2D Vector (diagonal matrix) that holds a variance value for each tissue type on each modality type in the order [T1 T1CE T2 FLAIR] (currently, it just does +/- 10 to mean)
\param check_orientation ???
\param b_valid_label ???
*/
BOOL InitializeMeansAndVariancesFromPoints(double (*scan_points_info)[4], int scan_points_num, char (*image_files)[1024], 
  MeanVectorType &pMeanVector, VarianceVectorType &pVarianceVector, int check_orientation, int* b_valid_label)
{
  std::cout << "<<< Entered Initializing function.. >>>\n\n";

	FVolume images[NumberOfImageChannels];
	double scan_points_info_out[MaxNumberOFPoints][4];
	//double color_mean[NumberOfPriorChannels][NumberOfImageChannels];
  std::vector< std::vector<double> > intensity_values_T1, intensity_values_T1CE, intensity_values_T2, intensity_values_T2FL;
  int color_num[NumberOfPriorChannels];
	int i, j;

  std::vector< std::vector< double > > color_mean, ourStDevs;
  color_mean.clear();
  ourStDevs.clear();

  // initialize intensity value vectors with the maximum possible tissue types
  intensity_values_T1.resize(NumberOfPriorChannels);
  intensity_values_T1CE.resize(NumberOfPriorChannels);
  intensity_values_T2.resize(NumberOfPriorChannels);
  intensity_values_T2FL.resize(NumberOfPriorChannels);

	for (i = 0; i < NumberOfImageChannels; i++) 
  {
		images[i].load(image_files[i], 1);
	}

	////////////////////////////////////////////////////////////////////////////////
	pMeanVector.clear();
	for (i = 0; i < NumberOfPriorChannels; i++) 
  {
		MeanType v(0.0);
		pMeanVector.push_back(v); 
	}
	////////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////////
	for (i = 0; i < NumberOfPriorChannels; i++) 
  {
		color_num[i] = 0;
		//for (j = 0; j < NumberOfImageChannels; j++) 
  //  {
		//	//color_mean[i][j] = 0;
		//}
	}

	if (!GetPoints(image_files[0], scan_points_info, scan_points_info_out, scan_points_num, check_orientation)) 
  {
		return FALSE;
	}

	for (i = 0; i < scan_points_num; i++) 
  {
    std::vector< double > test_vals_T1;
		int ix, iy, iz, type;
		ix = static_cast<int>(scan_points_info_out[i][0] + 0.1);
    iy = static_cast<int>(scan_points_info_out[i][1] + 0.1);
    iz = static_cast<int>(scan_points_info_out[i][2] + 0.1);
    type = static_cast<int>(scan_points_info_out[i][3] + 0.1);
		if (ix >= 0 && ix < images[0].m_vd_x && iy >= 0 && iy < images[0].m_vd_y && iz >= 0 && iz < images[0].m_vd_z) 
    {
			TRACE2("point %d: (%d, %d, %d)\n", i, ix, iy, iz);
			//
			for (j = 0; j < NumberOfImageChannels; j++) 
      {
				//color_mean[type][j] += images[j].m_pData[iz][iy][ix][0];

        //std::cout << "=====Switch statement starts=====\n";
        switch (j)
        {
        case 0:
          intensity_values_T1[type].push_back(images[j].m_pData[iz][iy][ix][0]);
          //std::cout << "imgIn:" << j << ", T1 value:" << images[j].m_pData[iz][iy][ix][0] << "\n";
          break;
        case 1:
          intensity_values_T1CE[type].push_back(images[j].m_pData[iz][iy][ix][0]);
          //std::cout << "imgIn:" << j << ", T1CE value:" << images[j].m_pData[iz][iy][ix][0] << "\n";
          break;
        case 2:
          intensity_values_T2[type].push_back(images[j].m_pData[iz][iy][ix][0]);
          //std::cout << "imgIn:" << j << ", T2 value:" << images[j].m_pData[iz][iy][ix][0] << "\n";
          break;
        case 3:
          intensity_values_T2FL[type].push_back(images[j].m_pData[iz][iy][ix][0]);
          //std::cout << "imgIn:" << j << ", T2 Flair value:" << images[j].m_pData[iz][iy][ix][0] << "\n";
          break;
        default:
          //std::cerr << "Number of image channels are inconsistent. End of world detected..\n";
          break;
        }
			}
			color_num[type]++;

		} 
    else 
    {
			TRACE("wrong point: (%d, %d, %d), use check orientation\n", ix, iy, iz);
			return FALSE;
		}
	}

  tempMeanAndVarIterator(color_mean, ourStDevs, intensity_values_T1, "T1");
  tempMeanAndVarIterator(color_mean, ourStDevs, intensity_values_T1CE, "T1CE");
  tempMeanAndVarIterator(color_mean, ourStDevs, intensity_values_T2, "T2");
  tempMeanAndVarIterator(color_mean, ourStDevs, intensity_values_T2FL, "T2FL");

  std::vector< std::vector < double > > tmpMeans;
  tmpMeans.resize(NumberOfPriorChannels);

  for (i = 0; i < NumberOfPriorChannels; i++)
  {
    for (j = 0; j < NumberOfImageChannels; j++)
    {
      tmpMeans[i].push_back(color_mean[j][i]);
    }
  }
  color_mean.clear();
  color_mean = tmpMeans;
  
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	if (color_num[ED] > 0 && color_num[NE] > 0) 
  {
		bool bSame = true;
		for (j = 0; j < NumberOfImageChannels; j++) 
    {
			if (color_mean[ED][j] != color_mean[NE][j]) 
      {
				bSame = false;
				break;
			}
		}
		if (bSame) 
    {
			// indicate NE is going to be merged with ED
			b_valid_label[NE] = 2;
			TRACE2("b_valid_label[NE] = 2\n");
		}
	}
#endif

	/*for (i = 0; i < NumberOfPriorChannels; i++) 
  {
		if (color_num[i] > 0) 
    {
			for (j = 0; j < NumberOfImageChannels; j++) 
      {
				color_mean[i][j] /= color_num[i];
			}
		}
	}*/
	if (color_num[BG] == 0) 
  {
		for (j = 0; j < NumberOfImageChannels; j++) 
    {
			color_mean[BG][j] = 0.05;
		}
		color_num[BG] = 1;
	}
	if (color_num[TU] == 0) 
  {
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
		if (color_num[NE] != 0) {
			for (j = 0; j < NumberOfImageChannels; j++) {
				color_mean[TU][j] = color_mean[NE][j];
			}
			color_num[TU] = color_num[NE];
		} else if (color_num[NCR] != 0) {
			for (j = 0; j < NumberOfImageChannels; j++) {
				color_mean[TU][j] = color_mean[NCR][j];
			}
			color_num[TU] = color_num[NCR];
		} else if (color_num[ED] != 0) {
			// in this case, we set tumor label using ed and switch to tumor label to ed afterwards
			for (j = 0; j < NumberOfImageChannels; j++) {
				color_mean[TU][j] = color_mean[ED][j];
			}
			color_num[TU] = color_num[ED];
		} else {
			TRACE("no point on tumor and edema labels\n");
			return FALSE;
		}
#else
		if (color_num[NCR] != 0) {
			for (j = 0; j < NumberOfImageChannels; j++) {
				color_mean[TU][j] = color_mean[NCR][j];
			}
			color_num[TU] = color_num[NCR];
		} else {
			TRACE("no point on tumor labels\n");
			return FALSE;
		}
#endif
	}
	if (color_num[NCR] == 0) {
		for (j = 0; j < NumberOfImageChannels; j++) {
			color_mean[NCR][j] = color_mean[TU][j];
		}
		color_num[NCR] = color_num[TU];
	}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	if (color_num[NE] == 0) {
		for (j = 0; j < NumberOfImageChannels; j++) {
			color_mean[NE][j] = color_mean[NCR][j];
		}
		color_num[NE] = color_num[NCR];
	}
#endif
	if (color_num[ED] == 0) {
		for (j = 0; j < NumberOfImageChannels; j++) {
			color_mean[ED][j] = color_mean[WM][j];
		}
		color_num[ED] = color_num[WM];
	}
#if defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	if (color_num[CB] == 0) {
		for (j = 0; j < NumberOfImageChannels; j++) {
			color_mean[CB][j] = color_mean[WM][j];
		}
		color_num[CB] = color_num[WM];
	}
#endif

	for (i = 0; i < NumberOfPriorChannels; i++) {
		if (color_num[i] == 0) {
			TRACE("no point on %s\n", label[i]);
			return FALSE;
		}
	}
	//
	for (i = 0; i < NumberOfPriorChannels; i++) {
		MeanType mean;
		for (j = 0; j < NumberOfImageChannels; j++) {
			mean(j) = color_mean[i][j];
		}
		pMeanVector.at(i) = mean;
	}
 
	////////////////////////////////////////////////////////////////////////////////
  
  // modelling the variance of each tissue type code goes here //
	
  ////////////////////////////////////////////////////////////////////////////////
	pVarianceVector.clear();
	/*for(i = 0; i < NumberOfPriorChannels; i++) 
  {
		VarianceType s;
		s.set_identity();
		s *= 10.0;
		pVarianceVector.push_back(s);
	}

  VarianceVectorType OurVarianceVector;
  OurVarianceVector.clear();*/

  for (i = 0; i < NumberOfPriorChannels; i++)
  {
    VarianceType s;
    s.set_identity();
    for (int rs = 0, cs = 0; rs < NumberOfImageChannels, cs < NumberOfImageChannels; rs++, cs++)
    {
      s(rs,cs) = ourStDevs[rs][i];
    }
    pVarianceVector.push_back(s);
  }

  /// Display the mean and variances obtained on the console START
  std::cout << "==Initialization Mean values== \n";
  for (i = 0; i < NumberOfPriorChannels; i++)
  {
    std::cout << "-Tissue:" << i << "\n";

    std::cout << "T1:      " << color_mean[i][0] << "\n";
    std::cout << "T1CE:    " << color_mean[i][1] << "\n";
    std::cout << "T2:      " << color_mean[i][2] << "\n";
    std::cout << "T2FLAIR: " << color_mean[i][2] << "\n";
  }

  std::cout << "==Initialization Variance values== \n";
  std::cout << "----------------------------------\n";
  std::cout << "(Layout)\n";
  std::cout << "\tT1\t0\t0\t0\n";
  std::cout << "\t0\tT1CE\t0\t0\n";
  std::cout << "\t0\t0\tT2\t0\n";
  std::cout << "\t0\t0\t0\tT2FLAIR\n";
  std::cout << "----------------------------------\n";
  for (int tissueTypes = 0; tissueTypes < pVarianceVector.size(); tissueTypes++)
  {
    std::cout << "-Tissue:" << tissueTypes << "\n";
    for (int rows = 0; rows < pVarianceVector.at(tissueTypes).rows(); rows++)
    {
      for (int cs = 0; cs < pVarianceVector.at(tissueTypes).cols(); cs++)
      {
        std::cout << pVarianceVector.at(tissueTypes)(rows, cs) << "\t";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
  /// Display the mean and variances obtained on the console END

	////////////////////////////////////////////////////////////////////////////////

	for (i = 0; i < NumberOfImageChannels; i++) {
		images[i].clear();
	}

	return TRUE;
}

static
inline void ComputeLikelihood4(double m[16], double ym[4], double* like) {
	double inv[16], det, det_1, ySIy;
	
  // calculate invert of 'm'
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

#if defined(USE_3_IMAGES) || defined(USE_5_IMAGES)
BOOL ComputePosteriors(FVolume* vd, FVolume* priors, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector, FVolume* posteriors)
{
	BOOL res = TRUE;
	//
	int i, j, l, m, n;
	//
	int vd_x = vd[0].m_vd_x;
	int vd_y = vd[0].m_vd_y;
	int vd_z = vd[0].m_vd_z;
	//
	for (n = 0; n < vd_z; n++) {
		for (m = 0; m < vd_y; m++) {
			for (l = 0; l < vd_x; l++) {
				float priors_val[NumberOfPriorChannels];
				MeanType y;
#ifndef USE_DISCARD_ZERO_AREA
				double ys = 0;
#else
				double ys = 1;
#endif
				//
				for (i = 0; i < NumberOfPriorChannels; i++) {
					priors_val[i] = priors[i].m_pData[n][m][l][0];
				}
				//
				for (i = 0; i < NumberOfImageChannels; i++) {
					y(i) = vd[i].m_pData[n][m][l][0];
#ifndef USE_DISCARD_ZERO_AREA
					ys += y(i);
#else
					if (y(i) <= 0) {
						ys = 0;
					}
#endif
				}
				//
				// computing likelihood and posterior values
				double like[NumberOfPriorChannels];
				// if foreground...
				if (ys > 0) {
#if 1
					// to the number of classes
					for (i = 0; i < NumberOfPriorChannels; i++) {
						MeanType SIy = vnl_qr<double>(pVarianceVector->at(i)).solve(y - pMeanVector->at(i));
						double ySIy = dot_product(y - pMeanVector->at(i), SIy);
						double detS = vnl_determinant(pVarianceVector->at(i)) + epss;
						like[i] = 1.0 / (PI2M4 * vcl_sqrt(detS)) * exp(-0.5 * ySIy) + epss;
					}
					// compute the denum
#ifndef USE_SUM_EPSS2
					double denum = 0.0;
#else
					double denum = epss;
#endif
					for (i = 0; i < NumberOfPriorChannels; i++) {
						denum +=  priors_val[i] * like[i];
					}
					// compute the posterior
					if (denum != 0) {
						for (i = 0; i < NumberOfPriorChannels; i++) {
							posteriors[i].m_pData[n][m][l][0] = priors_val[i] * like[i] / denum;
						}
					} else {
						for (i = 0; i < NumberOfPriorChannels; i++) {
							posteriors[i].m_pData[n][m][l][0] = 0.0;
						}
					}
#endif
#if 0
					// to the number of classes
					for (i = 0; i < NumberOfPriorChannels; i++) {
						MeanType SIy = vnl_qr<double>(pVarianceVector->at(i)).solve(y - pMeanVector->at(i));
						double ySIy = dot_product(y - pMeanVector->at(i), SIy);
						double detS = vnl_determinant(pVarianceVector->at(i)) + epss;
						like[i] = 1.0 / (PI2M4 * vcl_sqrt(detS)) * exp(-0.5 * ySIy);
					}
					// compute the denum
					double denum = epss;
					for (i = 0; i < NumberOfPriorChannels; i++) {
						denum +=  priors_val[i] * like[i];
					}
					// compute the posterior
					for (i = 0; i < NumberOfPriorChannels; i++) {
						posteriors[i].m_pData[n][m][l][0] = priors_val[i] * like[i] / denum;
					}
#endif
				// if background...
				} else {
					// compute the posterior
					for (i = 0; i < NumberOfPriorChannels; i++) {
						posteriors[i].m_pData[n][m][l][0] = 0.0;
					}
					posteriors[BG].m_pData[n][m][l][0] = 1.0;
				}
			} // l
		} // m
	} // n

	return res;
}
#elif defined(USE_4_IMAGES)
// faster version
BOOL ComputePosteriors(FVolume* vd, FVolume* priors, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector, FVolume* posteriors
#ifdef USE_TU_PRIOR_CUTOFF
	, float tu_prior_cutoff_tu_th, float tu_prior_cutoff_ncr_th
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	, float tu_prior_cutoff_ne_th
#endif
	, float tu_prior_cutoff_ed_th
#endif
#ifdef USE_PL_COST
	, FVolume* pl_costs
#endif
	, int* b_valid_label
	)
{
	double vv[NumberOfPriorChannels][16];
	double vv_inv[NumberOfPriorChannels][16];
	double vv_det[NumberOfPriorChannels], vv_c1[NumberOfPriorChannels], vv_c2[NumberOfPriorChannels];
	double mv[NumberOfPriorChannels][4];
	BOOL res = TRUE;
	//
	int i, j, k, l, m, n;
	//
	int vd_x = vd[0].m_vd_x;
	int vd_y = vd[0].m_vd_y;
	int vd_z = vd[0].m_vd_z;
	//
	// assume 4 channels for image
	if (NumberOfImageChannels != 4) {
		return FALSE;
	}
	//
	for (i = 0; i < NumberOfPriorChannels; i++) {
		for (j = 0; j < 4; j++) {
			for (k = 0; k < 4; k++) {
				vv[i][j*4+k] = pVarianceVector->at(i)(j, k);
			}
			mv[i][j] = pMeanVector->at(i)(j);
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
	//
	for (n = 0; n < vd_z; n++) {
		for (m = 0; m < vd_y; m++) {
			for (l = 0; l < vd_x; l++) {
				float priors_val[NumberOfPriorChannels];
				double like[NumberOfPriorChannels];
				double y[NumberOfImageChannels], ym[NumberOfImageChannels];
#ifndef USE_DISCARD_ZERO_AREA
				double ys = 0;
#else
				double ys = 1;
#endif
				//
				for (i = 0; i < NumberOfPriorChannels; i++) {
					priors_val[i] = priors[i].m_pData[n][m][l][0];
				}
#ifdef USE_TU_PRIOR_CUTOFF
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
				if (priors_val[TU ] < tu_prior_cutoff_tu_th ) { priors_val[TU ] = 0; }
				if (priors_val[NCR] < tu_prior_cutoff_ncr_th) { priors_val[NCR] = 0; }
				if (priors_val[NE ] < tu_prior_cutoff_ne_th ) { priors_val[NE ] = 0; }
				//
				if (priors_val[TU ] > tu_prior_cutoff_ed_th || 
					priors_val[NCR] > tu_prior_cutoff_ed_th || 
					priors_val[NE ] > tu_prior_cutoff_ed_th) 
				{ 
					for (i = 0; i < NumberOfPriorChannels; i++) {
						if (i != TU && i != NCR && i != NE) {
							priors_val[i] = 0; 
						}
					}
				}
#else
				if (priors_val[TU ] < tu_prior_cutoff_tu_th ) { priors_val[TU ] = 0; }
				if (priors_val[NCR] < tu_prior_cutoff_ncr_th) { priors_val[NCR] = 0; }
				//
				if (priors_val[TU ] > tu_prior_cutoff_ed_th || 
					priors_val[NCR] > tu_prior_cutoff_ed_th) 
				{ 
					for (i = 0; i < NumberOfPriorChannels; i++) {
						if (i != TU && i != NCR) {
							priors_val[i] = 0; 
						}
					}
				}
#endif
#endif
				//
				for (i = 0; i < NumberOfImageChannels; i++) {
					y[i] = vd[i].m_pData[n][m][l][0];
#ifndef USE_DISCARD_ZERO_AREA
					ys += y[i];
#else
					if (y[i] <= 0) {
						ys = 0;
					}
#endif
				}
				//
				// computing likelihood and posterior values
				// if foreground...
				if (ys > 0) {
					// to the number of classes
					for (i = 0; i < NumberOfPriorChannels; i++) {
						if (b_valid_label[i] == 0) {
							like[i] = 0;
							continue;
						}
						//
						ym[0] = y[0] - mv[i][0]; 
						ym[1] = y[1] - mv[i][1]; 
						ym[2] = y[2] - mv[i][2]; 
						ym[3] = y[3] - mv[i][3]; 
#ifdef USE_LIKE_TU_T1CE
						if (i == TU) {
							ComputeLikelihood1(vv[i][5], ym[1], &like[i]);
						} else {
							//ComputeLikelihood4(vv[i], ym, &like[i]);
							ComputeLikelihood4(vv_inv[i], vv_c1[i], vv_c2[i], ym, &like[i]);
						}
#else
						//ComputeLikelihood4(vv[i], ym, &like[i]);
						ComputeLikelihood4(vv_inv[i], vv_c1[i], vv_c2[i], ym, &like[i]);
#endif
					}
#ifndef USE_LIKE_TU_MAX
					// compute the denum
#ifndef USE_SUM_EPSS2
					double pl_val, denum = 0.0;
#else
					double pl_val, denum = epss;
#endif
					for (i = 0; i < NumberOfPriorChannels; i++) {
						pl_val = priors_val[i] * like[i];
						denum += pl_val;
						//
#ifdef USE_PL_COST
						pl_costs[i].m_pData[n][m][l][0] = pl_val;
#endif
					}
					// compute the posterior
					if (denum != 0) {
						for (i = 0; i < NumberOfPriorChannels; i++) {
							posteriors[i].m_pData[n][m][l][0] = priors_val[i] * like[i] / denum;
						}
					} else {
						for (i = 0; i < NumberOfPriorChannels; i++) {
							posteriors[i].m_pData[n][m][l][0] = 0.0;
						}
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
							denum += priors_val[i] * like[i];
						}
					}
					denum += priors_val[TU] * 3 * like_tu_max;
					// compute the posterior
					if (denum != 0) {
						for (i = 0; i < NumberOfPriorChannels; i++) {
							if (i != TU && i != NCR && i != NE) {
								posteriors[i].m_pData[n][m][l][0] = priors_val[i] * like[i] / denum;
							}
						}
						post_tu_max = priors_val[TU] * 3 * like_tu_max / denum;
						//
						denum_tu = like[TU];
						if (bValidNCR) {
							denum_tu += like[NCR];
						}
						if (bValidNE) {
							denum_tu += like[NE];
						}
						if (denum_tu != 0) {
							posteriors[TU].m_pData[n][m][l][0] = post_tu_max * like[TU] / denum_tu;
							if (bValidNCR) {
								posteriors[NCR].m_pData[n][m][l][0] = post_tu_max * like[NCR] / denum_tu;
							} else {
								posteriors[NCR].m_pData[n][m][l][0] = posteriors[TU].m_pData[n][m][l][0];
							}
							if (bValidNE) {
								posteriors[NE].m_pData[n][m][l][0] = post_tu_max * like[NE] / denum_tu;
							} else {
								posteriors[NE].m_pData[n][m][l][0] = posteriors[NCR].m_pData[n][m][l][0];
							}
						} else {
							posteriors[TU].m_pData[n][m][l][0] = 0;
							posteriors[NCR].m_pData[n][m][l][0] = 0;
							posteriors[NE].m_pData[n][m][l][0] = 0;
						}
					} else {
						for (i = 0; i < NumberOfPriorChannels; i++) {
							posteriors[i].m_pData[n][m][l][0] = 0.0;
						}
					}
#endif
				// if background...
				} else {
					// compute the posterior
					for (i = 0; i < NumberOfPriorChannels; i++) {
						posteriors[i].m_pData[n][m][l][0] = 0.0;
						//
#ifdef USE_PL_COST
						pl_costs[i].m_pData[n][m][l][0] = 0;
#endif
					}
					posteriors[BG].m_pData[n][m][l][0] = 1.0;
					//
#ifdef USE_PL_COST
					pl_costs[BG].m_pData[n][m][l][0] = 0;
#endif
				}
			} // l
		} // m
	} // n

	return res;
}
#endif
BOOL ComputeLabelMap(FVolume* posteriors, BVolume& label_map, int* b_valid_label) {
	int vd_x, vd_y, vd_z;
	int l, m, n, i;
	int b_valid_label_tmp[NumberOfPriorChannels];
	bool b_merge_ed = false;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	bool b_merge_ed_ne = false;
#endif
	
	vd_x = posteriors[0].m_vd_x;
	vd_y = posteriors[0].m_vd_y;
	vd_z = posteriors[0].m_vd_z;

	for (i = 0; i < NumberOfPriorChannels; i++) {
		b_valid_label_tmp[i] = b_valid_label[i];
	}
	if (b_valid_label[ED]) {
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
		if (b_valid_label[TU] == 0 && b_valid_label[NCR] == 0 && b_valid_label[NE] == 0) {
			b_valid_label_tmp[TU ] = 1;
			b_valid_label_tmp[NCR] = 1;
			b_valid_label_tmp[NE ] = 1;
#else
		if (b_valid_label[TU] == 0 && b_valid_label[NCR] == 0) {
			b_valid_label_tmp[TU ] = 1;
			b_valid_label_tmp[NCR] = 1;
#endif
			b_merge_ed = true;
		}
	}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
	if (b_valid_label[NE] == 2) {
		b_merge_ed_ne = true;
	}
#endif

	for (n = 0; n < vd_z; n++) {
		for (m = 0; m < vd_y; m++) {
			for (l = 0; l < vd_x; l++) {
				float max_l_val = 0;
				int max_l_idx = -1;
				for (i = 0; i < NumberOfPriorChannels; i++) {
					if (b_valid_label_tmp[i] == 0) {
						continue;
					}
					if (max_l_val < posteriors[i].m_pData[n][m][l][0]) {
						max_l_val = posteriors[i].m_pData[n][m][l][0];
						max_l_idx = i;
					} else if (max_l_val == posteriors[i].m_pData[n][m][l][0]) {
						// prefer TU or higher labels
						if ((max_l_idx == TU && i == NCR) || (max_l_idx == NCR && i == TU)) {
							max_l_idx = TU;
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
						} else if ((max_l_idx == TU && i == NE) || (max_l_idx == NE && i == TU)) {
							max_l_idx = TU;
						} else if ((max_l_idx == NCR && i == NE) || (max_l_idx == NE && i == NCR)) {
							max_l_idx = NCR;
#endif
						} else {
							max_l_idx = i;
						}
					}
				}
				if (max_l_idx >= 0) {
					if (b_merge_ed) {
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
						if (max_l_idx == TU || max_l_idx == NCR || max_l_idx == NE) {
							label_map.m_pData[n][m][l][0] = label_idx[ED];
#else
						if (max_l_idx == TU || max_l_idx == NCR) {
#endif
							label_map.m_pData[n][m][l][0] = label_idx[ED];
						} else {
							label_map.m_pData[n][m][l][0] = label_idx[max_l_idx];
						}
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
					} else if (b_merge_ed_ne) {
						if (max_l_idx == NE) {
							label_map.m_pData[n][m][l][0] = label_idx[ED];
						} else {
							label_map.m_pData[n][m][l][0] = label_idx[max_l_idx];
						}
#endif
					} else {
						label_map.m_pData[n][m][l][0] = label_idx[max_l_idx];
					}
				} else {
					label_map.m_pData[n][m][l][0] = 0;
				}
			}
		}
	}

	return TRUE;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
