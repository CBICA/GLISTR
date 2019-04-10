///////////////////////////////////////////////////////////////////////////////////////
// stdafx.h
// Developed by Dongjin Kwon
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2011-2014 Dongjin Kwon
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(WIN32) || defined(WIN64)
////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <Windows.h>
//
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
////////////////////////////////////////////////////////////////////////////////////////////////////////
#else
////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
//
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <float.h>
#include <fcntl.h>
#if !defined(__APPLE__)
#include <sys/sendfile.h>
#endif
#include <dirent.h>
#include <fnmatch.h>
////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

//#define BRATS_2013
#ifdef BRATS_2013 // this flag has become redundant. FUTURE WORK: check which parts of the code are useful
////////////////////////////////////////////////////////////////////////////////////////////////////////
#define USE_10A_PRIORS
#define USE_4_IMAGES
//
#define USE_OPTIM_DG
#define USE_OPTIM_LIMIT_XYZ_RANGES
#define OPTIM_DW_MIN 0.001
#define OPTIM_DW_MAX 0.75
#define USE_HOPS_INITIAL_X
#define USE_HOPS_RANDOM_ORDER
//
#define USE_DISCARD_ZERO_AREA
#define USE_WARPED_BG
//
#define USE_MASK_WEIGHT
//
#define USE_MASK_G
//
#define USE_PROB_PRIOR
#ifdef USE_PROB_PRIOR 
#define PROB_SMOOTHING
#endif
//
#define USE_ED_TU_DENS_NONE
//#define USE_ED_TU_DENS_NONE_MIX
#if defined(USE_ED_TU_DENS_NONE) || defined(USE_ED_TU_DENS_NONE_MIX)
#define ED_TU_DENS_TH 1e-8
#endif
//
#define USE_ED_NON_WM_PROB
#ifdef USE_ED_NON_WM_PROB
#define NON_WM_PROB 0.05
#endif
//
#define USE_MEAN_SHIFT_UPDATE
//#define USE_MEAN_SHIFT_UPDATE_ONCE
//#define USE_MEAN_SHIFT_UPDATE_STEP
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE) || defined(USE_MEAN_SHIFT_UPDATE_STEP)
#define USE_MS_TU_NCR_NE
#define USE_MS_WM_ED
#endif
//
#define USE_TU_PRIOR_CUTOFF
#ifdef USE_TU_PRIOR_CUTOFF
#define TU_PRIOR_CUTOFF_TU_TH 0.01
#define TU_PRIOR_CUTOFF_NCR_TH 0.5
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
#define TU_PRIOR_CUTOFF_NE_TH 0.5
#endif
#define TU_PRIOR_CUTOFF_ED_TH 0.8
#endif
//
#define USE_TU_MULTI_ADD
//
#define USE_COMPUTE_Q_ESTIMATE_VARIANCES
//
#define RWR_BG_PROB
#ifdef RWR_BG_PROB
#define RWR_USE_POST_TH
#ifdef RWR_USE_POST_TH
#define RWR_POST_TH 0.8
#endif
#endif
//
#define USE_FAST_LIKELIHOOD
//
#define SIMUL_ITER_STEP_0 100
#define SIMUL_ITER_STEP_1 100
#define SIMUL_ITER_STEP_N 100
//
#define USE_JSR_GLISTR
#ifdef USE_JSR_GLISTR
#define JSR_ITER_STEP_0 10
#define JSR_ITER_STEP_N 20
//
#define USE_JSR_ELASTIC
#define USE_JSR_PROB_WEIGHT_OLD
#endif
//
//#define USE_SUM_EPSS
//#define USE_SUM_EPSS2
//
#define USE_LABEL_NOISE_REDUCTION
#define CC_NUM_TH 100
#define USE_LABEL_CORRECTION
////////////////////////////////////////////////////////////////////////////////////////////////////////
#else
////////////////////////////////////////////////////////////////////////////////////////////////////////
//#define USE_8_PRIORS
//#define USE_9_PRIORS
//#define USE_10_PRIORS
#define USE_10A_PRIORS
//#define USE_11_PRIORS
//
//#define USE_3_IMAGES
#define USE_4_IMAGES
//#define USE_5_IMAGES
//
#define USE_OPTIM_DG
//#define USE_OPTIM_LIMIT_XYZ_RANGES
//#define USE_OPTIM_UPDATE_C
//#define USE_OPTIM_UPDATE_T
#define OPTIM_DW_MIN 0.001
#define OPTIM_DW_MAX 0.75
#define USE_HOPS_INITIAL_X
#define USE_HOPS_NO_SYNC_EVAL
#ifdef USE_HOPS_NO_SYNC_EVAL
#define USE_HOPS_MAX_RETURN
#endif
#define USE_HOPS_RANDOM_ORDER
//
#define USE_DISCARD_ZERO_AREA
#define USE_WARPED_BG
//#define USE_COST_BG
//
#define USE_MASK_WEIGHT
//
#define USE_MASK_G
//
#define USE_PROB_PRIOR
#ifdef USE_PROB_PRIOR 
//#define PROB_ESTIMATE_T
//#define PROB_PRIOR_USE_PREV_REGION
#define PROB_SMOOTHING
#define PROB_LOAD_TU_ONLY
#endif
//
//#define USE_ED_TU_DISP
//#define USE_ED_TU_DENS
#if defined(USE_ED_TU_DISP) || defined(USE_ED_TU_DENS)
//#define USE_OPTIM_ED
//#define USE_ED_FIND_RANGE
#ifdef USE_ED_FIND_RANGE
#define USE_ED_FIND_RANGE_MD
#endif
#endif
//#define USE_ED_TU_NONE
//#define USE_ED_TU_DISP_NONE
#ifdef USE_ED_TU_DISP_NONE
#define ED_TU_DISP_TH 1 
#endif
//#define USE_ED_TU_DENS_NONE
#define USE_ED_TU_DENS_NONE_MIX
#if defined(USE_ED_TU_DENS_NONE) || defined(USE_ED_TU_DENS_NONE_MIX)
#define ED_TU_DENS_TH 1e-8
#endif
//#define USE_ED_TU_DW
//
#define USE_ED_NON_WM_PROB
//
//#define USE_LIKE_TU_T1CE
//
//#define USE_OUTLIER_REJECTION_INIT_MEANS
#ifdef USE_OUTLIER_REJECTION_INIT_MEANS
//#define USE_INIT_MEANS_CSF
//#define USE_INIT_MEANS_VS
//#define USE_INIT_MEANS_VT
//#define USE_INIT_MEANS_ED
//#define USE_INIT_MEANS_GM
//#define USE_INIT_MEANS_WM
//#define USE_WM_UPDATE_SKIP_ED_REGION
#endif
//#define USE_OUTLIER_REJECTION_LAMBDA
#define USE_MEAN_SHIFT_UPDATE
//#define USE_MEAN_SHIFT_UPDATE_ONCE
//#define USE_MEAN_SHIFT_UPDATE_STEP
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE) || defined(USE_MEAN_SHIFT_UPDATE_STEP)
#define USE_MS_TU_NCR_NE
#define USE_MS_WM_ED
#endif
//
#define USE_TU_PRIOR_CUTOFF
#ifdef USE_TU_PRIOR_CUTOFF
#define TU_PRIOR_CUTOFF_TU_TH 0.01
#define TU_PRIOR_CUTOFF_NCR_TH 0.5
#if defined(USE_10_PRIORS) || defined(USE_10A_PRIORS) || defined(USE_11_PRIORS)
#define TU_PRIOR_CUTOFF_NE_TH 0.5
#endif
#define TU_PRIOR_CUTOFF_ED_TH 0.8
#endif
//
//#define USE_ED_EXPAND
//
//#define USE_PL_COST
//
//#define USE_TU_SIMUL_H0
//
#ifdef USE_11_PRIORS
#define USE_LIKE_TU_MAX
#endif
//
#define USE_TU_MULTI_ADD
//#define USE_TU_MULTI_MAX
//
//#define USE_COMPUTE_Q_ESTIMATE_VARIANCES
//
#define RWR_BG_PROB
#ifdef RWR_BG_PROB
#define RWR_USE_POST_TH
#ifdef RWR_USE_POST_TH
#define RWR_POST_TH 0.8
#endif
//#define RWR_USE_POST_COMP
#endif
#define RWR_ESTIMATE_RE_M
//
#define USE_FAST_LIKELIHOOD
//
#define SIMUL_ITER_STEP_0 40
//#define SIMUL_ITER_STEP_0 100
#define SIMUL_ITER_STEP_1 70
//#define SIMUL_ITER_STEP_1 100
#define SIMUL_ITER_STEP_N 100
//
#define USE_JSR_GLISTR
#ifdef USE_JSR_GLISTR
//#define JSR_ITER_STEP_0 5
#define JSR_ITER_STEP_0 10
#define JSR_ITER_STEP_N 20
//#define JSR_ITER_STEP_N 25
//
#define USE_JSR_ELASTIC
//#define USE_JSR_FLUID
//#define USE_JSR_MULTI_LEVEL
//#define USE_JSR_NORMALIZE_UPDATE_FIELD
//#define USE_JSR_WEGHT_IMAGE_PYRAMID
//#define USE_JSR_CHECK_PRIOR_MAG
//#define USE_JSR_PROB_WEIGHT_OLD
#define USE_JSR_NO_WARPED_BG
#endif
//
//#define USE_SUM_EPSS
#define USE_SUM_EPSS2
//
#define USE_LABEL_NOISE_REDUCTION
#define CC_NUM_TH 100
#define USE_LABEL_CORRECTION
//
#define UPDATE_DEFORMATION_FIELD
#ifdef UPDATE_DEFORMATION_FIELD 
#define USE_SYMM_REG
#define USE_CC_NCC
#define USE_DISCRETE_OPTIMIZATION
#define USE_CONTINUOUS_OPTIMIZATION
#define CONT_OPT_ITER_LEVEL_0 50
#define CONT_OPT_ITER_LEVEL_N 100
#define USE_INIT_FIELD
#define USE_INIT_FIELD_WEIGHT_ABNORMAL_REGION
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

#if defined(USE_3_IMAGES)
#define NumberOfImageChannels	3
#elif defined(USE_4_IMAGES)
#define NumberOfImageChannels	4
#elif defined(USE_5_IMAGES)
#define NumberOfImageChannels	5
#endif
// set number of tissues to consider
#if defined(USE_8_PRIORS)
#define NumberOfPriorChannels	8
#elif defined(USE_9_PRIORS)
#define NumberOfPriorChannels	9
#elif defined(USE_10_PRIORS) || defined(USE_10A_PRIORS)
#define NumberOfPriorChannels	10 // set number of tissues to consider
#elif defined(USE_11_PRIORS)
#define NumberOfPriorChannels	11
#endif

#if defined(USE_8_PRIORS)
// BG, CSF, GM, WM, VS, ED, NCR, TU
enum {
	BG = 0,
    CSF,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU
};
static const char label[NumberOfPriorChannels][32] = { "BG", "CSF", "GM", "WM", "VS", "ED", "NCR", "TU" };
static const char label2[NumberOfPriorChannels][32] = { "BG", "CSF", "GM", "WM", "VEINS", "EDEMA", "NCR", "TUMOR" };
static const char label3[NumberOfPriorChannels][32] = { "8", "4", "5", "6", "3", "7", "2", "1" };
static const int labeln[NumberOfPriorChannels] = { 7, 6, 4, 1, 2, 3, 5, 0 };
static const int label_idx[NumberOfPriorChannels] = { 0, 10, 150, 250, 25, 100, 175, 200 };
//static const int label_s_idx[NumberOfPriorChannels] = { 0, 10, 150, 250, 10, 250, 200, 200 };
static const int label_s_idx[NumberOfPriorChannels] = { 0, 10, 150, 250, 10, 250, 250, 250 };
#elif defined(USE_9_PRIORS)
// BG, CSF, VT, GM, WM, VS, ED, NCR, TU
enum {
	BG = 0,
    CSF,
    VT,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU
};
static const char label[NumberOfPriorChannels][32] = { "BG", "CSF", "VT", "GM", "WM", "VS", "ED", "NCR", "TU" };
static const int label_idx[NumberOfPriorChannels] = { 0, 10, 50, 150, 250, 25, 100, 175, 200 };
static const int label_s_idx[NumberOfPriorChannels] = { 0, 10, 10, 150, 250, 10, 250, 250, 250 };
#elif defined(USE_10_PRIORS)
// BG, CSF, VT, GM, WM, VS, ED, NCR, TU, NE
enum {
	BG = 0,
    CSF,
    VT,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU,
	NE
};
static const char label[NumberOfPriorChannels][32] = { "BG", "CSF", "VT", "GM", "WM", "VS", "ED", "NCR", "TU", "NE" };
static const int label_idx[NumberOfPriorChannels] = { 0, 10, 50, 150, 250, 25, 100, 175, 200, 185 };
static const int label_s_idx[NumberOfPriorChannels] = { 0, 10, 10, 150, 250, 10, 250, 250, 250, 250 };
#elif defined(USE_10A_PRIORS)
// BG, CSF, GM, WM, VS, ED, NCR, TU, NE, CB
enum {
	BG = 0,
    CSF,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU,
	NE,
	CB
};
static const char label[NumberOfPriorChannels][32] = { "BG", "CSF", "GM", "WM", "VS", "ED", "NCR", "TU", "NE", "CB" };
static const int label_idx[NumberOfPriorChannels] = { 0, 10, 150, 250, 25, 100, 175, 200, 185, 5 };
static const int label_s_idx[NumberOfPriorChannels] = { 0, 10, 150, 250, 10, 250, 250, 250, 250, 250 };
#elif defined(USE_11_PRIORS)
// BG, CSF, VT, GM, WM, VS, ED, NCR, TU, NE, CB
enum {
	BG = 0,
    CSF,
    VT,
    GM,
    WM,
    VS,
    ED,
    NCR,
    TU,
	NE,
	CB
};
static const char label[NumberOfPriorChannels][32] = { "BG", "CSF", "VT", "GM", "WM", "VS", "ED", "NCR", "TU", "NE", "CB" };
static const int label_idx[NumberOfPriorChannels] = { 0, 10, 50, 150, 250, 25, 100, 175, 200, 185, 5 };
static const int label_s_idx[NumberOfPriorChannels] = { 0, 10, 10, 150, 250, 10, 250, 250, 250, 250, 250 };
#endif

#if defined(WIN32) || defined(WIN64)
	#define DIR_SEP "\\"
	#define DIR_SEP_C '\\'
#else
	#define DIR_SEP "/"
	#define DIR_SEP_C '/'
#endif

#if (defined(WIN32) || defined(WIN64)) && defined(_DEBUG)
#if 1
#define MODULE_FOLDER					"D:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\bin"
#ifdef _DEBUG
#define EVALUATE_Q_PATH					"D:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\x64\\Debug\\EvaluateQ.exe"
#else
#define EVALUATE_Q_PATH					"D:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\x64\\Release\\EvaluateQ.exe"
#endif
//#define HOPSPACK_PATH					"D:\\LIBRARY\\HOPSPACK\\hopspack-2.0.2-win32\\HOPSPACK_main_serial.exe"
#define HOPSPACK_PATH					"D:\\LIBRARY\\HOPSPACK\\hopspack-2.0.2-win32\\HOPSPACK_main_threaded.exe"
#else
#define MODULE_FOLDER					"C:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\bin"
#ifdef _DEBUG
#define EVALUATE_Q_PATH					"C:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\x64\\Debug\\EvaluateQ.exe"
#else
#define EVALUATE_Q_PATH					"C:\\WORKSPACE\\PROJECT\\BrainTumorSegmentationS\\x64\\Release\\EvaluateQ.exe"
#endif
//#define HOPSPACK_PATH					MODULE_FOLDER ## "\\HOPSPACK_main_serial.exe"
#define HOPSPACK_PATH					MODULE_FOLDER ## "\\HOPSPACK_main_threaded.exe"
#endif
//
#define FLIRT_PATH						MODULE_FOLDER ## "\\flirt.exe"
#define CONVERT_XFM_PATH				MODULE_FOLDER ## "\\convert_xfm.exe"
#define TDCALC_PATH						MODULE_FOLDER ## "\\3dcalc.exe"
#define RESAMPLE_IMAGE_PATH				MODULE_FOLDER ## "\\ResampleImage.exe"
//
#define SIMULATOR_PATH					MODULE_FOLDER ## "\\ForwardSolverDiffusion.exe"
//
#define RESAMPLE_DEFORMATION_FIELD_PATH MODULE_FOLDER ## "\\ResampleDeformationField.exe"
#define REVERSE_DEFORMATION_FIELD_PATH	MODULE_FOLDER ## "\\ReverseDeformationField.exe"
#define WARP_IMAGE_PATH					MODULE_FOLDER ## "\\WarpImage.exe"
#define CONCATENATE_FIELD_PATH			MODULE_FOLDER ## "\\ConcatenateFields.exe"
//
#define SUSAN_PATH						MODULE_FOLDER ## "\\susan.exe"
#endif


#define USE_TRACE
#define USE_METAIO


#ifndef BOOL
#define BOOL int
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef BYTE
#define BYTE unsigned char
#endif
#ifndef DWORD
#define DWORD unsigned long
#endif
#ifndef LPCTSTR
#define LPCTSTR const char*
#endif

#define MIN(a,b)  (((a) < (b)) ? (a) : (b))
#define MAX(a,b)  (((a) > (b)) ? (a) : (b))

#ifdef USE_TRACE
//
//#ifdef _DEBUG
#if 1
#undef TRACE
#define TRACE Trace
#define TRACE1(...) {(g_verbose>=1)?Trace(__VA_ARGS__):((void)0);}
#define TRACE2(...) {(g_verbose>=2)?Trace(__VA_ARGS__):((void)0);}
#define TRACE3(...) {(g_verbose>=3)?Trace(__VA_ARGS__):((void)0);}
#ifdef __cplusplus
extern "C" {
#endif
extern BOOL g_bTrace;
extern FILE *g_fp_trace;
extern char g_trace_file[1024];
extern BOOL g_bTraceStdOut;
extern int g_verbose;
void Trace(const char* szFormat, ...);
#ifdef __cplusplus
}
#endif
#else
#ifndef TRACE
#define TRACE printf
#define TRACE1(...) {(g_verbose>=1)?printf(__VA_ARGS__):((void)0);}
#define TRACE2(...) {(g_verbose>=2)?printf(__VA_ARGS__):((void)0);}
#define TRACE3(...) {(g_verbose>=3)?printf(__VA_ARGS__):((void)0);}
#endif
#endif
//
#endif
