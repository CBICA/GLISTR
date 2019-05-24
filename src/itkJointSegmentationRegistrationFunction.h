/**
 * @file  itkJointSegmentationRegistrationFunction.h
 * @brief Function class implementing the EM based joint segmentation-registration algorithm.
 *
 * Copyright (c) 2011-2014 University of Pennsylvania. All rights reserved.<br />
 * See http://www.cbica.upenn.edu/sbia/software/license.html or COPYING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#ifndef __itkJointSegmentationRegistrationFunction_h
#define __itkJointSegmentationRegistrationFunction_h

#include <itkPDEDeformableRegistrationFunction.h>
#include <itkPoint.h>
#include <itkCovariantVector.h>
#include <itkInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkCentralDifferenceImageFunction.h>
#include <itkWarpImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageFileWriter.h>
#include <itkCastImageFilter.h>
#include <iostream>
#include <fstream>
#include <itkChangeLabelImageFilter.h>
#include <vector>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>


namespace itk {


/*!
 * \class JointSegmentationRegistrationFunction
 * \brief Computes voxel-wise updates for EM-based joint segmentation and
 *        registration.
 *
 * This class implements methods to compute the voxel-wise updates for the
 * deformation field and the posteriors of various classes for the joint
 * segementation and atlas registration of the images in the corresponding
 * itk::JointSegmentationRegistrationFilter class.
 *
 * \see itk::JointSegmentationRegistrationFilter
 */
template<class TFixedImage, class TMovingImage, class TDeformationField,unsigned int VNumberOfFixedChannels=4, unsigned int VNumberOfMovingChannels=6>
class ITK_EXPORT JointSegmentationRegistrationFunction : public PDEDeformableRegistrationFunction< TFixedImage, TMovingImage, TDeformationField>
{
public:
	// Standard class typedefs
	typedef JointSegmentationRegistrationFunction Self;
	typedef PDEDeformableRegistrationFunction<TFixedImage, TMovingImage, TDeformationField> Superclass;
	typedef SmartPointer<Self> Pointer;
	typedef SmartPointer<const Self> ConstPointer;
	// Method for creation through the object factory
	itkNewMacro(Self);
	// Run-time type information (and related methods)
	itkTypeMacro(JointSegmentationRegistrationFunction, PDEDeformableRegistrationFunction);
	// FixedImage image type
	typedef typename Superclass::FixedImageType       FixedImageType;
	typedef typename Superclass::FixedImagePointer    FixedImagePointer;
	typedef typename FixedImageType::IndexType        IndexType;
	typedef typename FixedImageType::SizeType         SizeType;
	typedef typename FixedImageType::SpacingType      SpacingType;
	typedef typename FixedImageType::DirectionType    DirectionType;
	// MovingImage image type
	typedef typename Superclass::MovingImageType      MovingImageType;
	typedef typename Superclass::MovingImagePointer   MovingImagePointer;
	typedef typename MovingImageType::PixelType       MovingPixelType;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	// WeightImage image type
	typedef typename Superclass::MovingImageType      WeightImageType;
	typedef typename Superclass::MovingImagePointer   WeightImagePointer;
	typedef typename MovingImageType::PixelType       WeightPixelType;
#endif
#ifdef USE_PROB_PRIOR
	// ProbImage image type
	typedef typename Superclass::MovingImageType      ProbImageType;
	typedef typename Superclass::MovingImagePointer   ProbImagePointer;
	typedef typename MovingImageType::PixelType       ProbPixelType;
#endif
	// Deformation field type
#if ITK_VERSION_MAJOR >= 4
	typedef typename Superclass::DisplacementFieldType DeformationFieldType;
	typedef typename Superclass::DisplacementFieldTypePointer DeformationFieldTypePointer;
#else
	typedef typename Superclass::DeformationFieldType DeformationFieldType;
	typedef typename Superclass::DeformationFieldTypePointer DeformationFieldTypePointer;
#endif
	// Inherit some enums from the superclass
	itkStaticConstMacro(ImageDimension, unsigned int,Superclass::ImageDimension);
	// NumberOfFixedChannels
	itkStaticConstMacro(NumberOfFixedChannels, unsigned int, VNumberOfFixedChannels);
	// NumberOfMovingChannels
	itkStaticConstMacro(NumberOfMovingChannels, unsigned int, VNumberOfMovingChannels);
	// Inherit some enums from the superclass
	typedef typename Superclass::PixelType            PixelType;
	typedef typename Superclass::RadiusType           RadiusType;
	typedef typename Superclass::NeighborhoodType     NeighborhoodType;
	typedef typename Superclass::FloatOffsetType      FloatOffsetType;
	typedef typename Superclass::TimeStepType         TimeStepType;
	// Interpolator type
	typedef double                                    CoordRepType;
	typedef InterpolateImageFunction<MovingImageType, CoordRepType> InterpolatorType;
	typedef typename InterpolatorType::Pointer        InterpolatorPointer;
	typedef typename InterpolatorType::PointType      PointType;
	typedef LinearInterpolateImageFunction<MovingImageType, CoordRepType> DefaultInterpolatorType;
	// Warper type
	typedef WarpImageFilter<MovingImageType, MovingImageType, DeformationFieldType> WarperType;
	typedef typename WarperType::Pointer              WarperPointer;
	// Covariant vector type
	typedef CovariantVector<double,itkGetStaticConstMacro(ImageDimension)> CovariantVectorType;
	// Moving image gradient (unwarped) calculator type
	typedef CentralDifferenceImageFunction<MovingImageType,CoordRepType> MovingImageGradientCalculatorType;
	typedef typename MovingImageGradientCalculatorType::Pointer MovingImageGradientCalculatorPointer;
	// Object to keep gradient calculators of moving images
	typedef std::vector<typename MovingImageGradientCalculatorType::Pointer> MovingImageGradientCalculatorVector;
	// Object to keep interpolators needed for image warpers
	typedef std::vector<typename InterpolatorType::Pointer> InterpolatorTypeVector;
	// Object to keep warper objects
	typedef std::vector<typename WarperType::Pointer> WarperTypeVector;
	// Declaring variance type and holder object
	typedef vnl_matrix_fixed<double, NumberOfFixedChannels, NumberOfFixedChannels> VarianceType;
	typedef std::vector<VarianceType> VarianceVectorType;
	// Declaring mean type and holder object
	typedef vnl_vector_fixed<double, NumberOfFixedChannels> MeanType;
	typedef std::vector<MeanType> MeanVectorType;

	// Set/Get methods to access init means vector object
	MeanVectorType* GetInitMeanVector(void) { return &m_InitMeanVector; }
	void SetInitMeanVector(MeanVectorType ptr) { m_InitMeanVector = ptr; }
	// Set/Get methods to access means vector object
	MeanVectorType* GetMeanVector(void) { return &m_MeanVector; }
	void SetMeanVector(MeanVectorType ptr) { m_MeanVector = ptr; }
	// Set/Get methods to access covariance matrix pointers vector object
	VarianceVectorType* GetVarianceVector(void) { return &m_VarianceVector; }
	void SetVarianceVector(VarianceVectorType ptr) { m_VarianceVector = ptr; }

	// Set/Get methods for nth moving image interpolator
	virtual void SetNthImageInterpolator(InterpolatorType* ptr, unsigned int n) { m_MovingImageInterpolatorVector[n-1] = ptr; }
	InterpolatorType* GetNthImageInterpolator(unsigned int n) { return m_MovingImageInterpolatorVector[n-1]; }

	// Set/Get methods for nth moving image warper
	virtual void SetNthImageWarper(WarperType* ptr, unsigned int n) { m_MovingImageWarperVector[n-1] = ptr; }
	WarperType* GetNthImageWarper(unsigned int n) { return m_MovingImageWarperVector[n-1]; }

	// This class uses a constant timestep of 1
	virtual TimeStepType ComputeGlobalTimeStep(void* itkNotUsed(GlobalData)) const { return m_TimeStep; }

	// Return a pointer to a global data structure that is passed to
	// this object from the solver at each calculation
	virtual void *GetGlobalDataPointer() const {
		GlobalDataStruct *global = new GlobalDataStruct();
		global->m_NumberOfPixelsProcessed = 0L;
		return global;
	}

	// Release memory for global data structure
	virtual void ReleaseGlobalDataPointer(void *GlobalData) const;

	// Set the object's state before each iteration
	virtual void InitializeIteration();

	// This method is called by a finite difference solver image filter at
	// each pixel that does not lie on a data set boundary
	virtual PixelType ComputeUpdate(const NeighborhoodType &neighborhood, void *globalData, const FloatOffsetType &offset = FloatOffsetType(0.0));

	// Get the metric value. The metric value is the KL difference
	// in intensity between the fixed image and transforming moving image
	// probability maps
	virtual double GetMetric() const { return m_Metric; }

	// Get the rms change in deformation field
	virtual const double &GetRMSChange() const { return m_RMSChange; }
	
	typedef std::vector<typename FixedImageType::Pointer> FixedImagePointerVectorType;
	typedef std::vector<typename MovingImageType::Pointer> MovingImagePointerVectorType;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	typedef std::vector<typename WeightImageType::Pointer> WeightImagePointerVectorType;
#endif
#ifdef USE_PROB_PRIOR
	typedef std::vector<typename ProbImageType::Pointer> ProbImagePointerVectorType;
#endif
	
	// Set/Get methods to access fixed image vector pointers
	FixedImagePointerVectorType GetFixedImagePointerVector(void) const { return m_FixedImageVector; }
	void SetFixedImagePointerVector(FixedImagePointerVectorType ptr) { m_FixedImageVector = ptr; }
	// Set/Get methods to access moving image vector pointers
	MovingImagePointerVectorType GetMovingImagePointerVector(void) const { return m_MovingImageVector; }
	void SetMovingImagePointerVector(MovingImagePointerVectorType ptr) { m_MovingImageVector = ptr; }
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	// Set/Get methods to access prob image vector pointers
	WeightImagePointerVectorType GetWeightImagePointerVector(void) const { return m_WeightImageVector; }
	void SetWeightImagePointerVector(WeightImagePointerVectorType ptr) { m_WeightImageVector = ptr; }
#endif
#ifdef USE_PROB_PRIOR
	// Set/Get methods to access prob image vector pointers
	ProbImagePointerVectorType GetProbImagePointerVector(void) const { return m_ProbImageVector; }
	void SetProbImagePointerVector(ProbImagePointerVectorType ptr) { m_ProbImageVector = ptr; }
#endif
	
	// Set/Get methods for fixed prior images
	virtual void SetNthFixedImage( typename FixedImageType::Pointer ptr, unsigned int n) { m_FixedImageVector[n-1] = ptr; } 
	const typename FixedImageType::Pointer  GetNthFixedImage(unsigned int n) const { return m_FixedImageVector[n-1]; }
	// Set/Get methods for moving prior images
	virtual void SetNthMovingImage( MovingImageType * ptr, unsigned int n) { m_MovingImageVector[n-1] = ptr; } 
	const MovingImageType* GetNthMovingImage(unsigned int n) const { return m_MovingImageVector[n-1]; }
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	// Set/Get methods for weight images
	virtual void SetNthWeightImage(typename WeightImageType::Pointer ptr, unsigned int n) { m_WeightImageVector.at(n-1) = ptr; } 
	const typename WeightImageType::Pointer GetNthWeightImage(unsigned int n) const { return m_WeightImageVector.at(n-1); }
#endif
#ifdef USE_PROB_PRIOR
	// Set/Get methods for prob images
	virtual void SetNthProbImage(typename ProbImageType::Pointer ptr, unsigned int n) { m_ProbImageVector[n-1] = ptr; } 
	const typename ProbImageType::Pointer GetNthProbImage(unsigned int n) const { return m_ProbImageVector[n-1]; }
#endif
	
	typedef ImageRegionConstIterator<const FixedImageType> FixedImageConstIteratorType;
	typedef ImageRegionConstIterator<const MovingImageType> MovingImageConstIteratorType;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	typedef ImageRegionConstIterator<const WeightImageType> WeightImageConstIteratorType;
#endif
#ifdef USE_PROB_PRIOR
	typedef ImageRegionConstIterator<const ProbImageType> ProbImageConstIteratorType;
#endif

	typedef ImageRegionIterator< FixedImageType> FixedImageIteratorType;
	typedef ImageRegionIterator< MovingImageType> MovingImageIteratorType;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	typedef ImageRegionIterator< WeightImageType> WeightImageIteratorType;
#endif
#ifdef USE_PROB_PRIOR
	typedef ImageRegionIterator< ProbImageType> ProbImageIteratorType;
#endif

#ifndef USE_JSR_WEGHT_IMAGE_PYRAMID
	typedef MovingImageType WeightImageType;

	typedef ImageRegionIterator< WeightImageType> WeightImageIteratorType;

	// Set/Get methods for weight images
	virtual void SetNthWeightImage(WeightImageType * ptr, unsigned int n) { m_WeightImageVector[n-1] = ptr; } 
	const WeightImageType* GetNthWeightImage(unsigned int n) const { return m_WeightImageVector[n-1]; }
	WeightImageType* GetNthWeightImage(unsigned int n) { return m_WeightImageVector[n-1]; }

	typedef std::vector<typename WeightImageType::Pointer> WeightImagePointerVectorType;
#endif
	
	// This function will compute the posteriors
	void ComputeWeightImages(bool bUpdateWeightImages);
	// This function will log the input parameters and cost at the end
	void LogVariables();

	int GetNumberOfElapsedIterations(void) { return m_iteration_number; }
	
	void SetNumberOfElapsedIterations( unsigned int n) { m_iteration_number = n; }
	unsigned int GetNumberOfElapsedIterations(void) const { return m_iteration_number; }
	
	int GetNumberOfMaximumIterations(void) { return m_max_iteration_number; }
	
	void SetNumberOfMaximumIterations(unsigned int n) { m_max_iteration_number = n; }
	unsigned int GetNumberOfMaximumIterations(void) const { return m_max_iteration_number; }
	
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
	BOOL ComputeMSMap(std::vector<FixedImageConstIteratorType>& f_it_vect, BVolume& label_map, FVolume* ms_map, int* r_labels, int r_num, int* c_labels, int c_num, int samp = 1);
#endif
	void UpdateMeansAndVariances(void);
	BOOL SaveMeansAndVariances(const char* means_file, const char* variances_file, MeanVectorType* pMeanVector, VarianceVectorType* pVarianceVector);
	
	// Set/Get functions for the Marquardt constant
	double GetSigma2() { return m_Sigma2; }
	void SetSigma2(double s) { m_Sigma2 = s; }
	// Used if a change in Sigma2 is needed
	double GetDeltaSigma2() { return m_dSigma2; }
	void SetDeltaSigma2(double s) { m_dSigma2 = s; }
#ifdef USE_PROB_PRIOR
	double GetProbWeight() { return m_ProbWeight; }
	void SetProbWeight(double w) { m_ProbWeight = w; }
#endif

	bool m_bUpdateMeansAndVariances;
#if defined(USE_MEAN_SHIFT_UPDATE) || defined(USE_MEAN_SHIFT_UPDATE_ONCE)
	bool m_bMeanShiftUpdate;
	//
	bool mean_shift_update;
	bool ms_tumor;
	bool ms_edema;
#endif
	
protected:
	// vector object holding init mean vectors
	MeanVectorType m_InitMeanVector;
	// vector object holding mean vectors
	MeanVectorType m_MeanVector;
	// vector object holding covariance matrices
	VarianceVectorType m_VarianceVector;
	// Marquardt coefficient
	double m_Sigma2;
	// Depending on the number of iterations this keeps the change rate in m_Sigma2
	double m_dSigma2;
#ifdef USE_PROB_PRIOR
	double m_ProbWeight;
#endif

	// The fixed mask image
	//FixedImagePointer m_FixedMaskImage;

	FixedImagePointerVectorType   m_FixedImageVector;
	MovingImagePointerVectorType  m_MovingImageVector;
	WeightImagePointerVectorType  m_WeightImageVector;
#ifdef USE_PROB_PRIOR
	ProbImagePointerVectorType    m_ProbImageVector;
#endif

	JointSegmentationRegistrationFunction();
	~JointSegmentationRegistrationFunction() {}
	void PrintSelf(std::ostream& os, Indent indent) const;

	// FixedImage image neighborhood iterator type
	typedef ConstNeighborhoodIterator<FixedImageType> FixedImageNeighborhoodIteratorType;

	// A global data type for this class of equation. Used to store iterators for the fixed image
	struct GlobalDataStruct {
		double          m_SumOfEMLogs;
		unsigned long   m_NumberOfPixelsProcessed;
		double          m_SumOfSquaredChange;
		//double          m_SumOfMassTransport;
	};

private:
	int                             m_iteration_number;
	int                             m_max_iteration_number;
	unsigned int                    m_NumberOfFixedChannels;
	unsigned int                    m_NumberOfMovingChannels;
	JointSegmentationRegistrationFunction(const Self&); //purposely not implemented
	void operator=(const Self&); //purposely not implemented

	// Cache fixed image information
	PointType                       m_FixedImageOrigin;
	SpacingType                     m_FixedImageSpacing;
	DirectionType                   m_FixedImageDirection;
	double                          m_Normalizer;

	// Cache fixed mask image information
	PointType                       m_FixedMaskImageOrigin;
	SpacingType                     m_FixedMaskImageSpacing;
	DirectionType                   m_FixedMaskImageDirection;
	
	// Function to interpolate the moving image
	InterpolatorPointer             m_MovingImageInterpolator;
	// Filter to warp moving image for fast gradient computation
	WarperPointer                   m_MovingImageWarper;
	// vector object for gradient calculators in moving images
	MovingImageGradientCalculatorVector m_MovingImageGradientCalculatorVector;
	// vector object for moving image interpolators
	InterpolatorTypeVector          m_MovingImageInterpolatorVector;
	// vector object for moving image warping
	WarperTypeVector                m_MovingImageWarperVector;
	
	// The global timestep
	TimeStepType                    m_TimeStep;

	// The metric value is the mean square difference in intensity between
	// the fixed image and transforming moving image computed over the
	// the overlapping region between the two images
	mutable double                  m_Metric;
	mutable double                  m_RMSChange;
	mutable unsigned long           m_NumberOfPixelsProcessed;
	mutable double                  m_SumOfEMLogs;
	mutable double                  m_SumOfSquaredChange;

	// Mutex lock to protect modification to metric
	mutable SimpleFastMutexLock     m_MetricCalculationLock;
};


} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkJointSegmentationRegistrationFunction.txx"
#endif

#endif
