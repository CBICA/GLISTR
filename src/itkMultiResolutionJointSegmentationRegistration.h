/**
 * @file  itkMultiResolutionJointSegmentationRegistration.h
 * @brief Function class implementing multi resolution joint segmentation-registration algorithm.
 *
 * Copyright (c) 2011-2014 University of Pennsylvania. All rights reserved.<br />
 * See http://www.cbica.upenn.edu/sbia/software/license.html or COPYING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#ifndef __itkMultiResolutionJointSegmentationRegistration_h
#define __itkMultiResolutionJointSegmentationRegistration_h

#include <vector>

#include <itkImage.h>
#include <itkImageToImageFilter.h>
#include <itkPDEDeformableRegistrationFilter.h>
#include <itkVectorResampleImageFilter.h>
#include <itkMultiResolutionPyramidImageFilter.h>

#include "itkJointSegmentationRegistrationFilter.h"


namespace itk {


/*!
 * \class MultiResolutionJointSegmentationRegistration
 * \brief Implements EM-based joint segmentation and registration using
 *        multi-resolution pyramids.
 *
 * This class implements multi-resolution joint segmentation and registration
 * algorithm by making multi-resolution pyramids of input fixed and moving
 * images. It embodies a member to itkJointSegmentationRegistrationFilter to
 * perform joint segmentation and atlas registration at each level. The computed
 * registration field and estimated mean and variances are used to initialize
 * the next higher level of resolutions.
 */
template <class TFixedImage, class TMovingImage, class TDeformationField, class  TRealType = float, unsigned int VNumberOfFixedChannels=4, unsigned int VNumberOfMovingChannels=6>
class ITK_EXPORT MultiResolutionJointSegmentationRegistration : public ImageToImageFilter <TDeformationField, TDeformationField>
{
public:
	/** Standard class typedefs */
	typedef MultiResolutionJointSegmentationRegistration Self;
	typedef ImageToImageFilter<TDeformationField, TDeformationField> Superclass;
	typedef SmartPointer<Self> Pointer;
	typedef SmartPointer<const Self> ConstPointer;

	/** Method for creation through the object factory. */
	itkNewMacro(Self);

	/** Run-time type information (and related methods). */
	itkTypeMacro(MultiResolutionJointSegmentationRegistration, ImageToImageFilter);

	/** Fixed image type. */
	typedef TFixedImage FixedImageType;
	typedef typename FixedImageType::Pointer FixedImagePointer;
	typedef typename FixedImageType::ConstPointer FixedImageConstPointer;

	/** Moving image type. */
	typedef TMovingImage MovingImageType;
	typedef typename MovingImageType::Pointer MovingImagePointer;
	typedef typename MovingImageType::ConstPointer MovingImageConstPointer;

#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	/** Weight image type. */
	typedef TMovingImage WeightImageType;
	typedef typename WeightImageType::Pointer WeightImagePointer;
	typedef typename WeightImageType::ConstPointer WeightImageConstPointer;
#endif

#ifdef USE_PROB_PRIOR
	/** Prob image type. */
	typedef TMovingImage ProbImageType;
	typedef typename ProbImageType::Pointer ProbImagePointer;
	typedef typename ProbImageType::ConstPointer ProbImageConstPointer;
#endif

	/** Deformation field image type. */
	typedef TDeformationField DeformationFieldType;
	typedef typename DeformationFieldType::Pointer DeformationFieldPointer;

	/** ImageDimension. */
	itkStaticConstMacro(ImageDimension, unsigned int, FixedImageType::ImageDimension);
	
	/** NumberOfFixedChannels. **/
	itkStaticConstMacro(NumberOfFixedChannels, unsigned int, VNumberOfFixedChannels);
	/** NumberOfMovingChannels. **/
	itkStaticConstMacro(NumberOfMovingChannels, unsigned int, VNumberOfMovingChannels);
	
	/** Internal float image type. */
	typedef Image<TRealType,itkGetStaticConstMacro(ImageDimension)> FloatImageType;

	/** The internal segmentation-registration type. */
	typedef JointSegmentationRegistrationFilter< FloatImageType, FloatImageType, DeformationFieldType,VNumberOfFixedChannels, VNumberOfMovingChannels> SegmentationRegistrationType;

	typedef typename SegmentationRegistrationType::Pointer SegmentationRegistrationPointer;

	/** The default registration type. */
	typedef JointSegmentationRegistrationFilter< FloatImageType, FloatImageType, DeformationFieldType, VNumberOfFixedChannels, VNumberOfMovingChannels> DefaultSegmentationRegistrationType;

	typedef typename DefaultSegmentationRegistrationType::Pointer DefaultSegmentationRegistrationPointerType;

	/** The fixed multi-resolution image pyramid type. */
	typedef MultiResolutionPyramidImageFilter< FixedImageType, FloatImageType > FixedImagePyramidType;
	typedef typename FixedImagePyramidType::Pointer FixedImagePyramidPointer;
	/** The moving multi-resolution image pyramid type. */
	typedef MultiResolutionPyramidImageFilter< MovingImageType, FloatImageType > MovingImagePyramidType;
	typedef typename MovingImagePyramidType::Pointer MovingImagePyramidPointer;
	//  typedef typename MovingImagePyramidType::Pointer MovingMaskImagePyramidPointer;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	typedef MultiResolutionPyramidImageFilter< WeightImageType, FloatImageType > WeightImagePyramidType;
	typedef typename WeightImagePyramidType::Pointer WeightImagePyramidPointer;
#endif
#ifdef USE_PROB_PRIOR
	typedef MultiResolutionPyramidImageFilter< ProbImageType, FloatImageType > ProbImagePyramidType;
	typedef typename ProbImagePyramidType::Pointer ProbImagePyramidPointer;
#endif

	/** The deformation field expander type. */
	typedef VectorResampleImageFilter< DeformationFieldType, DeformationFieldType > FieldExpanderType;
	typedef typename FieldExpanderType::Pointer FieldExpanderPointer;
	
	/** Set the fixed image. */
	virtual void SetNthFixedImage(const FixedImageType * ptr, unsigned int n);
	/** Set the fixed mask image. */
	// virtual void SetFixedMaskImage( const FixedImageType * ptr );
	/** Get the fixed image. */
	const FixedImageType * GetNthFixedImage(unsigned int n) const;
	/** Get the fixed image. */
	//const FixedImageType * GetFixedMaskImage(void) const;
	/** Set the moving image. */
	virtual void SetNthMovingImage(const MovingImageType * ptr, unsigned int n);
	/** Get the moving image. */
	const MovingImageType * GetNthMovingImage(unsigned int n) const;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	/** Set the prob image. */
	virtual void SetNthWeightImage(const WeightImageType * ptr, unsigned int n);
	/** Get the moving image. */
	const WeightImageType * GetNthWeightImage(unsigned int n) const;
#endif
#ifdef USE_PROB_PRIOR
	/** Set the prob image. */
	virtual void SetNthProbImage(const ProbImageType * ptr, unsigned int n);
	/** Get the moving image. */
	const ProbImageType * GetNthProbImage(unsigned int n) const;
#endif

	/** Set initial deformation field to be used as is (no smoothing, no
	 *  subsampling at the coarsest level of the pyramid. */
	virtual void SetInitialDeformationField(DeformationFieldType * ptr) { this->m_InitialDeformationField=ptr; }

	/** Set initial deformation field. No assumption is made on the
	 *  input. It will therefore be smoothed and resampled to match the
	 *  images characteristics at the coarsest level of the pyramid. */
	virtual void SetArbitraryInitialDeformationField( DeformationFieldType * ptr ) { this->SetInput(ptr); }

	/** Get output deformation field. */
	const DeformationFieldType * GetDeformationField(void) { return this->GetOutput(); }

	/** Get the number of valid inputs.  For
	 * MultiResolutionPDEDeformableRegistration, this checks whether the
	 * fixed and moving images have been set. While
	 * MultiResolutionPDEDeformableRegistration can take a third input
	 * as an initial deformation field, this input is not a required input.
	 */
	virtual std::vector<SmartPointer<DataObject> >::size_type GetNumberOfValidRequiredInputs() const;

	typedef std::vector< typename FixedImagePyramidType::Pointer> FixedImagePyramidVectorType;
	typedef std::vector< typename MovingImagePyramidType::Pointer> MovingImagePyramidVectorType;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	typedef std::vector< typename WeightImagePyramidType::Pointer> WeightImagePyramidVectorType;
#endif
#ifdef USE_PROB_PRIOR
	typedef std::vector< typename ProbImagePyramidType::Pointer> ProbImagePyramidVectorType;
#endif
	
	/** sets Nth fixed image pyramid **/
	virtual void SetNthFixedImagePyramid(FixedImagePyramidType * ptr, unsigned int n) { m_FixedImagePyramidVector.assign(n-1,ptr); }
	FixedImagePyramidType * GetNthFixedImagePyramid(unsigned int n) { return m_FixedImagePyramidVector[n-1]; }
	/** sets Nth moving image pyramid **/
	virtual void SetNthMovingImagePyramid(MovingImagePyramidType * ptr, unsigned int n) { m_MovingImagePyramidVector.assign(n-1, ptr); }
	MovingImagePyramidType * GetNthMovingImagePyramid(unsigned int n) { return m_MovingImagePyramidVector[n-1]; }
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	/** sets Nth weight image pyramid **/
	virtual void SetNthWeightImagePyramid(WeightImagePyramidType * ptr, unsigned int n) { m_WeightImagePyramidVector.assign(n-1, ptr); }
	WeightImagePyramidType * GetNthWeightImagePyramid(unsigned int n) { return m_WeightImagePyramidVector[n-1]; }
#endif
#ifdef USE_PROB_PRIOR
	/** sets Nth prob image pyramid **/
	virtual void SetNthProbImagePyramid(ProbImagePyramidType * ptr, unsigned int n) { m_ProbImagePyramidVector.assign(n-1, ptr); }
	ProbImagePyramidType * GetNthProbImagePyramid(unsigned int n) { return m_ProbImagePyramidVector[n-1]; }
#endif
	
	/** Set the internal registrator. */
	itkSetObjectMacro(SegmentationRegistrationFilter, SegmentationRegistrationType);
	/** Get the internal registrator. */
	itkGetObjectMacro(SegmentationRegistrationFilter, SegmentationRegistrationType);
	
	/** Set number of multi-resolution levels. */
	virtual void SetNumberOfLevels(unsigned int num);
	/** Get number of multi-resolution levels. */
	itkGetConstReferenceMacro(NumberOfLevels, unsigned int);
	/** Get the current resolution level being processed. */
	itkGetConstReferenceMacro(CurrentLevel, unsigned int);
	/** Set number of iterations per multi-resolution levels. */
	itkSetVectorMacro(NumberOfIterations, unsigned int, m_NumberOfLevels);

	/** Set the moving image pyramid. */
	itkSetObjectMacro(FieldExpander, FieldExpanderType);
	/** Get the moving image pyramid. */
	itkGetObjectMacro(FieldExpander, FieldExpanderType);

	/** Get number of iterations per multi-resolution levels. */
	virtual const unsigned int * GetNumberOfIterations() const { return &(m_NumberOfIterations[0]); }

	/** Stop the registration after the current iteration. */
	virtual void StopRegistration();
	
	/**Sets the number of fixed channels**/
	void SetNumberOfFixedChannels(unsigned int n) { m_NumberOfFixedChannels = n; }
	/**Sets the number of moving channels**/
	void SetNumberOfMovingChannels(unsigned int n) { m_NumberOfMovingChannels = n; }
	/**Returns the number of fixed channels if requested**/
	virtual unsigned int GetNumberOfFixedChannels() const { return m_NumberOfFixedChannels; }
	/**Returns the number of moving channels if requested**/
	virtual unsigned int GetNumberOfMovingChannels() const { return m_NumberOfMovingChannels; }

	/** returns the metric value **/
	double GetMetric() { return GetSegmentationRegistrationFilter()->GetMetric(); }

protected:
	MultiResolutionJointSegmentationRegistration();
	~MultiResolutionJointSegmentationRegistration() {}
	void PrintSelf(std::ostream& os, Indent indent) const;

	/** Generate output data by performing the registration
	* at each resolution level. */
	virtual void GenerateData();

	/** The current implementation of this class does not support
	* streaming. As such it requires the largest possible region
	* for the moving, fixed and input deformation field. */
	virtual void GenerateInputRequestedRegion();

	/** By default, the output deformation field has the same
	* spacing, origin and LargestPossibleRegion as the input/initial
	* deformation field.
	*
	* If the initial deformation field is not set, the output
	* information is copied from the fixed image. */
	virtual void GenerateOutputInformation();

	/** The current implementation of this class does not supprot
	* streaming. As such it produces the output for the largest
	* possible region. */
	virtual void EnlargeOutputRequestedRegion(DataObject *ptr);

	/** This method returns true to indicate that the registration should
	* terminate at the current resolution level. */
	virtual bool Halt();

private:
	MultiResolutionJointSegmentationRegistration(const Self&); //purposely not implemented
	void operator=(const Self&); //purposely not implemented

	SegmentationRegistrationPointer m_SegmentationRegistrationFilter;
	
	FixedImagePyramidVectorType  m_FixedImagePyramidVector;
	MovingImagePyramidVectorType m_MovingImagePyramidVector;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	WeightImagePyramidVectorType m_WeightImagePyramidVector;
#endif
#ifdef USE_PROB_PRIOR
	ProbImagePyramidVectorType   m_ProbImagePyramidVector;
#endif
	
	FieldExpanderPointer       m_FieldExpander;
	DeformationFieldPointer    m_InitialDeformationField;

	unsigned int               m_NumberOfLevels;
	unsigned int               m_CurrentLevel;
	unsigned int               m_NumberOfFixedChannels;
	unsigned int               m_NumberOfMovingChannels;
	std::vector<unsigned int>  m_NumberOfIterations;

	/** Flag to indicate user stop registration request. */
	bool                       m_StopRegistrationFlag;
};


} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiResolutionJointSegmentationRegistration.txx"
#endif

#endif
