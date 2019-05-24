/**
 * @file  itkJointSegmentationRegistrationFilter.h
 * @brief Filter class implementing the EM based joint segmentation-registration algorithm.
 *
 * Copyright (c) 2011-2014 University of Pennsylvania. All rights reserved.<br />
 * See http://www.cbica.upenn.edu/sbia/software/license.html or COYPING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#ifndef __itkJointSegmentationRegistrationFilter_h
#define __itkJointSegmentationRegistrationFilter_h

#include <itkPDEDeformableRegistrationFilter.h>
#include "itkJointSegmentationRegistrationFunction.h"

#if ITK_VERSION_MAJOR >= 4
#include "itkMultiplyImageFilter.h"
#include "itkVectorMagnitudeImageFilter.h"
#include "itkExponentialDisplacementFieldImageFilter.h"
#else
#include "itkMultiplyByConstantImageFilter.h"
#include "itkGradientToMagnitudeImageFilter.h"
#include "itkExponentialDeformationFieldImageFilter.h"
#endif
#include "itkMinimumMaximumImageCalculator.h"
#include <itkWarpVectorImageFilter.h>
#include <itkVectorLinearInterpolateNearestNeighborExtrapolateImageFunction.h>
#include <itkAddImageFilter.h>


namespace itk {

/*!
 * \class JointSegmentationRegistrationFilter
 * \brief Implements EM-based joint segmentation and registration.
 *
 * This class implements joint segmentation and registration for a given set
 * of reference images and atlas probability maps. It embodies a member of
 * itk::JointSegmentationRegistrationFunction class to compute the deformation
 * field. It smoothes and applies the computed deformation field it to warp
 * the set of moving images.
 *
 * \see itk::JointSegmentationRegistrationFunction
 */
template<class TFixedImage, class TMovingImage, class TDeformationField, unsigned int VNumberOfFixedChannels=4, unsigned int VNumberOfMovingChannels=6>
class ITK_EXPORT JointSegmentationRegistrationFilter : public PDEDeformableRegistrationFilter< TFixedImage, TMovingImage, TDeformationField>
{
public:
	/** Standard class typedefs. */
	typedef JointSegmentationRegistrationFilter Self;
	typedef PDEDeformableRegistrationFilter< TFixedImage, TMovingImage,TDeformationField > Superclass;
	typedef SmartPointer<Self> Pointer;
	typedef SmartPointer<const Self> ConstPointer;

	/** Method for creation through the object factory. */
	itkNewMacro(Self);

	/** Run-time type information (and related methods). */
	itkTypeMacro(JointSegmentationRegistrationFilter, PDEDeformableRegistrationFilter);

	/** NumberOfFixedChannels. **/
	itkStaticConstMacro(NumberOfFixedChannels, unsigned int, VNumberOfFixedChannels);
	/** NumberOfMovingChannels. **/
	itkStaticConstMacro(NumberOfMovingChannels, unsigned int, VNumberOfMovingChannels);

	/** FixedImage image type. */
	typedef typename Superclass::FixedImageType           FixedImageType;
	typedef typename Superclass::FixedImagePointer        FixedImagePointer;
	/** MovingImage image type. */
	typedef typename Superclass::MovingImageType          MovingImageType;
	typedef typename Superclass::MovingImagePointer       MovingImagePointer;
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	/** WeightImage image type. */
	typedef typename Superclass::MovingImageType          WeightImageType;
	typedef typename Superclass::MovingImagePointer       WeightImagePointer;
#endif
#ifdef USE_PROB_PRIOR
	/** ProbImage image type. */
	typedef typename Superclass::MovingImageType          ProbImageType;
	typedef typename Superclass::MovingImagePointer       ProbImagePointer;
#endif
	/** Deformation field type. */
#if ITK_VERSION_MAJOR >= 4
	typedef typename Superclass::DisplacementFieldType    DeformationFieldType;
	typedef typename Superclass::DisplacementFieldPointer DeformationFieldPointer;
#else
	typedef typename Superclass::DeformationFieldType     DeformationFieldType;
	typedef typename Superclass::DeformationFieldPointer  DeformationFieldPointer;
#endif

	/** FiniteDifferenceFunction type. */
	typedef typename Superclass::FiniteDifferenceFunctionType FiniteDifferenceFunctionType;

	/** Take timestep type from the FiniteDifferenceFunction. */
	typedef typename 
		FiniteDifferenceFunctionType::TimeStepType        TimeStepType;

	/** RegistrationFunction type. */
	typedef JointSegmentationRegistrationFunction< FixedImageType, MovingImageType, DeformationFieldType,VNumberOfFixedChannels,VNumberOfMovingChannels> JointSegmentationRegistrationFunctionType;

	/** Get the metric value.*/
	virtual double GetMetric() const;
	virtual const double &GetRMSChange() const;

	/** Use a first-order approximation of the exponential.
	 *  This amounts to using an update rule of the type
	 *  s <- s o (Id + u) instead of s <- s o exp(u) */
	itkSetMacro(UseFirstOrderExp, bool);
	itkGetMacro(UseFirstOrderExp, bool);
	itkBooleanMacro(UseFirstOrderExp);

	/** Function to return the number of valid required inputs **/
	virtual std::vector<SmartPointer<DataObject> >::size_type GetNumberOfValidRequiredInputs() const;

	/** Set/Get nth fixed image.  */
	virtual void SetNthFixedImage(const FixedImageType * ptr, unsigned int n);
	virtual const FixedImageType * GetNthFixedImage(unsigned int n) const;
	virtual FixedImageType * GetNthFixedImage(unsigned int n);
	/** Set/Get nth moving image.  */
	virtual void SetNthMovingImage(const MovingImageType * ptr, unsigned int n);
	virtual const MovingImageType * GetNthMovingImage(unsigned int n) const;
	virtual MovingImageType * GetNthMovingImage(unsigned int n);
	/** method to access the posterior maps made in function **/
#ifdef USE_JSR_WEGHT_IMAGE_PYRAMID
	virtual void SetNthWeightImage(const WeightImageType * ptr, unsigned int n);
	virtual const WeightImageType * GetNthWeightImage(unsigned int n) const;
	virtual WeightImageType * GetNthWeightImage(unsigned int n);
#else
	virtual MovingImageType * GetNthWeightImage(unsigned int n);
#endif
#ifdef USE_PROB_PRIOR
	/** Set/Get nth prob image.  */
	virtual void SetNthProbImage(const ProbImageType * ptr, unsigned int n);
	virtual const ProbImageType * GetNthProbImage(unsigned int n) const;
	virtual ProbImageType * GetNthProbImage(unsigned int n);
#endif

	virtual void SetUpdateMeansAndVariances(bool bUpdateMeansAndVariances);
	virtual bool GetUpdateMeansAndVariances();

protected:
	JointSegmentationRegistrationFilter();
	~JointSegmentationRegistrationFilter() {}
	void PrintSelf(std::ostream& os, Indent indent) const;

	/** Initialize the state of filter and equation before each iteration. */
	virtual void InitializeIteration();

	/** This method allocates storage in m_UpdateBuffer.  It is called from
	 * FiniteDifferenceFilter::GenerateData(). */
	virtual void AllocateUpdateBuffer();

	/** Apply update. */
#if ITK_VERSION_MAJOR >= 4
	virtual void ApplyUpdate(const TimeStepType& dt);
#else
	virtual void ApplyUpdate(TimeStepType dt);
#endif

private:
	JointSegmentationRegistrationFilter(const Self&); //purposely not implemented
	void operator=(const Self&); //purposely not implemented

	/** Downcast the DifferenceFunction using a dynamic_cast to ensure that it is of the correct type.
	* this method will throw an exception if the function is not of the expected type. */
	JointSegmentationRegistrationFunctionType *  DownCastDifferenceFunctionType();
	const JointSegmentationRegistrationFunctionType *  DownCastDifferenceFunctionType() const;
	
	/** Exp and composition typedefs */
#if ITK_VERSION_MAJOR >= 4
	typedef MultiplyImageFilter< DeformationFieldType, Image<TimeStepType, DeformationFieldType::ImageDimension>, DeformationFieldType > MultiplyByConstantType;
	typedef VectorMagnitudeImageFilter< DeformationFieldType, MovingImageType > GradientToMagnitudeType;
	typedef ExponentialDisplacementFieldImageFilter< DeformationFieldType, DeformationFieldType > FieldExponentiatorType;
#else
	typedef MultiplyByConstantImageFilter< DeformationFieldType, TimeStepType, DeformationFieldType > MultiplyByConstantType;
	typedef GradientToMagnitudeImageFilter< DeformationFieldType, MovingImageType > GradientToMagnitudeType;
	typedef ExponentialDeformationFieldImageFilter< DeformationFieldType, DeformationFieldType > FieldExponentiatorType;
#endif	
	typedef MinimumMaximumImageCalculator< MovingImageType > MinimumMaximumFieldType;
	typedef WarpVectorImageFilter< DeformationFieldType, DeformationFieldType, DeformationFieldType > VectorWarperType;
	typedef VectorLinearInterpolateNearestNeighborExtrapolateImageFunction< DeformationFieldType, double > FieldInterpolatorType;
	typedef AddImageFilter< DeformationFieldType, DeformationFieldType, DeformationFieldType> AdderType;
	
	typedef typename MultiplyByConstantType::Pointer      MultiplyByConstantPointer;
	typedef typename GradientToMagnitudeType::Pointer	  GradientToMagnitudePointer;
	typedef typename MinimumMaximumFieldType::Pointer	  MinimumMaximumFieldPointer;
	typedef typename FieldExponentiatorType::Pointer      FieldExponentiatorPointer;
	typedef typename VectorWarperType::Pointer            VectorWarperPointer; 
	typedef typename FieldInterpolatorType::Pointer       FieldInterpolatorPointer;
	typedef typename FieldInterpolatorType::OutputType    FieldInterpolatorOutputType;
	typedef typename AdderType::Pointer                   AdderPointer;
	
	MultiplyByConstantPointer m_Multiplier;
	GradientToMagnitudePointer m_MagnitudeCalculator;
	MinimumMaximumFieldPointer m_MinMaxCalculator;
	FieldExponentiatorPointer m_Exponentiator;
	VectorWarperPointer       m_Warper;
	AdderPointer              m_Adder;
	bool                      m_UseFirstOrderExp;
	unsigned int              m_NumberOfFixedChannels;
	unsigned int              m_NumberOfMovingChannels;
};


} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkJointSegmentationRegistrationFilter.txx"
#endif

#endif
