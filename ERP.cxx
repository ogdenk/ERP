//  ERP.cxx
//
//  Processing of epilepsy patient brain MRI scans to extract radiomics features for machine learning.
//
//
//
//   Copyright (c) 2016, Image Processing Lab, SUNY Upstate Medical University, Syracuse NY
//   All rights reserved.

#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkBinaryMedianImageFilter.h"
#include "itkScalarImageToTextureFeaturesFilter.h"
#include "itkBinaryImageToShapeLabelMapFilter.h"
#include "itkBinaryImageToStatisticsLabelMapFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkScalarImageToRunLengthFeaturesFilter.h"
#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkShiftScaleImageFilter.h"
#include "itkCastImageFilter.h"
#include "DeepCopy.h"
#include <stdio.h>
#include <vector>

// These are the Freesurfer label values for the structures of interest
#define LeftHippocampus 17
#define RightHippocampus 53
#define LeftThalamus 10
#define RightThalamus 49

int main(int argc, char * argv[])
{
  unsigned char labelValue; // holds the current label value during the loop
  std::string labelName;		 // holds the current label name during the loop
  std::string outputstring;		 // used for writing results to csv files
  const unsigned int Dimension = 3;
  int i, j, k; // used for general purpose indexing
  int radius=1;  // size of neighborhood for GLCM determination
  float sum=0, mean=0, factor=1;  // used for structure mean signal calculation for scaling
  bool found; // used for neighborhood generation
  bool writefiles = false;

  // image data and label types
  typedef  unsigned short  int						PixelType;
  typedef  unsigned char							LabelPixelType;
  typedef itk::Image< PixelType, Dimension >        ImageType;
  typedef itk::Image< LabelPixelType, Dimension >	LabelImageType;

  //  create some containers to hold the various versions of processed images
  //  don't need one for all eight versions as some of those will persist as filter output images
  ImageType::Pointer origGF = ImageType::New();
  ImageType::Pointer origRescaledGF = ImageType::New();
  ImageType::Pointer resampled = ImageType::New();
  ImageType::Pointer resampledRescaled = ImageType::New();
  ImageType::Pointer resampledGF = ImageType::New();
   
  std::vector<ImageType::Pointer> imagePointers;  // This will hold the pointers to the images to be processed
  imagePointers.reserve(8);

  std::vector<ImageType::Pointer>::iterator imageIterator;  // Iterator for the outside loop

  std::vector<LabelImageType::Pointer> labelImagePointers;  // This will hold the pointers to the label maps
  labelImagePointers.reserve(8);

  LabelImageType::SizeType medianRadius; // used for binary median image filter
  medianRadius[0] = 1;
  medianRadius[1] = 1;
  medianRadius[2] = 1;

  // Get the input directory specified on the command line
  std::string inputFile;
  std::string inputLabels;
  std::string inputDirectory;
  if (argc > 1)
  {
	 	inputDirectory = argv[1];
  }
  else
  {
  	std::cout << "Input a directory to process" << std::endl;
	std::cout << "The image and label files must be in the directory and have the names:" << std::endl;
	std::cout << "rawavg.nrrd and aseg.nrrd" << std::endl;
	exit(1);
  }

  if (argc > 2) // write the processed image files to disk
	  writefiles = true;

  inputFile = inputDirectory + "rawavg.nrrd";
  inputLabels = inputDirectory + "aseg.nrrd";
  	
  // Create vectors of label names and label values
  std::vector<std::string> labelNames;
  labelNames.reserve(4); //storage for four labels
  labelNames.push_back("LHip");
  labelNames.push_back("RHip");
  labelNames.push_back("LThal");
  labelNames.push_back("RThal");

  std::vector<unsigned char> labelValues;
  labelValues.reserve(4);  //storage for four values
  labelValues.push_back(LeftHippocampus);
  labelValues.push_back(RightHippocampus);
  labelValues.push_back(LeftThalamus);
  labelValues.push_back(RightThalamus);

  // Create iterators for the vectors.  We will iterate over the structures when doing the feature extraction
  std::vector<std::string>::iterator labelNamesIterator;
  std::vector<unsigned char>::iterator labelValuesIterator;

  // open the image data
  typedef itk::ImageFileReader< ImageType >  ReaderType;
  ReaderType::Pointer imageReader = ReaderType::New();
  imageReader->SetFileName(inputFile );
  imageReader->Update();
  imagePointers.push_back(imageReader->GetOutput());  // pointer to the first image to process

  // open the labels
  typedef itk::ImageFileReader< LabelImageType >  LabelReaderType;
  LabelReaderType::Pointer labelReader = LabelReaderType::New();
  labelReader->SetFileName(inputLabels);
  labelReader->Update();

  //  Create file writers for the processed image and label files.
  typedef itk::ImageFileWriter<ImageType> ImageWriterType;
  ImageWriterType::Pointer imageWriter = ImageWriterType::New();
  imageWriter->SetUseCompression(true);

  typedef itk::ImageFileWriter<LabelImageType> LabelWriterType;
  LabelWriterType::Pointer labelWriter = LabelWriterType::New();
  labelWriter->SetUseCompression(true);

  //  Get the input image and label geometries to use in the resamplers
  //  Input image is 'high resolution' (original scan resolution)
  //  Input label map is 'low resolution' (1 mm^3)
  ImageType::RegionType region = imageReader->GetOutput()->GetLargestPossibleRegion();
  ImageType::SizeType size = region.GetSize();
  ImageType::RegionType labelregion = labelReader->GetOutput()->GetLargestPossibleRegion();
  ImageType::SizeType labelsize = labelregion.GetSize();
  ImageType::SpacingType spacing = imageReader->GetOutput()->GetSpacing();
  ImageType::SpacingType labelspacing = labelReader->GetOutput()->GetSpacing();
  ImageType::PointType origin = imageReader->GetOutput()->GetOrigin();
  ImageType::PointType labelorigin = labelReader->GetOutput()->GetOrigin();
  ImageType::DirectionType direction = imageReader->GetOutput()->GetDirection();
  ImageType::DirectionType labeldirection = labelReader->GetOutput()->GetDirection();

  // Set up the pipeline components.
  /////////////////////////////////////////////////////////////////////////////////////////

  // Need a resampler to downsample the input image data to be the same as the label image
  // at 1 x 1 x 1 mm.  We will use the label geometry to do this
  // Use a bspline interpolator to resample the image
  typedef itk::BSplineInterpolateImageFunction<ImageType, double >  InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetSplineOrder(2);

  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetInterpolator(interpolator);
  resampler->SetSize(labelsize);
  resampler->SetOutputSpacing(labelspacing);
  resampler->SetOutputOrigin(labelorigin);
  resampler->SetOutputDirection(labeldirection);
    
  // Need a resampler to upsample the label map to use with the input image
  // Need an interpolator, should use nearest neighbor for label maps
  typedef itk::NearestNeighborInterpolateImageFunction<LabelImageType, double >  NNInterpolatorType;
  NNInterpolatorType::Pointer nninterpolator = NNInterpolatorType::New();

  typedef itk::ResampleImageFilter<LabelImageType, LabelImageType> LabelResamplerType;
  LabelResamplerType::Pointer labelResampler = LabelResamplerType::New();
  labelResampler->SetInput(labelReader->GetOutput());
  labelResampler->SetInterpolator(nninterpolator);
  labelResampler->SetSize(size);
  labelResampler->SetOutputSpacing(spacing);
  labelResampler->SetOutputOrigin(origin);
  labelResampler->SetOutputDirection(direction);
  labelResampler->Update();  // only needs to be done once

  //  Write the resampled label map to a file for QC purposes
  if (writefiles)
  {
	  labelWriter->SetInput(labelResampler->GetOutput());
	  labelWriter->SetFileName(inputDirectory + "aseg_resampled.nrrd");
	  labelWriter->Update();
  }

  // First four (non-resampled) images will use the high-res labels
  for (i = 1; i <= 4; i++)
	  labelImagePointers.push_back(labelResampler->GetOutput());
  
  //  Last four images will use the original labels
  for (i = 1; i <= 4; i++)
	  labelImagePointers.push_back(labelReader->GetOutput());

  //  Median filter to smooth the edges of the labels
  typedef itk::BinaryMedianImageFilter<LabelImageType, LabelImageType> MedianFilterType;
  MedianFilterType::Pointer medianFilter = MedianFilterType::New();
  medianFilter->SetRadius(medianRadius);
  medianFilter->SetBackgroundValue(0);
  medianFilter->SetInput(labelResampler->GetOutput()); // first use will be to generate scaling factor for original image data

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //  Need to calculate the mean signal in the structures of interest for use in rescaling
  //  Use a statistics filter to get the gray level statistics from the original MRI image
  typedef itk::BinaryImageToStatisticsLabelMapFilter<LabelImageType, ImageType> BinaryImageToStatisticsLabelMapFilterType;
  BinaryImageToStatisticsLabelMapFilterType::Pointer BinaryToStatisticsFilter = BinaryImageToStatisticsLabelMapFilterType::New();
  BinaryToStatisticsFilter->SetInput1(medianFilter->GetOutput());
  BinaryToStatisticsFilter->SetInput2(imageReader->GetOutput());
  BinaryToStatisticsFilter->SetComputeHistogram(TRUE);
  BinaryToStatisticsFilter->SetCoordinateTolerance(0.01);
  BinaryImageToStatisticsLabelMapFilterType::OutputImageType::LabelObjectType* StatlabelObject;

  std::cout << "Calculating scale factor " << std::endl;
  labelValuesIterator = labelValues.begin();
  i = 0;
  while (labelValuesIterator != labelValues.end())
  {
	  labelValue = labelValues[i];
	  medianFilter->SetForegroundValue(labelValue);
	  BinaryToStatisticsFilter->SetInputForegroundValue(labelValue);
	  BinaryToStatisticsFilter->UpdateLargestPossibleRegion();
	  StatlabelObject = BinaryToStatisticsFilter->GetOutput()->GetNthLabelObject(0);
	  sum += StatlabelObject->GetMean();

	  i++;
	  labelValuesIterator++;
  }
  mean = sum / 4.0;
  std::cout << "Mean signal in structures is  " << mean << std::endl;
  factor = 512. / mean;
  std::cout << "Scale factor is  " << factor << std::endl << std::endl << std::endl;
  
  // Now calculate the rescaled version of the original image
  typedef itk::ShiftScaleImageFilter<ImageType, ImageType> ScalerType;
  ScalerType::Pointer scaler = ScalerType::New();
  scaler->SetInput(imageReader->GetOutput());
  scaler->SetScale(factor);
  scaler->SetShift(0);
  std::cout << "Rescaling input image." << std::endl;
  scaler->UpdateLargestPossibleRegion();
  imagePointers.push_back(scaler->GetOutput());  // This is image number 2 to process

  if (writefiles)
  {
	  imageWriter->SetInput(scaler->GetOutput());
	  imageWriter->SetFileName(inputDirectory + "rawavg_rescaled.nrrd");
	  imageWriter->Update();
  }
  
  // Create smoother to Gaussian filter the image data
  typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> SmoothingType;
  SmoothingType::Pointer smoother = SmoothingType::New();
  smoother->SetVariance(0.5);
  smoother->SetUseImageSpacingOn(); // this is the default value but just in case
  smoother->SetMaximumKernelWidth(5);
  smoother->SetInput(imageReader->GetOutput());
  std::cout << "Calculating smoothed version of input image." << std::endl;
  smoother->UpdateLargestPossibleRegion(); // This creates the smoothed version of the original data 

  DeepCopy<ImageType>(smoother->GetOutput(), origGF.GetPointer()); // need a deep copy of this image
  imagePointers.push_back(origGF);  // This is image number 3

  if (writefiles)
  {
	  imageWriter->SetInput(origGF.GetPointer());
	  imageWriter->SetFileName(inputDirectory + "rawavg_GF.nrrd");
	  imageWriter->Update();
  }

  //  Reconfigure to get a smoothed version of the rescaled image
  smoother->SetInput(scaler->GetOutput());
  std::cout << "Calculating smoothed version of rescaled image." << std::endl;
  smoother->UpdateLargestPossibleRegion();
  DeepCopy<ImageType>(smoother->GetOutput(), origRescaledGF.GetPointer()); // need a deep copy of this image
  imagePointers.push_back(origRescaledGF.GetPointer());  // This is image number 4

  if (writefiles)
  {
	  imageWriter->SetInput(origRescaledGF.GetPointer());
	  imageWriter->SetFileName(inputDirectory + "rawavg_rescaledGF.nrrd");
	  imageWriter->Update();
  }

  // Now we need rescaled versions of the four images we created.  Resampler geometry is already set up.
  resampler->SetInput(imageReader->GetOutput());
  std::cout << "Calculating resampled version of input image." << std::endl;
  resampler->UpdateLargestPossibleRegion();
  DeepCopy<ImageType>(resampler->GetOutput(), resampled.GetPointer()); // need a deep copy of this image
  imagePointers.push_back(resampled.GetPointer());  // This is image number 5

  if (writefiles)
  {
	  imageWriter->SetInput(resampled.GetPointer());
	  imageWriter->SetFileName(inputDirectory + "resampled.nrrd");
	  imageWriter->Update();
  }

  resampler->SetInput(scaler->GetOutput());
  std::cout << "Calculating resampled version of rescaled input image." << std::endl;
  resampler->UpdateLargestPossibleRegion();
  DeepCopy<ImageType>(resampler->GetOutput(), resampledRescaled.GetPointer()); // need a deep copy of this image
  imagePointers.push_back(resampledRescaled.GetPointer());  // This is image number 6

  if (writefiles)
  {
	  imageWriter->SetInput(resampledRescaled.GetPointer());
	  imageWriter->SetFileName(inputDirectory + "resampled_rescaled.nrrd");
	  imageWriter->Update();
  }

  std::cout << "Calculating resampled version of smoothed input image." << std::endl;
  smoother->SetInput(imageReader->GetOutput());
  smoother->UpdateLargestPossibleRegion();
  resampler->SetInput(smoother->GetOutput());
  resampler->UpdateLargestPossibleRegion();
  DeepCopy<ImageType>(resampler->GetOutput(), resampledGF.GetPointer()); // need a deep copy of this image
  imagePointers.push_back(resampledGF.GetPointer());  // This is image number 7

  if (writefiles)
  {
	  imageWriter->SetInput(resampledGF.GetPointer());
	  imageWriter->SetFileName(inputDirectory + "resampledGF.nrrd");
	  imageWriter->Update();
  }

  std::cout << "Calculating resampled version of rescaled, smoothed input image." << std::endl << std::endl << std::endl;
  smoother->SetInput(scaler->GetOutput());
  smoother->UpdateLargestPossibleRegion();
  resampler->UpdateLargestPossibleRegion();
  // Don't need a deep copy we will just let the filter hold the image since we're done modifying the images
  imagePointers.push_back(resampler->GetOutput());  // This is image number 8

  if (writefiles)
  {
	  imageWriter->SetInput(resampler->GetOutput());
	  imageWriter->SetFileName(inputDirectory + "resampled_rescaledGF.nrrd");
	  imageWriter->Update();
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Set up the feature extraction filters
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  //  Shape filter to extract shape metrics from the labels
  typedef itk::BinaryImageToShapeLabelMapFilter<LabelImageType> BinaryImageToShapeLabelMapFilterType;
  BinaryImageToShapeLabelMapFilterType::Pointer binaryImageToShapeLabelMapFilter = BinaryImageToShapeLabelMapFilterType::New();
  binaryImageToShapeLabelMapFilter->SetComputeFeretDiameter(true);
  binaryImageToShapeLabelMapFilter->SetInput(medianFilter->GetOutput());
  BinaryImageToShapeLabelMapFilterType::OutputImageType::LabelObjectType* labelObject;
 
  //  Need a cast image filter since the texture filters requires same type for image and mask
  typedef itk::CastImageFilter<LabelImageType, ImageType> LabelCasterType;
  LabelCasterType::Pointer labelCaster = LabelCasterType::New();
  labelCaster->SetInput(medianFilter->GetOutput());

  //  Texture filter to get the Haralick features
  typedef itk::Statistics::ScalarImageToTextureFeaturesFilter<ImageType> TextureFilterType;
  TextureFilterType::Pointer textureFilter = TextureFilterType::New();
  textureFilter->SetMaskImage(labelCaster->GetOutput());
  textureFilter->SetFastCalculations(false);
  textureFilter->SetNumberOfBinsPerAxis(1024);
  textureFilter->SetPixelValueMinMax(0, 2048);

  TextureFilterType::OffsetType   offset, test;
  TextureFilterType::OffsetVectorPointer   offset1;
  offset1 = TextureFilterType::OffsetVector::New();
  TextureFilterType::OffsetVector::ConstIterator vIt; 
  const TextureFilterType::FeatureValueVector* output;
  const TextureFilterType::FeatureValueVector* outputSD;

  //  From itkGreyLevelCooccurrenceMatrixTextureCoefficientsCalculator.h
  //
  //  enum TextureFeatureName {Energy, Entropy, Correlation,
  //  InverseDifferenceMoment, Inertia, ClusterShade, ClusterProminence, HaralickCorrelation};
  
  const TextureFilterType::FeatureNameVector* featureNames;
  featureNames = textureFilter->GetRequestedFeatures();
  // featureNames shows that features 1,2,4,5,6,7 of the above list are default

  // Build the offset vector for the texture filter
  // This is the same as the default but shows the process of building a neighborhood
  radius = 1;
  for (i = -radius; i <= radius; i++)
  {
	  for (j = -radius; j <= radius; j++)
	  {
		  for (k = -radius; k <= radius; k++)
		  {
			  if (((i + j + k) <= 0) && !(i == 0 && j == 0 && k == 0))
			  {
				  found = false;
				  offset[0] = i;
				  offset[1] = j;
				  offset[2] = k;
				  test[0] = -i;
				  test[1] = -j;
				  test[2] = -k;
				  for (vIt = offset1->Begin(); vIt != offset1->End(); ++vIt)
				  {
					  if (vIt.Value() == test)
						  found = true;
				  }

				  if (!found)
					  offset1->push_back(offset);
			  }
		  }
	  }
  }
  // Now set the offset vector for the texture filter
  textureFilter->SetOffsets(offset1);

  //  Create the run-length feature filter
  //  We will use the default values for the features and neighborhood
  typedef itk::Statistics::ScalarImageToRunLengthFeaturesFilter<ImageType> RunLengthFeaturesFilterType;
  RunLengthFeaturesFilterType::Pointer runLengthFeaturesFilter = RunLengthFeaturesFilterType::New();
  runLengthFeaturesFilter->SetMaskImage(labelCaster->GetOutput());
  runLengthFeaturesFilter->SetFastCalculations(false);
  runLengthFeaturesFilter->SetNumberOfBinsPerAxis(1024);
  runLengthFeaturesFilter->SetPixelValueMinMax(0, 2048);

  RunLengthFeaturesFilterType::FeatureValueVectorPointer runLengthOutput;
  RunLengthFeaturesFilterType::FeatureValueVectorPointer runLengthOutputSD;

  // Run length features from itkHistogramToRunLengthFeaturesFilter.h
  // typedef enum{ ShortRunEmphasis, LongRunEmphasis, GreyLevelNonuniformity, RunLengthNonuniformity,
  //	   LowGreyLevelRunEmphasis, HighGreyLevelRunEmphasis, ShortRunLowGreyLevelEmphasis,
  //	   ShortRunHighGreyLevelEmphasis, LongRunLowGreyLevelEmphasis, LongRunHighGreyLevelEmphasis }


  ///////////////////////////////////////////////////////////////////////////////////////////////
  // Open the output files for writing the features.  There are 8 different feature files
  
  std::vector<FILE *> outputFiles;
  outputFiles.reserve(8); //storage for eight file handles
  outputFiles.push_back(fopen((inputDirectory + "rawavg_features.csv").c_str(), "wt"));
  outputFiles.push_back(fopen((inputDirectory + "rawavgRescaled_features.csv").c_str(), "wt"));
  outputFiles.push_back(fopen((inputDirectory + "rawavgGF_features.csv").c_str(), "wt"));
  outputFiles.push_back(fopen((inputDirectory + "rawavgRescaledGF_features.csv").c_str(), "wt"));
  outputFiles.push_back(fopen((inputDirectory + "resampled_features.csv").c_str(), "wt"));
  outputFiles.push_back(fopen((inputDirectory + "resampledRescaled_features.csv").c_str(), "wt"));
  outputFiles.push_back(fopen((inputDirectory + "resampledGF_features.csv").c_str(), "wt"));
  outputFiles.push_back(fopen((inputDirectory + "resampledRescaledGF_features.csv").c_str(), "wt"));

  // Put the input directory on the first line of the results file.  This lets us parse the patient ID out later
  outputstring = "Directory = " + inputDirectory + "\n";
  fprintf(outputFiles[0], outputstring.c_str());
  fprintf(outputFiles[1], outputstring.c_str());
  fprintf(outputFiles[2], outputstring.c_str());
  fprintf(outputFiles[3], outputstring.c_str());
  fprintf(outputFiles[4], outputstring.c_str());
  fprintf(outputFiles[5], outputstring.c_str());
  fprintf(outputFiles[6], outputstring.c_str());
  fprintf(outputFiles[7], outputstring.c_str());

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //        MAIN LOOP
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Iterate over the 8 image versions to generate the features
  imageIterator = imagePointers.begin();
  k = 0;
 
  while (imageIterator != imagePointers.end())
  {
	  std::cout << "Processing image  " << k+1 << std::endl << std::endl;
	  //  Configure the pipeline to use the appropriate inputs
	  medianFilter->SetInput(labelImagePointers[k]);
	  BinaryToStatisticsFilter->SetInput2(imagePointers[k]);
	  textureFilter->SetInput(imagePointers[k]);
	  runLengthFeaturesFilter->SetInput(imagePointers[k]);

	  // Iterate over the labels we want to process
	  //////////////////////////////////////////////////////////////////////////////////////////////////////////
	  i = 0;
	  labelNamesIterator = labelNames.begin();

	  while (labelNamesIterator != labelNames.end())
	  {
		  labelName = labelNames[i];
		  labelValue = labelValues[i];
		  std::cout << "Processing  " << labelName << std::endl << std::endl;

		  // First we want to extract the current label and median filter to remove rough edges.
		  // need to set the appropriate value in the median filter
		  medianFilter->SetForegroundValue(labelValue);
		  medianFilter->UpdateLargestPossibleRegion();
		  std::cout << "Median Filter Updated" << std::endl << std::endl;

		  // Now get the shape statistics for that label
		  binaryImageToShapeLabelMapFilter->SetInputForegroundValue(labelValue);
		  std::cout << "Calculating shape values for " << labelName << std::endl;
		  binaryImageToShapeLabelMapFilter->UpdateLargestPossibleRegion();

		  labelObject = binaryImageToShapeLabelMapFilter->GetOutput()->GetNthLabelObject(0);
		  std::cout << "Roundness = " << labelObject->GetRoundness() << std::endl;
    	  outputstring = labelName + "Roundness, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), labelObject->GetRoundness());
		  std::cout << "Flatness = " << labelObject->GetFlatness() << std::endl;
		  outputstring = labelName + "Flatness, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), labelObject->GetFlatness());
		  std::cout << "Feret Diameter = " << labelObject->GetFeretDiameter() << std::endl;
		  outputstring = labelName + "FeretDiameter, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), labelObject->GetFeretDiameter());
		  std::cout << "Physical Size = " << labelObject->GetPhysicalSize() << std::endl;
		  outputstring = labelName + "Volume, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), labelObject->GetPhysicalSize());
		  std::cout << "Elongation = " << labelObject->GetElongation() << std::endl;
		  outputstring = labelName + "Elongation, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), labelObject->GetElongation());
		  std::cout << std::endl;

		  // Now get the statistics
		  BinaryToStatisticsFilter->SetInputForegroundValue(labelValue);
		  std::cout << "Calculating statistics values for " << labelName << std::endl;
		  BinaryToStatisticsFilter->UpdateLargestPossibleRegion();

		  StatlabelObject = BinaryToStatisticsFilter->GetOutput()->GetNthLabelObject(0);
		  std::cout << "Mean = " << StatlabelObject->GetMean() << std::endl;
		  outputstring = labelName + "Mean, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), StatlabelObject->GetMean());
		  std::cout << "Median = " << StatlabelObject->GetMedian() << std::endl;
		  outputstring = labelName + "Median, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), StatlabelObject->GetMedian());
		  std::cout << "Skewness = " << StatlabelObject->GetSkewness() << std::endl;
		  outputstring = labelName + "Skewness, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), StatlabelObject->GetSkewness());
		  std::cout << "Kurtosis = " << StatlabelObject->GetKurtosis() << std::endl;
		  outputstring = labelName + "Kurtosis, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), StatlabelObject->GetKurtosis());
		  std::cout << "Sigma = " << StatlabelObject->GetStandardDeviation() << std::endl;
		  outputstring = labelName + "Sigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), StatlabelObject->GetStandardDeviation());
		  std::cout << std::endl;
		 
		  // Now get the texture features
		  //  enum TextureFeatureName {Energy, Entropy, Correlation,
		  //  InverseDifferenceMoment, Inertia, ClusterShade, ClusterProminence, HaralickCorrelation};
		  textureFilter->SetInsidePixelValue(labelValue);
		  std::cout << "Calculating GLCM texture values for " << labelName << std::endl;
		  textureFilter->UpdateLargestPossibleRegion();

		  output = textureFilter->GetFeatureMeans();
		  outputSD = textureFilter->GetFeatureStandardDeviations();

		  std::cout << "Energy = " << (*output)[0] << std::endl;
		  outputstring = labelName + "Energy, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*output)[0]);
		  std::cout << "EnergySigma = " << (*outputSD)[0] << std::endl;
		  outputstring = labelName + "EnergySigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*outputSD)[0]);

		  std::cout << "Entropy = " << (*output)[1] << std::endl;
		  outputstring = labelName + "Entropy, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*output)[1]);
		  std::cout << "EntropySigma = " << (*outputSD)[1] << std::endl;
		  outputstring = labelName + "EntropySigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*outputSD)[1]);

		  std::cout << "InverseDifferenceMoment = " << (*output)[2] << std::endl;
		  outputstring = labelName + "InverseDifferenceMoment, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*output)[2]);
		  std::cout << "InverseDifferenceMomentSigma = " << (*outputSD)[2] << std::endl;
		  outputstring = labelName + "InverseDifferenceMomentSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*outputSD)[2]);

		  std::cout << "Inertia = " << (*output)[3] << std::endl;
		  outputstring = labelName + "Inertia, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*output)[3]);
		  std::cout << "InertiaSigma = " << (*outputSD)[3] << std::endl;
		  outputstring = labelName + "InertiaSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*outputSD)[3]);

		  std::cout << "ClusterShade = " << (*output)[4] << std::endl;
		  outputstring = labelName + "ClusterShade, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*output)[4]);
		  std::cout << "ClusterShadeSigma = " << (*outputSD)[4] << std::endl;
		  outputstring = labelName + "ClusterShadeSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*outputSD)[4]);

		  std::cout << "ClusterProminence = " << (*output)[5] << std::endl;
		  outputstring = labelName + "ClusterProminence, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*output)[5]);
		  std::cout << "ClusterProminenceSigma = " << (*outputSD)[5] << std::endl;
		  outputstring = labelName + "ClusterProminenceSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*outputSD)[5]);

		  std::cout << std::endl;

		  //  Get the run-length features
		  runLengthFeaturesFilter->SetInsidePixelValue(labelValue);
		  std::cout << "Calculating run-length features for " << labelName << std::endl;  
		  runLengthFeaturesFilter->UpdateLargestPossibleRegion();

		  runLengthOutput = runLengthFeaturesFilter->GetFeatureMeans();
		  runLengthOutputSD = runLengthFeaturesFilter->GetFeatureStandardDeviations();

		  // Output the run-length features
		  // Run length features from itkHistogramToRunLengthFeaturesFilter.h
		  // typedef enum{ ShortRunEmphasis, LongRunEmphasis, GreyLevelNonuniformity, RunLengthNonuniformity,
		  //	   LowGreyLevelRunEmphasis, HighGreyLevelRunEmphasis, ShortRunLowGreyLevelEmphasis,
		  //	   ShortRunHighGreyLevelEmphasis, LongRunLowGreyLevelEmphasis, LongRunHighGreyLevelEmphasis }
		  		  
		  std::cout << "ShortRunEmphasis = " << (*runLengthOutput)[0] << std::endl;
		  outputstring = labelName + "ShortRunEmphasis, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutput)[0]);
		  std::cout << "ShortRunEmphasisSigma = " << (*runLengthOutputSD)[0] << std::endl;
		  outputstring = labelName + "ShortRunEmphasisSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutputSD)[0]);

		  std::cout << "LongRunEmphasis = " << (*runLengthOutput)[1] << std::endl;
		  outputstring = labelName + "LongRunEmphasis, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutput)[1]);
		  std::cout << "LongRunEmphasisSigma = " << (*runLengthOutputSD)[1] << std::endl;
		  outputstring = labelName + "LongRunEmphasisSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutputSD)[1]);
		  
		  std::cout << "GreyLevelNonuniformity = " << (*runLengthOutput)[2] << std::endl;
		  outputstring = labelName + "GreyLevelNonuniformity, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutput)[2]);
		  std::cout << "GreyLevelNonuniformitySigma = " << (*runLengthOutputSD)[2] << std::endl;
		  outputstring = labelName + "GreyLevelNonuniformitySigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutputSD)[2]);

		  std::cout << "RunLengthNonuniformity = " << (*runLengthOutput)[3] << std::endl;
		  outputstring = labelName + "RunLengthNonuniformity, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutput)[3]);
		  std::cout << "RunLengthNonuniformitySigma = " << (*runLengthOutputSD)[3] << std::endl;
		  outputstring = labelName + "RunLengthNonuniformitySigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutputSD)[3]);

		  std::cout << "LowGreyLevelRunEmphasis = " << (*runLengthOutput)[4] << std::endl;
		  outputstring = labelName + "LowGreyLevelRunEmphasis, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutput)[4]);
		  std::cout << "LowGreyLevelRunEmphasisSigma = " << (*runLengthOutputSD)[4] << std::endl;
		  outputstring = labelName + "LowGreyLevelRunEmphasisSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutputSD)[4]);

		  std::cout << "HighGreyLevelRunEmphasis = " << (*runLengthOutput)[5] << std::endl;
		  outputstring = labelName + "HighGreyLevelRunEmphasis, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutput)[5]);
		  std::cout << "HighGreyLevelRunEmphasisSigma = " << (*runLengthOutputSD)[5] << std::endl;
		  outputstring = labelName + "HighGreyLevelRunEmphasisSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutputSD)[5]);

		  std::cout << "ShortRunLowGreyLevelEmphasis = " << (*runLengthOutput)[6] << std::endl;
		  outputstring = labelName + "ShortRunLowGreyLevelEmphasis, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutput)[6]);
		  std::cout << "ShortRunLowGreyLevelEmphasisSigma = " << (*runLengthOutputSD)[6] << std::endl;
		  outputstring = labelName + "ShortRunLowGreyLevelEmphasisSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutputSD)[6]);

		  std::cout << "ShortRunHighGreyLevelEmphasis = " << (*runLengthOutput)[7] << std::endl;
		  outputstring = labelName + "ShortRunHighGreyLevelEmphasis, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutput)[7]);
		  std::cout << "ShortRunHighGreyLevelEmphasisSigma = " << (*runLengthOutputSD)[7] << std::endl;
		  outputstring = labelName + "ShortRunHighGreyLevelEmphasisSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutputSD)[7]);

		  std::cout << "LongRunLowGreyLevelEmphasis = " << (*runLengthOutput)[8] << std::endl;
		  outputstring = labelName + "LongRunLowGreyLevelEmphasis, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutput)[8]);
		  std::cout << "LongRunLowGreyLevelEmphasisSigma = " << (*runLengthOutputSD)[8] << std::endl;
		  outputstring = labelName + "LongRunLowGreyLevelEmphasisSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutputSD)[8]);
		  
		  std::cout << "LongRunHighGreyLevelEmphasis = " << (*runLengthOutput)[9] << std::endl;
		  outputstring = labelName + "LongRunHighGreyLevelEmphasis, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutput)[9]);
		  std::cout << "LongRunHighGreyLevelEmphasisSigma = " << (*runLengthOutputSD)[9] << std::endl;
		  outputstring = labelName + "LongRunHighGreyLevelEmphasisSigma, %f \n";
		  fprintf(outputFiles[k], outputstring.c_str(), (*runLengthOutputSD)[9]);

		  std::cout << std::endl << std::endl;
		  
		  i++;
		  labelNamesIterator++;
	  }
	  std::cout << std::endl << std::endl;
	  std::cout << std::endl << std::endl;
	  
	  k++;
	  imageIterator++;
  }  
  fclose(outputFiles[0]);
  fclose(outputFiles[1]);
  fclose(outputFiles[2]);
  fclose(outputFiles[3]);
  fclose(outputFiles[4]);
  fclose(outputFiles[5]);
  fclose(outputFiles[6]);
  fclose(outputFiles[7]);
}