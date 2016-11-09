#include "itkTestingExtractSliceImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryMedianImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkImageFileReader.h"
#include "itkScalarImageToTextureFeaturesFilter.h"
#include "itkBinaryImageToShapeLabelMapFilter.h"
#include "itkBinaryImageToStatisticsLabelMapFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkScalarImageToRunLengthFeaturesFilter.h"
#include "QuickView.h"
#include <stdio.h>
#include <vector>



// These are the Freesurfer label values for the structures of interest
#define LeftHippocampus 17
#define RightHippocampus 53
#define LeftThalamus 10
#define RightThalamus 49


int main(int argc, char * argv[])
{
  
  unsigned short int labelValue; // holds the current label value during the loop
  std::string labelName;		 // holds the current label name during the loop
  const unsigned int Dimension = 3;
  int i, j, k; // used for general purpose indexing
  int radius=1;  // size of neighborhood for GLCM determination
  bool found;  // used to build the texture neighborhood vector
  size_t foundchar;
  

  // both image data and labels can be opened as UINT's
  typedef  unsigned short  int								   PixelType;
  typedef itk::Image< PixelType, Dimension >                   ImageType;
  typedef itk::Image< PixelType, Dimension-1>				   SliceType;

  ImageType::SizeType medianRadius; // used for binary median image filter
  medianRadius[0] = 1;
  medianRadius[1] = 1;
  medianRadius[2] = 1;

  // Get the input directory specified on the command line
  std::string inputFile;
  if (argc > 1)
  {
	 	inputFile = argv[1];
  }
  else
  {
  	std::cout <<  "Input a file to process" << std::endl;
	std::cout << "The label file should be in the same directory and have the name:" << std::endl;
	std::cout << "<filename>_label.nrrd where <filename> is the input file." << std::endl;
	exit(1);
  }

  // the file names for the brain scan and the label image
  std::string imageData = inputFile, filename, labelData, Path, outputstring;

  foundchar = inputFile.find_last_of("/\\");
  Path = imageData.substr(0, foundchar+1);

  filename = imageData.substr(foundchar + 1);
  foundchar = filename.find_last_of(".");
  labelData = Path + filename.substr(0, foundchar) +"_label.nrrd";
  

  // Open the output files and put the directory name on the first line
  FILE * outputFile = fopen((Path + filename.substr(0, foundchar) + "_features.csv").c_str(), "wt");
  FILE * outputFileGF = fopen((Path + filename.substr(0, foundchar) + "_featuresGF.csv").c_str(), "wt");
  outputstring = "Output Directory = " + Path + "\n";
  fprintf(outputFile, outputstring.c_str());
  fprintf(outputFileGF, outputstring.c_str());
	
  // Create vectors of label names and label values
  std::vector<std::string> labelNames;
  labelNames.reserve(4); //storage for four labels
  labelNames.push_back("LeftHippocampus");
  labelNames.push_back("RightHippocampus");
  labelNames.push_back("LeftThalamus");
  labelNames.push_back("RightThalamus");

  std::vector<unsigned short int> labelValues;
  labelValues.reserve(4);  //storage for four values
  labelValues.push_back(LeftHippocampus);
  labelValues.push_back(RightHippocampus);
  labelValues.push_back(LeftThalamus);
  labelValues.push_back(RightThalamus);

  // Create iterators for the vectors.  We will iterate over the structures when doing the processing
  std::vector<std::string>::iterator labelNamesIterator = labelNames.begin();
  std::vector<unsigned short int>::iterator labelValuesIterator = labelValues.begin();

  // open the image data
  typedef itk::ImageFileReader< ImageType >  ReaderType;
  ReaderType::Pointer imageReader = ReaderType::New();
  imageReader->SetFileName(imageData );
  imageReader->Update();

  // open the labels
  typedef itk::ImageFileReader< ImageType >  LabelReaderType;
  LabelReaderType::Pointer labelReader = LabelReaderType::New();
  labelReader->SetFileName(labelData);
  labelReader->Update();

  // make sure the image geometries match
  ImageType::RegionType region = imageReader->GetOutput()->GetLargestPossibleRegion();
  ImageType::SizeType size = region.GetSize();
  ImageType::RegionType labelregion = labelReader->GetOutput()->GetLargestPossibleRegion();
  ImageType::SizeType labelsize = labelregion.GetSize();
  ImageType::SpacingType spacing = imageReader->GetOutput()->GetSpacing();
  ImageType::SpacingType labelspacing = labelReader->GetOutput()->GetSpacing();
  ImageType::PointType origin = imageReader->GetOutput()->GetOrigin();
  ImageType::PointType labelorigin = labelReader->GetOutput()->GetOrigin();

  labelReader->GetOutput()->SetSpacing(imageReader->GetOutput()->GetSpacing());
  labelReader->GetOutput()->SetOrigin(imageReader->GetOutput()->GetOrigin());

 /* // This is just to get a look at one of the slices of the image data.  This block can be commented out
  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  ImageType::SizeType imagesize = labelReader->GetOutput()->GetLargestPossibleRegion().GetSize();
  ImageType::RegionType extractRegion;
  extractRegion.SetSize(0,imagesize[0]);
  extractRegion.SetSize(1,imagesize[1]);
  extractRegion.SetSize(2,0);
  extractRegion.SetIndex(0,0);
  extractRegion.SetIndex(1,0);
  extractRegion.SetIndex(2,128); // extract image slice 128 for viewing purposes

  
  typedef itk::Testing::ExtractSliceImageFilter<ImageType, SliceType> ExtractSliceType;
  ExtractSliceType::Pointer SliceExtracter = ExtractSliceType::New();
  SliceExtracter->SetInput(labelReader->GetOutput());
  SliceExtracter->SetExtractionRegion(extractRegion);
  SliceExtracter->SetDirectionCollapseToIdentity();
  SliceExtracter->Update();

  // view a slice
  QuickView viewer;
  viewer.AddImage(SliceExtracter->GetOutput());
  viewer.Visualize();
  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  */

  // Create smoothed version of the image data
  typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> SmoothingType;
  SmoothingType::Pointer smoother = SmoothingType::New();
  smoother->SetInput(imageReader->GetOutput());
  smoother->SetVariance(0.5);
  smoother->SetUseImageSpacingOn(); // this is the default value but just in case
  smoother->SetMaximumKernelWidth(5);

  // Set up the pipeline then we will iterate over the labels to be processed
  /////////////////////////////////////////////////////////////////////////////////////////

  //  Median filter to smooth the edges of the labels
  typedef itk::BinaryMedianImageFilter<ImageType, ImageType> MedianFilterType;
  MedianFilterType::Pointer medianFilter = MedianFilterType::New();
  medianFilter->SetRadius(medianRadius);
  medianFilter->SetBackgroundValue(0);
  medianFilter->SetInput(labelReader->GetOutput());

  //  Shape filter to extract shape metrics from the labels
  typedef itk::BinaryImageToShapeLabelMapFilter<ImageType> BinaryImageToShapeLabelMapFilterType;
  BinaryImageToShapeLabelMapFilterType::Pointer binaryImageToShapeLabelMapFilter = BinaryImageToShapeLabelMapFilterType::New();
  binaryImageToShapeLabelMapFilter->SetComputeFeretDiameter(true);
  binaryImageToShapeLabelMapFilter->SetInput(medianFilter->GetOutput());
  BinaryImageToShapeLabelMapFilterType::OutputImageType::LabelObjectType* labelObject;

  //  Statistics filter to get the gray level statistics from the MRI image for a particular label location
  typedef itk::BinaryImageToStatisticsLabelMapFilter<ImageType, ImageType> BinaryImageToStatisticsLabelMapFilterType;
  BinaryImageToStatisticsLabelMapFilterType::Pointer BinaryToStatisticsFilter = BinaryImageToStatisticsLabelMapFilterType::New();
  BinaryToStatisticsFilter->SetInput1(medianFilter->GetOutput());
  BinaryToStatisticsFilter->SetComputeHistogram(TRUE);
  BinaryToStatisticsFilter->SetCoordinateTolerance(0.01);
  BinaryImageToStatisticsLabelMapFilterType::OutputImageType::LabelObjectType* StatlabelObject;
 
  //  Texture filter to get the Haralick features
  typedef itk::Statistics::ScalarImageToTextureFeaturesFilter<ImageType> TextureFilterType;
  TextureFilterType::Pointer textureFilter = TextureFilterType::New();
  textureFilter->SetMaskImage(medianFilter->GetOutput());
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
  runLengthFeaturesFilter->SetMaskImage(medianFilter->GetOutput());
  runLengthFeaturesFilter->SetFastCalculations(false);
  runLengthFeaturesFilter->SetNumberOfBinsPerAxis(1024);
  runLengthFeaturesFilter->SetPixelValueMinMax(0, 2048);

  RunLengthFeaturesFilterType::FeatureValueVectorPointer runLengthOutput;
  RunLengthFeaturesFilterType::FeatureValueVectorPointer runLengthOutputSD;

  // Run length features from itkHistogramToRunLengthFeaturesFilter.h
  // typedef enum{ ShortRunEmphasis, LongRunEmphasis, GreyLevelNonuniformity, RunLengthNonuniformity,
  //	   LowGreyLevelRunEmphasis, HighGreyLevelRunEmphasis, ShortRunLowGreyLevelEmphasis,
  //	   ShortRunHighGreyLevelEmphasis, LongRunLowGreyLevelEmphasis, LongRunHighGreyLevelEmphasis }

  // Iterate over the labels we want to process
  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  i = 0;
  while (labelNamesIterator != labelNames.end())
  {
	  labelName = labelNames[i];
	  labelValue = labelValues[i];

	  // First we want to extract the current label and median filter to remove rough edges.
	  // need to set the appropriate value in the median filter
	  medianFilter->SetForegroundValue(labelValue);
	  medianFilter->Update();

	  // Now get the shape statistics for that label
	  binaryImageToShapeLabelMapFilter->SetInputForegroundValue(labelValue);
	  binaryImageToShapeLabelMapFilter->Update();
	  std::cout << "There is " << binaryImageToShapeLabelMapFilter->GetOutput()->GetNumberOfLabelObjects() << " object." << std::endl;

	  // Loop over the regions (should only be 1)
	  std::cout << "Shape values for " << labelName << std::endl;
	  std::cout << "Roundness, Flatness, Feret Diameter, Number of Pixels, Physical Size " << std::endl;
	  for (j = 0; j < binaryImageToShapeLabelMapFilter->GetOutput()->GetNumberOfLabelObjects(); j++)
	  {
		  labelObject = binaryImageToShapeLabelMapFilter->GetOutput()->GetNthLabelObject(j);
		  std::cout << labelObject->GetRoundness() << std::endl;
		  outputstring = labelName + "Roundness, %f \n";
		  fprintf(outputFile, outputstring.c_str(), labelObject->GetRoundness());
		  fprintf(outputFileGF, outputstring.c_str(), labelObject->GetRoundness());
		  std::cout << labelObject->GetFlatness() << std::endl;
		  outputstring = labelName + "Flatness, %f \n";
		  fprintf(outputFile, outputstring.c_str(), labelObject->GetFlatness());
		  fprintf(outputFileGF, outputstring.c_str(), labelObject->GetFlatness());
		  std::cout << labelObject->GetFeretDiameter() << std::endl;
		  outputstring = labelName + "FeretDiameter, %f \n";
		  fprintf(outputFile, outputstring.c_str(), labelObject->GetFeretDiameter());
		  fprintf(outputFileGF, outputstring.c_str(), labelObject->GetFeretDiameter());
		  std::cout << labelObject->GetPhysicalSize() << std::endl;
		  outputstring = labelName + "Volume, %f \n";
		  fprintf(outputFile, outputstring.c_str(), labelObject->GetPhysicalSize());
		  fprintf(outputFileGF, outputstring.c_str(), labelObject->GetPhysicalSize());
		  std::cout << labelObject->GetElongation() << std::endl;
		  outputstring = labelName + "Elongation, %f \n";
		  fprintf(outputFile, outputstring.c_str(), labelObject->GetElongation());
		  fprintf(outputFileGF, outputstring.c_str(), labelObject->GetElongation());

		  //std::cout << "Object " << i << " has principal axes " << labelObject->GetPrincipalAxes() << std::endl;
		  //std::cout << "Object " << i << " has principal moments " << labelObject->GetPrincipalMoments() << std::endl;
		  
		  std::cout << std::endl << std::endl;
	  }

	  // Now get the statistics for the un-Blurred image data using the current label

	  BinaryToStatisticsFilter->SetInput2(imageReader->GetOutput());
	  BinaryToStatisticsFilter->SetInputForegroundValue(labelValue);
	  BinaryToStatisticsFilter->Update();

	  std::cout << "There is " << BinaryToStatisticsFilter->GetOutput()->GetNumberOfLabelObjects() << " object with statistics." << std::endl;
	  std::cout << "Statistics values from un-blurred image for " << labelName << std::endl;
	  std::cout << "Mean, Median, Skewness, Kurtosis, Standard Deviation " << std::endl;
	  for (k = 0; k < BinaryToStatisticsFilter->GetOutput()->GetNumberOfLabelObjects(); k++)
	  {
		  StatlabelObject = BinaryToStatisticsFilter->GetOutput()->GetNthLabelObject(k);
		  // Output the shape properties of the ith region
		  // Mean, median, skewness, kurtosis, sigma
		  std::cout << StatlabelObject->GetMean() << std::endl;
		  outputstring = labelName + "Mean, %f \n";
		  fprintf(outputFile, outputstring.c_str(), StatlabelObject->GetMean());
		  std::cout << StatlabelObject->GetMedian() << std::endl;
		  outputstring = labelName + "Median, %f \n";
		  fprintf(outputFile, outputstring.c_str(), StatlabelObject->GetMedian());
		  std::cout << StatlabelObject->GetSkewness() << std::endl;
		  outputstring = labelName + "Skewness, %f \n";
		  fprintf(outputFile, outputstring.c_str(), StatlabelObject->GetSkewness());
		  std::cout << StatlabelObject->GetKurtosis() << std::endl;
		  outputstring = labelName + "Kurtosis, %f \n";
		  fprintf(outputFile, outputstring.c_str(), StatlabelObject->GetKurtosis());
		  std::cout << StatlabelObject->GetStandardDeviation() << std::endl;
		  outputstring = labelName + "Sigma, %f \n";
		  fprintf(outputFile, outputstring.c_str(), StatlabelObject->GetStandardDeviation());
		  std::cout << std::endl << std::endl;
	  }

	  // Now get the texture features for the unblurred image
	  textureFilter->SetInput(imageReader->GetOutput());
	  textureFilter->SetInsidePixelValue(labelValue);
	  textureFilter->Update();

	  output = textureFilter->GetFeatureMeans();
	  outputSD = textureFilter->GetFeatureStandardDeviations();

	  //  enum TextureFeatureName {Energy, Entropy, Correlation,
	  //  InverseDifferenceMoment, Inertia, ClusterShade, ClusterProminence, HaralickCorrelation};
	  std::cout << "Radius 1 Texture Features for: " << labelName << std::endl;
	  
	  std::cout << "Energy = " << (*output)[0] << std::endl;
	  outputstring = labelName + "Energy, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*output)[0]);
	  std::cout << "EnergySigma = " << (*outputSD)[0] << std::endl;
	  outputstring = labelName + "EnergySigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*outputSD)[0]);

	  std::cout << "Entropy = " << (*output)[1] << std::endl;
	  outputstring = labelName + "Entropy, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*output)[1]);
	  std::cout << "EntropySigma = " << (*outputSD)[1] << std::endl;
	  outputstring = labelName + "EntropySigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*outputSD)[1]);

	  std::cout << "InverseDifferenceMoment = " << (*output)[2] << std::endl;
	  outputstring = labelName + "InverseDifferenceMoment, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*output)[2]);
	  std::cout << "InverseDifferenceMomentSigma = " << (*outputSD)[2] << std::endl;
	  outputstring = labelName + "InverseDifferenceMomentSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*outputSD)[2]);

	  std::cout << "Inertia = " << (*output)[3] << std::endl;
	  outputstring = labelName + "Inertia, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*output)[3]);
	  std::cout << "InertiaSigma = " << (*outputSD)[3] << std::endl;
	  outputstring = labelName + "InertiaSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*outputSD)[3]);

	  std::cout << "ClusterShade = " << (*output)[4] << std::endl;
	  outputstring = labelName + "ClusterShade, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*output)[4]);
	  std::cout << "ClusterShadeSigma = " << (*outputSD)[4] << std::endl;
	  outputstring = labelName + "ClusterShadeSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*outputSD)[4]);

	  std::cout << "ClusterProminence = " << (*output)[5] << std::endl;
	  outputstring = labelName + "ClusterProminence, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*output)[5]);
	  std::cout << "ClusterProminenceSigma = " << (*outputSD)[5] << std::endl;
	  outputstring = labelName + "ClusterProminenceSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*outputSD)[5]);
	  
	 
	  std::cout << std::endl << std::endl;

	  //  Get the run-length features for the original data 
	  runLengthFeaturesFilter->SetInput(imageReader->GetOutput());
	  runLengthFeaturesFilter->SetInsidePixelValue(labelValue);
	  runLengthFeaturesFilter->Update();
	  runLengthOutput = runLengthFeaturesFilter->GetFeatureMeans();
	  runLengthOutputSD = runLengthFeaturesFilter->GetFeatureStandardDeviations();

	  // Output the run-length features
	  // Run length features from itkHistogramToRunLengthFeaturesFilter.h
	  // typedef enum{ ShortRunEmphasis, LongRunEmphasis, GreyLevelNonuniformity, RunLengthNonuniformity,
	  //	   LowGreyLevelRunEmphasis, HighGreyLevelRunEmphasis, ShortRunLowGreyLevelEmphasis,
	  //	   ShortRunHighGreyLevelEmphasis, LongRunLowGreyLevelEmphasis, LongRunHighGreyLevelEmphasis }
	  std::cout << "Radius 1 Run-Length Features for: " << labelName << std::endl;

	  /*
	  std::cout << "ShortRunEmphasis = " << (*runLengthOutput)[0] << std::endl;
	  outputstring = labelName + "ShortRunEmphasis, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutput)[0]);
	  std::cout << "ShortRunEmphasisSigma = " << (*runLengthOutputSD)[0] << std::endl;
	  outputstring = labelName + "ShortRunEmphasisSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutputSD)[0]);

	  std::cout << "LongRunEmphasis = " << (*runLengthOutput)[1] << std::endl;
	  outputstring = labelName + "LongRunEmphasis, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutput)[1]);
	  std::cout << "LongRunEmphasisSigma = " << (*runLengthOutputSD)[1] << std::endl;
	  outputstring = labelName + "LongRunEmphasisSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutputSD)[1]);
	  */

	  std::cout << "GreyLevelNonuniformity = " << (*runLengthOutput)[2] << std::endl;
	  outputstring = labelName + "GreyLevelNonuniformity, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutput)[2]);
	  std::cout << "GreyLevelNonuniformitySigma = " << (*runLengthOutputSD)[2] << std::endl;
	  outputstring = labelName + "GreyLevelNonuniformitySigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutputSD)[2]);

	  std::cout << "RunLengthNonuniformity = " << (*runLengthOutput)[3] << std::endl;
	  outputstring = labelName + "RunLengthNonuniformity, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutput)[3]);
	  std::cout << "RunLengthNonuniformitySigma = " << (*runLengthOutputSD)[3] << std::endl;
	  outputstring = labelName + "RunLengthNonuniformitySigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutputSD)[3]);

	  /*
	  std::cout << "LowGreyLevelRunEmphasis = " << (*runLengthOutput)[4] << std::endl;
	  outputstring = labelName + "LowGreyLevelRunEmphasis, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutput)[4]);
	  std::cout << "LowGreyLevelRunEmphasisSigma = " << (*runLengthOutputSD)[4] << std::endl;
	  outputstring = labelName + "LowGreyLevelRunEmphasisSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutputSD)[4]);
	  */

	  std::cout << "HighGreyLevelRunEmphasis = " << (*runLengthOutput)[5] << std::endl;
	  outputstring = labelName + "HighGreyLevelRunEmphasis, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutput)[5]);
	  std::cout << "HighGreyLevelRunEmphasisSigma = " << (*runLengthOutputSD)[5] << std::endl;
	  outputstring = labelName + "HighGreyLevelRunEmphasisSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutputSD)[5]);

	  /*
	  std::cout << "ShortRunLowGreyLevelEmphasis = " << (*runLengthOutput)[6] << std::endl;
	  outputstring = labelName + "ShortRunLowGreyLevelEmphasis, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutput)[6]);
	  std::cout << "ShortRunLowGreyLevelEmphasisSigma = " << (*runLengthOutputSD)[6] << std::endl;
	  outputstring = labelName + "ShortRunLowGreyLevelEmphasisSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutputSD)[6]);
	  */

	  std::cout << "ShortRunHighGreyLevelEmphasis = " << (*runLengthOutput)[7] << std::endl;
	  outputstring = labelName + "ShortRunHighGreyLevelEmphasis, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutput)[7]);
	  std::cout << "ShortRunHighGreyLevelEmphasisSigma = " << (*runLengthOutputSD)[7] << std::endl;
	  outputstring = labelName + "ShortRunHighGreyLevelEmphasisSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutputSD)[7]);

	  /*
	  std::cout << "LongRunLowGreyLevelEmphasis = " << (*runLengthOutput)[8] << std::endl;
	  outputstring = labelName + "LongRunLowGreyLevelEmphasis, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutput)[8]);
	  std::cout << "LongRunLowGreyLevelEmphasisSigma = " << (*runLengthOutputSD)[8] << std::endl;
	  outputstring = labelName + "LongRunLowGreyLevelEmphasisSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutputSD)[8]);
	  */

	  std::cout << "LongRunHighGreyLevelEmphasis = " << (*runLengthOutput)[9] << std::endl;
	  outputstring = labelName + "LongRunHighGreyLevelEmphasis, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutput)[9]);
	  std::cout << "LongRunHighGreyLevelEmphasisSigma = " << (*runLengthOutputSD)[9] << std::endl;
	  outputstring = labelName + "LongRunHighGreyLevelEmphasisSigma, %f \n";
	  fprintf(outputFile, outputstring.c_str(), (*runLengthOutputSD)[9]);


	  // Now get the statistics for the Blurred image data using the current label
	  ////////////////////////////////////////////////////////////////////////////////////////////////
	  BinaryToStatisticsFilter->SetInput2(smoother->GetOutput());
	  BinaryToStatisticsFilter->SetInputForegroundValue(labelValue);
	  BinaryToStatisticsFilter->Update();

	  std::cout << "There is " << BinaryToStatisticsFilter->GetOutput()->GetNumberOfLabelObjects() << " object with statistics." << std::endl;
	  std::cout << "Statistics values from blurred image for " << labelName << std::endl;
	  std::cout << "Mean, Median, Skewness, Kurtosis, Standard Deviation " << std::endl;
	  for (k = 0; k < BinaryToStatisticsFilter->GetOutput()->GetNumberOfLabelObjects(); k++)
	  {
		  StatlabelObject = BinaryToStatisticsFilter->GetOutput()->GetNthLabelObject(k);
		  // Output the shape properties of the ith region
		  // Mean, median, skewness, kurtosis, sigma
		  std::cout << StatlabelObject->GetMean() << std::endl;
		  outputstring = labelName + "MeanGF, %f \n";
		  fprintf(outputFileGF, outputstring.c_str(), StatlabelObject->GetMean());
		  std::cout << StatlabelObject->GetMedian() << std::endl;
		  outputstring = labelName + "MedianGF, %f \n";
		  fprintf(outputFileGF, outputstring.c_str(), StatlabelObject->GetMedian());
		  std::cout << StatlabelObject->GetSkewness() << std::endl;
		  outputstring = labelName + "SkewnessGF, %f \n";
		  fprintf(outputFileGF, outputstring.c_str(), StatlabelObject->GetSkewness());
		  std::cout << StatlabelObject->GetKurtosis() << std::endl;
		  outputstring = labelName + "KurtosisGF, %f \n";
		  fprintf(outputFileGF, outputstring.c_str(), StatlabelObject->GetKurtosis());
		  std::cout << StatlabelObject->GetStandardDeviation() << std::endl;
		  outputstring = labelName + "SigmaGF, %f \n";
		  fprintf(outputFileGF, outputstring.c_str(), StatlabelObject->GetStandardDeviation());
		  std::cout << std::endl << std::endl;
	  }

	  // Now get the texture features for the blurred image
	  textureFilter->SetInput(smoother->GetOutput());
	  textureFilter->SetInsidePixelValue(labelValue);
	  textureFilter->Update();

	  output = textureFilter->GetFeatureMeans();
	  outputSD = textureFilter->GetFeatureStandardDeviations();

	  std::cout << "Radius 1 Texture Features (blurred image) for: " << labelName << std::endl;
	  //  enum TextureFeatureName {Energy, Entropy, Correlation,
	  //  InverseDifferenceMoment, Inertia, ClusterShade, ClusterProminence, HaralickCorrelation};

	  std::cout << "EnergyGF = " << (*output)[0] << std::endl;
	  outputstring = labelName + "EnergyGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*output)[0]);
	  std::cout << "EnergySigmaGF = " << (*outputSD)[0] << std::endl;
	  outputstring = labelName + "EnergySigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*outputSD)[0]);

	  std::cout << "EntropyGF = " << (*output)[1] << std::endl;
	  outputstring = labelName + "EntropyGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*output)[1]);
	  std::cout << "EntropySigmaGF = " << (*outputSD)[1] << std::endl;
	  outputstring = labelName + "EntropySigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*outputSD)[1]);
	  
	  std::cout << "InverseDifferenceMomentGF = " << (*output)[2] << std::endl;
	  outputstring = labelName + "InverseDifferenceMomentGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*output)[2]);
	  std::cout << "InverseDifferenceMomentSigmaGF = " << (*outputSD)[2] << std::endl;
	  outputstring = labelName + "InverseDifferenceMomentSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*outputSD)[2]);

	  std::cout << "InertiaGF = " << (*output)[3] << std::endl;
	  outputstring = labelName + "InertiaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*output)[3]);
	  std::cout << "InertiaSigmaGF = " << (*outputSD)[3] << std::endl;
	  outputstring = labelName + "InertiaSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*outputSD)[3]);

	  std::cout << "ClusterShadeGF = " << (*output)[4] << std::endl;
	  outputstring = labelName + "ClusterShadeGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*output)[4]);
	  std::cout << "ClusterShadeSigmaGF = " << (*outputSD)[4] << std::endl;
	  outputstring = labelName + "ClusterShadeSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*outputSD)[4]);

	  std::cout << "ClusterProminenceGF = " << (*output)[5] << std::endl;
	  outputstring = labelName + "ClusterProminenceGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*output)[5]);
	  std::cout << "ClusterProminenceSigmaGF = " << (*outputSD)[5] << std::endl;
	  outputstring = labelName + "ClusterProminenceSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*outputSD)[5]);

	  std::cout << std::endl << std::endl;

	  //  Get the run-length features for the Smoothed data 
	  runLengthFeaturesFilter->SetInput(smoother->GetOutput());
	  runLengthFeaturesFilter->SetInsidePixelValue(labelValue);
	  runLengthFeaturesFilter->Update();
	  runLengthOutput = runLengthFeaturesFilter->GetFeatureMeans();
	  runLengthOutputSD = runLengthFeaturesFilter->GetFeatureStandardDeviations();

	  // Output the run-length features
	  // Run length features from itkHistogramToRunLengthFeaturesFilter.h
	  // typedef enum{ ShortRunEmphasis, LongRunEmphasis, GreyLevelNonuniformity, RunLengthNonuniformity,
	  //	   LowGreyLevelRunEmphasis, HighGreyLevelRunEmphasis, ShortRunLowGreyLevelEmphasis,
	  //	   ShortRunHighGreyLevelEmphasis, LongRunLowGreyLevelEmphasis, LongRunHighGreyLevelEmphasis }
	  std::cout << "Radius 1 Run-Length Features (smoothed image) for: " << labelName << std::endl;

	  /*
	  std::cout << "ShortRunEmphasisGF = " << (*runLengthOutput)[0] << std::endl;
	  outputstring = labelName + "ShortRunEmphasisGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutput)[0]);
	  std::cout << "ShortRunEmphasisSigmaGF = " << (*runLengthOutputSD)[0] << std::endl;
	  outputstring = labelName + "ShortRunEmphasisSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutputSD)[0]);

	  std::cout << "LongRunEmphasisGF = " << (*runLengthOutput)[1] << std::endl;
	  outputstring = labelName + "LongRunEmphasisGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutput)[1]);
	  std::cout << "LongRunEmphasisSigmaGF = " << (*runLengthOutputSD)[1] << std::endl;
	  outputstring = labelName + "LongRunEmphasisSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutputSD)[1]);
	  */

	  std::cout << "GreyLevelNonuniformityGF = " << (*runLengthOutput)[2] << std::endl;
	  outputstring = labelName + "GreyLevelNonuniformityGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutput)[2]);
	  std::cout << "GreyLevelNonuniformitySigmaGF = " << (*runLengthOutputSD)[2] << std::endl;
	  outputstring = labelName + "GreyLevelNonuniformitySigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutputSD)[2]);

	  std::cout << "RunLengthNonuniformityGF = " << (*runLengthOutput)[3] << std::endl;
	  outputstring = labelName + "RunLengthNonuniformityGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutput)[3]);
	  std::cout << "RunLengthNonuniformitySigmaGF = " << (*runLengthOutputSD)[3] << std::endl;
	  outputstring = labelName + "RunLengthNonuniformitySigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutputSD)[3]);

	  /*
	  std::cout << "LowGreyLevelRunEmphasisGF = " << (*runLengthOutput)[4] << std::endl;
	  outputstring = labelName + "LowGreyLevelRunEmphasisGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutput)[4]);
	  std::cout << "LowGreyLevelRunEmphasisSigmaGF = " << (*runLengthOutputSD)[4] << std::endl;
	  outputstring = labelName + "LowGreyLevelRunEmphasisSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutputSD)[4]);
	  */

	  std::cout << "HighGreyLevelRunEmphasisGF = " << (*runLengthOutput)[5] << std::endl;
	  outputstring = labelName + "HighGreyLevelRunEmphasisGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutput)[5]);
	  std::cout << "HighGreyLevelRunEmphasisSigmaGF = " << (*runLengthOutputSD)[5] << std::endl;
	  outputstring = labelName + "HighGreyLevelRunEmphasisSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutputSD)[5]);

	  /*
	  std::cout << "ShortRunLowGreyLevelEmphasisGF = " << (*runLengthOutput)[6] << std::endl;
	  outputstring = labelName + "ShortRunLowGreyLevelEmphasisGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutput)[6]);
	  std::cout << "ShortRunLowGreyLevelEmphasisSigmaGF = " << (*runLengthOutputSD)[6] << std::endl;
	  outputstring = labelName + "ShortRunLowGreyLevelEmphasisSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutputSD)[6]);
	  */

	  std::cout << "ShortRunHighGreyLevelEmphasisGF = " << (*runLengthOutput)[7] << std::endl;
	  outputstring = labelName + "ShortRunHighGreyLevelEmphasisGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutput)[7]);
	  std::cout << "ShortRunHighGreyLevelEmphasisSigmaGF = " << (*runLengthOutputSD)[7] << std::endl;
	  outputstring = labelName + "ShortRunHighGreyLevelEmphasisSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutputSD)[7]);

	  /*
	  std::cout << "LongRunLowGreyLevelEmphasisGF = " << (*runLengthOutput)[8] << std::endl;
	  outputstring = labelName + "LongRunLowGreyLevelEmphasisGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutput)[8]);
	  std::cout << "LongRunLowGreyLevelEmphasisSigmaGF = " << (*runLengthOutputSD)[8] << std::endl;
	  outputstring = labelName + "LongRunLowGreyLevelEmphasisSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutputSD)[8]);
	  */

	  std::cout << "LongRunHighGreyLevelEmphasisGF = " << (*runLengthOutput)[9] << std::endl;
	  outputstring = labelName + "LongRunHighGreyLevelEmphasisGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutput)[9]);
	  std::cout << "LongRunHighGreyLevelEmphasisSigmaGF = " << (*runLengthOutputSD)[9] << std::endl;
	  outputstring = labelName + "LongRunHighGreyLevelEmphasisSigmaGF, %f \n";
	  fprintf(outputFileGF, outputstring.c_str(), (*runLengthOutputSD)[9]);

	  std::cout << std::endl << std::endl;

	  i++;
	  labelValuesIterator++;
	  labelNamesIterator++;
  }

  fclose(outputFile);
  fclose(outputFileGF);
 
  std::cout << std::endl;
  std::cout << std::endl;

}