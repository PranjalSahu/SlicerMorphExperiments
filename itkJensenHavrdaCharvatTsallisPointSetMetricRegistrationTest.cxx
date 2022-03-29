/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "itkJensenHavrdaCharvatTsallisPointSetToPointSetMetricv4.h"
#include "itkGradientDescentOptimizerv4.h"
#include "itkTransform.h"
#include "itkAffineTransform.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"
#include "itkCommand.h"
#include "itkMesh.h"
#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"
#include "itkDisplacementFieldTransform.h"

#include <fstream>

template <typename TFilter>
class itkJensenHavrdaCharvatTsallisPointSetMetricRegistrationTestCommandIterationUpdate : public itk::Command
{
public:
  using Self = itkJensenHavrdaCharvatTsallisPointSetMetricRegistrationTestCommandIterationUpdate;

  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

protected:
  itkJensenHavrdaCharvatTsallisPointSetMetricRegistrationTestCommandIterationUpdate() = default;

public:
  void
  Execute(itk::Object * caller, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)caller, event);
  }

  void
  Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    if (typeid(event) != typeid(itk::IterationEvent))
    {
      return;
    }
    const auto * optimizer = dynamic_cast<const TFilter *>(object);

    if (!optimizer)
    {
      itkGenericExceptionMacro("Error dynamic_cast failed");
    }
    std::cout << "It: " << optimizer->GetCurrentIteration() << " metric value: " << optimizer->GetCurrentMetricValue();
    std::cout << std::endl;
  }
};

int
itkJensenHavrdaCharvatTsallisPointSetMetricRegistrationTest(int argc, char * argv[])
{

  const unsigned int Dimension = 3;
  
  using MeshType = itk::Mesh<double, Dimension>;
  using PointsContainerPointer = MeshType::PointsContainerPointer;
  
  using ReaderType = itk::MeshFileReader<MeshType>;
  using WriterType = itk::MeshFileWriter<MeshType>;

  /*
  Initialize fixed mesh polydata reader
  */
  ReaderType::Pointer fixedPolyDataReader = ReaderType::New();
  ReaderType::Pointer movingPolyDataReader = ReaderType::New();

  fixedPolyDataReader->SetFileName(argv[1]);
  movingPolyDataReader->SetFileName(argv[2]);

  try
  {
    fixedPolyDataReader->Update();
    movingPolyDataReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Error during Fixed Mesh Update() " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }

  MeshType::Pointer fixedMesh = fixedPolyDataReader->GetOutput();
  MeshType::Pointer movingMesh = movingPolyDataReader->GetOutput();

  unsigned int numberOfIterations = std::stoi(argv[3]);
  
  using PointSetType = itk::PointSet<double, Dimension>;
  using PointType = PointSetType::PointType;

  // Create PointSets from Mesh
  auto fixedPoints = PointSetType::New();
  fixedPoints->Initialize();
  fixedPoints->SetPoints(fixedMesh->GetPoints());

  auto movingPoints = PointSetType::New();
  movingPoints->Initialize();
  movingPoints->SetPoints(movingMesh->GetPoints());


  std::cout << "Number of fixed points " << fixedPoints->GetNumberOfPoints() << std::endl;
  std::cout << "Number of moving points " << movingPoints->GetNumberOfPoints() << std::endl;
  
  using PixelType = double;
  using FixedImageType = itk::Image<PixelType, Dimension>;
  using MovingImageType = itk::Image<PixelType, Dimension>;


  FixedImageType::SizeType fixedImageSize;
  FixedImageType::PointType fixedImageOrigin;
  FixedImageType::DirectionType fixedImageDirection;
  FixedImageType::SpacingType fixedImageSpacing;


  // two ellipses, one rotated slightly
  /*
    // Having trouble with these, as soon as there's a slight rotation added.
    unsigned long count = 0;
    for( float theta = 0; theta < 2.0 * itk::Math::pi; theta += 0.1 )
      {
      float radius = 100.0;
      PointType fixedPoint;
      fixedPoint[0] = 2 * radius * std::cos( theta );
      fixedPoint[1] = radius * std::sin( theta );
      fixedPoints->SetPoint( count, fixedPoint );

      PointType movingPoint;
      movingPoint[0] = 2 * radius * std::cos( theta + (0.02 * itk::Math::pi) ) + 2.0;
      movingPoint[1] = radius * std::sin( theta + (0.02 * itk::Math::pi) ) + 2.0;
      movingPoints->SetPoint( count, movingPoint );

      count++;
      }
  */


  // using AffineTransformType = itk::AffineTransform<double, Dimension>;
  // auto transform = AffineTransformType::New();
  // transform->SetIdentity();

  using PointIdentifier = MeshType::PointIdentifier;
  using BoundingBoxType = itk::BoundingBox<PointIdentifier, Dimension>;
  BoundingBoxType::Pointer boundingBox = BoundingBoxType::New();
  PointsContainerPointer points = movingMesh->GetPoints();
  boundingBox->SetPoints(points);
  boundingBox->ComputeBoundingBox();
  typename BoundingBoxType::PointType minBounds = boundingBox->GetMinimum();
  typename BoundingBoxType::PointType maxBounds = boundingBox->GetMaximum();

  int imageDiagonal = 400;
  double spacing = sqrt(boundingBox->GetDiagonalLength2()) / imageDiagonal;
  auto diff = maxBounds - minBounds;
  fixedImageSize[0] = ceil( 1.2 * diff[0] / spacing );
  fixedImageSize[1] = ceil( 1.2 * diff[1] / spacing );
  fixedImageSize[2] = ceil( 1.2 * diff[2] / spacing );
  fixedImageOrigin[0] = minBounds[0] - 0.1*diff[0];
  fixedImageOrigin[1] = minBounds[1] - 0.1*diff[1];
  fixedImageOrigin[2] = minBounds[2] - 0.1*diff[2];
  fixedImageDirection.SetIdentity();
  fixedImageSpacing.Fill( spacing );

  FixedImageType::Pointer fixedImage = FixedImageType::New();
  fixedImage->SetRegions( fixedImageSize );
  fixedImage->SetOrigin( fixedImageOrigin );
  fixedImage->SetDirection( fixedImageDirection );
  fixedImage->SetSpacing( fixedImageSpacing );
  fixedImage->Allocate();


  using TransformType = itk::DisplacementFieldTransform<double, Dimension>;
  auto transform = TransformType::New();
  using  DisplacementFieldType = TransformType::DisplacementFieldType;
  DisplacementFieldType::Pointer field = DisplacementFieldType::New();
  field->SetRegions( fixedImageSize );
  field->SetOrigin( fixedImageOrigin );
  field->SetDirection( fixedImageDirection );
  field->SetSpacing( fixedImageSpacing );
  field->Allocate();
  transform->SetDisplacementField(field);


  // Instantiate the metric
  using PointSetMetricType = itk::JensenHavrdaCharvatTsallisPointSetToPointSetMetricv4<PointSetType>;
  auto metric = PointSetMetricType::New();
  metric->SetFixedPointSet(fixedPoints);
  metric->SetMovingPointSet(movingPoints);
  metric->SetPointSetSigma(1.0);
  metric->SetKernelSigma(10.0);
  metric->SetUseAnisotropicCovariances(false);
  metric->SetCovarianceKNeighborhood(5);
  metric->SetEvaluationKNeighborhood(10);
  metric->SetMovingTransform(transform);
  metric->SetAlpha(1.1);
  metric->Initialize();

  // scales estimator
  using RegistrationParameterScalesFromShiftType =
    itk::RegistrationParameterScalesFromPhysicalShift<PointSetMetricType>;
  RegistrationParameterScalesFromShiftType::Pointer shiftScaleEstimator =
    RegistrationParameterScalesFromShiftType::New();
  shiftScaleEstimator->SetMetric(metric);
  // needed with pointset metrics
  shiftScaleEstimator->SetVirtualDomainPointSet(metric->GetVirtualTransformedPointSet());

  // optimizer
  using OptimizerType = itk::GradientDescentOptimizerv4;
  auto optimizer = OptimizerType::New();
  optimizer->SetMetric(metric);
  optimizer->SetNumberOfIterations(numberOfIterations);
  optimizer->SetScalesEstimator(shiftScaleEstimator);
  optimizer->SetMaximumStepSizeInPhysicalUnits(std::stof(argv[4]));

  using CommandType = itkJensenHavrdaCharvatTsallisPointSetMetricRegistrationTestCommandIterationUpdate<OptimizerType>;
  auto observer = CommandType::New();
  optimizer->AddObserver(itk::IterationEvent(), observer);

  optimizer->SetMinimumConvergenceValue(0.0);
  optimizer->SetConvergenceWindowSize(100);
  optimizer->StartOptimization();

  std::cout << "numberOfIterations: " << numberOfIterations << std::endl;
  // std::cout << "Moving-source final value: " << optimizer->GetCurrentMetricValue() << std::endl;
  // std::cout << "Moving-source final position: " << optimizer->GetCurrentPosition() << std::endl;
  // std::cout << "Optimizer scales: " << optimizer->GetScales() << std::endl;
  // std::cout << "Optimizer learning rate: " << optimizer->GetLearningRate() << std::endl;

  // applying the resultant transform to moving points and verify result
  //std::cout << "Fixed\tMoving\tMovingTransformed\tFixedTransformed\tDiff" << std::endl;
  bool                                             passed = true;
  PointType::ValueType                             tolerance = 1e-2;
  TransformType::InverseTransformBasePointer movingInverse = metric->GetMovingTransform();//->GetInverseTransform();
  //TransformType::InverseTransformBasePointer fixedInverse = metric->GetFixedTransform()->GetInverseTransform();
  
  //std::cout << "Got the moving Inverse " << movingInverse << std::endl;

  for (unsigned int n = 0; n < movingPoints->GetNumberOfPoints(); n++)
  {
    movingMesh->SetPoint(n, movingInverse->TransformPoint(movingPoints->GetPoint(n)));
  }


  //   // compare the points in virtual domain
  //   PointType transformedMovingPoint = movingInverse->TransformPoint(movingPoints->GetPoint(n));
  //   movingPoints->SetPoint(n, )
  //   // PointType transformedFixedPoint = fixedInverse->TransformPoint(fixedPoints->GetPoint(n));
  //   // PointType difference;
  //   // difference[0] = transformedMovingPoint[0] - transformedFixedPoint[0];
  //   // difference[1] = transformedMovingPoint[1] - transformedFixedPoint[1];
  //   // std::cout << fixedPoints->GetPoint(n) << "\t" << movingPoints->GetPoint(n) << "\t" << transformedMovingPoint << "\t"
  //   //           << transformedFixedPoint << "\t" << difference << std::endl;
  //   // if (itk::Math::abs(difference[0]) > tolerance || itk::Math::abs(difference[1]) > tolerance)
  //   // {
  //   //   passed = false;
  //   // }
  // }
  // if (!passed)
  // {
  //   std::cerr << "Results do not match truth within tolerance." << std::endl;
  //   return EXIT_FAILURE;
  // }

  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(movingMesh);
  writer->SetFileName("affineMovingMesh.vtk");
  writer->Update();


  return EXIT_SUCCESS;
}
